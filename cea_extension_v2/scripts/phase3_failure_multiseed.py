# -*- coding: utf-8 -*-
"""Phase 3 (CEA high-impact): failure-condition analysis across matched seeds - inference only.

Upgrades the single-seed (seed 42) overlap / lighting analyses to matched-seed multi-seed using the
already-trained checkpoints. No training, no threshold tuning, no re-selection.

Two coherent rule implementations are computed side by side, because the locked TEXT and the
extension's scripts disagree at the boundaries (author-acceptance finding #13):

  variant A = "locked text / manuscript implementation"  (PRIMARY)
      levels: Low r < 0.3 | Medium 0.3 <= r <= 0.6 | High r > 0.6
      matching: IoU >= 0.5, greedy in prediction order, first-wins on ties
      -> exactly scripts/occlusion_analysis.py (the file behind the published overlap numbers)

  variant B = "extension implementation"                 (SENSITIVITY)
      levels: Low r < 0.3 | Medium 0.3 <= r < 0.6 | High r >= 0.6
      matching: IoU > 0.5
      -> what diagnose_overlap_threshold_variants.py / phase13* used

Overlap levels are a **geometric bounding-box overlap proxy**, never true visible occlusion.
Lighting buckets are the official OinkTrack scene names (D / DN / N), never regrouped.

Performance: the GT side (boxes + both level assignments) is model-independent and is cached once per
process; every IoU/matching step is vectorised with numpy and reproduces IEEE-double arithmetic
identical to the manuscript's scalar `iou` (same formula and operation order).

Disk discipline: prediction exports go to a scratch directory and are deleted immediately after each
checkpoint's metrics are computed; only CSV/JSON/MD are kept.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

PJ = Path(r"<PIG_PROJECT>")
EXP = PJ / "experiments" / "cea_extension_v2"
OUT = EXP / "16_failure_multiseed"
SCRATCH = OUT / "_scratch_predictions"
OINK = PJ / "data" / "oinktrack_detection"
sys.path.insert(0, str(PJ / "scripts"))
from occlusion_analysis import overlap_ratio  # noqa: E402  (manuscript proxy, unchanged)

CONF, IOU_THR = 0.25, 0.5
LEVELS = ("Low", "Medium", "High")
LIGHTS = ("D", "DN", "N")
# the exact dataset strings this script writes into its CSV outputs; used to replace a
# dataset's rows on re-run instead of appending a second generation of the same measurement
DATASET_LABELS = ("PigDetect_official_test_250", "OinkTrack_official_test_8569")
OINK_TEST = ["C2D-1", "C1DN-1", "C1D-2", "C1N-2", "C2N-1", "C2N-4", "C2DN-1"]
PILOT_SEEDS = [1, 21, 42, 101, 113]
ALL_SEEDS = [1, 7, 21, 42, 100, 101, 103, 107, 109, 113]
VARIANTS = ("A_locked_text_manuscript", "B_extension_strict")


def registry() -> dict:
    return {r["run_id"]: r for r in csv.DictReader(
        (EXP / "run_registry.csv").open(encoding="utf-8-sig"))}


def ckpt_for(reg: dict, arm: str, seed: int, rule: str):
    if seed in (42, 1, 7, 21, 100):
        rid = f"lc_{arm}_seed{seed}" + ("-2" if (arm == "E5" and seed == 7) else "")
    else:
        rid = f"{arm}_seed{seed}"
    row = reg.get(rid)
    if not row:
        return None
    p = Path(row["run_dir"]) / "weights" / f"{rule}.pt"
    return p if p.exists() else None


# ------------------------------------------------------------------ vectorised geometry
def iou_matrix(preds, gts) -> np.ndarray:
    """Same formula and operation order as the manuscript's scalar `iou`."""
    n, m = len(preds), len(gts)
    if n == 0 or m == 0:
        return np.zeros((n, m))
    p = np.asarray(preds, dtype=float)
    g = np.asarray(gts, dtype=float)
    ix1 = np.maximum(p[:, None, 0], g[None, :, 0])
    iy1 = np.maximum(p[:, None, 1], g[None, :, 1])
    ix2 = np.minimum(p[:, None, 2], g[None, :, 2])
    iy2 = np.minimum(p[:, None, 3], g[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    ap = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    ua = ap[:, None] + ag[None, :] - inter
    out = np.zeros_like(inter)
    nz = ua > 0
    out[nz] = inter[nz] / ua[nz]
    return out


def match(gxy, pxy, inclusive: bool):
    """Greedy one-to-one matching in prediction order, first-wins on ties.

    Returns (matched, tp) where `matched[i]` is True when GT box i was matched - the same semantics
    as the manuscript's own `matched` list (this was inverted in a first draft of this function and
    caught by the equivalence test against the scalar implementation).
    """
    if not gxy or not pxy:
        return [False] * len(gxy), 0
    M = iou_matrix(pxy, gxy)
    unmatched = np.ones(len(gxy), dtype=bool)
    matched = np.zeros(len(gxy), dtype=bool)
    tp = 0
    for i in range(M.shape[0]):
        row = M[i].copy()
        row[~unmatched] = -1.0
        if inclusive:
            row[row < IOU_THR] = -1.0
            ok = row.max() >= IOU_THR
        else:
            row[row <= IOU_THR] = -1.0
            ok = row.max() > IOU_THR
        if ok:
            gi = int(np.argmax(row))          # first maximum == first-wins on ties
            unmatched[gi] = False
            matched[gi] = True
            tp += 1
    return matched.tolist(), tp


def levels(boxes_xyxy, variant: str) -> list:
    """GT overlap level from the manuscript's geometric proxy (thresholds per variant)."""
    out = []
    for i, b in enumerate(boxes_xyxy):
        r = overlap_ratio(b, [o for j, o in enumerate(boxes_xyxy) if j != i])
        if r < 0.3:
            out.append("Low")
        elif variant == "A_locked_text_manuscript":
            out.append("Medium" if r <= 0.6 else "High")
        else:
            out.append("Medium" if r < 0.6 else "High")
    return out


# ------------------------------------------------------------------------------- ground truth
def pigdetect_gt() -> dict:
    """stem -> [[x, y, w, h] pixels] from the manuscript's official COCO ground truth."""
    coco = json.loads((PJ / "dataset" / "test_coco.json").read_text(encoding="utf-8"))
    stem_of = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    gt = {s: [] for s in stem_of.values()}
    for a in coco["annotations"]:
        gt[stem_of[a["image_id"]]].append(list(a["bbox"]))
    return gt


def oink_gt(seq: str) -> dict:
    lbl_dir = OINK / "test" / "labels" / seq
    gt = {}
    for lbl in sorted(lbl_dir.glob("*.txt")):
        boxes = []
        for ln in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = ln.split()
            if len(p) < 5:
                continue
            _, xc, yc, w, h = (float(v) for v in p[:5])
            boxes.append([(xc - w / 2) * 1280, (yc - h / 2) * 720, w * 1280, h * 720])
        gt[int(lbl.stem)] = boxes
    return gt


def lighting_of(seq: str) -> str:
    return seq.split("-")[0][2:]


def xyxy(boxes):
    return [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes]


# ----------------------------------------------------------------------------------- caches
def build_pig_cache() -> dict:
    gt = pigdetect_gt()
    cache = {}
    for variant in VARIANTS:
        per = []
        for stem, boxes in gt.items():
            g = xyxy(boxes)
            per.append((stem, g, levels(g, variant)))
        cache[variant] = per
    return cache


def build_oink_cache() -> dict:
    cache = {}
    for variant in VARIANTS:
        per_seq = {}
        for seq in OINK_TEST:
            gt = oink_gt(seq)
            per = []
            for fidx, boxes in gt.items():
                if not boxes:
                    continue
                g = xyxy(boxes)
                per.append((fidx, g, levels(g, variant)))
            per_seq[seq] = per
        cache[variant] = per_seq
    return cache


# ------------------------------------------------------------------------------ evaluation
def evaluate_frames(frame_entries, preds_by_key, variant: str, inclusive: bool) -> tuple:
    stat = {lv: {"instances": 0, "hits": 0} for lv in LEVELS}
    light = {"gt": 0, "tp": 0, "fp": 0, "images": 0}
    for key, g, lv in frame_entries:
        pxy = xyxy(preds_by_key.get(key, []))
        live, tp = match(g, pxy, inclusive)
        for i, l in enumerate(lv):
            stat[l]["instances"] += 1
            stat[l]["hits"] += int(live[i])
        light["gt"] += len(g)
        light["tp"] += tp
        light["fp"] += len(pxy) - tp
        light["images"] += 1
    return stat, light


def export_pigdetect(ckpt: Path) -> dict:
    from ultralytics import YOLO
    YOLO(str(ckpt)).val(data=str(PJ / "dataset" / "PigDetect_lc" / "pig_lc.yaml"), split="test",
                        imgsz=640, batch=8, workers=0, conf=CONF, iou=0.7, verbose=False,
                        plots=False, save_json=True, project=str(SCRATCH), name="pigdetect",
                        exist_ok=True)
    p = SCRATCH / "pigdetect" / "predictions.json"
    recs = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    out = defaultdict(list)
    for r in recs:
        out[Path(str(r.get("file_name", ""))).stem].append(r["bbox"])
    return out


def export_oink(ckpt: Path) -> dict:
    """One val() per sequence so the frame mapping is unambiguous (anomaly #20)."""
    from ultralytics import YOLO
    m = YOLO(str(ckpt))
    out = {}
    for seq in OINK_TEST:
        y = SCRATCH / f"oink_{seq.replace('-', '_')}.yaml"
        y.write_text(f"path: {OINK.as_posix()}\ntrain: test/images/{seq}\nval: test/images/{seq}\n"
                     f"test: test/images/{seq}\nnames:\n  0: pig\n", encoding="utf-8")
        m.val(data=str(y), split="test", imgsz=640, batch=8, workers=0, conf=CONF, iou=0.7,
              verbose=False, plots=False, save_json=True, project=str(SCRATCH), name=f"oink_{seq}",
              exist_ok=True)
        pf = SCRATCH / f"oink_{seq}" / "predictions.json"
        recs = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else []
        per = defaultdict(list)
        for r in recs:
            per[int(r["image_id"])].append(r["bbox"])
        out[seq] = per
        if pf.exists():
            pf.unlink()
    return out


# ---------------------------------------------------------------------------- statistics
def exact_p(deltas: list) -> dict:
    n = len(deltas)
    T = sum(deltas) / n
    obs = abs(T)
    greater = tie = one = 0
    for signs in itertools.product([1, -1], repeat=n):
        m = sum(s * d for s, d in zip(signs, deltas)) / n
        if abs(m) > obs:
            greater += 1
        elif abs(m) == obs:
            tie += 1
        if m >= obs:
            one += 1
    ge = greater + tie
    return {"n": n, "configurations": 2 ** n, "observed_mean_exact": str(T),
            "greater": greater, "tie": tie, "greater_or_equal": ge,
            "exact_p_two_sided_fraction": f"{ge}/{2 ** n}",
            "exact_p_two_sided_decimal": str(Fraction(ge, 2 ** n)),
            "exact_p_one_sided_decimal": str(Fraction(one, 2 ** n)),
            "arithmetic": "exact rational (recalls are hits/instances)"}


def bootstrap_ci(deltas: list, r: int = 100_000, seed: int = 20260916) -> dict:
    d = np.asarray([float(x) for x in deltas], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(r, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return {"resamples": r, "rng_seed": seed, "ci_low": lo, "ci_high": hi,
            "contains_zero": bool(lo <= 0 <= hi)}


def summarise(pairs: list) -> dict:
    deltas = [p["e5"] - p["e0"] for p in pairs]
    f = [float(d) for d in deltas]
    support = min(min(p["e0_support"], p["e5_support"]) for p in pairs)
    return {"n_seeds": len(deltas), "seeds": [p["seed"] for p in pairs],
            "deltas_exact": [str(d) for d in deltas], "deltas_decimal": [round(x, 6) for x in f],
            "mean_delta": statistics.mean(f), "median_delta": statistics.median(f),
            "sd_delta": statistics.stdev(f) if len(f) > 1 else 0.0,
            "min_delta": min(f), "max_delta": max(f),
            "positive": sum(1 for x in f if x > 0), "negative": sum(1 for x in f if x < 0),
            "zero": sum(1 for x in f if x == 0),
            "bootstrap": bootstrap_ci(deltas), "signflip": exact_p(deltas),
            "support_min_instances": support, "low_support_warning": support < 100}


# ------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="pilot", choices=["pilot", "all"])
    ap.add_argument("--datasets", default="pigdetect,oinktrack")
    ap.add_argument("--rules", default="best,last")
    args = ap.parse_args()
    seeds = PILOT_SEEDS if args.seeds == "pilot" else ALL_SEEDS
    rules = [r for r in args.rules.split(",") if r]
    datasets = [d for d in args.datasets.split(",") if d]

    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    reg = registry()

    t0 = time.time()
    pig_cache = build_pig_cache() if "pigdetect" in datasets else None
    oink_cache = build_oink_cache() if "oinktrack" in datasets else None
    print(f"  GT caches built in {time.time() - t0:.1f} s", flush=True)

    overlap_rows, lighting_rows, raw = [], [], {}
    for dataset in datasets:
        for rule in rules:
            for arm in ("E0", "E5"):
                for seed in seeds:
                    ck = ckpt_for(reg, arm, seed, rule)
                    if ck is None:
                        raw[(dataset, rule, arm, seed)] = None
                        print(f"  [{dataset}] {arm} seed {seed} {rule}: checkpoint unavailable",
                              flush=True)
                        continue
                    shutil.rmtree(SCRATCH, ignore_errors=True)
                    SCRATCH.mkdir(parents=True, exist_ok=True)
                    t1 = time.time()
                    if dataset == "pigdetect":
                        preds = export_pigdetect(ck)
                        raw[(dataset, rule, arm, seed)] = {}
                        for variant in VARIANTS:
                            stat, _ = evaluate_frames(pig_cache[variant], preds, variant,
                                                      variant.startswith("A"))
                            raw[(dataset, rule, arm, seed)][variant] = stat
                            for lv in LEVELS:
                                v = stat[lv]
                                overlap_rows.append({
                                    "dataset": "PigDetect_official_test_250", "seed": seed,
                                    "arm": arm, "checkpoint_rule": rule, "variant": variant,
                                    "condition": lv, "instances": v["instances"], "hits": v["hits"],
                                    "recall": (v["hits"] / v["instances"]) if v["instances"] else ""})
                    else:
                        per_seq = export_oink(ck)
                        raw[(dataset, rule, arm, seed)] = {}
                        for variant in VARIANTS:
                            agg = {lv: {"instances": 0, "hits": 0} for lv in LEVELS}
                            light = {k: {"gt": 0, "tp": 0, "fp": 0, "images": 0} for k in LIGHTS}
                            for seq in OINK_TEST:
                                stat, lt = evaluate_frames(oink_cache[variant][seq],
                                                           per_seq.get(seq, {}), variant,
                                                           variant.startswith("A"))
                                for lv in LEVELS:
                                    agg[lv]["instances"] += stat[lv]["instances"]
                                    agg[lv]["hits"] += stat[lv]["hits"]
                                L = light[lighting_of(seq)]
                                for k in ("gt", "tp", "fp", "images"):
                                    L[k] += lt[k]
                            raw[(dataset, rule, arm, seed)][variant] = {"overlap": agg,
                                                                        "lighting": light}
                            for lv in LEVELS:
                                v = agg[lv]
                                overlap_rows.append({
                                    "dataset": "OinkTrack_official_test_8569", "seed": seed,
                                    "arm": arm, "checkpoint_rule": rule, "variant": variant,
                                    "condition": lv, "instances": v["instances"],
                                    "hits": v["hits"],
                                    "recall": (v["hits"] / v["instances"]) if v["instances"] else ""})
                            for k in LIGHTS:
                                v = light[k]
                                lighting_rows.append({
                                    "dataset": "OinkTrack_official_test_8569", "seed": seed,
                                    "arm": arm, "checkpoint_rule": rule, "variant": variant,
                                    "condition": k, "images": v["images"], "gt_boxes": v["gt"],
                                    "tp": v["tp"], "fp": v["fp"],
                                    "recall": (v["tp"] / v["gt"]) if v["gt"] else "",
                                    "precision": (v["tp"] / (v["tp"] + v["fp"]))
                                    if (v["tp"] + v["fp"]) else ""})
                    print(f"  [{dataset}] {arm} seed {seed} {rule}: {time.time() - t1:.1f} s",
                          flush=True)
    shutil.rmtree(SCRATCH, ignore_errors=True)

    stats = {"timestamp": datetime.now().isoformat(timespec="seconds"), "seeds": seeds,
             "rules": rules, "datasets": datasets,
             "variants": {"A_locked_text_manuscript": {
                 "levels": "Low r<0.3 | Medium 0.3<=r<=0.6 | High r>0.6",
                 "matching": "IoU >= 0.5, greedy in prediction order, first-wins on ties",
                 "status": "PRIMARY - matches the locked text and scripts/occlusion_analysis.py"},
                 "B_extension_strict": {
                 "levels": "Low r<0.3 | Medium 0.3<=r<0.6 | High r>=0.6",
                 "matching": "IoU > 0.5",
                 "status": "SENSITIVITY - what diagnose_overlap_threshold_variants/phase13 used"}},
             "rules_note": {"overlap": "geometric bounding-box overlap proxy, NOT true occlusion",
                            "lighting": "official OinkTrack scene names D/DN/N, never regrouped",
                            "statistics": {"bootstrap_resamples": 100000, "rng_seed": 20260916,
                                           "signflip": "exact 2^n enumeration, T = mean paired delta, "
                                                       "ties included, exact rational arithmetic"}},
             "summaries": {}, "runtime_seconds": round(time.time() - t0, 1)}

    for dataset in datasets:
        for variant in VARIANTS:
            for rule in rules:
                for lv in LEVELS:
                    pairs = []
                    for seed in seeds:
                        a = raw.get((dataset, rule, "E0", seed))
                        b = raw.get((dataset, rule, "E5", seed))
                        if not a or not b:
                            continue
                        ao = a[variant]["overlap"] if dataset == "oinktrack" else a[variant]
                        bo = b[variant]["overlap"] if dataset == "oinktrack" else b[variant]
                        if not ao[lv]["instances"] or not bo[lv]["instances"]:
                            continue
                        pairs.append({"seed": seed,
                                      "e0": Fraction(ao[lv]["hits"], ao[lv]["instances"]),
                                      "e5": Fraction(bo[lv]["hits"], bo[lv]["instances"]),
                                      "e0_support": ao[lv]["instances"],
                                      "e5_support": bo[lv]["instances"]})
                    if pairs:
                        stats["summaries"][f"{dataset}|overlap|{lv}|{rule}|{variant}"] = \
                            summarise(pairs)
                if dataset != "oinktrack":
                    continue
                for k in LIGHTS:
                    pairs = []
                    for seed in seeds:
                        a = raw.get((dataset, rule, "E0", seed))
                        b = raw.get((dataset, rule, "E5", seed))
                        if not a or not b:
                            continue
                        la, lb = a[variant]["lighting"][k], b[variant]["lighting"][k]
                        if not la["gt"] or not lb["gt"]:
                            continue
                        pairs.append({"seed": seed, "e0": Fraction(la["tp"], la["gt"]),
                                      "e5": Fraction(lb["tp"], lb["gt"]),
                                      "e0_support": la["gt"], "e5_support": lb["gt"]})
                    if pairs:
                        stats["summaries"][f"{dataset}|lighting|{k}|{rule}|{variant}"] = \
                            summarise(pairs)

    def write_rows(path: Path, new: list) -> None:
        """Merge with rows of *other* datasets, keeping every column.

        Two defects were found in the first version and are fixed here:
        (a) the header was taken from the first row, so when a pre-existing file had
            fewer columns the new rows lost their `variant` column silently;
        (b) the previous-run filter compared shorthand dataset names ("pigdetect")
            against the row values ("PigDetect_official_test_250"), so superseded rows
            from an earlier, differently-derived run were never replaced and the file
            ended up holding two generations of the same measurement.
        The filter now keys on the dataset string that this script actually writes, so
        re-running a dataset replaces its rows instead of appending to them.
        """
        own = DATASET_LABELS
        old = []
        if path.exists():
            old = [dict(r) for r in csv.DictReader(path.open(encoding="utf-8-sig"))
                   if r.get("dataset") not in own]
        rows = old + [{k: str(v) for k, v in r.items()} for r in new]
        if not rows:
            return
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(rows)

    if overlap_rows:
        write_rows(OUT / "overlap_per_seed.csv", overlap_rows)
    if lighting_rows:
        write_rows(OUT / "lighting_per_seed.csv", lighting_rows)

    stats_path = OUT / "failure_multiseed_statistics.json"
    if stats_path.exists():
        try:
            old_stats = json.loads(stats_path.read_text(encoding="utf-8"))
            for k, v in old_stats.get("summaries", {}).items():
                if k.split("|")[0] not in datasets:
                    stats["summaries"].setdefault(k, v)
            stats["previous_runs"] = old_stats.get("previous_runs", []) + [
                {"timestamp": old_stats.get("timestamp"), "datasets": old_stats.get("datasets"),
                 "seeds": old_stats.get("seeds"), "rules": old_stats.get("rules")}]
        except Exception:                                               # noqa: BLE001
            pass

    def write_summary(path: Path, kind: str) -> None:
        old = []
        if path.exists():
            old = [dict(r) for r in csv.DictReader(path.open(encoding="utf-8-sig"))
                   if r.get("dataset") not in datasets]
        rows = old
        for key, s in stats["summaries"].items():
            ds, k2, cond, rule, variant = key.split("|")
            if k2 != kind or ds not in datasets:
                continue
            rows.append({"dataset": ds, "variant": variant, "condition": cond,
                         "checkpoint_rule": rule, "n": s["n_seeds"],
                         "mean_delta": s["mean_delta"], "median_delta": s["median_delta"],
                         "sd_delta": s["sd_delta"], "min": s["min_delta"], "max": s["max_delta"],
                         "pos": s["positive"], "neg": s["negative"], "zero": s["zero"],
                         "ci_low": s["bootstrap"]["ci_low"], "ci_high": s["bootstrap"]["ci_high"],
                         "ci_contains_zero": s["bootstrap"]["contains_zero"],
                         "exact_p2": s["signflip"]["exact_p_two_sided_decimal"],
                         "exact_p2_fraction": s["signflip"]["exact_p_two_sided_fraction"],
                         "ties": s["signflip"]["tie"],
                         "low_support_warning": s["low_support_warning"],
                         "support_min_instances": s["support_min_instances"]})
        fields = ["dataset", "variant", "condition", "checkpoint_rule", "n", "mean_delta",
                  "median_delta", "sd_delta", "min", "max", "pos", "neg", "zero", "ci_low",
                  "ci_high", "ci_contains_zero", "exact_p2", "exact_p2_fraction", "ties",
                  "low_support_warning", "support_min_instances"]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})

    write_summary(OUT / "overlap_summary.csv", "overlap")
    write_summary(OUT / "lighting_summary.csv", "lighting")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# Failure-condition analysis across matched seeds (Phase 3)", "",
          f"Run: {stats['timestamp']}  ",
          f"Seeds: {seeds}  ",
          f"Checkpoint rules: {rules}  ",
          f"Datasets: {datasets}  ",
          f"Inference-only runtime: {stats['runtime_seconds'] / 60:.1f} min", "",
          "Overlap levels use the **geometric bounding-box overlap proxy** and must not be described "
          "as true visible occlusion. Lighting buckets are the official OinkTrack scene names "
          "(D / DN / N), never regrouped. Matching: confidence >= 0.25, greedy one-to-one. Paired "
          "deltas are E5 - E0 per matched seed in **exact rational arithmetic** (recall = "
          "hits / instances), so the sign-flip enumeration contains no floating-point step.", "",
          "Two boundary conventions are reported because the locked text and the extension's scripts "
          "disagree (author-acceptance finding #13):", "",
          "| variant | levels | matching | status |", "|---|---|---|---|",
          "| A_locked_text_manuscript | Low r<0.3, Medium 0.3<=r<=0.6, High r>0.6 | IoU >= 0.5 | "
          "**PRIMARY** (locked text + `scripts/occlusion_analysis.py`) |",
          "| B_extension_strict | Low r<0.3, Medium 0.3<=r<0.6, High r>=0.6 | IoU > 0.5 | "
          "sensitivity |",
          "", "| dataset | variant | condition | rule | n | mean delta | median | SD | min | max | "
              "+/-/0 | 95% CI | exact p2 | low support |",
          "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|"]
    for key in sorted(stats["summaries"]):
        ds, kind, cond, rule, variant = key.split("|")
        s = stats["summaries"][key]
        md.append(f"| {ds} | {variant.split('_')[0]} | {kind}:{cond} | {rule} | {s['n_seeds']} | "
                  f"{s['mean_delta']:+.6f} | {s['median_delta']:+.6f} | {s['sd_delta']:.6f} | "
                  f"{s['min_delta']:+.6f} | {s['max_delta']:+.6f} | "
                  f"{s['positive']}/{s['negative']}/{s['zero']} | "
                  f"[{s['bootstrap']['ci_low']:+.6f}, {s['bootstrap']['ci_high']:+.6f}] | "
                  f"{s['signflip']['exact_p_two_sided_decimal']} "
                  f"({s['signflip']['exact_p_two_sided_fraction']}) | "
                  f"{'YES' if s['low_support_warning'] else 'no'} |")
    md += ["", "## Method notes", "",
           "* GT boxes and both level assignments are model-independent and were computed once per "
           "run; all IoU/matching work is vectorised with numpy using the manuscript's exact formula "
           "and operation order, so the greedy matching result equals the scalar implementation.",
           "* OinkTrack predictions are exported one sequence at a time, so the frame mapping is "
           "unambiguous (anomaly #20); exports are deleted immediately after each checkpoint.",
           "* buckets with fewer than 100 GT instances in any seed are flagged `low support`; no "
           "significance interpretation should be attached to them.",
           "* no threshold, split, seed or checkpoint rule was changed for this analysis; no model "
           "was trained."]
    (OUT / "FAILURE_MULTI_SEED_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\nsummaries: {len(stats['summaries'])} | runtime {(time.time() - t0) / 60:.1f} min")
    print(f"written -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
