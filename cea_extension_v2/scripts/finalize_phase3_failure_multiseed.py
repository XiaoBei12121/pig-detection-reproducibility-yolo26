"""A4 - finalise the failure-condition multi-seed analysis (read-only w.r.t. raw data).

Does four things, in order:

1. Input checks: seeds, checkpoint rules, checkpoint traceability to the registry,
   overlap thresholds and the GT bucket counts, lighting buckets, and the mandated
   "geometric bounding-box overlap proxy" wording.
2. Repairs the *derived* per-seed CSV, which the first pass had contaminated: its
   writer took the header from the first row of the existing file, so superseded rows
   from an earlier, label-derived ground truth were kept and the `variant` column of the
   new rows was silently dropped. The pre-repair file is copied into the audit trail
   before anything is rewritten. No inference is re-run and no raw data is touched.
3. Independently recomputes all 36 summaries from the repaired per-seed values with
   exact rational arithmetic (recall = hits/instances), the same bootstrap RNG call and
   the same exact sign-flip definition, and compares every field with
   `failure_multiseed_statistics.json`.
4. Checks the values that were already known before this run, so that they are confirmed
   rather than assumed.

Outputs (new files only):
  .../author_acceptance/06_failure_multiseed_verification.csv
  .../author_acceptance/06_failure_multiseed_verification.md
  .../author_acceptance/integrity_audit/phase3_overlap_per_seed_pre_repair.csv
"""

from __future__ import annotations

import csv
import itertools
import json
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np

PJ = Path(r"<PIG_PROJECT>")
EXP = PJ / "experiments" / "cea_extension_v2"
OUT = EXP / "16_failure_multiseed"
AUD = EXP / "08_report" / "author_acceptance"
AUDIT_TRAIL = AUD / "integrity_audit"
REGISTRY = EXP / "run_registry.csv"
GT_COUNTS = {"Low": 1715, "Medium": 1385, "High": 2336}   # frozen COCO GT, verified
OINK_GT = {"D": 19944, "DN": 168561, "N": 73217}
SEEDS = [1, 21, 42, 101, 113]
RULES = ["best", "last"]
VARIANTS = ["A_locked_text_manuscript", "B_extension_strict"]
LIGHTS = ["D", "DN", "N"]
LEVELS = ["Low", "Medium", "High"]


def rows(p: Path):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, data: list, fields: list) -> None:
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(data)


def exact_p(deltas: list) -> dict:
    """Mirror of the analysis' exact sign-flip: m >= obs for the one-sided count."""
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
    return {"two": Fraction(greater + tie, 2 ** n), "one": Fraction(one, 2 ** n),
            "tie": tie, "greater": greater}


def bootstrap_ci(deltas: list, r: int = 100_000, seed: int = 20260916) -> dict:
    """Identical RNG call sequence to the analysis itself."""
    d = np.asarray([float(x) for x in deltas], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(r, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return {"ci_low": lo, "ci_high": hi, "contains_zero": bool(lo <= 0 <= hi)}


def main() -> int:
    checks = []

    def chk(item, expected, found, verdict, note=""):
        checks.append({"item": item, "expected": str(expected), "found": str(found),
                       "verdict": verdict, "note": note})

    stats = json.loads((OUT / "failure_multiseed_statistics.json").read_text(encoding="utf-8"))
    reg = rows(REGISTRY)

    # ---------------------------------------------------------------- 1. input checks
    chk("seeds used", SEEDS, stats["seeds"], "PASS" if stats["seeds"] == SEEDS else "FAIL")
    chk("checkpoint rules", RULES, stats["rules"], "PASS" if stats["rules"] == RULES else "FAIL")
    chk("datasets", ["pigdetect", "oinktrack"], stats["datasets"],
        "PASS" if sorted(stats["datasets"]) == ["oinktrack", "pigdetect"] else "FAIL")

    # every evaluated checkpoint must exist with the hash the registry recorded
    missing, hash_bad, n_ck = [], [], 0
    by_id = {r["run_id"]: r for r in reg}
    for rule in RULES:
        for arm in ("E0", "E5"):
            for seed in SEEDS:
                rid = f"{arm}_seed{seed}" if seed >= 101 else f"lc_{arm}_seed{seed}"
                v = by_id.get(rid)
                if v is None:
                    missing.append(rid)
                    continue
                n_ck += 1
                ck = Path(v["run_dir"]) / "weights" / f"{rule}.pt"
                if not ck.exists():
                    missing.append(str(ck))
                    continue
                reg_hash = v[f"{rule}_checkpoint_sha256"]
                import hashlib
                h = hashlib.sha256(ck.read_bytes()).hexdigest()
                if not h.startswith(reg_hash):
                    hash_bad.append(f"{rid}:{rule}")
    chk("checkpoints used exist and match the registry hash", f"{n_ck} checkpoints",
        f"missing={len(missing)} hash_mismatch={len(hash_bad)}",
        "PASS" if not missing and not hash_bad else "FAIL")

    # GT bucket counts: the pre-repair file is *expected* to show both generations, which is the
    # defect being documented; the post-repair file must show only the frozen COCO counts.
    per_seed_path = OUT / "overlap_per_seed.csv"
    backup = AUDIT_TRAIL / "phase3_overlap_per_seed_pre_repair.csv"
    AUDIT_TRAIL.mkdir(parents=True, exist_ok=True)
    source = backup if backup.exists() else per_seed_path
    if not backup.exists():
        shutil.copy2(per_seed_path, backup)
    before = rows(source)

    def pig_counts(rs):
        seen = defaultdict(set)
        for r in rs:
            if r["dataset"].startswith("PigDetect"):
                seen[(r["checkpoint_rule"], r["condition"])].add(int(r["instances"]))
        return {lv: sorted({v for (_, l) in seen for v in seen[(_, l)] if l == lv}) for lv in LEVELS}

    pre = pig_counts(before)
    contaminated = any(len(pre[lv]) > 1 for lv in LEVELS)
    chk("pre-repair per-seed CSV contamination", "two ground-truth generations present",
        pre, "PASS" if contaminated else "WARNING",
        f"pre-repair copy kept at {backup.relative_to(PJ).as_posix()}")

    # ---------------------------------------------------------------- 2. repair the derived CSV
    # The pre-repair file is exactly `superseded_generation + final_generation` (that is how the
    # writer appended). The boundary is found from the content itself: the final generation is the
    # longest tail that forms exactly two rows per key (variant A and variant B). No reliance on git
    # state - an earlier version read the boundary from `HEAD`, which stopped working once this
    # script's own commit became HEAD.
    keys_of = lambda rs: Counter((r["dataset"], r["seed"], r["arm"], r["checkpoint_rule"],
                                  r["condition"]) for r in rs)  # noqa: E731
    split = None
    for k in range(0, len(before) + 1):
        tail = before[k:]
        kk = keys_of(tail)
        if kk and all(v == 2 for v in kk.values()) and len(kk) * 2 == len(tail):
            split = k
            break
    dropped = before[:split] if split is not None else []
    uniq = before[split:] if split is not None else []
    sig_keys = ["dataset", "seed", "arm", "checkpoint_rule", "condition", "instances", "hits",
                "recall"]
    old_has_label_gt = any(r["instances"] in ("1714", "1386") for r in dropped)
    new_has_label_gt = any(r["instances"] in ("1714", "1386") for r in uniq)
    positional_ok = split is not None and old_has_label_gt and not new_has_label_gt
    deduped = 0

    good_rows, pair_index = [], defaultdict(int)
    for r in uniq:
        key = (r["dataset"], r["seed"], r["arm"], r["checkpoint_rule"], r["condition"])
        variant = VARIANTS[pair_index[key] % 2]
        pair_index[key] += 1
        good_rows.append({"dataset": r["dataset"], "seed": r["seed"], "arm": r["arm"],
                          "checkpoint_rule": r["checkpoint_rule"], "variant": variant,
                          "condition": r["condition"], "instances": r["instances"],
                          "hits": r["hits"], "recall": r["recall"]})
    fields = ["dataset", "seed", "arm", "checkpoint_rule", "variant", "condition",
              "instances", "hits", "recall"]
    write_csv(per_seed_path, good_rows, fields)

    pairs_ok = all(v == 2 for v in pair_index.values()) and len(pair_index) == 120
    # variants A and B must carry identical numbers (no GT box lies on a level threshold)
    ab_ok = True
    for r in good_rows:
        if r["variant"] != VARIANTS[0]:
            continue
        partner = [x for x in good_rows
                   if x["dataset"] == r["dataset"] and x["seed"] == r["seed"]
                   and x["arm"] == r["arm"] and x["checkpoint_rule"] == r["checkpoint_rule"]
                   and x["condition"] == r["condition"] and x["variant"] == VARIANTS[1]]
        if len(partner) != 1 or partner[0]["recall"] != r["recall"]:
            ab_ok = False
            break
    post = pig_counts(good_rows)
    post_ok = all(post[lv] == [GT_COUNTS[lv]] for lv in LEVELS)
    dup_ok = len(good_rows) == len({(r["dataset"], r["seed"], r["arm"], r["checkpoint_rule"],
                                     r["variant"], r["condition"]) for r in good_rows})
    chk("derived per-seed CSV repaired",
        "superseded generation dropped; 240 rows = 120 keys x 2 variants; frozen COCO GT only",
        f"{len(before)} rows -> {len(good_rows)}; boundary found from content at {split}; "
        f"superseded slice carries the label-derived counts = {positional_ok}; deduplicated "
        f"{deduped}; 120 keys x 2 variants = {pairs_ok}; A==B = {ab_ok}; no duplicate rows = "
        f"{dup_ok}; counts {post}",
        "PASS" if (positional_ok and pairs_ok and ab_ok and post_ok and dup_ok) else "FAIL",
        f"pre-repair copy kept at {backup.relative_to(PJ).as_posix()}")

    report = (OUT / "FAILURE_MULTI_SEED_REPORT.md").read_text(encoding="utf-8")
    proxy_ok = ("geometric bounding-box overlap proxy" in report
                and "must not be described as true visible occlusion" in report)
    chk("overlap proxy wording", "geometric bounding-box overlap proxy + explicit prohibition",
        "ok" if proxy_ok else "missing", "PASS" if proxy_ok else "FAIL")

    lig = rows(OUT / "lighting_per_seed.csv")
    lg = defaultdict(set)
    for r in lig:
        lg[r["condition"]].add(int(r["gt_boxes"]))
    chk("lighting buckets and GT counts", OINK_GT,
        {k: sorted(v) for k, v in lg.items()},
        "PASS" if all(lg[k] == {OINK_GT[k]} for k in LIGHTS) else "FAIL",
        "official OinkTrack scene names, never regrouped")

    # ---------------------------------------------------------------- 2b. residue removed
    # An earlier draft of this script contained a second, value-based repair block here. It ran
    # after the positional repair and silently overwrote the same file, keeping the 20 superseded
    # High-overlap rows (whose instance count matches the frozen GT) and producing 260 physical rows
    # with 20 byte-identical duplicates while the acceptance text stated 240. The block is deleted;
    # re-running this script now rewrites exactly 240 rows (120 keys x 2 variants) from the
    # pre-repair copy.

    # ---------------------------------------------------------------- 3. independent recomputation
    ov = defaultdict(lambda: defaultdict(dict))     # (ds,variant,cond,rule)[arm][seed] -> recall
    ov_sup = defaultdict(lambda: defaultdict(dict))
    for r in rows(per_seed_path):
        key = (r["dataset"], r["variant"], r["condition"], r["checkpoint_rule"])
        ov[key][r["arm"]][int(r["seed"])] = Fraction(int(r["hits"]), int(r["instances"]))
        ov_sup[key][r["arm"]][int(r["seed"])] = int(r["instances"])
    lt = defaultdict(lambda: defaultdict(dict))
    lt_sup = defaultdict(lambda: defaultdict(dict))
    for r in rows(OUT / "lighting_per_seed.csv"):
        key = (r["dataset"], r["variant"], r["condition"], r["checkpoint_rule"])
        lt[key][r["arm"]][int(r["seed"])] = Fraction(int(r["tp"]), int(r["gt_boxes"]))
        lt_sup[key][r["arm"]][int(r["seed"])] = int(r["gt_boxes"])

    ds_short = {"PigDetect_official_test_250": "pigdetect", "OinkTrack_official_test_8569": "oinktrack"}
    comparable = {"n_seeds", "deltas_exact", "positive", "negative", "zero",
                  "support_min_instances"}
    mismatches = []
    for (ds, variant, cond, rule), arms in list(ov.items()) + list(lt.items()):
        kind = "overlap" if (ds, variant, cond, rule) in ov else "lighting"
        key = f"{ds_short[ds]}|{kind}|{cond}|{rule}|{variant}"
        rec = stats["summaries"].get(key)
        if rec is None:
            mismatches.append(f"{key}: absent from statistics.json")
            continue
        src = ov if kind == "overlap" else lt
        sup_src = ov_sup if kind == "overlap" else lt_sup
        seeds = sorted(set(src[(ds, variant, cond, rule)]["E0"]) &
                       set(src[(ds, variant, cond, rule)]["E5"]))
        deltas = [src[(ds, variant, cond, rule)]["E5"][s] -
                  src[(ds, variant, cond, rule)]["E0"][s] for s in seeds]
        f = [float(d) for d in deltas]
        mine = {
            "n_seeds": len(deltas),
            "deltas_exact": [str(d) for d in deltas],
            "mean_delta": statistics.mean(f),
            "median_delta": statistics.median(f),
            "sd_delta": statistics.stdev(f) if len(f) > 1 else 0.0,
            "min_delta": min(f), "max_delta": max(f),
            "positive": sum(1 for x in f if x > 0), "negative": sum(1 for x in f if x < 0),
            "zero": sum(1 for x in f if x == 0),
            "support_min_instances": min(min(sup_src[(ds, variant, cond, rule)][a][s]
                                             for s in seeds) for a in ("E0", "E5")),
            "bootstrap": bootstrap_ci(deltas), "signflip": exact_p(deltas)}
        for field in ("n_seeds", "deltas_exact", "positive", "negative", "zero",
                      "support_min_instances"):
            if mine[field] != rec[field]:
                mismatches.append(f"{key}.{field}: recomputed {mine[field]} != recorded {rec[field]}")
        for field in ("mean_delta", "median_delta", "sd_delta", "min_delta", "max_delta"):
            if abs(mine[field] - rec[field]) > 1e-12:
                mismatches.append(f"{key}.{field}: {mine[field]} != {rec[field]}")
        for field, sub in (("ci_low", "ci_low"), ("ci_high", "ci_high")):
            if abs(mine["bootstrap"][field] - rec["bootstrap"][sub]) > 1e-12:
                mismatches.append(f"{key}.bootstrap.{field}: {mine['bootstrap'][field]} != "
                                  f"{rec['bootstrap'][sub]}")
        if str(mine["signflip"]["two"]) != rec["signflip"]["exact_p_two_sided_decimal"]:
            mismatches.append(f"{key}.signflip p2: {mine['signflip']['two']} != "
                              f"{rec['signflip']['exact_p_two_sided_decimal']}")
        if str(mine["signflip"]["one"]) != rec["signflip"]["exact_p_one_sided_decimal"]:
            mismatches.append(f"{key}.signflip p1: {mine['signflip']['one']} != "
                              f"{rec['signflip']['exact_p_one_sided_decimal']}")

    n_sum = len(stats["summaries"])
    chk("independent recomputation of every summary", f"{n_sum} summaries, all fields",
        f"{n_sum - len({m.split(':')[0] for m in mismatches})} fully reproduced; "
        f"{len(mismatches)} field mismatches",
        "PASS" if not mismatches else "FAIL")

    # variant A and B must agree (no GT box sits on a level threshold)
    same = all(
        ov[(ds, VARIANTS[0], c, r)][a][s] == ov[(ds, VARIANTS[1], c, r)][a][s]
        for ds in ds_short for c in LEVELS for r in RULES for a in ("E0", "E5") for s in SEEDS)
    chk("variant A == variant B", "identical", "identical" if same else "different",
        "PASS" if same else "WARNING",
        "no GT box lies on a level threshold, so the two boundary conventions coincide")

    # ---------------------------------------------------------------- 4. previously known values
    pigdet = {k: v for k, v in stats["summaries"].items() if k.startswith("pigdetect|")}
    hi = pigdet["pigdetect|overlap|High|best|A_locked_text_manuscript"]
    ok_hi = (abs(hi["mean_delta"] + 0.0032534246575342462) < 1e-12
             and hi["positive"] == 0 and not hi["bootstrap"]["contains_zero"]
             and hi["signflip"]["exact_p_two_sided_decimal"] == "1/16")
    chk("PigDetect High overlap, best.pt (known values)",
        "mean -0.00325, 0/5 positive, CI excludes 0, p2 = 1/16",
        f"mean {hi['mean_delta']:+.6f}, {hi['positive']}/5 positive, "
        f"CI [{hi['bootstrap']['ci_low']:+.6f},{hi['bootstrap']['ci_high']:+.6f}] "
        f"contains0={hi['bootstrap']['contains_zero']}, p2 {hi['signflip']['exact_p_two_sided_decimal']}",
        "PASS" if ok_hi else "FAIL")

    oink = {k: v for k, v in stats["summaries"].items() if k.startswith("oinktrack|")}
    neg = [k for k, v in oink.items() if v["mean_delta"] >= 0]
    chk("OinkTrack buckets all negative", "0 non-negative buckets",
        f"{len(neg)} non-negative of {len(oink)}",
        "PASS" if not neg else "WARNING", "; ".join(neg[:4]))

    fails = [c for c in checks if c["verdict"] == "FAIL"]
    write_csv(AUD / "06_failure_multiseed_verification.csv", checks,
              ["item", "expected", "found", "verdict", "note"])

    md = ["# A4 — failure-condition multi-seed: verification of the existing products", "",
          "Read-only with respect to raw data: no checkpoint, `results.csv`, evaluation CSV, split "
          "or seed was touched, and no inference was re-run. The only file rewritten is the derived "
          "per-seed CSV, whose earlier version mixed two generations of ground truth; the pre-repair "
          "copy is preserved in the audit trail.", "",
          "| check | expected | found | verdict |", "|---|---|---|---|"]
    for c in checks:
        md.append(f"| {c['item']} | {c['expected']} | {c['found']} | {c['verdict']} |")
    md += ["", f"FAIL: **{len(fails)}**", "",
           "## Method of the independent recomputation", "",
           "Recalls are reconstructed as exact fractions (`hits/instances` for the overlap "
           "analysis, `tp/gt_boxes` for the lighting analysis), paired deltas are `E5 − E0` per "
           "matched seed in rational arithmetic, the bootstrap repeats the analysis' own RNG call "
           "(`numpy.random.default_rng(20260916)`, 100 000 resamples, percentile 2.5/97.5), and the "
           "sign-flip enumerates all 2^5 sign assignments with exact ties included "
           "(`p2 = count(|T_perm| >= |T_obs|)/2^n`).", ""]
    if mismatches:
        md += ["## Field mismatches", "", *[f"* {m}" for m in mismatches[:40]], ""]
    else:
        md += ["## Field mismatches", "", "None: every recorded field of all 36 summaries is "
                                           "reproduced from the per-seed values.", ""]
    (AUD / "06_failure_multiseed_verification.md").write_text("\n".join(md), encoding="utf-8")

    for c in checks:
        print(f"  [{c['verdict']:7s}] {c['item']} -> {c['found']}")
    print(f"\nFAILs: {len(fails)} | field mismatches: {len(mismatches)}")
    for m in mismatches[:10]:
        print("   ", m)
    print("written ->", AUD / "06_failure_multiseed_verification.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
