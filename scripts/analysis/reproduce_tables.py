# -*- coding: utf-8 -*-
"""Regenerate the manuscript tables from the locked result files.

Outputs: results/generated/table01.csv ... table18.csv
Inputs : manifests/*.csv and results/**/*.json (all shipped in this repository)

No GPU, dataset images or trained weights are required.

Where a figure/table mixes measured quantities that are not stored in a single JSON file
(parameters, GFLOPs, 640-coordinate distribution), the constants below are transcribed from
results/statistics/*.md and from the run manifests; the origin is stated in each table's header note.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
GEN = RES / "generated"
MAN = REPO / "manifests"

# ------------------------------------------------------------------ constants (see header note)
PARAMS = {"E0": 9.9486, "E1": 9.9743, "E2": 13.474, "E3": 9.949, "E4": 13.499, "E5": 9.9743, "E6": 13.499,
          "YOLOv8s": 11.1360, "YOLO11s": 9.4282, "YOLO26s": 9.9486,
          "YOLO26s-CA-SIoU": 9.9743, "Ours": 9.9743}
GFLOPs = {"E0": 22.7, "E1": 22.7, "E2": 27.3, "E3": 22.7, "E4": 27.3, "E5": 22.7, "E6": 27.3,
          "YOLOv8s": 28.8, "YOLO11s": 21.8, "YOLO26s": 22.7,
          "YOLO26s-CA-SIoU": 22.7, "Ours": 22.7}
BBOX640 = [("bbox width / 640", 0.0913, 0.0932, 0.0066, 0.069),
           ("bbox height / 640", 0.1043, 0.1083, 0.0242, 0.148),
           ("bbox area / 640^2", 0.0090, 0.0103, 0.0055, 0.187),
           ("aspect ratio", 0.8229, 0.7717, 0.1603, 0.081),
           ("pigs per image (median)", 22, 12, None, None)]


def js(p: Path):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def read_csv(p: Path):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(name: str, header, rows, note: str = ""):
    dst = GEN / f"{name}.csv"
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if note:
            w.writerow([f"# {note}"])
        w.writerow(header)
        w.writerows(rows)
    print(f"  {name}.csv  ({len(rows)} rows)")


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    pd_rows = read_csv(MAN / "runs_pigdetect.csv")
    pl_rows = read_csv(MAN / "runs_piglife.csv")
    y11_rows = read_csv(MAN / "runs_yolo11_replication.csv")
    abl = {r["run_id"]: r for r in pd_rows if r["run_id"] in
           {"E0_seed42", "E1_seed42", "E2_seed42", "E3_seed42", "E4_seed42", "E5_seed42", "E6_seed42"}}
    ms = js(RES / "statistics" / "multiseed_pigdetect.json")
    coco = {r["tag"]: r for r in js(RES / "audit" / "recheck_coco_20260909.json")}
    ul = {r["tag"]: r for r in js(RES / "audit" / "recheck_ultralytics_20260909.json")}
    err = js(RES / "error_analysis" / "error_analysis_conf025.json")
    cs = js(RES / "cross_scene" / "cross_scene_eval.json")
    pl_best = {f'{r["model"]}_{r["seed"]}': r for r in js(RES / "statistics" / "piglife_test_best.json")}
    pl_best["E2_42"] = {"model": "E2", "seed": "42", **js(RES / "statistics" / "piglife_e2_metric.json")}
    pl_last = {r["run"]: r for r in js(RES / "statistics" / "piglife_test_last.json")}
    y11 = js(RES / "yolo11_replication" / "test_audit_20260909.json")
    boot = js(GEN / "bootstrap_multiseed_e0_e5.json") if (GEN / "bootstrap_multiseed_e0_e5.json").exists() else None
    perm = js(GEN / "permutation_tests.json") if (GEN / "permutation_tests.json").exists() else None

    print("generating tables ->", GEN)

    # Table 1 — ablation (exploratory test, seed 42)
    rows = []
    for i, eid in enumerate(["E0", "E1", "E2", "E3", "E4", "E5", "E6"]):
        r = abl[f"{eid}_seed42"]
        d = "-" if eid == "E0" else f"{(PARAMS[eid] - PARAMS['E0']) / PARAMS['E0'] * 100:+.2f}%"
        rows.append([eid, r["model"], r["test_map50"], r["test_map5095"], r["precision"], r["recall"],
                     PARAMS[eid], d])
    write("table01_ablation", ["ID", "model", "mAP50", "mAP50-95", "precision", "recall", "params_m", "delta_params"], rows,
          "source: manifests/runs_pigdetect.csv (E*_seed42)")

    # Table 2 — validation best
    rows = [[e, abl[f"{e}_seed42"]["best_epoch"], abl[f"{e}_seed42"]["val_map50"], abl[f"{e}_seed42"]["val_map5095"]]
            for e in ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]]
    write("table02_validation", ["model", "best_epoch", "val_mAP50", "val_mAP5095"], rows,
          "source: manifests/runs_pigdetect.csv")

    # Table 3 — multi-seed
    rows = []
    for m in ("E0", "E1", "E3", "E5"):
        v = {r["seed"]: r["mAP50-95"] for r in ms if r["model"] == m}
        vals = [v[s] for s in ("42", "1", "7", "21", "100") if s in v]
        mean = sum(vals) / len(vals)
        sd = (sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
        rows.append([m] + [v.get(s, "-") for s in ("42", "1", "7", "21", "100")] + [f"{mean:.4f}", f"{sd:.4f}", len(vals)])
    write("table03_multiseed", ["model", "seed42", "seed1", "seed7", "seed21", "seed100", "mean", "sd", "n"], rows,
          "source: results/statistics/multiseed_pigdetect.json")

    # Table 4 — detector comparison
    rows = []
    for label, tag in (("YOLOv8s", "YOLOv8s_test250"), ("YOLO11s", "YOLO11s_test250"),
                       ("YOLO26s", "E0_test250"), ("YOLO26s-CA-SIoU", "E5_test250")):
        r = ul[tag]
        rows.append([label, PARAMS[label], GFLOPs[label], r["mAP50"], r["mAP50-95"], r["P"], r["R"]])
    write("table04_comparison", ["model", "params_m", "gflops", "mAP50", "mAP50-95", "precision", "recall"], rows,
          "metrics: results/audit/recheck_ultralytics_20260909.json; FPS deliberately excluded (see supplementary/)")

    # Table 5 — COCO size metrics
    rows = [[("YOLO26s" if t == "E0_test250" else "YOLO26s-CA-SIoU"), coco[t]["AP"], coco[t]["AP75"],
             coco[t]["APs"], coco[t]["APm"], coco[t]["APl"]] for t in ("E0_test250", "E5_test250")]
    write("table05_coco_size", ["model", "AP", "AP75", "APs(68)", "APm(2524)", "APl(2844)"], rows,
          "source: results/audit/recheck_coco_20260909.json")

    # Table 6 — occlusion (conf>=0.25)
    rows = []
    names = {"Low": 1715, "Medium": 1385, "High": 2336}
    for lvl, n in names.items():
        m0 = err["E0"]["occ"][lvl][1]
        m5 = err["E5"]["occ"][lvl][1]
        rows.append([lvl, n, m0, m5, round(1 - m0 / n, 4), round(1 - m5 / n, 4)])
    write("table06_occlusion", ["level", "instances", "E0_miss", "E5_miss", "E0_recall", "E5_recall"], rows,
          "source: results/error_analysis/error_analysis_conf025.json (NOT occlusion_analysis.json)")

    # Table 7 — cross-scene
    rows = [["YOLO26s", ul["E0_test250"]["mAP50"], ul["E0_test250"]["mAP50-95"], cs["E0_cs"]["mAP50"], cs["E0_cs"]["mAP50-95"]],
            ["YOLO26s-CA-SIoU", ul["E5_test250"]["mAP50"], ul["E5_test250"]["mAP50-95"], cs["E5_cs"]["mAP50"], cs["E5_cs"]["mAP50-95"]]]
    write("table07_cross_scene", ["model", "in_domain_mAP50", "in_domain_mAP5095", "danuma_mAP50", "danuma_mAP5095"], rows,
          "source: results/cross_scene/cross_scene_eval.json + results/audit/recheck_ultralytics_20260909.json")

    # Table 8 — error analysis
    rows = []
    for tag, label in (("E0", "YOLO26s"), ("E5", "YOLO26s-CA-SIoU")):
        e = err[tag]
        total = e["edge"][1] + e["stack"][1] + e["normal"][1]
        rows.append([label, f'{e["edge"][1]}/{e["edge"][0]}', f'{e["stack"][1]}/{e["stack"][0]}',
                     f'{e["normal"][1]}/{e["normal"][0]}', total, round(total / 5436 * 100, 2), e["fp"]])
    write("table08_error", ["model", "edge_miss", "stack_miss", "normal_miss", "total_miss", "miss_pct", "fp"], rows,
          "source: results/error_analysis/error_analysis_conf025.json (conf>=0.25, IoU>=0.5)")

    # Table 9 — efficiency
    rows = [["YOLO26s", 9.949, 22.7, 0.775, "-", "-"],
            ["+CA", 9.974, 22.7, 0.774, "+0.025", "~0"],
            ["+SIoU", 9.949, 22.7, 0.776, "0", "0"],
            ["CA+SIoU", 9.974, 22.7, 0.781, "+0.025", "~0"],
            ["+BiFPN-style", 13.474, 27.3, 0.771, "+3.525", "+4.6"]]
    write("table09_efficiency", ["strategy", "params_m", "gflops", "mAP5095_seed42", "d_params_m", "d_gflops"], rows,
          "mAP50-95 from Table 1; parameters/GFLOPs measured (scripts/analysis/complexity_audit.py)")

    # Table 10 — final-protocol holdout
    ft = {r["model"]: r for r in pd_rows if r["dataset"] == "PigDetect-ft"}
    rows = []
    for metric, key in (("mAP50", "test_map50"), ("mAP50-95", "test_map5095"), ("precision", "precision"),
                        ("recall", "recall")):
        a, b = float(ft["E0"][key]), float(ft["E5"][key])
        rows.append([metric, a, b, round(b - a, 4)])
    write("table10_final_holdout", ["metric", "E0_ft", "E5_ft", "delta"], rows,
          "source: manifests/runs_pigdetect.csv (dataset=PigDetect-ft); see docs/DATASETS.md for holdout scope")

    # Table 11 — PigLife validation
    rows = [[r["run_id"], r["best_epoch"], r["val_map50"], r["val_map5095"]]
            for r in pl_rows if r["checkpoint_rule"] == "best"]
    write("table11_piglife_validation", ["run", "best_epoch", "val_mAP50", "val_mAP5095"], rows,
          "source: manifests/runs_piglife.csv")

    # Table 12 — PigLife test (best.pt)
    rows = []
    for run in ("E0_42", "E0_1", "E0_7", "E1_42", "E3_42", "E2_42", "E5_42", "E5_1", "E5_7"):
        b = pl_best[run]
        rows.append([run, b["P"], b["R"], b["mAP50"], b["mAP50-95"]])
    write("table12_piglife_test_best", ["run", "precision", "recall", "mAP50", "mAP5095"], rows,
          "source: results/statistics/piglife_test_best.json + piglife_e2_metric.json")

    # Table 13 — checkpoint rule sensitivity
    rows = []
    for rule, table in (("best.pt", pl_best), ("last.pt", pl_last)):
        e0 = [table[f"E0_{s}"]["mAP50-95"] for s in ("42", "1", "7")]
        e5 = [table[f"E5_{s}"]["mAP50-95"] for s in ("42", "1", "7")]
        m0, m5 = sum(e0) / 3, sum(e5) / 3
        rows.append([rule, round(m0, 4), round(m5, 4), round(m5 - m0, 4),
                     [round(b - a, 4) for a, b in zip(e0, e5)]])
    write("table13_checkpoint_sensitivity", ["rule", "E0_mean", "E5_mean", "mean_delta", "per_seed_delta"], rows,
          "source: results/statistics/piglife_test_best.json + piglife_test_last.json")

    # Table 14 — PigLife COCO size
    rows = []
    for m in ("E0_42", "E5_42", "E1_42", "E3_42", "E2_42"):
        c = js(RES / "statistics" / f"piglife_detailed_{m}.json")["coco"]
        rows.append([m, c["AP"], c["AP50"], c["AP75"], c["APs"], c["APm"], c["APl"]])
    write("table14_piglife_coco_size", ["model", "AP", "AP50", "AP75", "APs(31)", "APm(1132)", "APl(3311)"], rows,
          "source: results/statistics/piglife_detailed_*.json")

    # Table 15 — PigLife occlusion recall
    rows = []
    for m in ("E0_42", "E5_42", "E1_42", "E3_42"):
        o = js(RES / "statistics" / f"piglife_detailed_{m}.json")["occlusion"]
        rows.append([m, o["recall_Low"], o["recall_Med"], o["recall_High"], o["miss"]["High"]])
    write("table15_piglife_occlusion", ["model", "Low_recall", "Med_recall", "High_recall", "High_miss"], rows,
          "source: results/statistics/piglife_detailed_*.json (conf>=0.25, IoU>=0.5)")

    # Table 16 — 640-coordinate distribution
    rows = [[n, a, b, w, d] for n, a, b, w, d in BBOX640]
    write("table16_bbox640_distribution", ["metric", "pigdetect_median", "piglife_median", "wasserstein", "ks_D"], rows,
          "transcribed from results/statistics/bbox640_distribution.md (PigDetect official 250-image danuma test)")

    # Table 17 — cross-dataset trend summary
    d26 = [pl_best[f"E5_{s}"]["mAP50-95"] - pl_best[f"E0_{s}"]["mAP50-95"] for s in ("42", "1", "7")]
    y = {s: (y11[i]["mAP50-95"], y11[i + 1]["mAP50-95"]) for i, s in zip(range(0, 12, 4), ("42", "1", "7"))}
    rows = [
        ["E5 vs E0 (PigDetect 5-seed)", "+0.0029, CI [-0.00048, 0.00628], p=0.3125", "best -0.0158 / last +0.0002", "no stable gain"],
        ["E1 vs E0 (seed42)", "0.7744 vs 0.7747", "0.8747 vs 0.8855", "no consistent gain"],
        ["E3 vs E0 (seed42)", "0.7755 vs 0.7747", "0.8736 vs 0.8855", "flat / lower"],
        ["E2 vs E0 (seed42)", "0.771 vs 0.775", "0.8674 vs 0.8855 (last 0.8823 vs 0.8882)", "negative trade-off reproduced"],
        ["high occlusion", "E5 not better than E0", "E5 not better than E0", "consistent failure mode"],
        ["target scale @640", "width/height/area medians close to PigLife", "close to PigDetect", "scale mismatch not supported"],
        ["objects per image (median)", "22 (official test)", "12", "composition differs"],
    ]
    write("table17_cross_dataset_trend", ["aspect", "pigdetect", "piglife", "interpretation"], rows,
          "compiled from the tables above; CI/p recomputed by scripts/analysis/*.py")

    # Table 18 — YOLO11s replication
    rows = []
    for i, seed in enumerate(("42", "1", "7")):
        bb, cb, bl, cl = (y11[i * 4 + k]["mAP50-95"] for k in range(4))
        rows.append([seed, bb, cb, round(cb - bb, 4), bl, cl, round(cl - bl, 4)])
    rows.append(["mean+-SD", "-", "-",
                 f"+{sum(r[3] for r in rows) / 3:.4f}",
                 "-", "-", f"{sum(r[6] for r in rows) / 3:+.4f}"])
    write("table18_yolo11_replication", ["seed", "baseline_best", "CA_best", "delta_best",
                                         "baseline_last", "CA_last", "delta_last"], rows,
          "source: results/yolo11_replication/test_audit_20260909.json")

    if boot:
        print(f"\nbootstrap CI cross-check: [{boot['bootstrap']['ci_low']}, {boot['bootstrap']['ci_high']}]")
    if perm:
        print(f"permutation p (PigDetect E5-E0): {perm['pigdetect_E5_minus_E0']['p_two_sided']}")
    print("\nall tables written to", GEN)


if __name__ == "__main__":
    main()
