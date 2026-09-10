# -*- coding: utf-8 -*-
"""Evaluate both checkpoint rules (validation-selected ``best.pt`` and epoch-100 ``last.pt``).

This is the checkpoint-rule sensitivity analysis used for PigLife (Table 13) and for the YOLO11s
replication (Table 18). ``last.pt`` is only a post-hoc sensitivity check: it is never used to
re-select a model, and the test split is never queried per epoch.

Example
-------
python scripts/eval/checkpoint_eval.py --run-dir runs/piglife/E5_42 --dataset piglife \
    --rules best,last --out results/generated/ckpt_E5_42.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

LAYOUTS = {
    "pigdetect": ("PIG_DATA_ROOT", "./data/PigDetect", "train/images", "val/images", "test/images"),
    "piglife": ("PIG_PIGLIFE_ROOT", "./data/PigLife_1280", "train/images", "val/images", "test/images"),
    "crossscene": ("PIG_CS_ROOT", "./data/PigDetect_cs", "train_cs/images", "val_cs/images", "test_cs/images"),
    "final": ("PIG_FT_ROOT", "./data/PigDetect_ft", "train_ft/images", "val_ft/images", "final_test/images"),
}


def best_epoch(results_csv: Path):
    if not results_csv.exists():
        return None, None
    with open(results_csv, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("metrics/mAP50-95(B)")]
    if not rows:
        return None, None
    b = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    return int(float(b["epoch"])), round(float(b["metrics/mAP50-95(B)"]), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", choices=sorted(LAYOUTS), default="pigdetect")
    ap.add_argument("--rules", default="best,last")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    env, default, tr, va, te = LAYOUTS[args.dataset]
    root = Path(os.environ.get(env, default)).resolve()
    yml = root / f"{args.dataset}_data.yaml"
    yml.write_text(f"path: {root.as_posix()}\ntrain: {tr}\nval: {va}\ntest: {te}\n"
                   "names:\n  0: pig\n", encoding="utf-8")

    run_dir = Path(args.run_dir)
    ep, val = best_epoch(run_dir / "results.csv")
    out = {"run": run_dir.name, "dataset": args.dataset, "best_epoch": ep,
           "best_val_map5095": val, "rules": {}}
    for rule in [r.strip() for r in args.rules.split(",") if r.strip()]:
        w = run_dir / "weights" / f"{rule}.pt"
        if not w.exists():
            print(f"[skip] {w} not found")
            continue
        m = YOLO(str(w)).val(data=str(yml), split=args.split, imgsz=args.imgsz, batch=args.batch,
                             device=args.device, workers=0, plots=False, project="./runs/eval",
                             name=f"{run_dir.name}_{rule}", exist_ok=True)
        b = m.box
        out["rules"][rule] = {"P": round(float(b.mp), 4), "R": round(float(b.mr), 4),
                              "mAP50": round(float(b.map50), 4), "mAP50-95": round(float(b.map), 4)}
        print(f"{run_dir.name} [{rule}] {out['rules'][rule]}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
