# -*- coding: utf-8 -*-
"""Evaluate one checkpoint on one frozen test split (Ultralytics protocol, imgsz=640).

Examples
--------
python scripts/eval/evaluate_test.py --weights runs/pigdetect/E5_seed42/weights/best.pt \
    --dataset pigdetect --split test --out results/generated/eval_E5_seed42.json

--dataset selects the split layout:
    pigdetect  -> <PIG_DATA_ROOT or data/PigDetect>/{train,val,test}/images   (2411/270/250)
    piglife    -> data/PigLife_1280/{train,val,test}/images                  (1441/263/426)
    crossscene -> data/PigDetect_cs/{train_cs,val_cs,test_cs}/images         (1098/129/450)
    final      -> data/PigDetect_ft/{train_ft,val_ft,final_test}/images      (2226/251/204)

The test split is only ever evaluated once per checkpoint (no per-epoch test queries).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

LAYOUTS = {
    "pigdetect": ("PIG_DATA_ROOT", "./data/PigDetect",
                  {"train": "train/images", "val": "val/images", "test": "test/images"}),
    "piglife": ("PIG_PIGLIFE_ROOT", "./data/PigLife_1280",
                {"train": "train/images", "val": "val/images", "test": "test/images"}),
    "crossscene": ("PIG_CS_ROOT", "./data/PigDetect_cs",
                   {"train": "train_cs/images", "val": "val_cs/images", "test": "test_cs/images"}),
    "final": ("PIG_FT_ROOT", "./data/PigDetect_ft",
              {"train": "train_ft/images", "val": "val_ft/images", "test": "final_test/images"}),
}


def write_data_yaml(dataset: str) -> Path:
    env, default, paths = LAYOUTS[dataset]
    root = Path(os.environ.get(env, default)).resolve()
    yml = root / f"{dataset}_data.yaml"
    yml.write_text(f"path: {root.as_posix()}\ntrain: {paths['train']}\nval: {paths['val']}\n"
                   f"test: {paths['test']}\nnames:\n  0: pig\n", encoding="utf-8")
    return yml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", choices=sorted(LAYOUTS), default="pigdetect")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--save-json", action="store_true", help="also dump COCO-style predictions")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    data_yaml = write_data_yaml(args.dataset)
    model = YOLO(args.weights)
    m = model.val(data=str(data_yaml), split=args.split, imgsz=args.imgsz, batch=args.batch,
                  device=args.device, workers=0, plots=False, save_json=args.save_json,
                  project="./runs/eval", name=Path(args.weights).parent.parent.name, exist_ok=True)
    b = m.box
    res = {"weights": args.weights.replace("\\", "/"), "dataset": args.dataset, "split": args.split,
           "imgsz": args.imgsz, "P": round(float(b.mp), 4), "R": round(float(b.mr), 4),
           "mAP50": round(float(b.map50), 4), "mAP50-95": round(float(b.map), 4)}
    print(res)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
