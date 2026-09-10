# -*- coding: utf-8 -*-
"""Single-source-group held-out (cross-scene) evaluation: danuma source group as test.

Protocol actually used in the manuscript (and reported in Table 7):
    train/val = the non-danuma source groups (1098 / 129 images)
    test      = the danuma source group (450 images)
This is *not* a full leave-one-source-out study; only the danuma group is held out.

Example
-------
python scripts/eval/cross_scene_eval.py \
    --weights runs/cross_scene/E0_cs/weights/best.pt --out results/cross_scene/E0_cs.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--root", default=os.environ.get("PIG_CS_ROOT", "./data/PigDetect_cs"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    root = Path(args.root).resolve()
    yml = root / "cross_scene_data.yaml"
    yml.write_text(f"path: {root.as_posix()}\ntrain: train_cs/images\nval: val_cs/images\n"
                   "test: test_cs/images\nnames:\n  0: pig\n", encoding="utf-8")

    model = YOLO(args.weights)
    m = model.val(data=str(yml), split="test", imgsz=args.imgsz, batch=args.batch,
                  device=args.device, workers=0, plots=False, project="./runs/eval",
                  name="cross_scene", exist_ok=True)
    b = m.box
    res = {"weights": args.weights.replace("\\", "/"), "dataset": "PigDetect_cs",
           "held_out_group": "danuma", "test_images": 450,
           "P": round(float(b.mp), 4), "R": round(float(b.mr), 4),
           "mAP50": round(float(b.map50), 4), "mAP50-95": round(float(b.map), 4)}
    print(res)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
