# -*- coding: utf-8 -*-
"""Train one PigLife run (E0/E1/E2/E3/E5) with the manuscript protocol.

Dataset layout expected under $PIG_DATA_ROOT (default ./data/PigLife_1280):
    train/images, train/labels, val/images, val/labels, test/images, test/labels
Produced by scripts/data/prepare_piglife.py.

Example
-------
python scripts/train/train_piglife.py --config configs/yolo26s_ca_siou.yaml --seed 42 \
    --weights weights/yolo26s_ca_aligned.pt --siou --name E5_42
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULTS = dict(epochs=100, imgsz=640, batch=4, optimizer="MuSGD", lr0=0.01, momentum=0.937)


def build_data_yaml(root: Path) -> Path:
    yml = root / "piglife_data.yaml"
    yml.write_text(
        f"path: {root.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        "names:\n  0: pig\n", encoding="utf-8")
    return yml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--data-root", default=os.environ.get("PIG_DATA_ROOT", "./data/PigLife_1280"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    ap.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    ap.add_argument("--optimizer", default=DEFAULTS["optimizer"])
    ap.add_argument("--lr0", type=float, default=DEFAULTS["lr0"])
    ap.add_argument("--momentum", type=float, default=DEFAULTS["momentum"])
    ap.add_argument("--siou", action="store_true")
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="./runs/piglife")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    if args.siou:
        os.environ["YOLO_SIOU"] = "1"

    from ultralytics import YOLO

    data_yaml = build_data_yaml(Path(args.data_root).resolve())
    src = args.weights or args.config
    model = YOLO(src, task="detect") if src.endswith(".pt") else YOLO(args.config, task="detect")
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                optimizer=args.optimizer, lr0=args.lr0, momentum=args.momentum, seed=args.seed,
                device=args.device, project=args.project, name=args.name, exist_ok=True)
    print(f"finished PigLife run {args.name} (seed={args.seed}, siou={args.siou})")


if __name__ == "__main__":
    main()
