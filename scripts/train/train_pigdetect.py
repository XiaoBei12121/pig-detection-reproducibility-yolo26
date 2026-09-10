# -*- coding: utf-8 -*-
"""Train one PigDetect run (E0–E6) with the manuscript protocol.

Defaults are exactly the protocol used in the paper:
    epochs=100, imgsz=640, batch=4, optimizer=MuSGD, lr0=0.01, momentum=0.937

Examples
--------
# baseline, seed 42
python scripts/train/train_pigdetect.py --config configs/yolo26s_baseline.yaml --seed 42 \
    --name E0_seed42

# CA + SIoU (E5), enabling the SIoU regression path
python scripts/train/train_pigdetect.py --config configs/yolo26s_ca_siou.yaml --seed 42 \
    --weights weights/yolo26s_ca_aligned.pt --siou --name E5_seed42

# BiFPN-style variant (E2)
python scripts/train/train_pigdetect.py --config configs/yolo26s_bifpn.yaml --seed 42 \
    --weights weights/yolo26s_bifpn_aligned.pt --name E2_seed42

The dataset root is taken from $PIG_DATA_ROOT (default ./data/PigDetect) and the data YAML is
materialised next to it, so no absolute paths are required.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULTS = dict(epochs=100, imgsz=640, batch=4, optimizer="MuSGD", lr0=0.01, momentum=0.937)


def build_data_yaml(root: Path) -> Path:
    """Write a dataset YAML pointing at <root>/{train,val,test}/images (no absolute paths)."""
    yml = root / "pigdetect_data.yaml"
    yml.write_text(
        f"path: {root.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        "names:\n  0: pig\n", encoding="utf-8")
    return yml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="model YAML (see configs/)")
    ap.add_argument("--weights", default=None,
                    help="initial weights; use the aligned checkpoint for modified architectures")
    ap.add_argument("--data-root", default=os.environ.get("PIG_DATA_ROOT", "./data/PigDetect"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    ap.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    ap.add_argument("--optimizer", default=DEFAULTS["optimizer"])
    ap.add_argument("--lr0", type=float, default=DEFAULTS["lr0"])
    ap.add_argument("--momentum", type=float, default=DEFAULTS["momentum"])
    ap.add_argument("--siou", action="store_true", help="enable the SIoU regression loss (YOLO_SIOU=1)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="./runs/pigdetect")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    if args.siou:
        # must be set before ultralytics.utils.loss is imported (module-level switch)
        os.environ["YOLO_SIOU"] = "1"

    from ultralytics import YOLO  # delayed import so the SIoU switch takes effect

    root = Path(args.data_root).resolve()
    data_yaml = build_data_yaml(root)
    model_src = args.weights or args.config

    model = YOLO(model_src, task="detect") if model_src.endswith(".pt") else YOLO(args.config, task="detect")
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                optimizer=args.optimizer, lr0=args.lr0, momentum=args.momentum, seed=args.seed,
                device=args.device, project=args.project, name=args.name, exist_ok=True)
    print(f"finished run {args.name} (seed={args.seed}, siou={args.siou}) -> {args.project}/{args.name}")


if __name__ == "__main__":
    main()
