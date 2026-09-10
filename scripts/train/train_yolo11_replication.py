# -*- coding: utf-8 -*-
"""Cross-generation replication on YOLO11s (baseline vs +CA) with the PigDetect protocol.

Manuscript runs: baseline and +CA × seeds {42, 1, 7}. All other settings are identical to the
YOLO26s experiments (epochs 100, imgsz 640, batch 4, MuSGD, lr0 0.01, momentum 0.937).

The +CA arm inserts Coordinate Attention after the backbone C2PSA (index 11 in
configs/yolo11s_ca.yaml) and must inherit the aligned pretrained weights so that the only
difference to the baseline is the added module.

Examples
--------
python scripts/train/train_yolo11_replication.py --variant baseline --seed 42 --name y11_e0_seed42
python scripts/train/train_yolo11_replication.py --variant CA --seed 42 \
    --weights weights/yolo11s_ca_aligned.pt --name y11_e1ca_seed42
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULTS = dict(epochs=100, imgsz=640, batch=4, optimizer="MuSGD", lr0=0.01, momentum=0.937)
HERE = Path(__file__).resolve().parents[2]


def build_data_yaml(root: Path) -> Path:
    yml = root / "pigdetect_data.yaml"
    yml.write_text(
        f"path: {root.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        "names:\n  0: pig\n", encoding="utf-8")
    return yml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "CA"], required=True)
    ap.add_argument("--weights", default=None,
                    help="baseline: official yolo11s.pt; CA: the aligned checkpoint")
    ap.add_argument("--config", default=None,
                    help="defaults to configs/yolo11s_ca.yaml for the CA variant")
    ap.add_argument("--data-root", default=os.environ.get("PIG_DATA_ROOT", "./data/PigDetect"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    ap.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    ap.add_argument("--optimizer", default=DEFAULTS["optimizer"])
    ap.add_argument("--lr0", type=float, default=DEFAULTS["lr0"])
    ap.add_argument("--momentum", type=float, default=DEFAULTS["momentum"])
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="./runs/yolo11_replication")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    os.environ.setdefault("YOLO_SIOU", "0")   # replication uses the native regression loss
    from ultralytics import YOLO

    data_yaml = build_data_yaml(Path(args.data_root).resolve())
    if args.variant == "CA":
        cfg = args.config or str(HERE / "configs" / "yolo11s_ca.yaml")
        src = args.weights or cfg
    else:
        cfg, src = args.config, (args.weights or "yolo11s.pt")

    model = YOLO(src, task="detect") if src.endswith(".pt") else YOLO(cfg, task="detect")
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                optimizer=args.optimizer, lr0=args.lr0, momentum=args.momentum, seed=args.seed,
                device=args.device, project=args.project, name=args.name, exist_ok=True)
    print(f"finished YOLO11s {args.variant} seed={args.seed} -> {args.project}/{args.name}")


if __name__ == "__main__":
    main()
