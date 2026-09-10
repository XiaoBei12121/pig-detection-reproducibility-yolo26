# -*- coding: utf-8 -*-
"""Complexity audit: parameters (total and fused) and GFLOPs at 640x640.

The manuscript reports parameters measured on the trained ``best.pt`` checkpoints:
  YOLOv8s 11.1360 M | YOLO11s 9.4282 M | YOLO26s 9.9486 M | YOLO26s-CA-SIoU 9.9743 M
  YOLO11s+CA 9.4538 M  (Δ +0.0256 M vs its baseline, i.e. +0.27 %)
  E2 BiFPN-style 13.474 M (Δ +3.525 M, +35.4 %) and 27.3 GFLOPs vs 22.7

Note: parameters of the *released* pretrained weights (e.g. yolo11s.pt = 9.4588 M) differ slightly
from the trained checkpoints; the manuscript consistently uses the trained-checkpoint convention.

Example
-------
python scripts/analysis/complexity_audit.py --weights a.pt b.pt --out results/generated/complexity.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path


def measure(path: str):
    from ultralytics import YOLO

    model = YOLO(path)
    total = sum(p.numel() for p in model.model.parameters())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model.model.info(verbose=True, imgsz=640)
    line = [l for l in buf.getvalue().splitlines() if "GFLOPs" in l]
    gflops = line[-1].split("GFLOPs")[0].split()[-1] if line else None
    fused_params = None
    try:
        import copy
        m2 = copy.deepcopy(model.model).fuse()
        fused_params = sum(p.numel() for p in m2.parameters())
    except Exception:
        pass
    return {"weights": path.replace("\\", "/"), "params": total, "params_m": round(total / 1e6, 4),
            "fused_params": fused_params,
            "fused_params_m": round(fused_params / 1e6, 4) if fused_params else None,
            "gflops_640": gflops}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [measure(w) for w in args.weights]
    print(f"{'weights':<46}{'params(M)':>11}{'fused(M)':>10}{'GFLOPs':>8}")
    for r in rows:
        print(f"{Path(r['weights']).name:<46}{r['params_m']:>11}{str(r['fused_params_m']):>10}"
              f"{str(r['gflops_640']):>8}")
    if len(rows) == 2:
        d = rows[1]["params"] - rows[0]["params"]
        print(f"\ndelta: +{d / 1e6:.4f} M ({d / rows[0]['params'] * 100:+.2f} % vs baseline)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
