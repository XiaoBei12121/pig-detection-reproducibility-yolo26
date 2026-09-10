# -*- coding: utf-8 -*-
"""Align official pretrained weights to a modified model YAML (key remap by layer index + shape).

Used to give every modified architecture (CA, BiFPN-style, YOLO11s+CA) the same pretrained starting
point as its baseline: layers whose index is unchanged are copied directly, layers after the
insertion point are shifted, and newly introduced layers keep the framework default initialisation.

Key-alignment rules
-------------------
* ``--random-idx``  : indices that are newly inserted / cannot be inherited (random init)
* ``--shift-from``  : first index at which the layer numbering differs from the source checkpoint
* ``--shift``       : offset applied from that index onwards (default -1: one inserted layer)

Example (YOLO11s + CA, CA inserted at index 11 → head indices shift by +1):
    python scripts/train/align_weights.py --yaml configs/yolo11s_ca.yaml --orig yolo11s.pt \
        --random-idx 11 --shift-from 12 --shift -1 --out weights/yolo11s_ca_aligned.pt

The script prints an audit report: number of copied tensors, skipped tensors, shape mismatches and
keys missing from the target state dict.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="target model YAML")
    ap.add_argument("--orig", required=True, help="source checkpoint (.pt) to inherit from")
    ap.add_argument("--random-idx", required=True, help="comma-separated layer indices: random init")
    ap.add_argument("--shift-from", type=int, default=999)
    ap.add_argument("--shift", type=int, default=-1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    random_idx = {int(x) for x in args.random_idx.split(",") if x.strip()}
    ckpt = torch.load(args.orig, map_location="cpu")
    sd_orig = (ckpt["model"].float().state_dict() if isinstance(ckpt.get("model"), torch.nn.Module)
               else ckpt["model"])

    model = YOLO(args.yaml)
    sd_new = model.model.state_dict()
    print(f"source tensors: {len(sd_orig)} | target tensors: {len(sd_new)} | random-init layers: {sorted(random_idx)}")
    print(f"layer shift: index >= {args.shift_from} -> index{args.shift:+d}")

    aligned, copied, skipped = {}, 0, 0
    for k in sd_new:
        parts = k.split(".")
        idx = int(parts[1])
        if idx in random_idx:
            skipped += 1
            continue
        if idx >= args.shift_from:
            parts[1] = str(idx + args.shift)
            orig_k = ".".join(parts)
        else:
            orig_k = k
        if orig_k in sd_orig and sd_orig[orig_k].shape == sd_new[k].shape:
            aligned[k] = sd_orig[orig_k]
            copied += 1
        else:
            skipped += 1
            if orig_k in sd_orig:
                print(f"[shape mismatch] {k} <- {orig_k} "
                      f"({tuple(sd_orig[orig_k].shape)} vs {tuple(sd_new[k].shape)})")

    missing, unexpected = model.model.load_state_dict(aligned, strict=False)
    n_new = sum(1 for m in missing if m.split(".")[1].isdigit() and int(m.split(".")[1]) in random_idx)
    print(f"copied {copied} | skipped {skipped} | load_state_dict missing={len(missing)} "
          f"(of which {n_new} belong to the new module) | unexpected={len(unexpected)}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print("aligned checkpoint saved ->", args.out)


if __name__ == "__main__":
    main()
