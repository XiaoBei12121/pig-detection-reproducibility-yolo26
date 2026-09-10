# -*- coding: utf-8 -*-
"""640-input-coordinate target-scale audit for the two datasets (manuscript Table 16).

Letterboxing to 640×640 preserves aspect ratio, so a box's size in the network input space is
``bbox_size_original * (640 / max(H, W))``. For every GT box we record:
normalised width/height/area in the 640 input, aspect ratio, centre coordinates and boxes per image,
then compare the two datasets with the 1-D Wasserstein distance and the KS statistic.

Example
-------
python scripts/analysis/bbox_scale_audit.py \
    --pigdetect-root data/PigDetect/test --piglife-root data/PigLife_1280/test \
    --out results/generated/bbox640_audit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

TARGET = 640


def collect(img_dir: Path, lbl_dir: Path):
    feats = {k: [] for k in ("w", "h", "a", "ar", "cx", "cy", "per_img")}
    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(f) as im:
            W, H = im.size
        r = TARGET / max(W, H)
        lbl = lbl_dir / (f.stem + ".txt")
        boxes = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) >= 5:
                    boxes.append(tuple(map(float, p[1:5])))
        for cx, cy, bw, bh in boxes:
            w640 = bw * W * r / TARGET
            h640 = bh * H * r / TARGET
            feats["w"].append(w640)
            feats["h"].append(h640)
            feats["a"].append(w640 * h640)
            feats["ar"].append(w640 / max(h640, 1e-9))
            feats["cx"].append(cx)
            feats["cy"].append((cy * H * r) / TARGET)
        feats["per_img"].append(len(boxes))
    return {k: np.asarray(v, dtype=float) for k, v in feats.items()}


def wasserstein_1d(a, b, grid=2000):
    t = np.linspace(0, 1, grid)
    return float(np.mean(np.abs(np.quantile(a, t) - np.quantile(b, t))))


def ks_stat(a, b):
    allv = np.sort(np.concatenate([a, b]))
    ea = np.searchsorted(np.sort(a), allv, side="right") / len(a)
    eb = np.searchsorted(np.sort(b), allv, side="right") / len(b)
    return float(np.max(np.abs(ea - eb)))


def ks_approx_p(d, na, nb):
    ne = na * nb / (na + nb)
    lam = (np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * d
    return float(2 * sum(((-1) ** (k - 1)) * np.exp(-2 * k * k * lam * lam) for k in range(1, 40)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pigdetect-root", default="data/PigDetect/test")
    ap.add_argument("--piglife-root", default="data/PigLife_1280/test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pd_root, pl_root = Path(args.pigdetect_root), Path(args.piglife_root)
    pd_d = collect(pd_root / "images", pd_root / "labels")
    pl_d = collect(pl_root / "images", pl_root / "labels")

    label = {"w": "bbox width / 640", "h": "bbox height / 640", "a": "bbox area / 640^2",
             "ar": "aspect ratio", "cx": "centre x (normalised)", "cy": "centre y @640"}
    stats = {}
    print(f"{'metric':<24}{'PigDetect':>12}{'PigLife':>12}{'Wasserstein':>14}{'KS D':>9}")
    for k in ("w", "h", "a", "ar"):
        a, b = pd_d[k], pl_d[k]
        wd, d = wasserstein_1d(a, b), ks_stat(a, b)
        stats[k] = {"label": label[k], "pigdetect_median": round(float(np.median(a)), 4),
                    "piglife_median": round(float(np.median(b)), 4),
                    "pigdetect_iqr": [round(float(np.percentile(a, 25)), 4), round(float(np.percentile(a, 75)), 4)],
                    "piglife_iqr": [round(float(np.percentile(b, 25)), 4), round(float(np.percentile(b, 75)), 4)],
                    "wasserstein": round(wd, 4), "ks_D": round(d, 3),
                    "ks_p_approx": round(ks_approx_p(d, len(a), len(b)), 3)}
        print(f"{label[k]:<24}{stats[k]['pigdetect_median']:>12}{stats[k]['piglife_median']:>12}"
              f"{stats[k]['wasserstein']:>14}{stats[k]['ks_D']:>9}")
    for name, d in (("pigdetect", pd_d), ("piglife", pl_d)):
        stats.setdefault("per_image", {})[name] = {
            "images": int(len(d["per_img"])), "boxes": int(len(d["a"])),
            "median_boxes_per_image": float(np.median(d["per_img"])),
            "iqr_boxes_per_image": [float(np.percentile(d["per_img"], 25)), float(np.percentile(d["per_img"], 75))]}
        print(f"{name}: {int(len(d['per_img']))} images, {int(len(d['a']))} boxes, "
              f"median {np.median(d['per_img']):.0f} boxes/image")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
