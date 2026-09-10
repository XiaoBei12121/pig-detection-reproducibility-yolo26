# -*- coding: utf-8 -*-
"""Overlap-proxy (occlusion) recall analysis.

IMPORTANT — two different configurations exist in the manuscript pipeline:

  * ``--conf 0.25`` → the configuration whose results are reported in manuscript Table 6 / Table 15
    (use scripts/analysis/error_analysis.py for the full error table; it also emits the same
    per-level miss counts).
  * ``--conf 0`` (default here, no confidence filtering) → the auxiliary file
    ``results/occlusion/occlusion_analysis.json``, which is kept for audit only and whose recall
    values are systematically higher. It is NOT the source of Table 6.

Level definition (identical for both datasets): GT box covered-area ratio by other GT boxes,
Low < 0.3 ≤ Medium < 0.6 ≤ High. Matching uses IoU >= 0.5 (greedy, one-to-one).

Example
-------
python scripts/analysis/occlusion_analysis.py --gt data/PigDetect/test_coco.json \
    --preds E0=predictions_E0.json E5=predictions_E5.json --conf 0.25 \
    --out results/generated/occlusion_recall.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def overlap_ratio(a, others):
    ax1, ay1, ax2, ay2 = a
    area = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    s = 0.0
    for o in others:
        ix1, iy1 = max(ax1, o[0]), max(ay1, o[1])
        ix2, iy2 = min(ax2, o[2]), min(ay2, o[3])
        if ix2 > ix1 and iy2 > iy1:
            s += (ix2 - ix1) * (iy2 - iy1)
    return min(1.0, s / area)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--preds", nargs="+", required=True, help="TAG=predictions.json")
    ap.add_argument("--conf", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt = json.load(open(args.gt, encoding="utf-8-sig"))
    anns_by_img = {}
    for a in gt["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    stem2id = {Path(im["file_name"]).stem: im["id"] for im in gt["images"]}

    levels, dist = {}, {}
    for img_id, anns in anns_by_img.items():
        boxes = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                 for a in anns]
        for i, a in enumerate(anns):
            r = overlap_ratio(boxes[i], [b for j, b in enumerate(boxes) if j != i])
            levels[a["id"]] = "Low" if r < 0.3 else ("Medium" if r < 0.6 else "High")
    for lv in levels.values():
        dist[lv] = dist.get(lv, 0) + 1
    print("overlap-proxy level distribution:", dist)

    models = {}
    for spec in args.preds:
        tag, path = spec.split("=", 1)
        preds = json.load(open(path, encoding="utf-8-sig"))
        pb = {}
        for p in preds:
            if p.get("score", 1.0) >= args.conf:
                key = str(p["image_id"])
                gid = stem2id.get(key, stem2id.get(Path(key).stem))
                if gid is not None:
                    pb.setdefault(gid, []).append(p)
        stat = {lv: {"total": 0, "hit": 0} for lv in ("Low", "Medium", "High")}
        for img_id, anns in anns_by_img.items():
            gb = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                  for a in anns]
            pboxes = [[p["bbox"][0], p["bbox"][1], p["bbox"][0] + p["bbox"][2], p["bbox"][1] + p["bbox"][3]]
                      for p in pb.get(img_id, [])]
            hit = [False] * len(gb)
            for pbx in pboxes:
                best, bi = -1, 0.5
                for gi, g in enumerate(gb):
                    if not hit[gi]:
                        v = iou(pbx, g)
                        if v > bi:
                            bi, best = v, gi
                if best >= 0:
                    hit[best] = True
            for i, a in enumerate(anns):
                lv = levels[a["id"]]
                stat[lv]["total"] += 1
                stat[lv]["hit"] += int(hit[i])
        for lv in ("Low", "Medium", "High"):
            t, h = stat[lv]["total"], stat[lv]["hit"]
            stat[lv]["recall"] = round(h / t, 4) if t else None
            stat[lv]["miss"] = t - h
        models[tag] = stat
        print(f"{tag}: " + " | ".join(f"{lv} recall {stat[lv]['recall']} (miss {stat[lv]['miss']}/{stat[lv]['total']})"
                                      for lv in ("Low", "Medium", "High")))

    res = {"distribution": dist, "conf": args.conf, "models": models}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
