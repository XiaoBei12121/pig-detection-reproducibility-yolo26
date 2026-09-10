# -*- coding: utf-8 -*-
"""Diagnostic error analysis at conf >= 0.25 (source of manuscript Tables 6 and 8).

Definitions (identical to the implementation used for the manuscript):
  * match: prediction with IoU >= 0.5 against a GT box (greedy, one-to-one)
  * edge:  GT centre lies in the outer 5 % band of the image (cx<0.05W | cx>0.95W | cy<0.05H | cy>0.95H)
  * stack: GT box has IoU > 0.3 with at least one other GT box in the same image
  * normal: neither edge nor stack (edge takes precedence over stack)
  * overlap proxy: GT box covered-area ratio by other GT boxes → Low < 0.3 ≤ Medium < 0.6 ≤ High
  * FP: predictions with conf >= 0.25 that remain unmatched

The (edge, stack, normal) misses are mutually exclusive and sum to the total miss count.
These numbers must NOT be compared with the Ultralytics confusion matrix (different filtering).

Example
-------
python scripts/analysis/error_analysis.py --gt data/PigDetect/test_coco.json \
    --preds E0=predictions_E0.json E5=predictions_E5.json --out results/generated/error_analysis.json
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
    ap.add_argument("--gt", required=True, help="COCO ground truth json")
    ap.add_argument("--preds", nargs="+", required=True, help="TAG=predictions.json (Ultralytics format)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt = json.load(open(args.gt, encoding="utf-8-sig"))
    anns_by_img = {}
    for a in gt["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    img_by_id = {im["id"]: im for im in gt["images"]}
    stem2id = {Path(im["file_name"]).stem: im["id"] for im in gt["images"]}

    cats = {}
    for img_id, anns in anns_by_img.items():
        boxes = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                 for a in anns]
        W, H = img_by_id[img_id]["width"], img_by_id[img_id]["height"]
        is_stack = [False] * len(boxes)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if iou(boxes[i], boxes[j]) > 0.3:
                    is_stack[i] = is_stack[j] = True
        for i, a in enumerate(anns):
            x, y, w, h = a["bbox"]
            cx, cy = x + w / 2, y + h / 2
            edge = cx < 0.05 * W or cx > 0.95 * W or cy < 0.05 * H or cy > 0.95 * H
            ov = overlap_ratio(boxes[i], [b for k, b in enumerate(boxes) if k != i])
            cats[a["id"]] = {"edge": edge, "stack": is_stack[i],
                             "occ": "High" if ov >= 0.6 else ("Medium" if ov >= 0.3 else "Low")}

    res = {}
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

        stat = {"edge": [0, 0], "stack": [0, 0], "normal": [0, 0],
                "occ": {"Low": [0, 0], "Medium": [0, 0], "High": [0, 0]}, "fp": 0}
        for img_id, anns in anns_by_img.items():
            gb = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                  for a in anns]
            pboxes = [[p["bbox"][0], p["bbox"][1], p["bbox"][0] + p["bbox"][2], p["bbox"][1] + p["bbox"][3]]
                      for p in pb.get(img_id, [])]
            matched = [False] * len(gb)
            for pbx in pboxes:
                best, bi = -1, 0.5
                for gi, g in enumerate(gb):
                    if not matched[gi]:
                        v = iou(pbx, g)
                        if v > bi:
                            bi, best = v, gi
                if best >= 0:
                    matched[best] = True
            stat["fp"] += len(pboxes) - sum(matched)
            for i, a in enumerate(anns):
                c = cats[a["id"]]
                miss = int(not matched[i])
                key = "edge" if c["edge"] else ("stack" if c["stack"] else "normal")
                stat[key][0] += 1
                stat[key][1] += miss
                stat["occ"][c["occ"]][0] += 1
                stat["occ"][c["occ"]][1] += miss
        res[tag] = stat
        total_gt = sum(v[0] for v in (stat["edge"], stat["stack"], stat["normal"]))
        total_miss = sum(v[1] for v in (stat["edge"], stat["stack"], stat["normal"]))
        print(f"{tag}: conf>={args.conf} | edge {stat['edge'][1]}/{stat['edge'][0]} | "
              f"stack {stat['stack'][1]}/{stat['stack'][0]} | normal {stat['normal'][1]}/{stat['normal'][0]} | "
              f"total miss {total_miss}/{total_gt} ({total_miss / total_gt * 100:.2f}%) | FP {stat['fp']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
