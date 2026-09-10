# -*- coding: utf-8 -*-
"""COCO (pycocotools) evaluation of a prediction file against a COCO ground-truth file.

Produces the size-stratified metrics reported in the manuscript (AP, AP50, AP75, APs, APm, APl).

Example
-------
python scripts/eval/coco_eval.py --gt data/PigDetect/test_coco.json \
    --pred predictions.json --out results/generated/coco_E5.json

Notes
-----
* `--pred` accepts either an Ultralytics ``predictions.json`` (image_id = file stem) or a standard
  COCO detection json whose ``image_id`` already matches the GT file.
* APs is reported for completeness only: the PigDetect test split contains just 68 small instances
  (1.3 %), so the manuscript does not interpret APs rankings.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = json.load(open(args.gt, encoding="utf-8-sig"))
    stem2id = {os.path.splitext(im["file_name"])[0]: im["id"] for im in gt["images"]}
    preds = json.load(open(args.pred, encoding="utf-8-sig"))
    for p in preds:
        key = str(p["image_id"])
        if key in stem2id:
            p["image_id"] = stem2id[key]
        elif os.path.splitext(key)[0] in stem2id:
            p["image_id"] = stem2id[os.path.splitext(key)[0]]

    coco_gt = COCO(args.gt)
    coco_dt = coco_gt.loadRes(preds)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    res = {"gt": args.gt, "pred": args.pred, "AP": round(float(s[0]), 4), "AP50": round(float(s[1]), 4),
           "AP75": round(float(s[2]), 4), "APs": round(float(s[3]), 4), "APm": round(float(s[4]), 4),
           "APl": round(float(s[5]), 4)}
    print(res)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
