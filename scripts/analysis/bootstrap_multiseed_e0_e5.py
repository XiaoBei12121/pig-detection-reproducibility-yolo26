# -*- coding: utf-8 -*-
"""Exhaustive paired bootstrap for the PigDetect E0/E5 five-seed comparison.

Reads:   results/statistics/multiseed_pigdetect.json
Writes:  results/generated/bootstrap_multiseed_e0_e5.json (+ console table)

All 5**5 = 3125 ordered bootstrap resamples of the five paired deltas are enumerated, so the
percentile confidence interval is exact and does not depend on a random seed. No GPU / dataset needed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results" / "statistics" / "multiseed_pigdetect.json"
OUT = REPO / "results" / "generated"


def load(path: Path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", default="E5", help="treatment model")
    ap.add_argument("--model-b", default="E0", help="baseline model")
    ap.add_argument("--alpha", type=float, default=0.025, help="tail probability for the percentile CI")
    args = ap.parse_args()

    data = load(SRC)
    a = {r["seed"]: r["mAP50-95"] for r in data if r["model"] == args.model_a}
    b = {r["seed"]: r["mAP50-95"] for r in data if r["model"] == args.model_b}
    seeds = [s for s in ("42", "1", "7", "21", "100") if s in a and s in b]
    diffs = [a[s] - b[s] for s in seeds]
    n = len(diffs)

    means = sorted(sum(diffs[i] for i in combo) / n
                   for combo in itertools.product(range(n), repeat=n))
    lo = means[int(args.alpha * (len(means) - 1))]
    hi = means[int((1 - args.alpha) * (len(means) - 1))]
    mean_d = statistics.mean(diffs)
    sd_d = statistics.stdev(diffs)

    res = {
        "comparison": f"{args.model_a} - {args.model_b}",
        "metric": "mAP@0.5:0.95 (PigDetect exploratory test, 250 images)",
        "seeds": seeds,
        "per_seed_delta": [round(d, 4) for d in diffs],
        "mean_delta": round(mean_d, 4),
        "sd_delta": round(sd_d, 4),
        "bootstrap": {
            "method": "exhaustive paired bootstrap, percentile CI, all n**n ordered resamples",
            "resamples": len(means),
            "ci_level": 0.95,
            "ci_low": round(lo, 5),
            "ci_high": round(hi, 5),
            "contains_zero": lo <= 0.0 <= hi,
        },
        "cohens_dz_exploratory": round(mean_d / sd_d, 2),
        "note": ("n = 5 is small: the interval quantifies uncertainty about a stable positive gain and "
                 "must not be read as evidence that the two models are equivalent."),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "bootstrap_multiseed_e0_e5.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"{res['comparison']}  seeds={seeds}")
    print(f"per-seed deltas : {res['per_seed_delta']}")
    print(f"mean +{mean_d:.4f}  SD {sd_d:.4f}")
    print(f"exhaustive bootstrap ({len(means)} resamples): 95% CI [{lo:.5f}, {hi:.5f}] "
          f"(contains 0: {lo <= 0.0 <= hi})")
    print(f"Cohen's d_z (exploratory): {res['cohens_dz_exploratory']}")
    print("saved ->", OUT / "bootstrap_multiseed_e0_e5.json")


if __name__ == "__main__":
    main()
