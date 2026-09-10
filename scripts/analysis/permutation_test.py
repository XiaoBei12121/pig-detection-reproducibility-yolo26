# -*- coding: utf-8 -*-
"""Exact paired sign-flip permutation tests (no sampling, exhaustive enumeration).

Covered comparisons (each is a paired, same-seed comparison):
  * PigDetect E5 − E0                      (n = 5 → 2^5 = 32 configurations)
  * PigLife   E5 − E0, best.pt and last.pt (n = 3 → 2^3 = 8 configurations)
  * YOLO11s   CA − baseline, best and last (n = 3 → 2^3 = 8 configurations)

Reads only the locked result files shipped in results/. Writes
results/generated/permutation_tests.json and prints a table.

Interpretation note: with n = 3 or 5 the p-value grid is coarse (the smallest attainable two-sided
p is 2/2^n). A non-significant result is therefore reported as "no evidence of a stable gain" and
must not be turned into a claim of equivalence.
"""
from __future__ import annotations

import itertools
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
OUT = RES / "generated"


def load(p: Path):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def signflip(diffs):
    obs = abs(statistics.mean(diffs))
    two = one = total = 0
    for signs in itertools.product([1, -1], repeat=len(diffs)):
        total += 1
        m = sum(s * d for s, d in zip(signs, diffs)) / len(diffs)
        if abs(m) >= obs:
            two += 1
        if m >= obs:
            one += 1
    return {"mean_delta": round(statistics.mean(diffs), 4),
            "sd_delta": round(statistics.stdev(diffs), 4) if len(diffs) > 1 else 0.0,
            "per_seed_delta": [round(d, 4) for d in diffs],
            "configurations": total,
            "p_two_sided": round(two / total, 4),
            "p_one_sided": round(one / total, 4),
            "min_attainable_two_sided_p": round(2 / total, 4)}


def main():
    out = {}

    # PigDetect 5-seed
    ms = load(RES / "statistics" / "multiseed_pigdetect.json")
    e0 = {r["seed"]: r["mAP50-95"] for r in ms if r["model"] == "E0"}
    e5 = {r["seed"]: r["mAP50-95"] for r in ms if r["model"] == "E5"}
    seeds = [s for s in ("42", "1", "7", "21", "100") if s in e0 and s in e5]
    out["pigdetect_E5_minus_E0"] = signflip([e5[s] - e0[s] for s in seeds])

    # PigLife best / last
    best = {f'{r["model"]}_{r["seed"]}': r["mAP50-95"] for r in load(RES / "statistics" / "piglife_test_best.json")}
    last = {r["run"]: r["mAP50-95"] for r in load(RES / "statistics" / "piglife_test_last.json")}
    for rule, table in (("best", best), ("last", last)):
        diffs = [table[f"E5_{s}"] - table[f"E0_{s}"] for s in ("42", "1", "7")]
        out[f"piglife_E5_minus_E0_{rule}"] = signflip(diffs)

    # YOLO11s replication best / last
    y = load(RES / "yolo11_replication" / "test_audit_20260909.json")
    rows, pos = {}, 0
    for s in ("42", "1", "7"):
        rows[s] = {"base_best": y[pos]["mAP50-95"], "ca_best": y[pos + 1]["mAP50-95"],
                   "base_last": y[pos + 2]["mAP50-95"], "ca_last": y[pos + 3]["mAP50-95"]}
        pos += 4
    for rule in ("best", "last"):
        out[f"yolo11_CA_minus_baseline_{rule}"] = signflip(
            [rows[s][f"ca_{rule}"] - rows[s][f"base_{rule}"] for s in ("42", "1", "7")])

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "permutation_tests.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"{'comparison':<34}{'n':>3}{'mean':>9}{'p(two-sided)':>14}{'min p':>8}")
    for k, v in out.items():
        n = len(v["per_seed_delta"])
        print(f"{k:<34}{n:>3}{v['mean_delta']:>+9.4f}{v['p_two_sided']:>14.4f}"
              f"{v['min_attainable_two_sided_p']:>8.4f}")
    print("\nsaved ->", OUT / "permutation_tests.json")


if __name__ == "__main__":
    main()
