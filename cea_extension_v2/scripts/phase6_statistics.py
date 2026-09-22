# -*- coding: utf-8 -*-
"""Phase 6 / phase 12 statistics: paired E5 - E0 analysis for any checkpoint-evaluation CSV.

Rules are fixed by experiments/cea_extension_v2/PROTOCOL_LOCK.md:
  * per-seed raw paired deltas are always reported
  * mean, median, sample SD (n-1), min, max, positive/negative/zero pair counts
  * paired bootstrap: 100,000 resamples (with replacement) over the paired deltas,
    RNG seed 20260916, percentile 95% CI  (exhaustive 10^10 enumeration is explicitly not used)
  * exact paired sign-flip permutation: full enumeration of 2^n configurations, one- and two-sided
    (the enumeration function is imported from scripts/stats_lc_confirmatory.py, i.e. the same
    implementation the manuscript already uses)
  * best.pt and last.pt are analysed separately and never pooled

Usage
-----
python phase6_statistics.py --input <eval csv> --out-dir <dir> --label pigdetect_corrected
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np

PJ = Path(r"<PIG_PROJECT>")
sys.path.insert(0, str(PJ / "scripts"))
from stats_lc_confirmatory import signflip_counts  # noqa: E402

BOOTSTRAP_R = 100_000
BOOTSTRAP_SEED = 20260916


def paired_bootstrap_random(deltas, rng_seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_R):
    """Percentile 95% CI from `resamples` paired bootstrap means (sampling the deltas with
    replacement). The RNG seed is fixed by the protocol lock and returned for the record."""
    d = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(d), size=(resamples, len(d)))
    means = d[idx].mean(axis=1)
    means.sort()
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return lo, hi, means, rng_seed, resamples


def analyse(pairs, exact_pairs, metric="mAP50_95"):
    """Float pairs serve descriptive/bootstrap statistics; decimal pairs serve sign-flip."""
    deltas = [b - a for _, a, b in pairs]
    exact_deltas = [b - a for _, a, b in exact_pairs]
    e0 = [a for _, a, _ in pairs]
    e5 = [b for _, _, b in pairs]
    lo, hi, means, seed, r = paired_bootstrap_random(deltas)
    two_count, one_count, ncfg = signflip_counts(exact_deltas)
    out = {
        "n_seeds": len(deltas),
        "seeds": [s for s, _, _ in pairs],
        "per_seed": {str(s): {"E0": a, "E5": b, "delta": round(b - a, 6)} for s, a, b in pairs},
        "E0_mean": round(statistics.mean(e0), 6),
        "E0_sample_sd": round(statistics.stdev(e0), 6) if len(e0) > 1 else 0.0,
        "E5_mean": round(statistics.mean(e5), 6),
        "E5_sample_sd": round(statistics.stdev(e5), 6) if len(e5) > 1 else 0.0,
        "per_seed_delta": [round(d, 6) for d in deltas],
        "mean_delta": round(statistics.mean(deltas), 6),
        "median_delta": round(statistics.median(deltas), 6),
        "sample_SD_delta": round(statistics.stdev(deltas), 6) if len(deltas) > 1 else 0.0,
        "min_delta": round(min(deltas), 6),
        "max_delta": round(max(deltas), 6),
        "positive_pairs": sum(1 for d in deltas if d > 0),
        "negative_pairs": sum(1 for d in deltas if d < 0),
        "zero_pairs": sum(1 for d in deltas if d == 0),
        "bootstrap": {"type": "paired bootstrap, sampling the paired deltas with replacement",
                      "resamples": r, "rng_seed": seed, "method": "percentile 95% CI",
                      "ci_low": round(lo, 6), "ci_high": round(hi, 6),
                      "contains_zero": lo <= 0 <= hi,
                      "resample_means_file": None},
        "signflip": {"type": "exact paired sign-flip permutation, full enumeration; decimal/rational arithmetic; ties included",
                     "configurations": ncfg,
                     "one_sided_extreme_count": one_count,
                     "two_sided_extreme_count": two_count,
                     "p_one_sided": one_count / ncfg, "p_two_sided": two_count / ncfg,
                     "min_attainable_two_sided_p": round(2 / ncfg, 6)},
    }
    return out, means, deltas, exact_deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="checkpoint evaluation CSV")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="pigdetect_corrected")
    ap.add_argument("--prefix", default="statistics_summary",
                    help="output filename stem (use e.g. oinktrack_statistics_summary for OinkTrack)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.input, encoding="utf-8-sig")))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"generated": datetime.now().isoformat(timespec="seconds"),
               "input": str(Path(args.input)), "label": args.label,
               "rules": {"bootstrap_resamples": BOOTSTRAP_R, "bootstrap_rng_seed": BOOTSTRAP_SEED,
                         "signflip": "exact decimal/rational enumeration of 2^n configurations; |T_perm| >= |T_obs|; ties included",
                         "checkpoint_rules_analysed_separately": ["best", "last"]},
               "checkpoint_rules": {}}

    for rule in ("best", "last"):
        sub = [r for r in rows if r["checkpoint_rule"] == rule]
        if not sub:
            continue
        by_seed = {}
        exact_by_seed = {}
        for r in sub:
            by_seed.setdefault(int(r["seed"]), {})[r["model"]] = float(r["mAP50_95"])
            exact_by_seed.setdefault(int(r["seed"]), {})[r["model"]] = Decimal(r["mAP50_95"])
        pairs = [(s, v["E0"], v["E5"]) for s, v in sorted(by_seed.items())
                 if "E0" in v and "E5" in v]
        exact_pairs = [(s, exact_by_seed[s]["E0"], exact_by_seed[s]["E5"])
                       for s, _, _ in pairs]
        if not pairs:
            # Single-arm family (e.g. the RT-DETR boundary baseline has no E0/E5 pairing).
            # Previously this crashed inside the exact sign-flip enumeration with
            # "mean requires at least one data point" (anomaly #19). Report it explicitly.
            models = sorted({m for v in by_seed.values() for m in v})
            summary["checkpoint_rules"][rule] = {
                "status": "no_paired_E0_E5_seeds",
                "models_present": models,
                "seeds_present": sorted(by_seed),
                "note": "this script analyses PAIRED E5-E0 deltas; a single-arm family needs "
                        "descriptive statistics instead (see phase7e_rtdetr_boundary_stats.py)",
            }
            print(f"[skip] {rule}: no paired E0/E5 seeds (models present: {models}) - "
                  f"descriptive statistics only")
            continue
        res, means, deltas, exact_deltas = analyse(pairs, exact_pairs)
        summary["checkpoint_rules"][rule] = res

        with open(out_dir / f"paired_deltas_{rule}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["seed", "E0_mAP50_95", "E5_mAP50_95", "delta_E5_minus_E0"])
            for (s, a, b), d in zip(pairs, deltas):
                w.writerow([s, a, b, round(d, 6)])
        with open(out_dir / f"bootstrap_{rule}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["resample_index", "bootstrap_mean_delta"])
            for i, m in enumerate(means):
                w.writerow([i, round(float(m), 8)])
        with open(out_dir / f"signflip_exact_{rule}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["configuration", "signs", "mean_delta", "abs_mean_delta"])
            import itertools
            for k, signs in enumerate(itertools.product([1, -1], repeat=len(exact_deltas))):
                m = sum(sg * d for sg, d in zip(signs, exact_deltas)) / len(exact_deltas)
                w.writerow([k, "".join("+" if sg > 0 else "-" for sg in signs),
                            round(m, 8), round(abs(m), 8)])
        # Label-prefixed copies: the generic names above are shared by every family, so running
        # this script for PigDetect and then for OinkTrack used to overwrite the same three files
        # with no trace of which analysis they belong to. The copies keep both (anomaly #19).
        import shutil
        for stem in ("paired_deltas", "bootstrap", "signflip_exact"):
            shutil.copy2(out_dir / f"{stem}_{rule}.csv",
                         out_dir / f"{args.prefix}_{stem}_{rule}.csv")
        summary["checkpoint_rules"][rule]["bootstrap"]["resample_means_file"] = \
            f"bootstrap_{rule}.csv"

    (out_dir / f"{args.prefix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# Paired statistics — {args.label}", "",
             f"Input: `{Path(args.input).name}`  ",
             f"Generated: {summary['generated']}  ",
             f"Bootstrap: {BOOTSTRAP_R:,} paired resamples, RNG seed {BOOTSTRAP_SEED}, percentile 95% CI  ",
             f"Sign-flip: exact decimal/rational enumeration (2^n configurations); "
             f"|T_perm| >= |T_obs|, ties included. "
             f"best.pt and last.pt are reported separately; no seed was dropped or selected.", ""]
    for rule, v in summary["checkpoint_rules"].items():
        lines += [f"## {rule}.pt ({v['n_seeds']} matched seeds)", "",
                  "| seed | E0 | E5 | E5-E0 |", "|---|---:|---:|---:|"]
        for s in v["seeds"]:
            d = v["per_seed"][str(s)]
            lines.append(f"| {s} | {d['E0']:.4f} | {d['E5']:.4f} | {d['delta']:+.4f} |")
        lines += ["",
                  f"- E0 {v['E0_mean']:.4f} ± {v['E0_sample_sd']:.4f} (sample SD) | "
                  f"E5 {v['E5_mean']:.4f} ± {v['E5_sample_sd']:.4f}",
                  f"- paired deltas: {[round(d, 4) for d in v['per_seed_delta']]}",
                  f"- mean {v['mean_delta']:+.4f} | median {v['median_delta']:+.4f} | "
                  f"sample SD {v['sample_SD_delta']:.4f} | range "
                  f"[{v['min_delta']:+.4f}, {v['max_delta']:+.4f}] | "
                  f"positive/negative/zero = {v['positive_pairs']}/{v['negative_pairs']}/"
                  f"{v['zero_pairs']}",
                  f"- bootstrap 95% CI [{v['bootstrap']['ci_low']:+.5f}, "
                  f"{v['bootstrap']['ci_high']:+.5f}] (contains zero: "
                  f"{v['bootstrap']['contains_zero']})",
                  f"- exact sign-flip p (two-sided) {v['signflip']['p_two_sided']:.4f}, "
                  f"(one-sided) {v['signflip']['p_one_sided']:.4f}; minimum attainable two-sided p "
                  f"{v['signflip']['min_attainable_two_sided_p']:.4f}",
                  ""]
    (out_dir / f"{args.prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("written ->", out_dir)


if __name__ == "__main__":
    main()
