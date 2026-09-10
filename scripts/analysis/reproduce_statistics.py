# -*- coding: utf-8 -*-
"""Recompute every statistic reported in the manuscript from the locked result files.

No GPU, no PyTorch, no Ultralytics and no dataset images are required: this script only reads the
locked result files shipped inside ``results/``.

Outputs (``results/generated/``):
  statistics.json   machine-readable summary
  statistics.md     human-readable summary (all numbers quoted in the manuscript)

Usage:
  python scripts/analysis/reproduce_statistics.py
  python scripts/analysis/reproduce_statistics.py --figure   # also regenerate Figure 10

Algorithmic note: the computations here are the same as those used for the manuscript
(exhaustive enumeration, no random sampling):
  * paired bootstrap: all 5**5 = 3125 ordered resamples of the 5 paired deltas
  * sign-flip permutation: all 2**5 = 32 (and 2**3 = 8) sign configurations
  * sample standard deviation (n-1 denominator)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
OUT = RES / "generated"


# ----------------------------------------------------------------------------- helpers
def load_json(path: Path):
    # utf-8-sig tolerates an optional BOM (some editors write one)
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def paired_bootstrap_ci(diffs, iters=None, alpha=0.025):
    """Exhaustive paired bootstrap over all n**n ordered resamples (percentile CI)."""
    n = len(diffs)
    means = []
    for combo in itertools.product(range(n), repeat=n):
        means.append(sum(diffs[i] for i in combo) / n)
    means.sort()
    lo = means[int(alpha * (len(means) - 1))]
    hi = means[int((1 - alpha) * (len(means) - 1))]
    return lo, hi, len(means)


def signflip_p(diffs):
    """Exact paired sign-flip permutation test (two-sided and one-sided p)."""
    obs = abs(statistics.mean(diffs))
    total = two = one = 0
    for signs in itertools.product([1, -1], repeat=len(diffs)):
        total += 1
        m = sum(s * d for s, d in zip(signs, diffs)) / len(diffs)
        if abs(m) >= obs:
            two += 1
        if m >= obs:
            one += 1
    return two / total, one / total, total


def mean_sd(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def spearman(a, b):
    ra = {v: i for i, v in enumerate(sorted(a))}
    rb = {v: i for i, v in enumerate(sorted(b))}
    x = [ra[v] + 1 for v in a]
    y = [rb[v] + 1 for v in b]
    mx, my = statistics.mean(x), statistics.mean(y)
    cov = sum((i - mx) * (j - my) for i, j in zip(x, y))
    vx = sum((i - mx) ** 2 for i in x)
    vy = sum((j - my) ** 2 for j in y)
    return cov / (vx * vy) ** 0.5 if vx and vy else 0.0


# ----------------------------------------------------------------------------- analyses
def pigdetect_multiseed():
    data = load_json(RES / "statistics" / "multiseed_pigdetect.json")
    out = {}
    for model in ("E0", "E1", "E3", "E5"):
        vals = {r["seed"]: r["mAP50-95"] for r in data if r["model"] == model}
        if not vals:
            continue
        m, sd = mean_sd(list(vals.values()))
        out[model] = {"per_seed": vals, "mean": round(m, 4), "sd": round(sd, 4), "n": len(vals)}

    e0 = out["E0"]["per_seed"]
    e5 = out["E5"]["per_seed"]
    seeds = [s for s in ("42", "1", "7", "21", "100") if s in e0 and s in e5]
    diffs = [e5[s] - e0[s] for s in seeds]
    lo, hi, n_res = paired_bootstrap_ci(diffs)
    p_two, p_one, n_signs = signflip_p(diffs)
    sd_d = statistics.stdev(diffs)
    out["E5_minus_E0"] = {
        "seeds": seeds,
        "per_seed_delta": [round(d, 4) for d in diffs],
        "mean_delta": round(statistics.mean(diffs), 4),
        "sd_delta": round(sd_d, 4),
        "bootstrap_ci95": [round(lo, 5), round(hi, 5)],
        "bootstrap_resamples": n_res,
        "signflip_p_two_sided": round(p_two, 4),
        "signflip_p_one_sided": round(p_one, 4),
        "signflip_configurations": n_signs,
        "cohens_dz_exploratory": round(statistics.mean(diffs) / sd_d, 2),
    }
    return out


def piglife_best_last():
    best = load_json(RES / "statistics" / "piglife_test_best.json")
    last = load_json(RES / "statistics" / "piglife_test_last.json")
    e2 = load_json(RES / "statistics" / "piglife_e2_metric.json")

    b = {f'{r["model"]}_{r["seed"]}': r["mAP50-95"] for r in best}
    l = {r["run"]: r["mAP50-95"] for r in last}
    seeds = ("42", "1", "7")
    out = {"checkpoint_rules": {}}

    for rule, table in (("best", b), ("last", l)):
        e0 = [table[f"E0_{s}"] for s in seeds]
        e5 = [table[f"E5_{s}"] for s in seeds]
        deltas = [x - y for x, y in zip(e5, e0)]
        m0, s0 = mean_sd(e0)
        m5, s5 = mean_sd(e5)
        p_two, p_one, n_signs = signflip_p(deltas)
        out["checkpoint_rules"][rule] = {
            "E0_per_seed": [round(v, 4) for v in e0],
            "E5_per_seed": [round(v, 4) for v in e5],
            "E0_mean": round(m0, 4), "E0_sd": round(s0, 4),
            "E5_mean": round(m5, 4), "E5_sd": round(s5, 4),
            "per_seed_delta": [round(d, 4) for d in deltas],
            "mean_delta": round(statistics.mean(deltas), 4),
            "sd_delta": round(statistics.stdev(deltas), 4),
            "signflip_p_two_sided": round(p_two, 4),
            "signflip_configurations": n_signs,
        }

    out["single_module_seed42"] = {
        "E1": b.get("E1_42"), "E2": e2.get("mAP50-95"), "E3": b.get("E3_42"), "E0": b.get("E0_42"),
        "E2_last": l.get("E2_42"),
    }
    return out


def yolo11_replication():
    raw = load_json(RES / "yolo11_replication" / "test_audit_20260909.json")
    rows, pos = {}, 0
    for s in ("42", "1", "7"):
        rows[s] = {"base_best": raw[pos]["mAP50-95"], "ca_best": raw[pos + 1]["mAP50-95"],
                   "base_last": raw[pos + 2]["mAP50-95"], "ca_last": raw[pos + 3]["mAP50-95"]}
        pos += 4
    out = {"per_seed": {s: {k: round(v, 4) for k, v in r.items()} for s, r in rows.items()}}
    for rule in ("best", "last"):
        d = [rows[s][f"ca_{rule}"] - rows[s][f"base_{rule}"] for s in ("42", "1", "7")]
        m, sd = mean_sd(d)
        p_two, _, n_signs = signflip_p(d)
        out[rule] = {"per_seed_delta": [round(x, 4) for x in d], "mean_delta": round(m, 4),
                     "sd_delta": round(sd, 4), "signflip_p_two_sided": round(p_two, 4),
                     "signflip_configurations": n_signs,
                     "positive_directions": sum(1 for x in d if x > 0)}
    out["rank_flip"] = {
        s: ("flip" if ((rows[s]["ca_best"] > rows[s]["base_best"]) !=
                       (rows[s]["ca_last"] > rows[s]["base_last"])) else "no_flip")
        for s in ("42", "1", "7")}
    return out


def rank_stability_pigdetect():
    data = load_json(RES / "statistics" / "multiseed_pigdetect.json")
    models = ["E0", "E1", "E3", "E5"]
    seeds = ["42", "1", "7"]
    table = {m: {s: v["mAP50-95"] for v in data if v["model"] == m and (s := v["seed"]) in seeds}
             for m in models}
    order, ranks = {}, {m: [] for m in models}
    for s in seeds:
        o = sorted(models, key=lambda m: -table[m][s])
        order[s] = o
        for i, m in enumerate(o):
            ranks[m].append(i + 1)
    return {
        "order_per_seed": order,
        "mean_rank": {m: round(statistics.mean(ranks[m]), 3) for m in models},
        "rank_sd": {m: round(statistics.stdev(ranks[m]), 3) for m in models},
        "rank1_count": {m: sum(1 for r in ranks[m] if r == 1) for m in models},
        "spearman": {
            "42_vs_1": round(spearman([table[m]["42"] for m in models], [table[m]["1"] for m in models]), 2),
            "42_vs_7": round(spearman([table[m]["42"] for m in models], [table[m]["7"] for m in models]), 2),
            "1_vs_7": round(spearman([table[m]["1"] for m in models], [table[m]["7"] for m in models]), 2),
        },
    }


def make_figure(stats):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[figure] matplotlib unavailable ({exc}); skipped")
        return None
    d26 = stats["pigdetect_multiseed"]["E5_minus_E0"]["per_seed_delta"]
    ci = stats["pigdetect_multiseed"]["E5_minus_E0"]["bootstrap_ci95"]
    y11 = stats["yolo11_replication"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.axhspan(ci[0], ci[1], color="#C0392B", alpha=0.12)
    ax.plot(range(len(d26)), d26, "o-", color="#C0392B")
    ax.axhline(statistics.mean(d26), color="#C0392B", ls=":", lw=1)
    ax.set_xticks(range(len(d26)))
    ax.set_xticklabels([f"seed{s}" for s in stats["pigdetect_multiseed"]["E5_minus_E0"]["seeds"]])
    ax.set_title("(a) YOLO26s: E5 − E0 mAP@0.5:0.95")
    ax.set_ylabel("Δ mAP@0.5:0.95")
    ax = axes[1]
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.plot(range(3), y11["best"]["per_seed_delta"], "s-", color="#1F618D", label="best.pt")
    ax.plot(range(3), y11["last"]["per_seed_delta"], "^-", color="#7D3C98", label="last.pt")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["seed42", "seed1", "seed7"])
    ax.set_title("(b) YOLO11s: CA − baseline mAP@0.5:0.95")
    ax.legend(fontsize=8)
    fig.tight_layout()
    dst = REPO / "figures" / "fig10_cross_detector_effects.png"
    fig.savefig(dst, dpi=200)
    plt.close(fig)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true", help="also regenerate Figure 10")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    stats = {
        "pigdetect_multiseed": pigdetect_multiseed(),
        "piglife_best_last": piglife_best_last(),
        "yolo11_replication": yolo11_replication(),
        "rank_stability_pigdetect": rank_stability_pigdetect(),
    }
    with open(OUT / "statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    pm = stats["pigdetect_multiseed"]
    pl = stats["piglife_best_last"]["checkpoint_rules"]
    y11 = stats["yolo11_replication"]
    lines = [
        "# Recomputed statistics (generated from locked result files)",
        "",
        "## PigDetect multi-seed (mAP@0.5:0.95, test 250)",
        "",
        "| Model | n | mean | SD | per-seed |",
        "|---|---:|---:|---:|---|",
    ]
    for m in ("E0", "E1", "E3", "E5"):
        if m in pm:
            v = pm[m]
            lines.append(f"| {m} | {v['n']} | {v['mean']} | {v['sd']} | {v['per_seed']} |")
    d = pm["E5_minus_E0"]
    lines += [
        "",
        f"Paired E5−E0 deltas: {d['per_seed_delta']} (mean {d['mean_delta']:+.4f}, SD {d['sd_delta']:.4f})",
        f"Exhaustive paired bootstrap over {d['bootstrap_resamples']} resamples: 95% percentile CI "
        f"[{d['bootstrap_ci95'][0]:.5f}, {d['bootstrap_ci95'][1]:.5f}]",
        f"Exact sign-flip permutation ({d['signflip_configurations']} configurations): "
        f"two-sided p = {d['signflip_p_two_sided']:.4f}",
        f"Cohen's d_z (exploratory only): {d['cohens_dz_exploratory']}",
        "",
        "## PigLife (test 426, E0 vs E5)",
        "",
        "| Rule | E0 mean±SD | E5 mean±SD | per-seed Δ | mean Δ | sign-flip p |",
        "|---|---|---|---:|---:|---:|",
    ]
    for rule, v in pl.items():
        lines.append(f"| {rule}.pt | {v['E0_mean']}±{v['E0_sd']} | {v['E5_mean']}±{v['E5_sd']} | "
                     f"{v['per_seed_delta']} | {v['mean_delta']:+.4f} | {v['signflip_p_two_sided']:.4f} |")
    lines += [
        "",
        "## YOLO11s replication (CA − baseline, test 250)",
        "",
        "| Rule | per-seed Δ | mean±SD | positive | sign-flip p |",
        "|---|---|---:|---:|---:|",
    ]
    for rule in ("best", "last"):
        v = y11[rule]
        lines.append(f"| {rule}.pt | {v['per_seed_delta']} | {v['mean_delta']:+.4f}±{v['sd_delta']:.4f} | "
                     f"{v['positive_directions']}/3 | {v['signflip_p_two_sided']:.4f} |")
    lines += ["", f"Rank flips (best vs last): {y11['rank_flip']}", "",
              "## YOLO26s rank stability (E0/E1/E3/E5, common seeds 42/1/7)", ""]
    rs = stats["rank_stability_pigdetect"]
    for s, order in rs["order_per_seed"].items():
        lines.append(f"- seed{s}: " + " > ".join(order))
    lines.append(f"- mean rank: {rs['mean_rank']}; rank-1 counts: {rs['rank1_count']}")
    lines.append(f"- Spearman: {rs['spearman']}")
    with open(OUT / "statistics.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nsaved -> {OUT / 'statistics.json'} , {OUT / 'statistics.md'}")
    if args.figure:
        dst = make_figure(stats)
        if dst:
            print("figure ->", dst)


if __name__ == "__main__":
    main()
