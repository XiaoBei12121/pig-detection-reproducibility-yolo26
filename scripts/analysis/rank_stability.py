# -*- coding: utf-8 -*-
"""Rank-stability analysis and Figure 11.

Outputs
  results/generated/rank_stability.json          (tables, machine readable)
  results/generated/rank_stability.md            (tables, human readable)
  figures/fig11_rank_stability.png               (a: YOLO26s rank heatmap, b: YOLO11s rank flip)

Inputs (locked result files, no GPU/dataset needed):
  results/statistics/multiseed_pigdetect.json
  results/yolo11_replication/test_audit_20260909.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
OUT = RES / "generated"
MODELS = ["E0", "E1", "E3", "E5"]
SEEDS = ["42", "1", "7"]


def load(p: Path):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    ms = load(RES / "statistics" / "multiseed_pigdetect.json")
    table = {m: {} for m in MODELS}
    for r in ms:
        if r["model"] in table and r["seed"] in SEEDS:
            table[r["model"]][r["seed"]] = r["mAP50-95"]

    order, ranks = {}, {m: [] for m in MODELS}
    for s in SEEDS:
        o = sorted(MODELS, key=lambda m: -table[m][s])
        order[s] = o
        for i, m in enumerate(o):
            ranks[m].append(i + 1)

    y = load(RES / "yolo11_replication" / "test_audit_20260909.json")
    rows, pos = {}, 0
    for s in SEEDS:
        rows[s] = {"base_best": y[pos]["mAP50-95"], "ca_best": y[pos + 1]["mAP50-95"],
                   "base_last": y[pos + 2]["mAP50-95"], "ca_last": y[pos + 3]["mAP50-95"]}
        pos += 4
    flips = {}
    for s in SEEDS:
        v = rows[s]
        best_winner = "+CA" if v["ca_best"] > v["base_best"] else "baseline"
        last_winner = "+CA" if v["ca_last"] > v["base_last"] else "baseline"
        flips[s] = {"best_winner": best_winner, "last_winner": last_winner,
                    "flip": best_winner != last_winner,
                    "delta_best": round(v["ca_best"] - v["base_best"], 4),
                    "delta_last": round(v["ca_last"] - v["base_last"], 4)}

    res = {
        "pigdetect_rank_order": order,
        "pigdetect_mean_rank": {m: round(statistics.mean(ranks[m]), 3) for m in MODELS},
        "pigdetect_rank_sd": {m: round(statistics.stdev(ranks[m]), 3) for m in MODELS},
        "pigdetect_rank1_count": {m: sum(1 for r in ranks[m] if r == 1) for m in MODELS},
        "pigdetect_spearman": {
            "42_vs_1": round(spearman([table[m]["42"] for m in MODELS], [table[m]["1"] for m in MODELS]), 2),
            "42_vs_7": round(spearman([table[m]["42"] for m in MODELS], [table[m]["7"] for m in MODELS]), 2),
            "1_vs_7": round(spearman([table[m]["1"] for m in MODELS], [table[m]["7"] for m in MODELS]), 2),
        },
        "yolo11_best_vs_last": flips,
        "yolo11_flip_count": sum(1 for v in flips.values() if v["flip"]),
        "note": ("Only 4 models and 3 common seeds are available for the YOLO26s ranking and only 2 arms "
                 "for YOLO11s; Spearman coefficients and flip counts are descriptive only."),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "rank_stability.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    lines = ["# Rank stability", "", "## YOLO26s (E0/E1/E3/E5, common seeds 42/1/7)", "",
             "| Seed | rank1 | rank2 | rank3 | rank4 |", "|---|---|---|---|---|"]
    for s in SEEDS:
        lines.append(f"| {s} | " + " | ".join(order[s]) + " |")
    lines += ["", f"- mean rank: {res['pigdetect_mean_rank']}",
              f"- rank SD: {res['pigdetect_rank_sd']}",
              f"- rank-1 counts: {res['pigdetect_rank1_count']}",
              f"- Spearman: {res['pigdetect_spearman']}", "",
              "## YOLO11s best.pt vs last.pt (baseline vs +CA)", "",
              "| Seed | better (best.pt) | better (last.pt) | flip | Δbest | Δlast |",
              "|---|---|---|---|---:|---:|"]
    for s in SEEDS:
        v = flips[s]
        lines.append(f"| {s} | {v['best_winner']} | {v['last_winner']} | "
                     f"{'YES' if v['flip'] else 'no'} | {v['delta_best']:+.4f} | {v['delta_last']:+.4f} |")
    lines += ["", f"- flips: {res['yolo11_flip_count']}/3 seeds", "", res["note"]]
    with open(OUT / "rank_stability.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    if not args.no_figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            rank = np.array([[ranks[m][i] for i in range(3)] for m in MODELS], dtype=float)
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
            ax = axes[0]
            ax.imshow(rank, cmap="RdYlGn_r", vmin=1, vmax=4)
            ax.set_xticks(range(3), [f"seed{s}" for s in SEEDS])
            ax.set_yticks(range(len(MODELS)), MODELS)
            for i in range(len(MODELS)):
                for j in range(3):
                    ax.text(j, i, int(rank[i, j]), ha="center", va="center")
            ax.set_title("(a) YOLO26s rank (1 = best)")
            ax = axes[1]
            for k, s in enumerate(SEEDS):
                v = rows[s]
                rb = {"baseline": 1 if v["base_best"] >= v["ca_best"] else 2,
                      "+CA": 1 if v["ca_best"] > v["base_best"] else 2}
                rl = {"baseline": 1 if v["base_last"] >= v["ca_last"] else 2,
                      "+CA": 1 if v["ca_last"] > v["base_last"] else 2}
                style = "-" if rb == rl else "--"
                for name, c in (("baseline", "#1F618D"), ("+CA", "#7D3C98")):
                    ax.plot([0, 1], [rb[name], rl[name]], style, marker="o", color=c,
                            label=name if k == 0 else None)
            ax.set_xticks([0, 1], ["best.pt", "last.pt"])
            ax.set_yticks([1, 2], ["rank 1", "rank 2"])
            ax.invert_yaxis()
            ax.set_title("(b) YOLO11s rank flip (dashed = flip)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            dst = REPO / "figures" / "fig11_rank_stability.png"
            fig.savefig(dst, dpi=200)
            plt.close(fig)
            print("figure ->", dst)
        except Exception as exc:
            print(f"[figure] skipped ({exc})")


if __name__ == "__main__":
    main()
