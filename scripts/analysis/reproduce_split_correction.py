# -*- coding: utf-8 -*-
"""Reproduce the split-correction deliverables: table19, table20 and figure 12.

Reads only committed result files:
    results/leakage_corrected_confirmatory/test_eval_all.csv   per-checkpoint test metrics (20 checkpoints)
    results/leakage_corrected_confirmatory/statistics.json     paired statistics of the corrected split
    results/original_split_last_eval/fig12_data.json           mean delta and interval per split and rule
    results/original_split_last_eval/statistics.json           paired statistics of the original split (last.pt)

Writes:
    results/generated/table19_leakage_corrected_confirmatory.csv
    results/generated/table20_corrected_statistics.csv
    figures/fig12_split_correction_effect.png

The two splits use the same frozen protocol and the same official test split (250 images); the
original split's best.pt values are the published five-seed table, its last.pt values were measured
afterwards from the same frozen checkpoints (results/original_split_last_eval/), and both
leakage-corrected checkpoint rules come from the confirmatory runs. No value is recomputed or
re-estimated here beyond formatting.

Run:  python scripts/analysis/reproduce_split_correction.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
LC = ROOT / "results" / "leakage_corrected_confirmatory"
ORIG = ROOT / "results" / "original_split_last_eval"
GEN = ROOT / "results" / "generated"
FIG = ROOT / "figures"
SEEDS = ["42", "1", "7", "21", "100"]


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def build_table19():
    recs = list(csv.DictReader((LC / "test_eval_all.csv").open(encoding="utf-8-sig")))
    idx = {(r["variant"], r["seed"], r["rule"]): float(r["mAP50-95"]) for r in recs}
    rows = []
    for s in SEEDS:
        e0b, e5b = idx[("E0", s, "best")], idx[("E5", s, "best")]
        e0l, e5l = idx[("E0", s, "last")], idx[("E5", s, "last")]
        rows.append([s, f"{e0b:.4f}", f"{e5b:.4f}", f"{e5b - e0b:+.4f}",
                     f"{e0l:.4f}", f"{e5l:.4f}", f"{e5l - e0l:+.4f}"])
    out = GEN / "table19_leakage_corrected_confirmatory.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        f.write("# source: results/leakage_corrected_confirmatory/test_eval_all.csv\n")
        w = csv.writer(f)
        w.writerow(["seed", "E0_best", "E5_best", "Delta_best", "E0_last", "E5_last", "Delta_last"])
        w.writerows(rows)
    print("wrote", out.relative_to(ROOT))
    return rows


def build_table20():
    st = load_json(LC / "statistics.json")["checkpoint_rules"]
    out = GEN / "table20_corrected_statistics.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        f.write("# source: results/leakage_corrected_confirmatory/statistics.json\n")
        w = csv.writer(f)
        w.writerow(["checkpoint", "mean_delta", "median_delta", "sample_SD", "bootstrap_CI_lower",
                    "bootstrap_CI_upper", "sign_flip_p_two_sided", "positive_negative_count"])
        for rule in ("best", "last"):
            v = st[rule]
            w.writerow([f"{rule}.pt", f"{v['mean_delta']:.4f}", f"{v['median_delta']:.4f}",
                        f"{v['sd_delta']:.4f}", f"{v['bootstrap']['ci_low']:.5f}",
                        f"{v['bootstrap']['ci_high']:.5f}", f"{v['signflip']['p_two_sided']:.4f}",
                        f"{v['positive_deltas']}/{v['negative_deltas']}"])
    print("wrote", out.relative_to(ROOT))
    return st


def build_fig12():
    data = load_json(ORIG / "fig12_data.json")["series"]
    splits = ["Original split", "Leakage-corrected\nsplit"]
    keys = {"best.pt": ("original_best", "corrected_best"),
            "last.pt": ("original_last", "corrected_last")}
    colors = {"best.pt": "#4C72B0", "last.pt": "#DD8452"}
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    x = [0, 1]
    width = 0.34
    for k, (rule, (ko, kc)) in enumerate(keys.items()):
        means = [data[ko]["mean_delta"], data[kc]["mean_delta"]]
        los = [data[ko]["ci95"][0], data[kc]["ci95"][0]]
        his = [data[ko]["ci95"][1], data[kc]["ci95"][1]]
        err = [[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]]
        pos = [xi + (k - 0.5) * width for xi in x]
        ax.bar(pos, means, width, label=rule, color=colors[rule],
               yerr=err, capsize=5, error_kw={"elinewidth": 1.2, "ecolor": "#333333"})
        for p, m in zip(pos, means):
            ax.annotate(f"{m:+.4f}", (p, m), textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=9)
    ax.axhline(0, color="#666666", linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Mean $\\Delta$mAP@0.5:0.95 (E5 $-$ E0)")
    ax.set_title("Effect of CA + SIoU under the original and the\nleakage-corrected split",
                 fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_ylim(-0.003, 0.008)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    out = FIG / "fig12_split_correction_effect.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    build_table19()
    build_table20()
    build_fig12()
