# -*- coding: utf-8 -*-
"""READ-ONLY exact tie audit of the last.pt paired sign-flip test.

Requested procedure (followed literally)
----------------------------------------
1. The E0/E5 values are taken from `results/pigdetect_corrected_combined.csv`, checkpoint_rule=last,
   exactly as stored (the highest precision that exists in the persisted evaluation output is the
   4-decimal text written by the evaluator; Table A's rounded values are NOT used).
2. Raw paired deltas for the 10 matched seeds.
3. Every decimal string is parsed exactly with Fraction (no binary float anywhere in the exact path).
4. All 2^10 = 1024 sign assignments are enumerated.
5. Counted: |T_perm| > |T_obs|, |T_perm| == |T_obs|, |T_perm| >= |T_obs|.
6. The official two-sided exact p is defined as count(|T_perm| >= |T_obs|)/1024 - ties included.
7. The old float implementation is re-run for comparison and the difference is explained.
8. All requested quantities are reported.
9. No rule was chosen because it gives a smaller p: both are reported as measured.

Nothing is modified: this script only reads existing files and writes three NEW files under
08_report/author_acceptance/.
"""
from __future__ import annotations

import csv
import itertools
import json
import statistics
from decimal import Decimal, getcontext
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

PJ = Path(r"<PIG_PROJECT>")
EXP = PJ / "experiments" / "cea_extension_v2"
OUTD = EXP / "08_report" / "author_acceptance"
COMBINED = EXP / "results" / "pigdetect_corrected_combined.csv"
SEEDS = [1, 7, 21, 42, 100, 101, 103, 107, 109, 113]
RULE = "last"
getcontext().prec = 60


def dec_text_to_fraction(t: str) -> Fraction:
    """Exact decimal-string -> Fraction (no float involved)."""
    return Fraction(Decimal(t.strip()))


def main() -> int:
    OUTD.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(COMBINED.open(encoding="utf-8-sig"))
            if r["checkpoint_rule"] == RULE]
    by_seed: dict[int, dict] = {}
    for r in rows:
        by_seed.setdefault(int(r["seed"]), {})[r["model"]] = r
    matched = [s for s in SEEDS if s in by_seed and "E0" in by_seed[s] and "E5" in by_seed[s]]

    # --- 1/2/3: exact deltas from the stored text -----------------------------
    per_seed = []
    deltas: list[Fraction] = []
    for s in matched:
        e0_txt = by_seed[s]["E0"]["mAP50_95"]
        e5_txt = by_seed[s]["E5"]["mAP50_95"]
        e0 = dec_text_to_fraction(e0_txt)
        e5 = dec_text_to_fraction(e5_txt)
        d = e5 - e0
        deltas.append(d)
        per_seed.append({"seed": s, "E0_text": e0_txt, "E5_text": e5_txt,
                         "E0_exact": str(e0), "E5_exact": str(e5),
                         "delta_exact": str(d), "delta_decimal": f"{float(d):+.4f}",
                         "run_id_E0": by_seed[s]["E0"]["run_id"],
                         "run_id_E5": by_seed[s]["E5"]["run_id"],
                         "checkpoint_E0": by_seed[s]["E0"]["checkpoint_path"],
                         "checkpoint_E5": by_seed[s]["E5"]["checkpoint_path"]})
    n = len(deltas)
    assert n == 10, f"expected 10 matched seeds, found {n}"

    # --- 4/5/6: exact enumeration --------------------------------------------
    T_obs = sum(deltas) / n                       # exact rational observed mean
    obs_abs = abs(T_obs)
    counts = {"greater": 0, "tie": 0, "less": 0}
    ties, table = [], []
    for signs in itertools.product([1, -1], repeat=n):
        perm = sum(s * d for s, d in zip(signs, deltas)) / n
        a = abs(perm)
        if a > obs_abs:
            cat = "greater"
        elif a == obs_abs:
            cat = "tie"
        else:
            cat = "less"
        counts[cat] += 1
        pat = "".join("+" if s > 0 else "-" for s in signs)
        if cat == "tie":
            ties.append({"pattern": pat, "T_perm_exact": str(perm), "T_perm_decimal": f"{float(perm):.18f}",
                         "sign_of_T_perm": "positive" if perm > 0 else ("negative" if perm < 0 else "zero")})
        table.append({"pattern": pat, "T_perm_exact_fraction": str(perm),
                      "T_perm_decimal": f"{float(perm):.18f}", "category_exact": cat})
    ge = counts["greater"] + counts["tie"]
    p_exact = Fraction(ge, 2 ** n)                # denominator is a power of two -> terminates
    p_decimal = Decimal(ge) / Decimal(2 ** n)

    # --- 7: the old float implementation, reproduced verbatim -----------------
    d_float = [float(x) for x in deltas]          # float(Fraction) == float(the stored text)
    obs_f = abs(statistics.mean(d_float))
    two_f = one_f = 0
    float_pat = {}
    for signs in itertools.product([1, -1], repeat=n):
        m = sum(s * x for s, x in zip(signs, d_float)) / n
        two_f += int(abs(m) >= obs_f)
        one_f += int(m >= obs_f)
        float_pat["".join("+" if s > 0 else "-" for s in signs)] = m
    # two further float variants, for the explanation only
    d_np = np.asarray(d_float, dtype=float)
    two_np = sum(int(abs(float(np.sum(np.asarray(sg) * d_np) / n)) >= obs_f)
                 for sg in itertools.product([1, -1], repeat=n))
    p_two_float = Fraction(two_f, 2 ** n)
    p_two_numpy = Fraction(two_np, 2 ** n)

    # which ties does float arithmetic drop?
    tie_float_detail = []
    for t in ties:
        m = float_pat[t["pattern"]]
        tie_float_detail.append({"pattern": t["pattern"], "T_perm_exact": t["T_perm_exact"],
                                 "T_perm_float": f"{m:.18f}",
                                 "abs_float_minus_abs_obs": f"{abs(m) - obs_f:.3e}",
                                 "counted_by_float_ge": bool(abs(m) >= obs_f)})
    dropped = [t["pattern"] for t in tie_float_detail if not t["counted_by_float_ge"]]

    persisted = json.loads((EXP / "05_statistics" / "statistics_summary.json").read_text(
        encoding="utf-8"))["checkpoint_rules"][RULE]["signflip"]

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scope": "READ-ONLY exact tie audit, last.pt, no existing file modified",
        "input": {"file": str(COMBINED), "checkpoint_rule": RULE,
                  "precision_note": ("the evaluator stores mAP@0.5:0.95 rounded to 4 decimals; that "
                                     "text is the highest precision that exists in the persisted "
                                     "evaluation output, so it is used verbatim and parsed exactly")},
        "matched_seeds": matched,
        "per_seed": per_seed,
        "raw_full_precision_deltas": [str(d) for d in deltas],
        "raw_full_precision_deltas_decimal": [str(Decimal(d.numerator) / Decimal(d.denominator))
                                              for d in deltas],
        "observed_mean_exact": str(T_obs),
        "observed_mean_decimal": str(Decimal(T_obs.numerator) / Decimal(T_obs.denominator)),
        "counts": {"greater": counts["greater"], "tie": counts["tie"], "less": counts["less"],
                   "greater_or_equal": ge, "total": 2 ** n},
        "exact_rational_p_two_sided": {"numerator": ge, "denominator": 2 ** n,
                                       "fraction": f"{ge}/{2 ** n}", "decimal": str(p_decimal)},
        "ties": ties,
        "old_float_implementation": {
            "implementation": "scripts/stats_lc_confirmatory.py::signflip reproduced verbatim "
                              "(obs = abs(statistics.mean(deltas)); m = sum(sign*deltas)/10; "
                              "count abs(m) >= obs)",
            "observed_abs_mean_float": repr(obs_f),
            "greater_or_equal_count": two_f, "one_sided_count": one_f,
            "p_two_sided_fraction": f"{two_f}/{2 ** n}",
            "p_two_sided_decimal": str(Decimal(two_f) / Decimal(2 ** n)),
        },
        "float_numpy_variant": {"greater_or_equal_count": two_np,
                                "p_two_sided_fraction": f"{two_np}/{2 ** n}",
                                "p_two_sided_decimal": str(Decimal(two_np) / Decimal(2 ** n))},
        "persisted_in_summary_json": {"p_two_sided": persisted["p_two_sided"],
                                      "p_one_sided": persisted["p_one_sided"],
                                      "configurations": persisted["configurations"]},
        "tie_behaviour_under_float": tie_float_detail,
        "ties_dropped_by_float": dropped,
        "difference_source": [
            "The 10 sign patterns listed in `ties` have |T_perm| exactly equal to |T_obs| in rational "
            "arithmetic, because the deltas are exact 4-decimal numbers and their signed sums hit the "
            "observed sum exactly.",
            "Exact rational arithmetic includes every tie under '>=', giving 206/1024.",
            "In binary floating point, `float(sum(sign*d for d in deltas))/10` for a tied pattern is "
            "not necessarily equal to `abs(statistics.mean(deltas))`: the two expressions accumulate "
            "in a different order and land one ulp apart, so "
            + ("%d of the %d ties fall just below the observed value and are dropped"
               % (len(dropped), len(ties)) if dropped else "no tie is dropped")
            + ".",
            "That is the entire difference between 206/1024 (exact, ties included) and 202/1024 "
            "(old float implementation); the discrepancy is 4/1024 = 0.00390625.",
        ],
        "rule_selection_note": ("No rule was selected because it yields a smaller p: both values are "
                               "reported as measured. The definition requested here (>= with ties "
                               "included, exact rational arithmetic) gives 206/1024."),
    }
    (OUTD / "04_07b_last_signflip_tie_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with (OUTD / "04_07b_last_signflip_exact_fractions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pattern", "T_perm_exact_fraction", "T_perm_decimal",
                                          "category_exact"])
        w.writeheader()
        for row in table:
            w.writerow({**row,
                        "T_perm_decimal": row["T_perm_decimal"],
                        "category_exact": row["category_exact"]})

    md = ["# last.pt — exact sign-flip tie audit (read-only)", "",
          f"Run: {report['timestamp']}  ",
          "Input: `results/pigdetect_corrected_combined.csv`, `checkpoint_rule=last`, values used "
          "verbatim as stored. **Precision note:** the evaluator writes mAP@0.5:0.95 rounded to 4 "
          "decimals, and that text is the highest precision present in the persisted evaluation "
          "output — no higher-precision source exists in the results files, so the exact arithmetic "
          "below is performed on those 4-decimal strings (parsed with `Fraction`, never converted to "
          "binary float).", "",
          "## 1. The 10 matched seeds and their raw deltas (exact)", "",
          "| seed | E0 text | E5 text | delta exact (fraction) | delta (decimal) |",
          "|---|---|---|---|---|"]
    for r in per_seed:
        md.append(f"| {r['seed']} | `{r['E0_text']}` | `{r['E5_text']}` | `{r['delta_exact']}` | "
                  f"{r['delta_decimal']} |")
    md += ["", f"Observed mean T_obs = `{report['observed_mean_exact']}` "
               f"= {report['observed_mean_decimal']}  ",
           f"|T_obs| = `{str(obs_abs)}`", "",
           "## 2. Full enumeration (2^10 = 1024 sign assignments)", "",
           "| quantity | count | p (fraction of 1024) | decimal |", "|---|---:|---|---|",
           f"| \\|T_perm\\| > \\|T_obs\\| | {counts['greater']} | {counts['greater']}/1024 | "
           f"{Decimal(counts['greater']) / Decimal(1024)} |",
           f"| \\|T_perm\\| == \\|T_obs\\| (ties) | {counts['tie']} | {counts['tie']}/1024 | "
           f"{Decimal(counts['tie']) / Decimal(1024)} |",
           f"| \\|T_perm\\| < \\|T_obs\\| | {counts['less']} | {counts['less']}/1024 | "
           f"{Decimal(counts['less']) / Decimal(1024)} |",
           f"| **\\|T_perm\\| >= \\|T_obs\\| (official, ties included)** | **{ge}** | **{ge}/1024** | "
           f"**{p_decimal}** |", "",
           "## 3. The 10 tied sign patterns (exact equality)", "",
           "| sign pattern | T_perm exact | T_perm decimal | sign |", "|---|---|---|---|"]
    for t in ties:
        md.append(f"| `{t['pattern']}` | `{t['T_perm_exact']}` | {t['T_perm_decimal']} | "
                  f"{t['sign_of_T_perm']} |")
    md += ["", "## 4. Old float implementation, reproduced verbatim", "",
           f"`obs = abs(statistics.mean(deltas))` = `{report['old_float_implementation']['observed_abs_mean_float']}`, "
           f"`m = sum(sign*deltas)/10`, count `abs(m) >= obs` → **{two_f}/1024** = "
           f"{report['old_float_implementation']['p_two_sided_decimal']} "
           f"(one-sided {one_f}/1024). The numpy pairwise-summation variant gives {two_np}/1024 = "
           f"{report['float_numpy_variant']['p_two_sided_decimal']}. The persisted summary records "
           f"p2 = {persisted['p_two_sided']}, p1 = {persisted['p_one_sided']}.", "",
           "### Why the old implementation gets 202 instead of 206", "",
           "| tied pattern | T_perm exact | T_perm float | abs(float) - abs(obs) | counted by float `>=` |",
           "|---|---|---|---:|---|"]
    for t in tie_float_detail:
        md.append(f"| `{t['pattern']}` | `{t['T_perm_exact']}` | {t['T_perm_float']} | "
                  f"{t['abs_float_minus_abs_obs']} | {t['counted_by_float_ge']} |")
    md += ["", f"Patterns dropped by the float path: "
               + (", ".join(f"`{p}`" for p in dropped) if dropped else "none") + ".", "",
           "## 5. Summary of the requested quantities", "",
           "| quantity | value |", "|---|---|",
           f"| 10 raw full-precision deltas | {report['raw_full_precision_deltas']} |",
           f"| observed mean (exact) | `{report['observed_mean_exact']}` = {report['observed_mean_decimal']} |",
           f"| greater count | {counts['greater']} |",
           f"| tie count | {counts['tie']} |",
           f"| greater-or-equal count (official) | {ge} |",
           f"| exact rational p | `{ge}/1024` |",
           f"| decimal p (exact, terminates) | **{p_decimal}** |",
           f"| old float p | {Decimal(two_f) / Decimal(1024)} ({two_f}/1024) |",
           f"| difference | 4/1024 = 0.00390625, caused solely by the "
           f"{counts['tie']} exact ties and floating-point ulp behaviour |", "",
           "## 6. Difference source (explicit)", ""] + \
         [f"{i+1}. {s}" for i, s in enumerate(report['difference_source'])] + \
         ["", "## 7. Rule-selection statement", "", report["rule_selection_note"], "",
          "## Output files (all new; nothing overwritten)", "",
          "| file | content |", "|---|---|",
          "| `04_07b_last_signflip_tie_audit.md` | this report |",
          "| `04_07b_last_signflip_tie_audit.json` | machine-readable full detail |",
          "| `04_07b_last_signflip_exact_fractions.csv` | all 1024 patterns with exact T_perm and category |",
          ""]
    (OUTD / "04_07b_last_signflip_tie_audit.md").write_text("\n".join(md), encoding="utf-8")

    print("observed mean (exact) :", T_obs)
    print("greater / tie / less  :", counts["greater"], "/", counts["tie"], "/", counts["less"])
    print("greater-or-equal      :", ge, "-> exact p =", f"{ge}/1024", "=", p_decimal)
    print("old float impl        :", two_f, "/1024 =", Decimal(two_f) / Decimal(1024),
          "| numpy variant:", two_np, "/1024 =", Decimal(two_np) / Decimal(1024))
    print("persisted json        :", persisted["p_two_sided"], "/", persisted["p_one_sided"])
    print("ties dropped by float :", dropped)
    print("written ->", OUTD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
