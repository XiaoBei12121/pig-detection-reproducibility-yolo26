"""Read-only verification of the exact sign-flip reconciliation.

Independent of the files it checks: the PigDetect 10-seed paired deltas are read
from the persisted evaluation CSV, parsed as exact decimals (Fraction/Decimal, no
binary float), and the two-sided / one-sided sign-flip probabilities are fully
enumerated over all 2^n sign assignments with exact ties included. The resulting
values are then compared, at each artefact's own precision, with every file that
quotes a PigDetect sign-flip p-value.

Writes only:
  .../author_acceptance/integrity_audit/exact_signflip_reconciliation_verification.{md,csv}
"""

from __future__ import annotations

import csv
import itertools
import json
import re
import zipfile
from decimal import ROUND_HALF_EVEN, Decimal
from fractions import Fraction
from pathlib import Path

PJ = Path(r"<PIG_PROJECT>")
EXP = PJ / "experiments" / "cea_extension_v2"
AUD = EXP / "08_report" / "author_acceptance"
OUT = AUD / "integrity_audit"

OLD = ["0.197266", "0.1973", "0.098633"]
NEW = ["0.201171875", "0.201172", "0.2012", "0.1005859375", "0.100586"]

# files that quote a sign-flip p as a CURRENT formal value
FORMAL = [
    EXP / "05_statistics/statistics_summary.json",
    EXP / "05_statistics/pigdetect_last_statistics.json",
    EXP / "05_statistics/exact_signflip_results.csv",
    EXP / "05_statistics/statistics_summary.md",
    EXP / "07_tables/tableB_checkpoint_statistics.csv",
    EXP / "08_report/experiment_results_summary.md",
    EXP / "08_report/manuscript_change_recommendations.md",
    AUD / "CEA_HIGH_IMPACT_EXPERIMENT_AUDIT.md",
]
# files that must KEEP the historical float value as audit evidence
HISTORY = [
    AUD / "04_07_traceability_and_statistics_audit.md",
    AUD / "04_07b_last_signflip_tie_audit.md",
    AUD / "04_07b_last_signflip_tie_audit.json",
    AUD / "04_07b_last_signflip_exact_fractions.csv",
    AUD / "04_07c_global_signflip_consistency_audit.md",
    AUD / "04_07c_global_signflip_consistency_audit.csv",
    AUD / "EXPERIMENT_DATA_INTEGRITY_AUDIT.md",
    AUD / "integrity_audit/numeric_revision_history.md",
    AUD / "integrity_audit/numeric_revision_history.csv",
    AUD / "integrity_audit/numeric_mutation_direction_audit.md",
    AUD / "integrity_audit/derived_statistics_recompute_audit.md",
    AUD / "integrity_audit/derived_statistics_recompute_audit.csv",
]


def dec(x) -> Decimal:
    if isinstance(x, Fraction):
        return Decimal(x.numerator) / Decimal(x.denominator)
    return Decimal(str(x))


def q6(x) -> Decimal:
    return dec(x).quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)


def exact_signflip(deltas: list) -> dict:
    n = len(deltas)
    obs = abs(sum(deltas) / n)
    two = one = 0
    for signs in itertools.product((1, -1), repeat=n):
        m = sum(s * d for s, d in zip(signs, deltas)) / n
        if abs(m) >= obs:      # ties included
            two += 1
        if m >= obs:
            one += 1
    return {"n": n, "obs": obs, "p_two": Fraction(two, 2 ** n),
            "p_one": Fraction(one, 2 ** n), "ties_free_two": two, "one_count": one}


def read_deltas() -> dict:
    rows = list(csv.DictReader((EXP / "results/pigdetect_corrected_combined.csv")
                               .open(encoding="utf-8-sig")))
    per = {}
    for r in rows:
        if r["model"] not in ("E0", "E5"):
            continue
        per.setdefault(r["checkpoint_rule"], {}).setdefault(int(r["seed"]), {})[r["model"]] = \
            Fraction(Decimal(r["mAP50_95"].strip()))
    out = {}
    for rule, seeds in per.items():
        ds = [seeds[s]["E5"] - seeds[s]["E0"] for s in sorted(seeds)
              if "E0" in seeds[s] and "E5" in seeds[s]]
        out[rule] = exact_signflip(ds)
    return out


def has(path: Path, needle: str):
    if not path.exists():
        return None
    return needle in path.read_text(encoding="utf-8", errors="replace")


def docx_text() -> str:
    p = AUD / "01_CEA_Submission_Ready_Manuscript_exact_signflip_corrected.docx"
    if not p.exists():
        return ""
    with zipfile.ZipFile(p) as z:
        return re.sub(r"<[^>]+>", " ", z.read("word/document.xml").decode("utf-8", "replace"))


def main() -> int:
    sf = read_deltas()
    last = sf["last"]
    best = sf["best"]
    rows = []

    def rec(artefact, item, expected, found, verdict, note=""):
        rows.append({"artefact": artefact, "item": item, "expected": str(expected),
                     "found": str(found), "verdict": verdict, "note": note})

    # --- the recomputation itself
    rec("recomputation from evaluation CSV", "PigDetect 10-seed last p_two",
        "206/1024 = 0.201171875", f"{last['p_two']} = {dec(last['p_two'])} (= 206/1024)",
        "REFERENCE", "full 2^10 enumeration, ties included; 206/1024 reduces to 103/512")
    rec("recomputation from evaluation CSV", "PigDetect 10-seed last p_one",
        f"103/1024 = 0.1005859375", f"{last['p_one']} = {dec(last['p_one'])}",
        "REFERENCE", "count(T_perm >= |T_obs|)")
    rec("recomputation from evaluation CSV", "PigDetect 10-seed best p_two",
        "289/512 = 0.564453125", f"{best['p_two']} = {dec(best['p_two'])}", "REFERENCE",
        "unchanged, included as a control")

    exp_last_two, exp_last_one = last["p_two"], last["p_one"]
    # canonical notation is count/2^n (206/1024), which reduces to 103/512 - same rational
    den = 2 ** last["n"]
    two_str = f"{last['ties_free_two']}/{den}"
    one_str = f"{last['one_count']}/{den}"

    # --- JSON artefacts
    d = json.loads((EXP / "05_statistics/statistics_summary.json").read_text(encoding="utf-8"))
    s = d["checkpoint_rules"]["last"]["signflip"]
    rec("05_statistics/statistics_summary.json", "last p_two_sided", q6(exp_last_two),
        s["p_two_sided"], "PASS" if dec(s["p_two_sided"]) == q6(exp_last_two) else "FAIL")
    rec("05_statistics/statistics_summary.json", "last p_one_sided", q6(exp_last_one),
        s["p_one_sided"], "PASS" if dec(s["p_one_sided"]) == q6(exp_last_one) else "FAIL")
    rule_text = d["rules"]["signflip"]
    ok_rule = all(k in rule_text for k in ("exact", "2^n", "|T_perm| >= |T_obs|", "ties"))
    rec("05_statistics/statistics_summary.json", "rules.signflip wording",
        "exact rational/decimal + >= + ties included",
        rule_text[:70] + "...", "PASS" if ok_rule else "WARNING")
    # exact-value fields (added so the file carries both the exact decimal/fraction and the 6-dp display value)
    for item, key, expect in (("last p_two_sided_exact", "p_two_sided_exact", exp_last_two),
                              ("last p_one_sided_exact", "p_one_sided_exact", exp_last_one),
                              ("last p_two_sided_fraction", "p_two_sided_fraction", two_str),
                              ("last p_one_sided_fraction", "p_one_sided_fraction", str(exp_last_one))):
        got = s.get(key, "")
        ok = str(got) == str(expect) if "fraction" in key else \
            (got != "" and dec(got) == dec(expect))
        rec("05_statistics/statistics_summary.json", item, expect, got, "PASS" if ok else "FAIL")
    rec("05_statistics/statistics_summary.json", "ties_included flag", True, s.get("ties_included"),
        "PASS" if s.get("ties_included") is True else "FAIL")

    d2 = json.loads((EXP / "05_statistics/pigdetect_last_statistics.json").read_text(encoding="utf-8"))
    s2 = d2["signflip"]
    rec("05_statistics/pigdetect_last_statistics.json", "p_two_sided", q6(exp_last_two),
        s2["p_two_sided"], "PASS" if dec(s2["p_two_sided"]) == q6(exp_last_two) else "FAIL")
    rec("05_statistics/pigdetect_last_statistics.json", "p_one_sided", q6(exp_last_one),
        s2["p_one_sided"], "PASS" if dec(s2["p_one_sided"]) == q6(exp_last_one) else "FAIL")
    for item, key, expect in (("p_two_sided_exact", "p_two_sided_exact", exp_last_two),
                              ("p_one_sided_exact", "p_one_sided_exact", exp_last_one),
                              ("p_two_sided_fraction", "p_two_sided_fraction", two_str),
                              ("p_one_sided_fraction", "p_one_sided_fraction", str(exp_last_one))):
        got = s2.get(key, "")
        ok = str(got) == str(expect) if "fraction" in key else \
            (got != "" and dec(got) == dec(expect))
        rec("05_statistics/pigdetect_last_statistics.json", item, expect, got,
            "PASS" if ok else "FAIL")

    # --- exact_signflip_results.csv (records the exact fraction)
    p = EXP / "05_statistics/exact_signflip_results.csv"
    for r in csv.DictReader(p.open(encoding="utf-8-sig")):
        if r.get("dataset") == "PigDetect" and r.get("rule") == "last":
            rec("05_statistics/exact_signflip_results.csv", "PigDetect last p_two_sided",
                str(exp_last_two), r.get("p_two_sided"),
                "PASS" if dec(r["p_two_sided"]) == dec(exp_last_two) else "FAIL")
            rec("05_statistics/exact_signflip_results.csv", "PigDetect last p_one_sided",
                str(exp_last_one), r.get("p_one_sided"),
                "PASS" if dec(r["p_one_sided"]) == dec(exp_last_one) else "FAIL")

    # --- table B
    for r in csv.DictReader((EXP / "07_tables/tableB_checkpoint_statistics.csv")
                            .open(encoding="utf-8-sig")):
        if r["checkpoint"] == "last.pt":
            rec("07_tables/tableB_checkpoint_statistics.csv", "last.pt p_two", q6(exp_last_two),
                r["signflip_p_two_sided"],
                "PASS" if dec(r["signflip_p_two_sided"]) == q6(exp_last_two) else "FAIL")
            rec("07_tables/tableB_checkpoint_statistics.csv", "last.pt p_one", q6(exp_last_one),
                r["signflip_p_one_sided"],
                "PASS" if dec(r["signflip_p_one_sided"]) == q6(exp_last_one) else "FAIL")
        if r["checkpoint"] == "best.pt":
            rec("07_tables/tableB_checkpoint_statistics.csv", "best.pt p_two (must be untouched)",
                q6(best["p_two"]), r["signflip_p_two_sided"],
                "PASS" if dec(r["signflip_p_two_sided"]) == q6(best["p_two"]) else "FAIL")

    # --- markdown reports
    md = EXP / "05_statistics/statistics_summary.md"
    txt = md.read_text(encoding="utf-8") if md.exists() else ""
    ok_md = any(x in txt for x in ("0.201171875", "0.201172", "0.2012"))
    rec("05_statistics/statistics_summary.md", "p present in an agreed representation",
        "0.201171875 | 0.201172 | 0.2012",
        ", ".join(x for x in ("0.201171875", "0.201172", "0.2012") if x in txt) or "none",
        "PASS" if ok_md else "FAIL")
    rec("05_statistics/statistics_summary.md", "contains stale 0.197266", "no",
        "yes" if "0.197266" in txt else "no",
        "PASS" if "0.197266" not in txt else "FAIL")

    h = AUD / "CEA_HIGH_IMPACT_EXPERIMENT_AUDIT.md"
    ht = h.read_text(encoding="utf-8") if h.exists() else ""
    rec("08_report/author_acceptance/CEA_HIGH_IMPACT_EXPERIMENT_AUDIT.md", "current p (4dp)",
        "0.2012", "0.2012" if "0.2012" in ht else ("0.1973" if "0.1973" in ht else "?"),
        "PASS" if ("0.2012" in ht and "0.1973" not in ht) else "FAIL")

    dt = docx_text()
    if dt:
        rec("01_CEA_Submission_Ready_Manuscript_exact_signflip_corrected.docx", "manuscript p (4dp)",
            "0.2012", f"0.2012 x{dt.count('0.2012')}, 0.1973 x{dt.count('0.1973')}",
            "PASS" if ("0.2012" in dt and "0.1973" not in dt) else "FAIL")
        rec("...docx", "manuscript has no stale float p", "absent",
            "absent" if not any(x in dt for x in OLD) else "PRESENT",
            "PASS" if not any(x in dt for x in OLD) else "FAIL")

    # --- untouched families (must not have moved)
    for name, path, key, expect in (
            ("OinkTrack summary", EXP / "05_statistics/oinktrack_statistics_summary.json", 0.185547, 0.113281),
            ("5-seed exploratory", PJ / "results/leakage_corrected_confirmatory/statistics.json",
             0.3125, 0.25)):
        if not path.exists():
            rec(name, "present", "yes", "no", "WARNING")
            continue
        t = path.read_text(encoding="utf-8")
        rec(name, "sign-flip p values unchanged", f"{expect}",
            "unchanged" if str(expect) in t or str(key) in t else "CHECK",
            "PASS" if (str(expect) in t or str(key) in t) else "WARNING")

    # --- global scan of the old/new strings
    HISTORY_EXTRA = {
        AUD / "04_07_independent_summary.json",
        AUD / "05_e5_selection_timeline_audit.csv",
        AUD / "05_e5_selection_timeline_audit.json",
        AUD / "05_e5_selection_timeline_audit.md",
    }
    scan = []
    for f in sorted(EXP.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in (".md", ".json", ".csv", ".txt", ".yaml"):
            continue
        if f.stat().st_size > 3_000_000:
            continue
        try:
            t = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        old_hits = [k for k in OLD if k in t]
        new_hits = [k for k in NEW if k in t]
        if old_hits or new_hits:
            rp = f.relative_to(PJ).as_posix()
            if any(f == x for x in FORMAL):
                cls = "FORMAL"
            elif any(f == x for x in HISTORY) or f in HISTORY_EXTRA:
                cls = "HISTORY"
            elif "predictions.json" in rp:
                cls = "PREDICTIONS"
            elif "/03_runs/" in rp or rp.endswith("_log.json"):
                cls = "LOG" if "p_value_convention_note" not in t else "LOG (annotated)"
            elif "integrity_audit/" in rp:
                cls = "audit-output"
            else:
                cls = "OTHER"
            ctx = ""
            if old_hits:
                i = t.find(old_hits[0])
                ctx = " ".join(t[max(0, i - 110):i + 110].split())
            scan.append({"file": rp, "class": cls, "old_values": ";".join(old_hits),
                         "new_values": ";".join(new_hits), "bytes": f.stat().st_size,
                         "context": ctx})

    bad = [r for r in rows if r["verdict"] == "FAIL"]
    precedence = {"FORMAL": 0, "OTHER": 1, "PREDICTIONS": 2, "LOG": 3, "HISTORY": 4,
                  "audit-output": 5}
    formal_bad = [s for s in scan if s["class"] in ("FORMAL", "OTHER") and s["old_values"]]
    unresolved = [s for s in scan if s["class"] in ("PREDICTIONS", "LOG") and s["old_values"]]

    md_lines = ["# Exact sign-flip reconciliation — independent verification", "",
                "Recomputed from `results/pigdetect_corrected_combined.csv` with exact rational "
                "arithmetic (highest-precision persisted decimal text, `Fraction`/`Decimal`), full "
                "2^n sign enumeration, two-sided p = count(|T_perm| >= |T_obs|)/2^n, ties included.",
                "",
                f"PigDetect 10-seed **last.pt**: p_two = **{last['p_two']}** "
                f"(= {dec(last['p_two'])}, 6 dp {q6(last['p_two'])}, 4 dp 0.2012), "
                f"p_one = **{last['p_one']}** (= {dec(last['p_one'])}, 6 dp {q6(last['p_one'])}).",
                "",
                f"PigDetect 10-seed best.pt (control, must be unchanged): p_two = {best['p_two']} "
                f"(= {dec(best['p_two'])}).", "",
                "| artefact | item | expected | found | verdict |", "|---|---|---|---|---|"]
    for r in rows:
        md_lines.append(f"| {r['artefact']} | {r['item']} | {r['expected']} | {r['found']} | "
                        f"{r['verdict']} |")
    md_lines += ["", "## Global scan for the superseded values", "",
                 "| file | class | old values present | new values present |", "|---|---|---|---|"]
    for s in sorted(scan, key=lambda x: (precedence.get(x["class"], 9), x["file"])):
        md_lines.append(f"| `{s['file']}` | {s['class']} | {s['old_values'] or '-'} | "
                        f"{s['new_values'] or '-'} |")
    md_lines += ["", "### Context of every remaining superseded-value occurrence", ""]
    for s in sorted(scan, key=lambda x: (precedence.get(x["class"], 9), x["file"])):
        if not s["old_values"]:
            continue
        md_lines += [f"* **{s['class']}** `{s['file']}` — old: `{s['old_values']}`", "",
                     f"  > ...{s['context']}...", ""]
    md_lines += ["", f"FAIL verdicts: **{len(bad)}**; formal/other files still carrying a "
                     f"superseded value: **{len(formal_bad)}**; occurrences needing semantic "
                     f"judgement (log / prediction files): **{len(unresolved)}**.", "",
                 "HISTORY-class files keep the old float value on purpose: they are the audit "
                 "evidence that documents the superseded convention and its cause. LOG files are "
                 "historical execution records of what the pipeline printed at that time and are "
                 "likewise not rewritten. Predictions files are matched only by numeric "
                 "coincidence (detection scores/coordinates), not by a p-value."]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exact_signflip_reconciliation_verification.md").write_text("\n".join(md_lines),
                                                                      encoding="utf-8")
    with (OUT / "exact_signflip_reconciliation_verification.csv").open("w", newline="",
                                                                      encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["artefact", "item", "expected", "found", "verdict", "note"],
                           restval="")
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        print(f"  [{r['verdict']:9s}] {r['artefact']} :: {r['item']} -> {r['found']}")
    print()
    print("scan hits:")
    for s in scan:
        print(f"  {s['class']:13s} {s['file']}  old=[{s['old_values']}] new=[{s['new_values']}]")
    print()
    print(f"FAILs: {len(bad)} | formal/other files with superseded value: {len(formal_bad)}")
    print("written ->", OUT / "exact_signflip_reconciliation_verification.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
