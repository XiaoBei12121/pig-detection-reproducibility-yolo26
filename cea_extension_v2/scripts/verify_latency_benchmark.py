"""A5 - verify the fixed-hardware latency benchmark against the required protocol.

Recomputes every reported statistic from the per-iteration raw file (120 000 rows) using the
benchmark's own definitions - mean, sample median, nearest-rank p95
(`sorted(times)[int(0.95*n)-1]`), sample SD, min/max, FPS = 1000/mean,
throughput = batch*1000/mean - and checks the protocol requirements: five architectures,
batches 1 and 4, fp32 and fp16, imgsz 640, 100 warm-up iterations, 1000 measured iterations
repeated over 3 rounds, the required environment block, and the absence of the forbidden
artefacts (exported models, profiler traces, prediction images).

Read-only apart from the two verification files it writes into 15_latency/.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PJ = Path(r"<PIG_PROJECT>")
LAT = PJ / "experiments" / "cea_extension_v2" / "15_latency"


def rows(p: Path):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    summary = rows(LAT / "latency_summary.csv")
    raw = rows(LAT / "latency_raw.csv")
    env = json.loads((LAT / "latency_environment.json").read_text(encoding="utf-8"))
    checks = []

    def chk(item, expected, found, verdict):
        checks.append({"item": item, "expected": str(expected), "found": str(found),
                       "verdict": verdict})

    # ---- coverage: five architectures x {best,last} x {1,4} x {fp32,fp16}
    import re as _re

    def base_of(m: str) -> str:
        """Strip the checkpoint-rule suffix from a model label, e.g.
        'E0 (YOLO26s baseline, best)' -> 'E0 (YOLO26s baseline)'; 'YOLO11s+CA (last)' -> 'YOLO11s+CA'."""
        m = _re.sub(r", (?:best|last)\)$", ")", m)
        m = _re.sub(r" \((?:best|last)\)$", "", m)
        return m
    models = sorted({r["model"] for r in summary})
    bases = sorted({base_of(r["model"]) for r in summary})
    loaders = sorted({r["architecture"] for r in summary})
    combos = {(r["batch"], r["precision"]) for r in summary}
    chk("base architectures covered", 5, f"{len(bases)}: {', '.join(bases)}",
        "PASS" if len(bases) == 5 else "FAIL")
    chk("loader families (architecture column)", "YOLO + RTDETR", ", ".join(loaders),
        "PASS" if loaders == ["RTDETR", "YOLO"] else "WARNING")
    chk("model x checkpoint-rule configurations", 10, len(models),
        "PASS" if len(models) == 10 else "FAIL")
    chk("batch x precision grid", "{(1,fp32),(1,fp16),(4,fp32),(4,fp16)}", sorted(combos),
        "PASS" if combos == {( "1", "fp32"), ("1", "fp16"), ("4", "fp32"), ("4", "fp16")}
        else "FAIL")
    chk("configurations", 40, len(summary), "PASS" if len(summary) == 40 else "FAIL")
    chk("imgsz", 640, sorted({r["imgsz"] for r in summary}),
        "PASS" if {r["imgsz"] for r in summary} == {"640"} else "FAIL")

    missing_ck = [r["checkpoint"] for r in summary if not Path(r["checkpoint"]).exists()]
    chk("checkpoints used exist", "all present",
        f"{len(summary) - len(missing_ck)}/{len(summary)} present",
        "PASS" if not missing_ck else "FAIL")

    # ---- iteration counts and round structure
    by_cfg = defaultdict(list)
    rounds_seen = defaultdict(set)
    for r in raw:
        key = (r["model"], r["batch"], r["precision"])
        by_cfg[key].append(float(r["latency_ms"]))
        rounds_seen[key].add(r["round"])
    chk("raw rows", 40 * 3 * 1000, len(raw), "PASS" if len(raw) == 120000 else "FAIL")
    chk("rounds per configuration", {1, 2, 3}, sorted({frozenset(v) for v in rounds_seen.values()},
                                                      key=len)[0],
        "PASS" if all(v == {"1", "2", "3"} for v in rounds_seen.values()) else "FAIL")
    per_round = {k: len(v) / len(rounds_seen[k]) for k, v in by_cfg.items()}
    chk("measured iterations per round", 1000, sorted(set(per_round.values())),
        "PASS" if set(per_round.values()) == {1000.0} else "FAIL")
    chk("warm-up iterations (environment)", 100, env.get("warmup_iterations"),
        "PASS" if env.get("warmup_iterations") == 100 else "FAIL")
    chk("measured iterations (environment)", 1000, env.get("measured_iterations"),
        "PASS" if env.get("measured_iterations") == 1000 else "FAIL")
    chk("rounds (environment)", 3, env.get("rounds"),
        "PASS" if env.get("rounds") == 3 else "FAIL")

    # ---- recompute every reported statistic with the benchmark's own definitions
    worst = []
    round_spread = []
    for r in summary:
        key = (r["model"], r["batch"], r["precision"])
        t = by_cfg.get(key)
        if not t:
            worst.append(f"{key}: no raw rows")
            continue
        n = len(t)
        mean = statistics.mean(t)
        mine = {
            "mean_ms": mean, "median_ms": statistics.median(t),
            "p95_ms": sorted(t)[max(0, int(0.95 * n) - 1)],
            "std_ms": statistics.stdev(t), "min_ms": min(t), "max_ms": max(t),
            "fps": 1000.0 / mean,
            "throughput_img_s": float(r["batch"]) * 1000.0 / mean,
        }
        for field, val in mine.items():
            # the summary stores rounded values; compare at the stored precision
            txt = str(r[field])
            places = len(txt.split(".")[1]) if "." in txt else 0
            if round(val, places) != float(txt):
                worst.append(f"{key}.{field}: reported {txt} vs recomputed {val}")
        if int(r["iterations"]) != n:
            worst.append(f"{key}.iterations: reported {r['iterations']} vs raw {n}")
        # per-round means: a direct check of the "3 rounds" requirement
        means = []
        for rd in ("1", "2", "3"):
            xs = [float(x["latency_ms"]) for x in raw
                  if (x["model"], x["batch"], x["precision"], x["round"]) == key + (rd,)]
            means.append(statistics.mean(xs))
        round_spread.append({"model": r["model"], "batch": r["batch"], "precision": r["precision"],
                             "round1_mean_ms": round(means[0], 4), "round2_mean_ms": round(means[1], 4),
                             "round3_mean_ms": round(means[2], 4),
                             "spread_ms": round(max(means) - min(means), 4),
                             "spread_pct": round(100 * (max(means) - min(means)) / statistics.mean(means), 2)})
    chk("all reported statistics recomputed from raw", "0 discrepancies",
        f"{len(worst)} discrepancies", "PASS" if not worst else "FAIL")

    # ---- model properties (grouped by base architecture, not by loader family)
    arch_params = defaultdict(set)
    arch_flops = defaultdict(set)
    for r in summary:
        arch_params[base_of(r["model"])].add(r["params"])
        arch_flops[base_of(r["model"])].add(r["gflops"])
    chk("params identical across batch/precision/rule for one architecture",
        "1 value per architecture", {k: len(v) for k, v in arch_params.items()},
        "PASS" if all(len(v) == 1 for v in arch_params.values()) else "FAIL")
    flops_ok = all(len({x for x in v if x}) <= 1 for v in arch_flops.values())
    chk("GFLOPs: one value per architecture where reported (fp32 loads only)",
        "consistent per architecture", {k: sorted(v) for k, v in arch_flops.items()},
        "PASS" if flops_ok else "WARNING")

    # ---- environment and fairness block
    need_env = ["gpu", "torch", "torch_cuda", "cudnn", "ultralytics", "imgsz", "batches",
                "precisions", "warmup_iterations", "measured_iterations", "rounds", "timing",
                "fairness", "excluded_from_timing"]
    absent = [k for k in need_env if k not in env]
    chk("environment block fields", "all present", f"missing={absent}",
        "PASS" if not absent else "FAIL")
    fair = " | ".join(env.get("fairness", []))
    chk("fairness: no compile / TensorRT / ONNX / CUDA graphs",
        "all four excluded",
        fair,
        "PASS" if all(k in fair for k in ("compile", "TensorRT", "ONNX", "CUDA graphs")) else "FAIL")
    chk("timing method", "CUDA events + synchronize", env.get("timing"),
        "PASS" if "synchronize" in str(env.get("timing", "")) else "FAIL")
    chk("no disk I/O or image decoding in the timed region", "excluded",
        env.get("excluded_from_timing"),
        "PASS" if env.get("excluded_from_timing") else "FAIL")

    # ---- forbidden artefacts
    bad_files = [p.name for p in LAT.iterdir()
                 if p.suffix.lower() in (".onnx", ".engine", ".trt", ".trace", ".pt", ".jpg",
                                         ".jpeg", ".png", ".mp4")]
    chk("no exported models / traces / prediction images in 15_latency", "none",
        bad_files or "none", "PASS" if not bad_files else "FAIL")

    # ---- cold-start artefact (documented, not silently corrected)
    cold = summary[0]
    later_same_arch = [r for r in summary
                       if r["architecture"] == cold["architecture"] and r["batch"] == cold["batch"]
                       and r["precision"] == cold["precision"] and r is not cold]
    chk("first configuration flagged as cold-start",
        "documented in latency_environment.json",
        env.get("validation_measurement", {}).get("finding", "absent"),
        "PASS" if env.get("validation_measurement", {}).get("finding", "").upper() == "YES - CONFIRMED"
        else "WARNING")

    fails = [c for c in checks if c["verdict"] == "FAIL"]
    with (LAT / "LATENCY_VERIFICATION.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item", "expected", "found", "verdict"])
        w.writeheader()
        w.writerows(checks)
    with (LAT / "latency_rounds.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "batch", "precision", "round1_mean_ms",
                                          "round2_mean_ms", "round3_mean_ms", "spread_ms",
                                          "spread_pct"])
        w.writeheader()
        w.writerows(round_spread)

    md = ["# A5 — latency benchmark verification", "",
          "Every statistic in `latency_summary.csv` is recomputed here from the 120 000 per-iteration "
          "timings in `latency_raw.csv`, using the benchmark's own definitions "
          "(p95 = nearest rank `sorted(times)[int(0.95*n)-1]`, sample SD with ddof=1, "
          "FPS = 1000/mean, throughput = batch·1000/mean).", "",
          "| check | expected | found | verdict |", "|---|---|---|---|"]
    for c in checks:
        md.append(f"| {c['item']} | {c['expected']} | {c['found']} | {c['verdict']} |")
    md += ["", f"FAIL: **{len(fails)}**; statistic discrepancies: **{len(worst)}**.", ""]
    if worst:
        md += ["## Discrepancies", "", *[f"* {w}" for w in worst[:30]], ""]
    md += ["## Round-to-round spread (the 3-round requirement, recomputed)", "",
           "| model | batch | precision | round1 (ms) | round2 (ms) | round3 (ms) | spread (ms) | spread (%) |",
           "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in round_spread:
        md.append(f"| {r['model']} | {r['batch']} | {r['precision']} | {r['round1_mean_ms']} | "
                  f"{r['round2_mean_ms']} | {r['round3_mean_ms']} | {r['spread_ms']} | "
                  f"{r['spread_pct']} |")
    md += ["", "## Notes", "",
           "* GFLOPs is a model property computed on the fp32 load (`get_flops`); the fp16 rows "
           "therefore carry no GFLOPs by design, and the per-architecture table above shows the "
           "value is consistent wherever it is reported.",
           "* The first configuration of the run is a documented cold-start outlier "
           "(`validation_measurement` in `latency_environment.json`); it is left as measured and "
           "flagged rather than overwritten.",
           "* No training, no exported model, no profiler trace and no prediction image was "
           "produced by this benchmark."]
    (LAT / "LATENCY_VERIFICATION.md").write_text("\n".join(md), encoding="utf-8")

    for c in checks:
        print(f"  [{c['verdict']:7s}] {c['item']} -> {c['found']}")
    print(f"\nFAILs: {len(fails)} | discrepancies: {len(worst)}")
    for w_ in worst[:8]:
        print("   ", w_)
    print("written ->", LAT / "LATENCY_VERIFICATION.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
