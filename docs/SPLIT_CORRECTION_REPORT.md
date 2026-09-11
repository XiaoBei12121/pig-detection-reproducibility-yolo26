# Split-correction report: leakage-corrected confirmatory experiment

**Scope:** this document integrates the leakage-corrected confirmatory experiment into the reported
results. It modifies no original result, weight, split or log, and the `v1.0.1` release is unchanged.

| Item | Value |
|---|---|
| Corrected split | `splits/pigdetect/leakage_corrected_{train,val}.txt` (built from the 2681-image development pool) |
| Split audit | `results/leakage_corrected_confirmatory/split_audit.{json,csv}` |
| Per-checkpoint test metrics | `results/leakage_corrected_confirmatory/test_eval_all.{json,csv}` |
| Run manifest summary | `results/leakage_corrected_confirmatory/runs_manifest.csv` |
| Paired statistics | `results/leakage_corrected_confirmatory/statistics.{json,md}` |
| Run notes (incl. the incident below) | `results/leakage_corrected_confirmatory/run_notes.md` |
| Original split, last.pt measurement | `results/original_split_last_eval/` |
| Deliverables | `results/generated/table19_leakage_corrected_confirmatory.csv`, `results/generated/table20_corrected_statistics.csv`, `figures/fig12_split_correction_effect.png` |

## 1. Why a corrected split was built

The file-level audit of the original PigDetect splits (`docs/PIGDETECT_SPLIT_AUDIT.md`) found that the
original split procedure separates camera prefixes for the `--`-style file names but degrades to
file-level grouping for names without that marker. As a consequence the original **main training and
validation splits are image-disjoint but not strictly clip-disjoint**: they share **17 clip keys**,
about **1 %** of the 1729 clip keys in train + val, all coming from `psota`/`alameer`/`bergamini`
naming patterns whose file names expose frame indices. In other words, a few frames of the same
recording can occur on both sides — potential **clip-level overlap caused by incomplete sequence
identifiers**. That overlap touches the validation split only (the test splits stay image-disjoint
from their training pools), but it can inflate a validation-selected checkpoint, so the effect of the
proposed modifications was re-measured under a split that removes it.

## 2. The leakage-corrected split

Built only from the original 2681-image development pool, with a frame-aware clip key
(text before `--` → text before `_frame_` → trailing `-<digits>` removed → trailing `_<digits>`
removed → base name) and greedy whole-clip-group assignment:

| | Images | Instances | Clip keys |
|---|---:|---:|---:|
| train | 2411 | 33127 | 1484 |
| val | 270 | 3614 | 228 |
| development pool | 2681 | — | 1712 groups |

Verified guarantees written by the split script:

* **train ∩ val image overlap = 0**
* **train ∩ val clip overlap = 0**
* no image of train or val occurs in the official test split, and the official test split was neither
  split nor trained on nor used for checkpoint selection
  (`"modified": false, "used_for_splitting": false, "used_for_checkpoint_selection": false`)

**Stated limitation.** `danuma` file names carry no frame or video marker, so for that source the clip
key degenerates to the file name; clip-level disjointness can therefore only be established for the
sources whose names expose sequence structure. The guarantee above is a statement about the clip keys
that the file names allow.

## 3. Test isolation and protocol

Ten runs — E0 (YOLO26s baseline) and E5 (CA + SIoU) × seeds {42, 1, 7, 21, 100} — were trained with
the frozen protocol used everywhere in this study (epochs 100, imgsz 640, batch 4, MuSGD, lr0 0.01,
momentum 0.937, single class `pig`). The official benchmark test split (250 images) **was not accessed
until all training procedures were finalized**: it took no part in splitting, training, early stopping
or checkpoint selection. After training finished, each of the 20 checkpoints (10 runs × `best.pt` /
`last.pt`) was evaluated once on that split, and no structure or hyperparameter was re-selected from
the test results.

`best.pt` is the first epoch attaining the maximum validation fitness (ultralytics updates the best
checkpoint only on a strictly greater value); this matters for `lc_E0_seed7`, where the maximum
0.8595 is attained at both epoch 85 and epoch 90, so the reported best epoch is 85.

**Disclosed incident.** The first attempt of run `lc_E5_seed7` terminated after about one minute with
an incidental CUDA error (out of memory) and produced no checkpoints. It was rerun from scratch under
the identical predefined configuration; because the empty original run directory was left untouched
rather than deleted or renamed, ultralytics wrote the rerun to `lc_E5_seed7-2`, which is the
reportable seed-7 E5 run. The failed directory is retained for transparency and contains no weights.
All other nine runs trained without incident. As an additional reproducibility check, the 18
checkpoints already evaluated before that rerun were re-evaluated afterwards and reproduced
identically (0 differences).

## 4. Results

Per-seed values are in **`table19`** (`seed`, `E0_best`, `E5_best`, `Delta_best`, `E0_last`, `E5_last`,
`Delta_last`) and the paired statistics in **`table20`**; both are plotted in
**`figures/fig12_split_correction_effect.png`**.

| Split / checkpoint rule | mean Δ | 95 % bootstrap percentile interval | sign-flip p (two-sided) |
|---|---:|---|---:|
| Original split, `best.pt` (published) | +0.0029 | [−0.00048, +0.00628] | 0.3125 |
| Original split, `last.pt` (measured, §5) | +0.0017 | [−0.00096, +0.00366] | 0.2500 |
| Leakage-corrected split, `best.pt` | +0.0021 | [−0.00084, +0.00556] | 0.3125 |
| Leakage-corrected split, `last.pt` | +0.0032 | [−0.00016, +0.00642] | 0.2500 |

*Corrected split, `best.pt`* (n = 5 paired seeds): E0 0.7750 ± 0.0025, E5 0.7770 ± 0.0025, per-seed
Δ [+0.0001, +0.0088, +0.0032, −0.0025, +0.0007], median +0.0007, sample SD 0.0043.
*Corrected split, `last.pt`*: E0 0.7737 ± 0.0022, E5 0.7769 ± 0.0022, per-seed Δ [+0.0043, +0.0083,
+0.0056, −0.0021, −0.0001], median +0.0043, sample SD 0.0042. Intervals use the exhaustive paired
bootstrap (all 5⁵ = 3125 resamples, percentile method); p values are exact paired sign-flip
permutation tests over all 2⁵ = 32 sign configurations (the smallest attainable two-sided p is 0.0625
at n = 5).

**Conclusion.** Removing the clip-level overlap leaves the conclusion unchanged: CA + SIoU yields a
small positive mean difference (+0.0021 for `best.pt`, +0.0032 for `last.pt`), every 95 % interval
contains zero, and no test is significant at n = 5 seeds (p ≥ 0.25). The corrected split therefore
supports the manuscript's statement that the lightweight modifications produce small gains of
uncertain statistical significance rather than a validated improvement.

## 5. Producing the original split's `last.pt` values

The original five-seed report used `best.pt` only, so the split-correction comparison needed the
`last.pt` values of the same ten frozen runs. They were measured — never retrained — with the
manuscript protocol (`dataset/pig.yaml`, `split=test`, imgsz 640, batch 8) by
`results/original_split_last_eval/` (`eval_results.csv`, `statistics.json`, `fig12_data.json`).
Provenance check: the ten `best.pt` checkpoints of those runs were re-evaluated in the same pass and
reproduced the published five-seed table exactly (`provenance_best_check.csv`, all differences 0.0),
which confirms the run-to-checkpoint mapping. The official test split had already been used for the
original (exploratory) report, so this is a checkpoint-sensitivity analysis on an unchanged split, not
a new selection step.

## 6. Limitations to keep in the manuscript

1. The official 250-image test split is **entirely `danuma`**, and the original main training split
   contains 1319 `danuma` images, so this evaluation is **not** source-held-out and remains
   **exploratory**; the corrected split removes clip-level overlap but does not change this.
2. Source prefixes are dataset/file-name prefixes, not physical farms or pens.
3. `danuma` file names expose no sequence structure, so clip-level guarantees are limited to the
   sources whose names are informative.
4. n = 5 seeds: the exact sign-flip test cannot return p < 0.0625, and the bootstrap intervals are
   wide; the results are reported as effect sizes with uncertainty, not as significance claims.

## 7. Reproduction

```bash
python scripts/analysis/reproduce_split_correction.py   # regenerates table19, table20 and fig12
```

The script reads only the committed result files listed at the top of this document.
