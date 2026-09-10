# Reproducibility notes

## What this artifact contains — and what it does not

**Contains**: model configurations, the CA / BiFPN-style / SIoU source files and change description,
dataset split lists (relative file names only), preprocessing scripts, training/evaluation scripts,
statistical analysis scripts, run manifests, locked result files, manuscript figures, and the
environment snapshot.

**Does not contain**: PigDetect/PigLife images or annotations (obtain from the providers, see
`DATASETS.md`), trained weights (`*.pt`), server logs, or any absolute local paths, credentials or
personal data. See `../.gitignore`.

## Three levels of "reproducibility" distinguished in the paper

| Level | Definition | Evidence in this repository |
|---|---|---|
| Engineering reproducibility | Same seed, same environment → identical output | E0/E5 were each re-run 3× at seed 42; bit-identical `predictions.json` (`manifests/runs_pigdetect.csv`, rows `E0_rep*`, `E5_rep*`) |
| Statistical stability | Behaviour across random seeds | PigDetect E0/E5 × 5 seeds, E1/E3 × 3 seeds; PigLife × 3 seeds; exhaustive bootstrap CI and exact sign-flip permutation |
| Cross-condition reproducibility | Does the ranking survive a dataset change, a checkpoint-rule change and a detector-generation change? | PigLife (independent dataset), best.pt vs last.pt sensitivity, YOLO11s replication |

## Checkpoint policy (strictly enforced)

1. `best.pt` = validation-selected checkpoint; used for all primary results and model selection.
2. `last.pt` = epoch-100 checkpoint; used **only** as a post-hoc checkpoint-rule sensitivity check.
3. The test split is evaluated **once per checkpoint after training**; it is never queried per epoch
   and never used for hyper-parameter or model selection.
4. In the final-protocol experiment the protocol (weights md5, git commit, conf/IoU) was frozen before
   the single official evaluation.

## Analysis without retraining

The statistics, tables, rank-stability figures and the effect-size figure are recomputed from the
locked JSON/CSV files shipped in `results/`:

```bash
python scripts/analysis/reproduce_statistics.py --figure
python scripts/analysis/reproduce_tables.py
python scripts/analysis/rank_stability.py
python scripts/analysis/bootstrap_multiseed_e0_e5.py
python scripts/analysis/permutation_test.py
```

These need only Python + numpy (matplotlib for figures) — no GPU, no dataset, no Ultralytics.

## Known limitations of the artifact

- No checkpoints are redistributed: re-training is required to reproduce absolute mAP values from
  scratch (the analysis above still reproduces every reported statistic).
- PigDetect train/val is a project-level re-split, not the benchmark default (documented in
  `DATASETS.md`).
- The 204-image final-protocol holdout is drawn from the historical development pool (186/18 frames
  previously seen in the exploratory train/val splits) and is only unseen for the retrained
  final-protocol models.
- PigLife validation contains 6 camera groups only; checkpoint selection on it is noisy, which is
  exactly why the best/last sensitivity analysis is reported.
- Runtime (FPS) is environment sensitive: repeated laptop-GPU measurements varied by >10 % and
  reversed the E0/E5 ordering, so FPS appears only in `supplementary/`.
- Hyper-parameters were not exhaustively tuned: CA reduction `r=32`, SIoU `theta=4`, and the two
  local BiFPN-style fusion nodes were fixed a priori.
