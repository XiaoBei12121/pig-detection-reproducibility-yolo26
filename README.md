# Reproducibility Artifact

**Multi-seed, checkpoint-sensitive, cross-dataset and cross-generation evaluation of lightweight YOLO modifications for pig detection**

This repository is the **reproducibility artifact of a manuscript**, not a "better YOLO" project.
The main conclusion is **not** that CA+SIoU consistently improves YOLO26s. Instead, the experiments
evaluate whether small apparent gains remain stable across random seeds, checkpoint-selection
rules, datasets, and two YOLO generations — and the answer is largely negative.

### Version you should cite

| Reference | Meaning |
|---|---|
| **Release `v1.0.2`** | **the submission version pinned by the manuscript's Code availability statement** (commit `5b94518`); adds the leakage-corrected confirmatory experiment, `docs/SPLIT_CORRECTION_REPORT.md`, table19/table20 and fig12 |
| Release `v1.0.1` | author-metadata correction of the first frozen version; preserved for history (commit `4c72d27`) |
| Release `v1.0.0` | first frozen publication (author metadata still `TBD` in `CITATION.cff`); kept for history |
| `main` | development branch, may receive documentation-only updates — do **not** cite it in the paper |

`v1.0.2` is a strict extension of `v1.0.1`: no original result, split, weight or log was changed, and
both earlier releases remain unchanged. The citation metadata in `CITATION.cff` still declares
version `1.0.1`; only the Code availability statement was moved to `v1.0.2`, so the cited artifact
contains the confirmatory experiment and the split-correction audit.

### What can be reproduced without retraining — and what cannot

| Task | Requires | Status |
|---|---|---|
| Recompute every statistic (5-seed mean/SD, paired deltas, exhaustive bootstrap CI, exact sign-flip permutation, PigLife best/last, YOLO11s best/last, rank stability) | Python + the files in `results/` | ✅ fully reproducible offline |
| Regenerate Tables 1–20 and Figures 10–12 | Python + numpy/matplotlib | ✅ fully reproducible offline |
| Re-evaluate the reported metrics | the trained checkpoints (not redistributed) + the datasets | ⚠️ requires retraining or your own checkpoints |
| Retrain any run | PigDetect/PigLife data obtained from the providers + `manifests/` + `configs/` | ⚠️ not a one-command reproduction: build the dataset layout, then follow `manifests/*.csv` |

The original images, annotations and trained checkpoints are **not** redistributed — see
`docs/DATASETS.md` for the official sources. Third-party (Ultralytics-derived) files and their
licensing are listed in `THIRD_PARTY_NOTICES.md`.

---

## Paper

**Authors:** Shuo Kang; Lifeng Yin

**Affiliation:**
School of Rail Transit Intelligent Engineering,
Dalian Jiaotong University,
Dalian 116028, Liaoning, China

**Corresponding author:** Lifeng Yin — yinlifeng1030@djtu.edu.cn

Manuscript: *"… evaluation of lightweight YOLO modifications for pig detection in densely occluded pens"* (in preparation; no journal, volume or DOI is asserted).
The manuscript submission version of this artifact is pinned to release **`v1.0.2`**; see `PUBLIC_RELEASE_AUDIT.md` for the exact release tag and commit of every published version.

## Overview

We evaluate three commonly used lightweight modifications on YOLO26s:

- **CA** — Coordinate Attention inserted after the backbone C2PSA block (P5/32)
- **SIoU** — bounding-box regression loss (`YOLO_SIOU=1`, no topology change)
- **BiFPN-style** — local learnable weighted fusion replacing two bottom-up Concat nodes of the YOLO26s neck

Protocol: 7 ablations on PigDetect (E0–E6), 5 seeds for E0/E5, 3 seeds for E1/E3, a second dataset
(PigLife), a final-protocol held-out split, and a cross-generation replication on YOLO11s
(baseline vs +CA, 3 seeds). All test evaluations use the same frozen splits and the same evaluation
protocol; the test split is never queried per epoch.

## Main finding

| Evaluation axis | Result |
|---|---|
| PigDetect, 5 matched seeds (E5−E0, mAP@0.5:0.95) | mean **+0.0029**; exhaustive paired bootstrap (5⁵=3125) 95% CI **[−0.00048, 0.00628]**; exact sign-flip permutation (2⁵=32) two-sided **p = 0.3125** |
| PigLife, 3 seeds | validation-best rule: E0 **0.8841±0.0031** vs E5 **0.8682±0.0133**; fixed-end `last.pt` rule: **0.8882 vs 0.8884** → ranking is checkpoint-rule sensitive |
| YOLO11s replication, 3 seeds (CA − baseline) | best.pt **+0.0005±0.0039**, last.pt **−0.0001±0.0039**, 1/3 seeds show a best↔last rank flip |
| BiFPN-style fusion | negative accuracy–complexity trade-off on **both** datasets (+35.4% params, +4.6 GFLOPs, no accuracy gain) |
| Occlusion / overlap | the weakest condition on both datasets; CA+SIoU did not improve the high-overlap group |
| Runtime (FPS) | **not** used to rank models: repeated laptop-GPU measurements varied by >10% across sessions and reversed the E0/E5 ordering (see `supplementary/`) |

## Leakage-corrected confirmatory experiment

After auditing the original PigDetect split, we performed an additional clip-disjoint confirmatory
experiment. The corrected evaluation preserved the original conclusion that lightweight modifications
produced small and statistically uncertain gains.

The audit (`docs/PIGDETECT_SPLIT_AUDIT.md`) found that the original main train/val split is
image-disjoint but shares 17 clip keys (~1% of 1729), i.e. a small sequence-level overlap caused by
incomplete sequence identifiers in part of the file names. The corrected split was rebuilt from the
same 2681-image development pool with a frame-aware clip key, giving **train ∩ val image overlap = 0**
and **train ∩ val clip overlap = 0**; the official 250-image test split was not accessed until all
training procedures were finalized and took no part in splitting, training or checkpoint selection.

| Split / checkpoint rule | mean Δ (E5−E0) | 95% bootstrap interval | exact sign-flip p |
|---|---:|---|---:|
| Original split, `best.pt` (published) | +0.0029 | [−0.00048, +0.00628] | 0.3125 |
| Original split, `last.pt` (measured afterwards from the same frozen checkpoints) | +0.0017 | [−0.00096, +0.00366] | 0.2500 |
| Leakage-corrected split, `best.pt` | +0.0021 | [−0.00084, +0.00556] | 0.3125 |
| Leakage-corrected split, `last.pt` | +0.0032 | [−0.00016, +0.00642] | 0.2500 |

Every 95% interval contains zero and the sign-flip test is not significant at n = 5 seeds
(p ≥ 0.25); removing the clip-level overlap changes neither the sign nor the magnitude of the effect.
Details, including the disclosed retraining incident of one run and the limitations that must be kept,
are in `docs/SPLIT_CORRECTION_REPORT.md` (deliverables: table19, table20, `figures/fig12_split_correction_effect.png`).

## Repository structure

```
configs/        YOLO26s / YOLO11s model configurations (E0–E6 + cross-generation)
modules/        CA, BiFPN-style fusion, SIoU loss — source files and change description
scripts/        data preparation, training, evaluation, analysis and figure/table generation
splits/         exact image lists (relative names only) for every split used in the paper
manifests/      one row per run: config, seed, hyper-parameters, metrics, result file
results/        locked result files (tables, statistics, cross-scene, error/occlusion analysis, audit)
figures/        manuscript figures
docs/           REPRODUCIBILITY / DATASETS / EXPERIMENT_MATRIX / RESULTS_TRACEABILITY / ENVIRONMENT /
                PIGDETECT_SPLIT_AUDIT / SPLIT_CORRECTION_REPORT
supplementary/  runtime (FPS) measurements and notes
```

## Environment

Python 3.10.21 · PyTorch 2.0.1+cu118 · CUDA 11.8 · Ultralytics 8.4.135 · NVIDIA RTX 4050 Laptop GPU (6 GB).
See `docs/ENVIRONMENT.md`; core packages in `requirements.txt`, full snapshot in `requirements_full.txt`.

> The statistics/table/figure regeneration scripts do **not** need a GPU or PyTorch — they run on the
> locked result files shipped in `results/`.

## Datasets

Original PigDetect and PigLife images/annotations are **not redistributed** here. See `docs/DATASETS.md`
for the official sources (DOI / portal), the expected local directory layout, and the license terms.

## Installation

```bash
git clone https://github.com/XiaoBei12121/pig-detection-reproducibility-yolo26.git
cd pig-detection-reproducibility-yolo26
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# CUDA 11.8 PyTorch (if training/evaluating):
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

## Reproduce PigDetect experiments

```bash
export PIG_DATA_ROOT=./data/PigDetect          # Windows: set PIG_DATA_ROOT=.\data\PigDetect
python scripts/train/train_pigdetect.py \
  --config configs/yolo26s_ca_siou.yaml --seed 42 \
  --epochs 100 --imgsz 640 --batch 4 --optimizer MuSGD --lr0 0.01 --momentum 0.937 \
  --siou --project ./runs/pigdetect --name E5_CA_SIoU
python scripts/eval/evaluate_test.py --weights runs/pigdetect/E5_CA_SIoU/weights/best.pt --split test
```

All 20 PigDetect runs (E0–E6 + multi-seed + repeats) are enumerated with their exact parameters in
`manifests/runs_pigdetect.csv`.

## Reproduce PigLife experiments

```bash
python scripts/data/prepare_piglife.py --raw-root ./data/PigLife/raw --out-root ./data/PigLife_1280
python scripts/train/train_piglife.py --config configs/yolo26s_baseline.yaml --seed 42 --name E0_42
python scripts/eval/checkpoint_eval.py --run-dir runs/piglife/E0_42 --rule best,last   # both checkpoint rules
```

Runs: E0/E5 × {42,1,7}; E1/E2/E3 × {42}. See `manifests/runs_piglife.csv`.

## Reproduce YOLO11s replication

```bash
python scripts/train/train_yolo11_replication.py --variant baseline --seed 42
python scripts/train/train_yolo11_replication.py --variant CA       --seed 1
```

Runs: baseline and +CA × {42,1,7}. See `manifests/runs_yolo11_replication.csv`.

## Statistical analysis

No retraining required — these scripts recompute every statistic in the paper from the locked
result files inside `results/`:

```bash
python scripts/analysis/reproduce_statistics.py    # 5-seed mean/SD, paired deltas, bootstrap CI, permutation p,
                                                   # PigLife best/last, YOLO11 best/last, rank stability
python scripts/analysis/bootstrap_multiseed_e0_e5.py   # exhaustive 5^5 paired bootstrap
python scripts/analysis/permutation_test.py            # exact sign-flip permutation (2^5, 2^3)
python scripts/analysis/rank_stability.py              # rank tables + Figure 11
```

## Generate manuscript tables

```bash
python scripts/analysis/reproduce_tables.py        # -> results/generated/table01.csv ... table18.csv
python scripts/analysis/reproduce_split_correction.py   # -> results/generated/table19.csv, table20.csv
```

## Generate manuscript figures

```bash
python scripts/analysis/rank_stability.py                       # figures/fig11_rank_stability.png
python scripts/analysis/reproduce_statistics.py --figure        # figures/fig10_cross_detector_effects.png
python scripts/analysis/reproduce_split_correction.py           # figures/fig12_split_correction_effect.png
python scripts/analysis/bbox_scale_audit.py --figure            # 640-coordinate scale distributions
```

Figures 1–9 are produced from the trained models / dataset statistics by the scripts listed in
`docs/RESULTS_TRACEABILITY.md`.

## Expected results

- PigDetect test (250 images): E0 mAP@0.5:0.95 = 0.775, E5 = 0.781 (seed 42); 5-seed means 0.7732±0.0032 and 0.7761±0.0030.
- PigLife test (426 images): E0 = 0.8855, E5 = 0.8834 (seed 42); 3-seed means 0.8841±0.0031 vs 0.8682±0.0133 (best.pt rule), 0.8882 vs 0.8884 (last.pt rule).
- YOLO11s replication test: baseline 0.7707/0.7685/0.7724 vs +CA 0.7679/0.7733/0.7719 (best.pt, seeds 42/1/7).
- Leakage-corrected confirmatory split (E0/E5 × seeds 42/1/7/21/100, official test 250): mean Δ **+0.0021** (`best.pt`, CI [−0.00084, +0.00556], p = 0.3125) and **+0.0032** (`last.pt`, CI [−0.00016, +0.00642], p = 0.2500).
- Full numbers with sources: `docs/RESULTS_TRACEABILITY.md`.

## Reproducibility notes

- **Checkpoint policy** — validation split selects `best.pt`; `last.pt` (epoch 100) is used only as a
  post-hoc checkpoint-rule sensitivity analysis; test is never queried per epoch and never used for model selection.
- **Fixed-seed reruns** — E0 and E5 were each re-run 3× at seed 42 and produced identical predictions
  (engineering reproducibility, not statistical stability).
- **Multi-seed** — PigDetect E0/E5: {42,1,7,21,100}; E1/E3: {42,1,7}; PigLife E0/E5: {42,1,7}; YOLO11s: {42,1,7}.
- **Statistics** — n = 5 (PigDetect) and n = 3 (PigLife / YOLO11s) are small: we report descriptive
  statistics and exhaustive enumeration (bootstrap CI, exact sign-flip permutation) and do **not**
  claim statistical significance or equivalence.
- **Runtime** — single-image Python-API FPS on a laptop GPU varied by >10% between sessions and even
  reversed the E0/E5 ordering, so FPS is reported only in `supplementary/` and is not used for model ranking.
- **Paths** — all scripts read the dataset root from `PIG_DATA_ROOT` (or CLI arguments); no local
  absolute paths are required.
- **Leakage-corrected confirmatory experiment** — one of its ten runs (`lc_E5_seed7`) first terminated
  after ~1 min with an incidental CUDA out-of-memory error and was rerun from scratch under the
  identical predefined configuration; the failed (weight-free) run directory is retained and the rerun
  is reported under `lc_E5_seed7-2`. The official test split was evaluated once per checkpoint after
  training, and the 18 checkpoints already measured before that rerun reproduced identically. See
  `results/leakage_corrected_confirmatory/run_notes.md`.

## Data availability

PigDetect: https://doi.org/10.25625/I6UYE9 (not redistributed here).
PigLife: https://data.aifarms.org/view/piglife (not redistributed here; obtain under the provider's license).
Split lists used in this paper are provided in `splits/`.

## Code availability

This repository (AGPL-3.0) with the tagged release `v1.0.2` that the manuscript's Code availability
statement pins, plus the earlier releases `v1.0.1` (author-metadata correction) and `v1.0.0`.
Experimental audit snapshot of the development repository: commit `0f51028` (local, not public).

## Citation

If you use this artifact, please cite it as described in `CITATION.cff`:

> Kang, S., & Yin, L. (2026). *Reproducibility artifact for lightweight YOLO modifications for pig
> detection* (version v1.0.1). School of Rail Transit Intelligent Engineering, Dalian Jiaotong
> University. https://github.com/XiaoBei12121/pig-detection-reproducibility-yolo26/releases/tag/v1.0.1

Corresponding author: Lifeng Yin (yinlifeng1030@djtu.edu.cn).
The associated manuscript is still in preparation, so no bibliographic metadata for it is asserted
here. When referring to the YOLO26 model itself, cite the official publication (Jocher et al., 2026,
arXiv:2606.03748) rather than this repository — see `THIRD_PARTY_NOTICES.md`.

## License

**AGPL-3.0** — see `LICENSE`. Files that are derived from or modified relative to
[Ultralytics](https://github.com/ultralytics/ultralytics) (`modules/coordatt.py`, `modules/bifpn.py`,
`modules/siou_loss.py`, `configs/*`) retain their original copyright and license headers and are
distributed under the same AGPL-3.0 terms.
