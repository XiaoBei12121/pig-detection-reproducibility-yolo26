# PUBLIC RELEASE AUDIT

| Item | Value |
|---|---|
| Repository | https://github.com/XiaoBei12121/pig-detection-reproducibility-yolo26 (public) |
| Repository URL | https://github.com/XiaoBei12121/pig-detection-reproducibility-yolo26 |
| **Submission release (cited by the manuscript)** | **`v1.0.1`** (annotated) |
| Submission release URL | https://github.com/XiaoBei12121/pig-detection-reproducibility-yolo26/releases/tag/v1.0.1 |
| Submission release commit | the commit tagged `v1.0.1` — resolve with `git rev-list -n1 v1.0.1` (author-metadata commit: `41d0f3a`; this audit file intentionally does not hard-code its own final SHA) |
| First published release (history) | `v1.0.0` (annotated) = `84a7f08` — author metadata was still `TBD`; kept for history, **not** cited by the manuscript |
| Authors | Shuo Kang; Lifeng Yin — School of Rail Transit Intelligent Engineering, Dalian Jiaotong University, Dalian 116028, Liaoning, China |
| Corresponding author | Lifeng Yin — yinlifeng1030@djtu.edu.cn |
| Experimental audit commit (development repository) | **0f51028** (`git rev-parse HEAD` of the private development repo at freeze time) |
| Date | 2026-09-10 |
| License | AGPL-3.0-only (files derived from Ultralytics keep their original headers) |
| Tracked files | 127 |

## Included files

- `configs/` — 8 model configurations: E0–E6 (baseline, CA, BiFPN-style, SIoU and combinations) + `yolo11s_ca.yaml`
- `modules/` — `coordatt.py`, `bifpn.py`, `siou_loss.py` + `README.md` describing exactly what was changed
- `scripts/train/` — `train_pigdetect.py`, `train_piglife.py`, `train_yolo11_replication.py`, `align_weights.py`
- `scripts/eval/` — `evaluate_test.py`, `coco_eval.py`, `cross_scene_eval.py`, `checkpoint_eval.py`
- `scripts/analysis/` — `reproduce_statistics.py`, `reproduce_tables.py`, `bootstrap_multiseed_e0_e5.py`,
  `permutation_test.py`, `rank_stability.py`, `error_analysis.py`, `occlusion_analysis.py`,
  `bbox_scale_audit.py`, `complexity_audit.py`
- `scripts/data/` — `prepare_pigdetect.py`, `prepare_piglife.py`, `split_final_test.py`, `build_manifests.py`
- `splits/` — exact split lists (relative file names only): PigDetect train/val/test, cross-scene,
  final-protocol; PigLife camera split + `group_manifest.csv`
- `manifests/` — 27 PigDetect runs, 18 PigLife rows (9 runs × best/last), 12 YOLO11s rows (6 × best/last)
- `results/` — locked result files: multi-seed, PigLife best/last, YOLO11 replication, cross-scene,
  error analysis (conf ≥ 0.25), occlusion (with an explicit note on the two different configurations),
  audits, and the 18 regenerated tables under `results/generated/`
- `figures/` — fig01 … fig11
- `docs/` — `REPRODUCIBILITY.md`, `DATASETS.md`, `EXPERIMENT_MATRIX.md`, `RESULTS_TRACEABILITY.md`, `ENVIRONMENT.md`
- `supplementary/` — runtime (FPS) measurements and notes
- `README.md`, `LICENSE` (AGPL-3.0), `CITATION.cff`, `requirements.txt`, `requirements_full.txt`,
  `environment.yml`, `.gitignore`, `.gitattributes`, `environment/` snapshots

## Excluded files (deliberately)

| Excluded | Reason |
|---|---|
| PigDetect / PigLife images and annotations | Not redistributable; obtain from the providers (see `docs/DATASETS.md`) |
| `*.pt` / `*.pth` / `*.onnx` / `*.engine` | Model weights are not distributed in this version (kept repository small; training commands are documented) |
| `args.yaml` of individual runs, server logs, archives | Environment-specific artefacts; their content is summarised in `manifests/` |
| Ultralytics framework source | Third-party dependency; the modified files are shipped in `modules/` |

## Verification performed before publishing

| Check | Command / method | Result |
|---|---|---|
| Working tree clean | `git status --short` | **clean** (before the final commit) |
| No dataset images / annotations | scan for `data/`, `datasets/`, `images/`, `labels/`, `*.jpg` | **none present** |
| No checkpoints | scan for `*.pt`, `*.pth`, `*.onnx`, `*.engine` | **none present** |
| No archives / logs / secrets files | scan for `*.zip`, `*.log`, `.env`, `*.key`, `*.pem` | **none present** |
| No large files | `Get-ChildItem -Recurse | Where Length > 10MB` | **none** |
| No absolute local paths | regex scan for Windows/Unix absolute-path prefixes and the local user-id string over all text files | **0 hits** (2 pip-freeze lines with local editable-install paths were replaced by path-free notes) |
| No credentials | regex scan for common credential patterns (provider-style access tokens, private-key headers, api-key/password/token assignments) | **0 hits** |
| Statistics reproduction | `scripts/analysis/reproduce_statistics.py --figure` | reproduces 5-seed mean/SD (E0 0.7732 ± 0.0032; E5 0.7761 ± 0.0030), paired Δ +0.0029, exhaustive bootstrap CI [−0.00048, 0.00628] (3125 resamples), exact sign-flip p = 0.3125, PigLife best/last, YOLO11s best/last, rank stability; Figure 10 regenerated |
| Bootstrap / permutation scripts | `bootstrap_multiseed_e0_e5.py`, `permutation_test.py` | run successfully; values match the manuscript |
| Table regeneration | `scripts/analysis/reproduce_tables.py` | all 18 tables regenerated (`results/generated/table01…table18.csv`) |
| Rank stability + Figure 11 | `scripts/analysis/rank_stability.py` | tables + `figures/fig11_rank_stability.png` regenerated |
| Training smoke test | `scripts/train/train_pigdetect.py --epochs 1 --seed 42` (YOLO26s baseline) | **passed** (1 epoch completed, validation mAP50 = 0.913; output written to a temporary directory, not committed) |

## Data redistribution status

- PigDetect: **not redistributed**. Source: https://doi.org/10.25625/I6UYE9
- PigLife: **not redistributed** (consistent with the manuscript statement that the original data must
  be obtained under the provider's license). Source: https://data.aifarms.org/view/piglife
- Only split lists (file names), preprocessing scripts and derived statistics are shipped.

## Known limitations of the artifact

- No checkpoints are included: absolute metrics require retraining; every reported *statistic* is
  nevertheless reproducible offline from the shipped result files.
- PigDetect train/val is a project-level re-split, not the benchmark default.
- The 204-image final-protocol holdout comes from the historical development pool (186/18 frames were
  previously in the exploratory train/val splits); it is unseen only for the retrained final-protocol
  models and was evaluated once after freezing the protocol.
- PigLife validation has 6 camera groups; checkpoint selection on it is noisy (best/last sensitivity is
  therefore reported explicitly).
- Runtime (FPS) is environment dependent and is reported only in `supplementary/`.
- Hyper-parameters were fixed a priori (CA r = 32; SIoU θ = 4; two local BiFPN-style fusion nodes) and
  were not exhaustively tuned.

## Publication status

### Release `v1.0.1` — manuscript submission version (author metadata completed)

- Tag: `v1.0.1` (annotated), published 2026-09-10.
- Author-metadata commit: `41d0f3a` ("Complete author metadata for manuscript release v1.0.1").
- Final release commit: the commit tagged `v1.0.1` — resolve with `git rev-list -n1 v1.0.1`
  (this audit document is updated by a follow-up commit, whose SHA is therefore deliberately not
  hard-coded here).
- Release URL: https://github.com/XiaoBei12121/pig-detection-reproducibility-yolo26/releases/tag/v1.0.1
- Authors: Shuo Kang; Lifeng Yin — School of Rail Transit Intelligent Engineering,
  Dalian Jiaotong University, Dalian 116028, Liaoning, China. Corresponding author: Lifeng Yin
  (yinlifeng1030@djtu.edu.cn). ORCID iDs are intentionally not asserted.
- Changes relative to `v1.0.0`: `CITATION.cff` (authors, affiliations, `AGPL-3.0-only` license
  identifier, `url`, `version`), `README.md` (authors, affiliation, corresponding author, citation
  guidance), this audit document and the submission checklist.
- **Unchanged relative to `v1.0.0`**: every file under `results/`, `splits/`, `figures/` and
  `scripts/` — no experimental value, split list, manifest, figure or statistic was modified, and the
  raw audit records (`*_raw.json`) are retained.

### Release `v1.0.0` — first publication (history, author metadata was `TBD`)

- Tag: `v1.0.0` (annotated) = `84a7f08`; kept unchanged so the published history stays auditable.
- Release URL: https://github.com/XiaoBei12121/pig-detection-reproducibility-yolo26/releases/tag/v1.0.0
- Upload method: `gh repo create … --public --source … --remote origin --push`, followed by
  `git push origin v1.0.0` and `gh release create v1.0.0 …` (through the local HTTPS proxy).
- Commit identity for the public repository: `XiaoBei12121 <XiaoBei12121@users.noreply.github.com>`
  (no personal e-mail is exposed in the git history).
