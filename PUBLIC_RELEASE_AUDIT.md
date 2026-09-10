# PUBLIC RELEASE AUDIT

| Item | Value |
|---|---|
| Repository | `pig-detection-reproducibility-yolo26` (local; **not yet pushed**) |
| Repository URL | _to be filled after `git push` — `https://github.com/<USERNAME>/pig-detection-reproducibility-yolo26`_ |
| Release tag | **v1.0.0** (annotated) |
| Release commit SHA | the commit that tag `v1.0.0` points to — resolve with `git rev-list -n1 v1.0.0` (it is the tip of `main` at freeze time; this file intentionally avoids hard-coding its own SHA) |
| Experimental audit commit (development repository) | **0f51028** (`git rev-parse HEAD` of the private development repo at freeze time) |
| Date | 2026-09-09 |
| License | AGPL-3.0 (files derived from Ultralytics keep their original headers) |
| Tracked files | 125 |

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
| No absolute local paths | regex scan `C:\Users\`, `[A-Z]:\`, `/home/`, `/root/`, `26079` over all text files | **0 hits** (2 pip-freeze lines with local editable-install paths were replaced by path-free notes) |
| No credentials | regex scan `ghp_`, `sk-…`, `BEGIN … PRIVATE KEY`, `api_key`, `password`, `token` | **0 hits** |
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

## Remaining steps (require the repository owner's GitHub login)

```bash
# after creating an empty public repository named pig-detection-reproducibility-yolo26
cd pig-detection-reproducibility-yolo26
git remote add origin https://github.com/<USERNAME>/pig-detection-reproducibility-yolo26.git
git branch -M main
git push -u origin main
git push origin v1.0.0
```

Then create the GitHub Release for tag `v1.0.0` and fill the repository URL / release URL above
(and the placeholder `<USERNAME>` in `README.md` and `CITATION.cff`).
