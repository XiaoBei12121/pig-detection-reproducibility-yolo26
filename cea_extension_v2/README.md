# CEA extension evidence snapshot

This directory publishes the code, locked protocol, derived evaluation tables, and audit summaries for the reliability-oriented pig-detection model-selection study. It extends, but does not rewrite, the earlier `v1.0.2` reproducibility artifact. The associated manuscript is under author review; these files are research evidence, not a claim of journal acceptance or prospective farm validation.

## Read the results in order

1. `PROTOCOL_LOCK.md` and `01_protocol/protocol_lock.yaml` describe the corrected split, ten matched E0/E5 seeds, 100-epoch runs, and checkpoint rules. The PigDetect official test is image-disjoint but same-source (`danuma`), not source-held-out or cross-farm.
2. `07_tables/tableA_pigdetect_10seed_corrected.csv` contains the 10 paired seed rows. `07_tables/tableB_checkpoint_statistics.csv` gives the paired summary, bootstrap intervals, and sign-flip probabilities. Under the evaluated conditions, evidence was insufficient to justify replacing E0 with E5.
3. `05_statistics/exact_signflip_results.csv` is the formal exact-rational result. PigDetect `last.pt` two-sided p is **206/1024 = 0.201171875** (manuscript display **0.2012**); the historical float-derived 0.1973 is not the formal result. Exact ties are included in the `|T_perm| >= |T_obs|` comparison.
4. `results/oinktrack_all_evaluations.csv` and `07_tables/tableD_oinktrack_external.csv` document frozen retrospective evaluation on the public OinkTrack official test. Mean E5-E0 mAP@0.5:0.95 differences are approximately -0.0050 (`best.pt`) and -0.0048 (`last.pt`), reversing the small positive internal means. This does not establish E5 inferiority or field performance.
5. `16_failure_multiseed/overlap_per_seed.csv` is the 240-row, 240-unique-key derived per-seed table also prepared as manuscript Supplementary Data 1. It holds aggregate counts and recalls, not images, annotations, image paths, or frame identifiers. `overlap_summary.csv` and `lighting_summary.csv` are descriptive condition summaries. Overlap is a geometric box-overlap proxy, not observed visible occlusion.
6. `15_latency/` contains fixed-hardware timing summaries and verification notes. The full 120,000-row raw timing file is retained in the private project archive, not this GitHub subset. The E0 `best.pt`, batch-1 FP32 row in `latency_summary.csv` is a documented cold-start outlier; use the independently verified warm steady-state rescan of **18.010 ms / 55.53 images/s** for the manuscript comparison. These are model-forward timings on one RTX 4050 Laptop GPU, not end-to-end deployment latency.

The public copies normalize local absolute path prefixes to `<PIG_PROJECT>`; metric values, counts, seeds, and statistical results are not changed. The `run_registry.csv` records checkpoint identities, but the trained checkpoint binaries themselves are not redistributed.

## Scope and provenance

- E5 was chosen after an exploratory official-test result was available, so the PigDetect comparison remains post-selection and vulnerable to selection bias. The corrected split, locked protocol, matched seeds, and later OinkTrack evaluation do not erase that history.
- OinkTrack was not used for training, candidate choice, checkpoint choice, or threshold tuning. It is a retrospective public benchmark, not a prospective deployment cohort.
- The original images and annotations from PigDetect, PigLife, and OinkTrack are not included. Obtain them from their providers. Their licenses remain with the providers.
- No checkpoint, original evaluation CSV, original split, or experiment output was modified to assemble this public subset. The local full-project transfer archive remains private and contains the larger retained evidence.

See `audit/corrected_split_summary.md`, `16_failure_multiseed/FAILURE_MULTI_SEED_REPORT.md`, and `15_latency/LATENCY_VERIFICATION.md` for additional audit detail. Scripts under `scripts/` are retained for inspecting or reproducing calculations from the supplied results; any retraining or re-evaluation requires separately obtained datasets and checkpoints.
