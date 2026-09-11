# Leakage-corrected confirmatory experiment — run notes

Purpose: verify the CA + SIoU (E5) against the YOLO26s baseline (E0) on a split that removes the
clip-level train/val overlap of the original development split, and evaluate both variants once on
the untouched official PigDetect benchmark test split (250 images).

## Split

* Built only from the original 2681-image development pool.
* train 2411 images / val 270 images, filled by greedy clip-group assignment.
* Guarantees (verified on the written files): train/val image intersection = 0, train/val clip-key
  intersection = 0; the official test split was not split, trained on, or used for checkpoint
  selection, and does not overlap train or val.
* Audit: split_audit.json, split_audit.csv (this directory).
* Limitation: danuma file names carry no frame marker, so for those images the clip key degenerates
  to the file name.

## Runs

* 10 runs = 2 variants x seeds {42, 1, 7, 21, 100}; protocol frozen and identical to the original
  experiment (epochs 100, imgsz 640, batch 4, MuSGD, lr0 0.01, momentum 0.937, single class pig).
* E0 initialised from yolo26s.pt; E5 initialised from weights/yolo26s_ca_aligned.pt with SIoU enabled.
* best.pt = validation-selected checkpoint; last.pt = epoch 100. Both retained for every run.
* No structure or hyperparameter was selected from test-set results.

## Incident: lc_E5_seed7

* The first attempt of run lc_E5_seed7 terminated after about one minute with an incidental CUDA
  error (out of memory) and produced no checkpoints.
* It was rerun from scratch under the identical predefined configuration (same model, seed, epochs,
  image size, batch size, optimizer, loss setting). The rerun completed all 100 epochs.
* Because the original run directory was deliberately left in place rather than deleted or renamed,
  ultralytics wrote the rerun to the directory lc_E5_seed7-2. That directory is the reportable seed-7
  E5 run; scripts/eval_lc_test.py records the mapping in RUN_DIR_OVERRIDES.
* The failed directory lc_E5_seed7 is retained for transparency; it contains only args.yaml,
  labels.jpg and three train_batch images, and no weights.
* All other 9 runs trained without incident.

## Evaluation and reproducibility check

* Every one of the 20 checkpoints (10 runs x best/last) was evaluated on the official test split
  after training: test_eval_all.json, test_eval_all.csv, test_eval_<run>_<rule>.json.
* The 18 checkpoints that had already been evaluated before the seed-7 rerun were evaluated again
  afterwards as a check: all 18 mAP@0.5:0.95 values reproduced exactly (0 differences at 4 decimals).
* runs_manifest.csv collects configuration, seed, best epoch, validation and test metrics per run;
  every manifest was cross-checked against the corresponding results.csv.

## Best-epoch convention

best.pt corresponds to the first epoch attaining the maximum validation fitness, because ultralytics
updates the best checkpoint only on a strictly greater value. This matters for lc_E0_seed7, where
0.8595 is attained at both epoch 85 and epoch 90: the reported best epoch is 85.

## Paired statistics

statistics.json / statistics.md: per-seed differences, mean, median, sample SD, sign counts,
exhaustive paired bootstrap percentile 95% interval (3125 resamples) and exact sign-flip
permutation test (32 configurations), for both checkpoint rules, plus the unchanged original
main-split reference values.
