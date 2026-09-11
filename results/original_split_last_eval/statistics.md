# Original main split — last.pt evaluation on the official test split

The original five-seed report used best.pt only; this file adds the epoch-100 (last.pt)
checkpoints of the same ten frozen runs, evaluated once with the manuscript protocol
(dataset/pig.yaml, split=test, imgsz=640, batch=8). No retraining, no checkpoint or
original-result modification.

| seed | E0 (last.pt) | E5 (last.pt) | E5-E0 |
|---|---:|---:|---:|
| 42 | 0.7756 | 0.7791 | +0.0035 |
| 1 | 0.7727 | 0.7746 | +0.0019 |
| 7 | 0.7766 | 0.7732 | -0.0034 |
| 21 | 0.7727 | 0.777 | +0.0043 |
| 100 | 0.7711 | 0.7731 | +0.0020 |

- E0 0.7737 ± 0.0023 ｜ E5 0.7754 ± 0.0026
- mean Δ +0.0017 ｜ median +0.0020 ｜ sample SD 0.0030 ｜ range [-0.0034, +0.0043] ｜ +/− 4/1
- exhaustive paired bootstrap (3125 resamples): 95% percentile interval [-0.00096, +0.00366] (contains 0: True)
- exact sign-flip permutation (32 configurations): two-sided p = 0.2500, one-sided p = 0.1250

Provenance: the ten best.pt checkpoints of the same runs were re-evaluated as a check and
reproduced the published main-split table exactly (see provenance_best_check.csv).
