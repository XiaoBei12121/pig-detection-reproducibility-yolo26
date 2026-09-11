# Leakage-corrected confirmatory experiment — statistics

Official PigDetect test split (250 images), evaluated once per checkpoint after training.
E5 = CA + SIoU, E0 = YOLO26s baseline; seeds {42,1,7,21,100}; identical training protocol.

## Checkpoint rule: 5 matched seeds (best.pt)

| seed | E0 | E5 | E5-E0 |
|---|---:|---:|---:|
| 42 | 0.778 | 0.7781 | +0.0001 |
| 1 | 0.7712 | 0.78 | +0.0088 |
| 7 | 0.7749 | 0.7781 | +0.0032 |
| 21 | 0.7761 | 0.7736 | -0.0025 |
| 100 | 0.7747 | 0.7754 | +0.0007 |

- E0: 0.775 ± 0.0025 ｜ E5: 0.777 ± 0.0025
- paired deltas: [0.0001, 0.0088, 0.0032, -0.0025, 0.0007]
- mean +0.0021 ｜ median +0.0007 ｜ SD 0.0043 ｜ range [-0.0025, +0.0088] ｜ +/− 4/1
- exhaustive paired bootstrap (3125 resamples): 95% percentile interval [-0.00084, +0.00556] (contains 0: True)
- exact sign-flip permutation (32 configurations): two-sided p = 0.3125, one-sided p = 0.1562 (minimum attainable two-sided p = 0.0625)

## Checkpoint rule: 5 matched seeds (last.pt)

| seed | E0 | E5 | E5-E0 |
|---|---:|---:|---:|
| 42 | 0.7729 | 0.7772 | +0.0043 |
| 1 | 0.7712 | 0.7795 | +0.0083 |
| 7 | 0.7723 | 0.7779 | +0.0056 |
| 21 | 0.7756 | 0.7735 | -0.0021 |
| 100 | 0.7763 | 0.7762 | -0.0001 |

- E0: 0.7737 ± 0.0022 ｜ E5: 0.7769 ± 0.0022
- paired deltas: [0.0043, 0.0083, 0.0056, -0.0021, -0.0001]
- mean +0.0032 ｜ median +0.0043 ｜ SD 0.0042 ｜ range [-0.0021, +0.0083] ｜ +/− 3/2
- exhaustive paired bootstrap (3125 resamples): 95% percentile interval [-0.00016, +0.00642] (contains 0: True)
- exact sign-flip permutation (32 configurations): two-sided p = 0.2500, one-sided p = 0.1250 (minimum attainable two-sided p = 0.0625)

## Reference: original main split (unchanged)

- E0 0.7732 +/- 0.0032, E5 0.7761 +/- 0.0030, mean Δ +0.0029, CI [-0.00048, 0.00628], p = 0.3125
