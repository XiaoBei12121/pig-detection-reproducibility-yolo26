# Recomputed statistics (generated from locked result files)

## PigDetect multi-seed (mAP@0.5:0.95, test 250)

| Model | n | mean | SD | per-seed |
|---|---:|---:|---:|---|
| E0 | 5 | 0.7732 | 0.0032 | {'42': 0.7747, '1': 0.7739, '7': 0.7768, '21': 0.7683, '100': 0.7724} |
| E1 | 3 | 0.7769 | 0.0023 | {'42': 0.7744, '1': 0.7774, '7': 0.779} |
| E3 | 3 | 0.7744 | 0.0014 | {'42': 0.7755, '1': 0.7748, '7': 0.7728} |
| E5 | 5 | 0.7761 | 0.003 | {'42': 0.7806, '1': 0.7725, '7': 0.7759, '21': 0.777, '100': 0.7746} |

Paired E5−E0 deltas: [0.0059, -0.0014, -0.0009, 0.0087, 0.0022] (mean +0.0029, SD 0.0044)
Exhaustive paired bootstrap over 3125 resamples: 95% percentile CI [-0.00048, 0.00628]
Exact sign-flip permutation (32 configurations): two-sided p = 0.3125
Cohen's d_z (exploratory only): 0.67

## PigLife (test 426, E0 vs E5)

| Rule | E0 mean±SD | E5 mean±SD | per-seed Δ | mean Δ | sign-flip p |
|---|---|---|---:|---:|---:|
| best.pt | 0.8841±0.0031 | 0.8682±0.0133 | [-0.0021, -0.0218, -0.0236] | -0.0158 | 0.2500 |
| last.pt | 0.8882±0.0016 | 0.8884±0.002 | [0.0013, 0.0008, -0.0015] | +0.0002 | 1.0000 |

## YOLO11s replication (CA − baseline, test 250)

| Rule | per-seed Δ | mean±SD | positive | sign-flip p |
|---|---|---:|---:|---:|
| best.pt | [-0.0028, 0.0048, -0.0005] | +0.0005±0.0039 | 1/3 | 1.0000 |
| last.pt | [-0.0042, 0.0004, 0.0036] | -0.0001±0.0039 | 2/3 | 1.0000 |

Rank flips (best vs last): {'42': 'no_flip', '1': 'no_flip', '7': 'flip'}

## YOLO26s rank stability (E0/E1/E3/E5, common seeds 42/1/7)

- seed42: E5 > E3 > E0 > E1
- seed1: E1 > E3 > E0 > E5
- seed7: E1 > E0 > E5 > E3
- mean rank: {'E0': 2.667, 'E1': 2, 'E3': 2.667, 'E5': 2.667}; rank-1 counts: {'E0': 0, 'E1': 2, 'E3': 0, 'E5': 1}
- Spearman: {'42_vs_1': -0.8, '42_vs_7': -0.8, '1_vs_7': 0.4}
