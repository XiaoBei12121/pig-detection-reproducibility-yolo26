# Rank stability

## YOLO26s (E0/E1/E3/E5, common seeds 42/1/7)

| Seed | rank1 | rank2 | rank3 | rank4 |
|---|---|---|---|---|
| 42 | E5 | E3 | E0 | E1 |
| 1 | E1 | E3 | E0 | E5 |
| 7 | E1 | E0 | E5 | E3 |

- mean rank: {'E0': 2.667, 'E1': 2, 'E3': 2.667, 'E5': 2.667}
- rank SD: {'E0': 0.577, 'E1': 1.732, 'E3': 1.155, 'E5': 1.528}
- rank-1 counts: {'E0': 0, 'E1': 2, 'E3': 0, 'E5': 1}
- Spearman: {'42_vs_1': -0.8, '42_vs_7': -0.8, '1_vs_7': 0.4}

## YOLO11s best.pt vs last.pt (baseline vs +CA)

| Seed | better (best.pt) | better (last.pt) | flip | Δbest | Δlast |
|---|---|---|---|---:|---:|
| 42 | baseline | baseline | no | -0.0028 | -0.0042 |
| 1 | +CA | +CA | no | +0.0048 | +0.0004 |
| 7 | baseline | +CA | YES | -0.0005 | +0.0036 |

- flips: 1/3 seeds

Only 4 models and 3 common seeds are available for the YOLO26s ranking and only 2 arms for YOLO11s; Spearman coefficients and flip counts are descriptive only.
