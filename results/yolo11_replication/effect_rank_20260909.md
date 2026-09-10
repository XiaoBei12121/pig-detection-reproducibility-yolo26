# YOLO11s 复现实验：effect 与 rank 分析

## 1. YOLO11s best/last paired effect（test mAP50-95）

| Seed | base best | +CA best | Δbest | base last | +CA last | Δlast |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.7707 | 0.7679 | -0.0028 | 0.7684 | 0.7642 | -0.0042 |
| 1 | 0.7685 | 0.7733 | +0.0048 | 0.7686 | 0.7690 | +0.0004 |
| 7 | 0.7724 | 0.7719 | -0.0005 | 0.7681 | 0.7717 | +0.0036 |
- Δbest: [-0.0028, 0.0048, -0.0005] → mean +0.0005 ± 0.0039（正 1/3）
- Δlast: [-0.0042, 0.0004, 0.0036] → mean -0.0001 ± 0.0039（正 2/3）

- exact sign-flip（2^3=8, best）: 双侧 p = 1.000

- exact sign-flip（2^3=8, last）: 双侧 p = 1.000

## 2. YOLO26s rank stability（共同 seed 42/1/7，E0/E1/E3/E5）

| Seed | rank1 | rank2 | rank3 | rank4 |
|---|---|---|---|---|
| 42 | E5 | E3 | E0 | E1 |
| 1 | E1 | E3 | E0 | E5 |
| 7 | E1 | E0 | E5 | E3 |

| Model | mean rank | rank SD | Rank1 次数 |
|---|---:|---:|---:|
| E0 | 2.67 | 0.58 | 0 |
| E1 | 2.00 | 1.73 | 2 |
| E3 | 2.67 | 1.15 | 0 |
| E5 | 2.67 | 1.53 | 1 |

Spearman rank correlation（4 模型，仅描述性）：
- seed42 vs seed1: -0.80

- seed42 vs seed7: -0.80

- seed1 vs seed7: +0.40

## 3. YOLO11s rank flip（best vs last，+CA vs baseline）

| Seed | best 更高 | last 更高 | flip? |
|---|---|---|---|
| 42 | baseline | baseline | no |
| 1 | +CA | +CA | no |
| 7 | baseline | +CA | YES |

- 3 seeds 中 flip 次数: 1
