# Occlusion / overlap analysis — which file is which

Both files below were produced by overlap-proxy analysis of ground-truth boxes
(a box's covered-area ratio by other GT boxes). **They use different confidence thresholds and
must not be mixed.**

| File | Confidence / IoU settings | Role in the manuscript |
|---|---|---|
| `../error_analysis/error_analysis_conf025.json` | predictions filtered at **conf ≥ 0.25**, IoU ≥ 0.5 | **This is the source of manuscript Tables 6 and 8** (per-level misses and recall; edge/stack/normal misses; false positives) |
| `occlusion_analysis.json` (this folder) | different (lower) confidence configuration | **NOT** the source of manuscript Table 6. Kept only for completeness/audit; its recall values are systematically higher because far more low-confidence predictions are retained |

Verification of the correct pairing (PigDetect exploratory test, 5436 GT instances):

| Level | Instances | `error_analysis_conf025.json` misses (E0 / E5) | Implied recall (E0 / E5) |
|---|---:|---|---|
| Low | 1715 | 24 / 21 | 0.9860 / 0.9878 |
| Medium | 1385 | 38 / 32 | 0.9726 / 0.9769 |
| High | 2336 | 208 / 210 | 0.9110 / 0.9101 |

Level definitions (identical in both datasets, verified in the code):
`ratio < 0.3 → Low`, `0.3 ≤ ratio < 0.6 → Medium`, `ratio ≥ 0.6 → High`.

PigLife equivalent (manuscript Table 15) is produced by
`scripts/analysis/occlusion_analysis.py --dataset piglife` and stored in
`../statistics/` via the PigLife detailed-evaluation outputs.
