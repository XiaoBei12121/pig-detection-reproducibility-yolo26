# Corrected PigDetect split — verification (CEA extension v2, phase 1)

- Timestamp: 2026-09-16T01:21:51
- Project commit: `9918d0878e6087be3f4fd99927c8ab4d52393ede`
- Dataset root: `<PIG_PROJECT>\dataset`
- YAML: `pig_lc.yaml` sha256 `e29520cdeb2ece4762b316e550d373b8e2dc34f4012ca2c16391c98dc3bd3564`

| split | images | instances | danuma | psota | alameer | bergamini | clip keys | content sha256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| corrected_train | 2411 | 33127 | 1259 | 733 | 257 | 162 | 1484 | `f8aa741990ac6525…` |
| corrected_val | 270 | 3614 | 195 | 0 | 51 | 24 | 228 | `3964365ba15f0455…` |
| official_test | 250 | 5436 | 250 | 0 | 0 | 0 | 250 | `7b243f9ef760e910…` |

## Checks

| check | value | pass |
|---|---|---|
| train_images_2411 | 2411 | PASS |
| val_images_270 | 270 | PASS |
| test_images_250_official | 250 | PASS |
| train_val_image_intersection_zero | 0 | PASS |
| train_val_clip_intersection_zero | 0 | PASS |
| test_vs_train_image_intersection_zero | 0 | PASS |
| test_vs_val_image_intersection_zero | 0 | PASS |
| test_vs_train_clip_intersection_zero | 0 | PASS |
| test_vs_val_clip_intersection_zero | 0 | PASS |
| no_duplicate_image_content_across_splits | 0 | PASS |
| yaml_test_entry_points_at_official_test | {"path": "<PIG_PROJECT>/dataset/PigDetect_lc", "test": "../PigDetect/test/images", "resolved": "<PIG_PROJECT>\\dataset\\PigDetect\\test\\images", "expected": "<PIG_PROJECT>\\dataset\\PigDetect\\test\\images"} | PASS |
| official_test_is_danuma_only | {"danuma": 250} | PASS |
| train_contains_danuma_so_test_not_cross_farm | 1259 | PASS |

**All checks pass: True**

## Intersections

| pair | images | clip keys |
|---|---:|---:|
| corrected_train ∩ corrected_val | 0 | 0 |
| corrected_train ∩ official_test | 0 | 0 |
| corrected_val ∩ official_test | 0 | 0 |

## Interpretation constraints (must hold in the manuscript)

- The official 250-image test split is entirely `danuma` and the corrected training split contains 1259 `danuma` images, so this test is **not** cross-farm / source-held-out and must be described as an exploratory same-source test.
- The corrected split removes the clip-level train/val overlap that the file names permit (image intersection 0, clip intersection 0). For `danuma` no clip-level guarantee is possible because its file names expose no frame/video marker.
- Content-level SHA256 shows no duplicate image across the three splits.
## 重新扫描核验（2026-09-21T16:06:24）

独立重扫 2,931 张图并重算全部哈希：**MATCH**。

* train 2411 / val 270 / test 250（与记录一致）
* 三个划分两两交集：文件名级 0，**图像内容级 0**
* split manifest / per-split content / yaml / 逐图哈希清单，逐值与 `corrected_split_hashes.json` 一致
* 详见 `audit/corrected_split_rescan.md` 与 `audit/corrected_split_rescan.json`
