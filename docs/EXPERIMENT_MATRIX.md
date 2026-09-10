# Experiment Matrix

## E0–E6 controlled ablation on YOLO26s (PigDetect)

| ID | Model | CA | BiFPN-style | SIoU | Config | Main seed |
|---|---|---:|---:|---:|---|---:|
| E0 | YOLO26s | No | No | No | `configs/yolo26s_baseline.yaml` | 42 |
| E1 | YOLO26s + CA | Yes | No | No | `configs/yolo26s_ca.yaml` | 42 |
| E2 | YOLO26s + BiFPN-style | No | Yes | No | `configs/yolo26s_bifpn.yaml` | 42 |
| E3 | YOLO26s + SIoU | No | No | Yes | `configs/yolo26s_siou.yaml` | 42 |
| E4 | YOLO26s + CA + BiFPN-style | Yes | Yes | No | `configs/yolo26s_ca_bifpn.yaml` | 42 |
| E5 | YOLO26s + CA + SIoU | Yes | No | Yes | `configs/yolo26s_ca_siou.yaml` | 42 |
| E6 | YOLO26s + CA + BiFPN-style + SIoU | Yes | Yes | Yes | `configs/yolo26s_ca_bifpn_siou.yaml` | 42 |

SIoU does not change the topology: the three SIoU configs are structurally identical to their
non-SIoU counterparts and the loss is switched on with the environment variable `YOLO_SIOU=1`.

## Seed coverage

| Experiment | Variants | Seeds |
|---|---|---|
| PigDetect E0/E5 (multi-seed) | E0, E5 | 42, 1, 7, 21, 100 |
| PigDetect E1/E3 (multi-seed) | E1, E3 | 42, 1, 7 |
| PigDetect E2/E4/E6 | E2, E4, E6 | 42 |
| PigDetect repeats (engineering reproducibility) | E0, E5 | 42 (×3 runs each; identical predictions) |
| PigLife | E0, E5 | 42, 1, 7 |
| PigLife | E1, E2, E3 | 42 |
| YOLO11s cross-generation replication | baseline, +CA | 42, 1, 7 |

## Checkpoint rules

| Rule | Definition | Used for |
|---|---|---|
| `best` | validation-selected `best.pt` | all primary results (model selection) |
| `last` | epoch-100 `last.pt` | post-hoc checkpoint-rule sensitivity analysis only (PigLife E0/E5; YOLO11s baseline/+CA) |

The test split is never queried per epoch and is never used to select models or hyper-parameters.

## Datasets and splits

| Experiment | Train / Val / Test | Split files |
|---|---|---|
| PigDetect main | 2411 / 270 / 250 | `splits/pigdetect/{train,val,test}.txt` |
| PigDetect cross-scene (danuma held out) | 1098 / 129 / 450 | `splits/pigdetect/cross_scene_*.txt` |
| PigDetect final-protocol holdout | 2226 / 251 / 204 | `splits/pigdetect/final_*.txt` |
| PigLife | 1441 / 263 / 426 | `splits/piglife/{train,val,test}.txt` + `group_manifest.csv` |

## Training protocol (all runs)

`epochs=100`, `imgsz=640`, `batch=4`, `optimizer=MuSGD`, `lr0=0.01`, `momentum=0.937`, single class `pig`,
pretrained initialisation from the official checkpoint with key alignment for modified architectures.
