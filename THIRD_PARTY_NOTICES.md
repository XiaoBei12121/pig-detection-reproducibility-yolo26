# Third-party notices

## Ultralytics (AGPL-3.0-only)

Portions of this repository **derive from or modify code and configuration files distributed with
[Ultralytics](https://github.com/ultralytics/ultralytics)**, which is licensed under
**AGPL-3.0** (the full license text is in [`LICENSE`](LICENSE)).

Files that contain Ultralytics-derived material (each retains its original Ultralytics AGPL-3.0
copyright/license header):

| File | Relationship to Ultralytics |
|---|---|
| `modules/coordatt.py` | New module added to the Ultralytics module zoo (`CoordAtt`), registered in `ultralytics/nn/modules/__init__.py` |
| `modules/bifpn.py` | New module added to the Ultralytics module zoo (`BiFPNAdd`), handled for list inputs in `ultralytics/nn/tasks.py` |
| `modules/siou_loss.py` | Drop-in replacement of the bounding-box regression term inside `ultralytics/utils/loss.py` (`BboxLoss.forward`) |
| `configs/yolo26s_*.yaml` | Derived from the official `ultralytics/cfg/models/26/yolo26.yaml` |
| `configs/yolo11s_ca.yaml` | Derived from the official `ultralytics/cfg/models/11/yolo11.yaml` |

**The Ultralytics framework itself is not redistributed here.** The modifications above are applied to
an unmodified local installation of `ultralytics==8.4.135`; see `docs/ENVIRONMENT.md` for the exact
version and `docs/REPRODUCIBILITY.md` for how the modules are wired in.

No Ultralytics trademarks or logos are used. This repository is an independent research artifact and
is **not** endorsed by or affiliated with Ultralytics.

## Datasets (not redistributed)

| Dataset | Provider | Reference |
|---|---|---|
| PigDetect / PigBench | PigBench data repository | https://doi.org/10.25625/I6UYE9 — Henrich et al., *Computers and Electronics in Agriculture* 241 (2026) 111264 |
| PigLife | AIFARMS data portal | https://data.aifarms.org/view/piglife |

Neither dataset's images or annotations are redistributed in this repository. Only split lists
(file names), preprocessing scripts and derived aggregate statistics are included.

## Citation of YOLO26

This repository's `CITATION.cff` describes **this reproducibility artifact only**. If you refer to the
YOLO26 model itself, cite the official publication in your manuscript:

> Jocher, G., Qiu, J., Liu, M., Lyu, S., Akyon, F. C., & Kalfaoglu, M. E. (2026).
> *Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models.* arXiv preprint arXiv:2606.03748.

The same applies to the other third-party methods used in the study (Coordinate Attention, SIoU loss,
BiFPN, FPN/PANet, COCO evaluation), which must be cited through their original papers in the
manuscript reference list.
