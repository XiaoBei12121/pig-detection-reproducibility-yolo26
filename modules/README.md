# Modules — what was actually changed

All files in this directory are **derived from or modified relative to Ultralytics**
(<https://github.com/ultralytics/ultralytics>) and keep their original AGPL-3.0 headers.
They are distributed under the same license (see `../LICENSE`).

The three candidate modifications evaluated in the manuscript are described below. Nothing else in
the detector was changed: backbone, neck, detection head, label assignment and training schedule
stay identical to the official YOLO26s / YOLO11s implementation.

---

## 1. Coordinate Attention (CA)

- **Source file**: `coordatt.py` (`class CoordAtt`)
- **Registration**: imported in `ultralytics/nn/modules/__init__.py` and referenced by name in the
  model YAML (`CoordAtt`).
- **Insertion location (YOLO26s)** — `configs/yolo26s_ca.yaml`:
  - backbone index 10 = `C2PSA` → inserted at **index 11**, i.e. after the last C2PSA block and
    before the neck (P5/32 stage);
  - head indices shift by +1 because the backbone gains one layer; the neck concatenation that
    previously consumed the C2PSA output now consumes the CA output (`[[-1, 11], 1, Concat, [1]]`);
  - `Detect` inputs become `[17, 20, 23]` (P3-small / P4-medium / P5-large).
- **Insertion location (YOLO11s)** — `configs/yolo11s_ca.yaml`: the YOLO11s backbone is
  topologically identical at its end (`… C3k2 → SPPF (9) → C2PSA (10)`), so CA is inserted at
  **index 11** as well and the same +1 index shift is applied (`Detect` inputs `[17, 20, 23]`).
- **Channels**: input = output = 512 at the `s` scale (C2PSA output, 1024 × width 0.5).
- **Reduction ratio**: `r = 32`, i.e. bottleneck width `mip = max(8, 512 // 32) = 16`
  (verified: `conv1.weight` shape `(16, 512, 1, 1)`, `conv_h/conv_w` shape `(512, 16, 1, 1)`).
- **Added parameters** (measured on the trained checkpoints, `scripts/analysis/complexity_audit.py`):
  - YOLO26s: 9.9486 M → 9.9743 M (**+0.0256 M, +0.25 %**)
  - YOLO11s: 9.4282 M → 9.4538 M (**+0.0256 M, +0.27 %**)
  - GFLOPs at 640×640: YOLO26s 22.7 → 22.7; YOLO11s 21.8 → 21.8
- **Weight initialisation**: shared (backbone/neck/head) parameters are inherited from the official
  pretrained weights by key alignment (layer-index shift + shape match, `scripts/train/align_weights.py`);
  the newly added CA parameters keep the framework default initialisation. No variant is trained from
  scratch while its baseline uses pretrained weights.

## 2. SIoU bounding-box regression loss

- **Source file**: `siou_loss.py` (`def siou_loss`), patched into
  `ultralytics/utils/loss.py` inside `BboxLoss.forward`.
- **Enabled by environment variable**: `YOLO_SIOU=1` (default `0` = native YOLO26 regression loss).
- **Only the bounding-box regression path is modified.** Classification loss, label assignment,
  progressive/STAL-related training logic and every other training component remain untouched.
- **No additional trainable parameters are introduced.**
- Loss form implemented (matching the original SIoU paper):

  `L = 1 − IoU + (Δ + Ω) / 2`

  where Δ is the angle-modulated distance cost (`γ = 2 − Λ`, `Λ = cos(2·arcsin(sin α) − π/2)`) and
  Ω is the shape cost with `θ = 4`. The angle term modulates the distance cost; it is not an
  additional additive term.
- **Configs**: `configs/yolo26s_siou.yaml` (E3), `configs/yolo26s_ca_siou.yaml` (E5),
  `configs/yolo26s_ca_bifpn_siou.yaml` (E6) are topology-identical to their non-SIoU counterparts —
  they exist only so that every run maps to an explicit config file.

## 3. BiFPN-style weighted fusion (BiFPNAdd)

- **Source file**: `bifpn.py` (`class BiFPNAdd`), registered in `ultralytics/nn/modules/__init__.py`
  and handled for list inputs in `ultralytics/nn/tasks.py`.
- **This repository does not implement the complete EfficientDet BiFPN topology.** It implements a
  **local BiFPN-style weighted-fusion substitution at two bottom-up fusion nodes of the YOLO26s
  neck** (the two top-down Concat nodes are unchanged).
- **Forward pass** (exact code path):

  ```
  w_i^+ = ReLU(w_i)                                  # non-negative mapping (ReLU, not softplus)
  alpha_i = w_i^+ / (sum_j w_j^+ + eps),  eps = 1e-4 # normalisation
  x_fuse = sum_i alpha_i * x_i                       # weighted fusion
  y = SiLU(BN(Conv3x3(x_fuse)))                      # 3x3, padding=1, bias=False → BN → SiLU
  ```

  Learnable weights are initialised to `torch.ones(num_inputs)`; each replaced node has
  **`num_inputs = 2`**.
- **Replaced nodes**:
  - `configs/yolo26s_bifpn.yaml` (E2): head index **18** — `[[-1, 13], 1, BiFPNAdd, [512]]` (P4/16)
    and head index **21** — `[[-1, 10], 1, BiFPNAdd, [1024]]` (P5/32, consuming the backbone C2PSA output).
    `Detect` inputs stay `[16, 19, 22]`.
  - `configs/yolo26s_ca_bifpn.yaml` (E4/E6): with CA inserted, the same two nodes shift to indices
    **19** and **22**; the P5 node consumes the CA output (`index 11`). Everything else is identical.
- **Added complexity**: +3.525 M parameters (9.949 → 13.474 M, +35.4 %) and +4.6 GFLOPs
  (22.7 → 27.3).
- **Scope of the negative result**: the conclusion in the manuscript applies only to these two local
  weighted-fusion substitutions inside the YOLO26s neck, under this training protocol. It is **not**
  evidence against the full multi-scale BiFPN architecture of EfficientDet.
