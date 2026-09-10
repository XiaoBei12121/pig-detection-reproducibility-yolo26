# Supplementary — runtime (FPS) measurements

Runtime is **not** used to rank models in the manuscript. On the laptop GPU used for this study,
single-image Python-API latency varied by more than 10 % between benchmarking sessions and even
reversed the relative ordering of the two near-identical models E0 and E5:

| Session | E0 FP32 | E5 FP32 | apparent conclusion |
|---|---:|---:|---|
| session 1 (single measurement, 100 iterations) | 60.4 | 56.2 | E5 ≈ 7 % slower |
| session 2 (best of 3×80 iterations, mean) | 51.8 | 57.9 | E5 ≈ 12 % faster |

Consequently the manuscript reports efficiency through **deterministic** quantities
(parameters, GFLOPs) and states explicitly that the claimed "lightweight" property refers to model
size/computation, not to a guaranteed end-to-end latency advantage.

## Files

- `fps_runtime_notes.csv` — every raw measurement: model, precision, the three per-session repeats,
  mean, SD, and the measurement protocol (`RTX 4050 Laptop | imgsz 640 | batch 1 | Python predict API
  | warmup 20 | iters 80 | CUDA-synchronised | image decoding excluded`).
- `../docs/ENVIRONMENT.md` — hardware/software context.

## Observations (qualified to this environment)

- FP16 did not improve measured FPS under this PyTorch pipeline on this laptop GPU; this is
  implementation- and hardware-specific and is not a general statement about FP16.
- At batch size 1 the per-call Python overhead dominates, so the measured slowdown of the
  higher-GFLOPs variant (E2) is smaller than its +20 % GFLOPs would suggest.
- Any deployment decision should re-measure latency on the target device, batch size, precision and
  inference backend (e.g. TensorRT), which is outside the scope of this artifact.
