# Fixed-hardware latency benchmark (Phase 2)

Run: 2026-09-21T19:27:25  
GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 592.82, 6141 MiB  
torch 2.0.1+cu118 / CUDA 11.8 / cuDNN 8700 / ultralytics 8.4.135 / python 3.10.21

Protocol: imgsz 640; batches [1, 4]; 100 warm-up + 1000 measured iterations 脳 3 rounds; fixed in-memory tensor; CUDA-event timing with synchronize; no compile/TensorRT/CUDA-graph/caching shortcuts.

Measured configurations: 40 of 40 (0 unavailable, see the error column).

| model | batch | precision | mean (ms) | median | p95 | std | FPS | img/s | peak VRAM (MiB) | params | GFLOPs |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 (YOLO26s baseline, best) | 1 | fp32 | 28.4657 | 30.4937 | 39.6175 | 8.2525 | 35.13 | 35.13 | 233.3 | 9948638 | 22.7 |
| E0 (YOLO26s baseline, best) | 1 | fp16 | 35.9318 | 35.3833 | 44.4969 | 5.2086 | 27.83 | 27.83 | 196.1 | 9948638 |  |
| E0 (YOLO26s baseline, best) | 4 | fp32 | 28.2926 | 28.3028 | 29.2731 | 1.2476 | 35.34 | 141.38 | 332.7 | 9948638 | 22.7 |
| E0 (YOLO26s baseline, best) | 4 | fp16 | 21.7917 | 20.9969 | 29.229 | 4.1211 | 45.89 | 183.56 | 263.6 | 9948638 |  |
| E0 (YOLO26s baseline, last) | 1 | fp32 | 18.4626 | 17.8611 | 24.3456 | 3.3111 | 54.16 | 54.16 | 233.6 | 9948638 | 22.7 |
| E0 (YOLO26s baseline, last) | 1 | fp16 | 21.5781 | 20.9997 | 28.1027 | 3.698 | 46.34 | 46.34 | 196.1 | 9948638 |  |
| E0 (YOLO26s baseline, last) | 4 | fp32 | 27.4369 | 27.3407 | 28.0371 | 0.9359 | 36.45 | 145.79 | 332.7 | 9948638 | 22.7 |
| E0 (YOLO26s baseline, last) | 4 | fp16 | 20.7378 | 19.8785 | 26.9599 | 3.4583 | 48.22 | 192.88 | 302.0 | 9948638 |  |
| E5 (YOLO26s+CA+SIoU, best) | 1 | fp32 | 19.0269 | 18.5539 | 24.5585 | 3.1736 | 52.56 | 52.56 | 271.0 | 9974286 | 22.7 |
| E5 (YOLO26s+CA+SIoU, best) | 1 | fp16 | 22.4522 | 21.8282 | 28.9567 | 3.7059 | 44.54 | 44.54 | 196.1 | 9974286 |  |
| E5 (YOLO26s+CA+SIoU, best) | 4 | fp32 | 27.6372 | 27.5348 | 28.244 | 0.9515 | 36.18 | 144.73 | 333.7 | 9974286 | 22.7 |
| E5 (YOLO26s+CA+SIoU, best) | 4 | fp16 | 21.1581 | 20.3997 | 27.5763 | 3.5697 | 47.26 | 189.05 | 263.7 | 9974286 |  |
| E5 (YOLO26s+CA+SIoU, last) | 1 | fp32 | 18.8518 | 18.2277 | 24.6793 | 3.4539 | 53.05 | 53.05 | 233.7 | 9974286 | 22.7 |
| E5 (YOLO26s+CA+SIoU, last) | 1 | fp16 | 22.2061 | 21.5751 | 28.5929 | 3.5664 | 45.03 | 45.03 | 196.1 | 9974286 |  |
| E5 (YOLO26s+CA+SIoU, last) | 4 | fp32 | 27.7418 | 27.5671 | 28.5449 | 1.2247 | 36.05 | 144.19 | 333.7 | 9974286 | 22.7 |
| E5 (YOLO26s+CA+SIoU, last) | 4 | fp16 | 21.3085 | 20.675 | 27.3285 | 3.3286 | 46.93 | 187.72 | 263.7 | 9974286 |  |
| YOLO11s baseline (best) | 1 | fp32 | 12.9908 | 12.4148 | 17.2759 | 2.4251 | 76.98 | 76.98 | 220.1 | 9428179 | 21.7 |
| YOLO11s baseline (best) | 1 | fp16 | 15.228 | 14.5397 | 20.3561 | 2.8474 | 65.67 | 65.67 | 190.9 | 9428179 |  |
| YOLO11s baseline (best) | 4 | fp32 | 24.4495 | 24.3784 | 24.7665 | 0.7464 | 40.9 | 163.6 | 287.0 | 9428179 | 21.7 |
| YOLO11s baseline (best) | 4 | fp16 | 15.8998 | 15.2904 | 19.4806 | 1.7653 | 62.89 | 251.58 | 263.4 | 9428179 |  |
| YOLO11s baseline (last) | 1 | fp32 | 13.1655 | 12.4335 | 18.1712 | 2.6225 | 75.96 | 75.96 | 220.1 | 9428179 | 21.7 |
| YOLO11s baseline (last) | 1 | fp16 | 15.2693 | 14.4277 | 20.4501 | 2.811 | 65.49 | 65.49 | 190.9 | 9428179 |  |
| YOLO11s baseline (last) | 4 | fp32 | 24.4823 | 24.3978 | 24.926 | 0.6506 | 40.85 | 163.38 | 287.0 | 9428179 | 21.7 |
| YOLO11s baseline (last) | 4 | fp16 | 16.0808 | 15.3196 | 20.1533 | 2.1874 | 62.19 | 248.74 | 263.4 | 9428179 |  |
| YOLO11s+CA (best) | 1 | fp32 | 13.9067 | 13.2214 | 18.8908 | 2.7649 | 71.91 | 71.91 | 220.2 | 9453827 | 21.7 |
| YOLO11s+CA (best) | 1 | fp16 | 15.7817 | 14.9185 | 21.1476 | 2.9267 | 63.36 | 63.36 | 227.2 | 9453827 |  |
| YOLO11s+CA (best) | 4 | fp32 | 24.6938 | 24.5893 | 25.2099 | 0.9633 | 40.5 | 161.98 | 323.4 | 9453827 | 21.7 |
| YOLO11s+CA (best) | 4 | fp16 | 16.5659 | 15.6017 | 21.4598 | 2.4498 | 60.37 | 241.46 | 299.2 | 9453827 |  |
| YOLO11s+CA (last) | 1 | fp32 | 13.7239 | 13.1086 | 18.6081 | 2.6001 | 72.87 | 72.87 | 256.8 | 9453827 | 21.7 |
| YOLO11s+CA (last) | 1 | fp16 | 15.856 | 15.0277 | 21.2247 | 2.9588 | 63.07 | 63.07 | 227.2 | 9453827 |  |
| YOLO11s+CA (last) | 4 | fp32 | 24.6504 | 24.5535 | 25.0184 | 0.8675 | 40.57 | 162.27 | 287.1 | 9453827 | 21.7 |
| YOLO11s+CA (last) | 4 | fp16 | 16.3876 | 15.6538 | 20.2261 | 1.9855 | 61.02 | 244.09 | 263.5 | 9453827 |  |
| RT-DETR-L (best) | 1 | fp32 | 36.0648 | 35.4196 | 42.2492 | 3.6615 | 27.73 | 27.73 | 321.4 | 32808131 | 109.9 |
| RT-DETR-L (best) | 1 | fp16 | 36.9199 | 36.1938 | 45.3604 | 5.3396 | 27.09 | 27.09 | 254.3 | 32808131 |  |
| RT-DETR-L (best) | 4 | fp32 | 113.4836 | 113.0148 | 117.8194 | 2.568 | 8.81 | 35.25 | 520.2 | 32808131 | 109.9 |
| RT-DETR-L (best) | 4 | fp16 | 61.8729 | 61.1583 | 67.8995 | 3.5828 | 16.16 | 64.65 | 378.9 | 32808131 |  |
| RT-DETR-L (last) | 1 | fp32 | 36.5356 | 35.8537 | 42.4645 | 3.7165 | 27.37 | 27.37 | 321.4 | 32808131 | 109.9 |
| RT-DETR-L (last) | 1 | fp16 | 37.2265 | 36.6296 | 45.738 | 5.1406 | 26.86 | 26.86 | 254.3 | 32808131 |  |
| RT-DETR-L (last) | 4 | fp32 | 113.678 | 113.1716 | 118.442 | 2.6345 | 8.8 | 35.19 | 520.2 | 32808131 | 109.9 |
| RT-DETR-L (last) | 4 | fp16 | 61.9716 | 61.2018 | 68.1892 | 3.6285 | 16.14 | 64.55 | 378.9 | 32808131 |  |

## Unavailable configurations

* none 鈥?every requested configuration ran.

## What is and is not included in the timing

* included: one forward pass of the model on a fixed GPU tensor (the deployment-relevant compute);
* excluded: image decode, letterbox/pre-processing, NMS/post-processing, disk I/O;
* the numbers are therefore *model* latency, not end-to-end pipeline latency;
* E5 uses the same checkpoint-loading path as E0; the SIoU switch affects training loss only and is irrelevant at inference (documented in the protocol lock), so E0/E5 differ only by the CA block;
* batch=4 FP16 for RT-DETR-L is the configuration most likely to exceed the 6 GB card; if it is marked unavailable the reason is recorded verbatim.


## Measurement artefact that must be kept in mind (validated)

The **first** configuration measured in this run — E0 (baseline, best), batch 1, fp32 — reports
28.466 ms, while the **same architecture** measured later in the same run (E0 (baseline, last),
batch 1, fp32) reports 18.463 ms: a 54 % spread for identical compute. This was investigated rather
than assumed:

* re-measuring E0 (baseline, best) in a **fresh process after a 60 s GPU pre-warm**, with the same
  protocol (100 warm-up + 1000 measured iterations × 3 rounds, CUDA events, synchronize), gives
  **18.010 ms** (median 17.425, p95 16.151, 55.53 FPS);
* i.e. the cold-start number is **10.456 ms (37 %) higher** than its steady-state value and matches the
  later last measurement.

**Consequence for reading the table:** the first row of latency_summary.csv is a *cold-start*
measurement (GPU clocks not yet boosted); every other configuration was measured at steady state.
Comparisons between models must therefore use the steady-state rows, and a deployment-relevant
latency figure should be quoted from the pre-warmed measurement recorded in
latency_environment.json → validation_measurement.

The validated steady-state numbers for the paired E0 comparison are:

| configuration | cold-start (as stored) | steady state (validated) |
|---|---:|---:|
| E0 best, b=1 fp32 | 28.466 ms | 18.010 ms |
| E0 last, b=1 fp32 | — | 18.463 ms |

No stored measurement was edited or replaced.
