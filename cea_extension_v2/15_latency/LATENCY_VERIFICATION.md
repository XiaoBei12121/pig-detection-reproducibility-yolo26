# A5 — latency benchmark verification

Every statistic in `latency_summary.csv` is recomputed here from the 120 000 per-iteration timings in `latency_raw.csv`, using the benchmark's own definitions (p95 = nearest rank `sorted(times)[int(0.95*n)-1]`, sample SD with ddof=1, FPS = 1000/mean, throughput = batch·1000/mean).

| check | expected | found | verdict |
|---|---|---|---|
| base architectures covered | 5 | 5: E0 (YOLO26s baseline), E5 (YOLO26s+CA+SIoU), RT-DETR-L, YOLO11s baseline, YOLO11s+CA | PASS |
| loader families (architecture column) | YOLO + RTDETR | RTDETR, YOLO | PASS |
| model x checkpoint-rule configurations | 10 | 10 | PASS |
| batch x precision grid | {(1,fp32),(1,fp16),(4,fp32),(4,fp16)} | [('1', 'fp16'), ('1', 'fp32'), ('4', 'fp16'), ('4', 'fp32')] | PASS |
| configurations | 40 | 40 | PASS |
| imgsz | 640 | ['640'] | PASS |
| checkpoints used exist | all present | 40/40 present | PASS |
| raw rows | 120000 | 120000 | PASS |
| rounds per configuration | {1, 2, 3} | frozenset({'3', '2', '1'}) | PASS |
| measured iterations per round | 1000 | [1000.0] | PASS |
| warm-up iterations (environment) | 100 | 100 | PASS |
| measured iterations (environment) | 1000 | 1000 | PASS |
| rounds (environment) | 3 | 3 | PASS |
| all reported statistics recomputed from raw | 0 discrepancies | 0 discrepancies | PASS |
| params identical across batch/precision/rule for one architecture | 1 value per architecture | {'E0 (YOLO26s baseline)': 1, 'E5 (YOLO26s+CA+SIoU)': 1, 'YOLO11s baseline': 1, 'YOLO11s+CA': 1, 'RT-DETR-L': 1} | PASS |
| GFLOPs: one value per architecture where reported (fp32 loads only) | consistent per architecture | {'E0 (YOLO26s baseline)': ['', '22.7'], 'E5 (YOLO26s+CA+SIoU)': ['', '22.7'], 'YOLO11s baseline': ['', '21.7'], 'YOLO11s+CA': ['', '21.7'], 'RT-DETR-L': ['', '109.9']} | PASS |
| environment block fields | all present | missing=[] | PASS |
| fairness: no compile / TensorRT / ONNX / CUDA graphs | all four excluded | no torch.compile | no TensorRT/ONNX | no CUDA graphs | each configuration loads its own weights once | one fixed GPU tensor reused for every iteration | no other GPU workload running during the benchmark | PASS |
| timing method | CUDA events + synchronize | torch.cuda.Event around one forward pass, torch.cuda.synchronize() each iteration | PASS |
| no disk I/O or image decoding in the timed region | excluded | ['image decode', 'letterbox/pre-processing', 'NMS post-processing', 'disk I/O', 'python-level result formatting beyond the forward call'] | PASS |
| no exported models / traces / prediction images in 15_latency | none | none | PASS |
| first configuration flagged as cold-start | documented in latency_environment.json | YES - confirmed | PASS |

FAIL: **0**; statistic discrepancies: **0**.

## Round-to-round spread (the 3-round requirement, recomputed)

| model | batch | precision | round1 (ms) | round2 (ms) | round3 (ms) | spread (ms) | spread (%) |
|---|---|---|---:|---:|---:|---:|---:|
| E0 (YOLO26s baseline, best) | 1 | fp32 | 32.8511 | 27.6292 | 24.9169 | 7.9343 | 27.87 |
| E0 (YOLO26s baseline, best) | 1 | fp16 | 36.1161 | 37.6568 | 34.0226 | 3.6342 | 10.11 |
| E0 (YOLO26s baseline, best) | 4 | fp32 | 27.7023 | 28.1346 | 29.041 | 1.3387 | 4.73 |
| E0 (YOLO26s baseline, best) | 4 | fp16 | 21.2932 | 21.3773 | 22.7045 | 1.4113 | 6.48 |
| E0 (YOLO26s baseline, last) | 1 | fp32 | 18.5397 | 18.3346 | 18.5133 | 0.2051 | 1.11 |
| E0 (YOLO26s baseline, last) | 1 | fp16 | 21.0109 | 22.4759 | 21.2475 | 1.465 | 6.79 |
| E0 (YOLO26s baseline, last) | 4 | fp32 | 27.2885 | 27.5428 | 27.4795 | 0.2542 | 0.93 |
| E0 (YOLO26s baseline, last) | 4 | fp16 | 20.8414 | 20.526 | 20.8459 | 0.3199 | 1.54 |
| E5 (YOLO26s+CA+SIoU, best) | 1 | fp32 | 19.6622 | 18.6975 | 18.7209 | 0.9647 | 5.07 |
| E5 (YOLO26s+CA+SIoU, best) | 1 | fp16 | 22.7386 | 22.1557 | 22.4625 | 0.5829 | 2.6 |
| E5 (YOLO26s+CA+SIoU, best) | 4 | fp32 | 27.4937 | 27.6882 | 27.7297 | 0.236 | 0.85 |
| E5 (YOLO26s+CA+SIoU, best) | 4 | fp16 | 20.6175 | 21.5059 | 21.3508 | 0.8885 | 4.2 |
| E5 (YOLO26s+CA+SIoU, last) | 1 | fp32 | 19.2045 | 19.0591 | 18.2917 | 0.9128 | 4.84 |
| E5 (YOLO26s+CA+SIoU, last) | 1 | fp16 | 22.2726 | 22.56 | 21.7858 | 0.7742 | 3.49 |
| E5 (YOLO26s+CA+SIoU, last) | 4 | fp32 | 27.8171 | 27.5949 | 27.8135 | 0.2221 | 0.8 |
| E5 (YOLO26s+CA+SIoU, last) | 4 | fp16 | 20.9652 | 21.2499 | 21.7104 | 0.7453 | 3.5 |
| YOLO11s baseline (best) | 1 | fp32 | 12.9794 | 13.0704 | 12.9225 | 0.1479 | 1.14 |
| YOLO11s baseline (best) | 1 | fp16 | 15.4997 | 14.9056 | 15.2787 | 0.5942 | 3.9 |
| YOLO11s baseline (best) | 4 | fp32 | 24.4192 | 24.4134 | 24.5159 | 0.1025 | 0.42 |
| YOLO11s baseline (best) | 4 | fp16 | 15.9782 | 15.8067 | 15.9146 | 0.1715 | 1.08 |
| YOLO11s baseline (last) | 1 | fp32 | 13.3819 | 13.145 | 12.9697 | 0.4122 | 3.13 |
| YOLO11s baseline (last) | 1 | fp16 | 15.3264 | 15.2681 | 15.2134 | 0.113 | 0.74 |
| YOLO11s baseline (last) | 4 | fp32 | 24.4019 | 24.5386 | 24.5063 | 0.1367 | 0.56 |
| YOLO11s baseline (last) | 4 | fp16 | 15.8642 | 16.3826 | 15.9956 | 0.5184 | 3.22 |
| YOLO11s+CA (best) | 1 | fp32 | 13.7104 | 13.9113 | 14.0986 | 0.3882 | 2.79 |
| YOLO11s+CA (best) | 1 | fp16 | 15.7991 | 15.6178 | 15.9284 | 0.3106 | 1.97 |
| YOLO11s+CA (best) | 4 | fp32 | 24.6027 | 24.6592 | 24.8195 | 0.2168 | 0.88 |
| YOLO11s+CA (best) | 4 | fp16 | 16.2097 | 16.3171 | 17.1708 | 0.9611 | 5.8 |
| YOLO11s+CA (last) | 1 | fp32 | 13.7361 | 13.7033 | 13.7323 | 0.0328 | 0.24 |
| YOLO11s+CA (last) | 1 | fp16 | 16.2387 | 15.6053 | 15.724 | 0.6335 | 4.0 |
| YOLO11s+CA (last) | 4 | fp32 | 24.6507 | 24.5868 | 24.7138 | 0.127 | 0.52 |
| YOLO11s+CA (last) | 4 | fp16 | 16.5639 | 16.314 | 16.2847 | 0.2792 | 1.7 |
| RT-DETR-L (best) | 1 | fp32 | 36.1024 | 36.1968 | 35.8952 | 0.3016 | 0.84 |
| RT-DETR-L (best) | 1 | fp16 | 37.0632 | 37.4376 | 36.2587 | 1.1789 | 3.19 |
| RT-DETR-L (best) | 4 | fp32 | 112.7781 | 113.644 | 114.0288 | 1.2507 | 1.1 |
| RT-DETR-L (best) | 4 | fp16 | 61.6689 | 62.03 | 61.9197 | 0.3611 | 0.58 |
| RT-DETR-L (last) | 1 | fp32 | 36.7088 | 36.2649 | 36.6332 | 0.4439 | 1.21 |
| RT-DETR-L (last) | 1 | fp16 | 37.8716 | 36.6141 | 37.1938 | 1.2575 | 3.38 |
| RT-DETR-L (last) | 4 | fp32 | 112.9174 | 113.9808 | 114.1359 | 1.2186 | 1.07 |
| RT-DETR-L (last) | 4 | fp16 | 61.6906 | 61.9094 | 62.3149 | 0.6244 | 1.01 |

## Notes

* GFLOPs is a model property computed on the fp32 load (`get_flops`); the fp16 rows therefore carry no GFLOPs by design, and the per-architecture table above shows the value is consistent wherever it is reported.
* The first configuration of the run is a documented cold-start outlier (`validation_measurement` in `latency_environment.json`); it is left as measured and flagged rather than overwritten.
* No training, no exported model, no profiler trace and no prediction image was produced by this benchmark.