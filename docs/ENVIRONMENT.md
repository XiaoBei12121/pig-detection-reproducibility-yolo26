# Environment

Environment used for the experiments reported in the manuscript (snapshot files copied verbatim
into `environment/`).

| Item | Value |
|---|---|
| Python | 3.10.21 |
| PyTorch | 2.0.1+cu118 |
| CUDA | 11.8 |
| Ultralytics | 8.4.135 |
| GPU | NVIDIA RTX 4050 Laptop GPU (6 GB) |
| OS | Windows (see `environment/gpu_info.txt` for the exact driver/OS report) |
| Primary training settings | epochs 100 · imgsz 640 · batch 4 · MuSGD · lr0 0.01 · momentum 0.937 · single class `pig` |

Snapshot files:

```
environment/python_version.txt    # python --version
environment/torch_cuda_info.txt   # torch.__version__, torch.version.cuda, device name
environment/gpu_info.txt          # nvidia-smi output
requirements.txt                  # core dependencies (recommended install)
requirements_full.txt             # full `pip freeze` snapshot at submission time
environment.yml                   # conda environment definition
```

Install:

```bash
pip install -r requirements.txt
# CUDA 11.8 PyTorch build used in the paper:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

## Which parts need which environment

| Task | Requirements |
|---|---|
| Recompute statistics / tables / rank stability / effect figures from the shipped result files | Python + numpy (matplotlib for figures). **No GPU, no PyTorch, no ultralytics.** |
| Re-evaluate shipped checkpoints (not redistributed) | PyTorch + ultralytics + `pycocotools` |
| Retrain models | PyTorch (CUDA 11.8 build) + ultralytics + a CUDA-capable GPU |

## Notes on determinism

- Fixed-seed reruns (E0/E5 at seed 42, three runs each) produced bit-identical `predictions.json`
  files → the execution path is deterministic; this is engineering reproducibility and is **not**
  the same as statistical stability across seeds.
- Changing the random seed changes mAP@0.5:0.95 across a span of ≈0.008 (PigDetect E0/E5 5-seed data).
- Absolute FPS is **not** reproducible across sessions on this hardware: repeated measurements varied
  by >10% and even reversed the E0/E5 ordering (see `supplementary/`).
