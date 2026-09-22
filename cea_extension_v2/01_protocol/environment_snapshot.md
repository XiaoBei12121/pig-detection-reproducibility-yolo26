# Environment snapshot (phase 0)

```
python_executable: <OLD_USER_HOME>\miniconda3\envs\pig_exp\python.exe
python_version: 3.10.21 | packaged by Anaconda, Inc. | (main, Aug 27 2026, 14:35:39) [MSC v.1942 64 bit (AMD64)]
platform: Windows-10-10.0.26200-SP0
conda_default_env: None
conda_prefix: None

$ python -c <probe>
import sys, torch, torchvision, ultralytics
print('sys.executable:', sys.executable)
print('torch:', torch.__version__)
print('torch.version.cuda:', torch.version.cuda)
print('cudnn:', torch.backends.cudnn.version())
print('torchvision:', torchvision.__version__)
print('ultralytics:', ultralytics.__version__)
print('cuda_available:', torch.cuda.is_available())
print('device_count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device_name:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
    print('total_mem_GB:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2))
print('cudnn_benchmark:', torch.backends.cudnn.benchmark, 'deterministic:', torch.backends.cudnn.deterministic)
print('matmul_allow_tf32:', torch.backends.cuda.matmul.allow_tf32)

sys.executable: <OLD_USER_HOME>\miniconda3\envs\pig_exp\python.exe
torch: 2.0.1+cu118
torch.version.cuda: 11.8
cudnn: 8700
torchvision: 0.15.2+cu118
ultralytics: 8.4.135
cuda_available: True
device_count: 1
device_name: NVIDIA GeForce RTX 4050 Laptop GPU
capability: (8, 9)
total_mem_GB: 6.0
cudnn_benchmark: False deterministic: False
matmul_allow_tf32: False
```

```
$ nvidia-smi
Wed Sep 16 01:20:30 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 592.82                 Driver Version: 592.82         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4050 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   50C    P8              2W /   50W |       0MiB /   6141MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           31624    C+G   ...2p2nqsd0c76g0\app\ChatGPT.exe      N/A      |
+-----------------------------------------------------------------------------------------+

$ nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,compute_cap --format=csv
name, memory.total [MiB], memory.used [MiB], driver_version, compute_cap
NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB, 0 MiB, 592.82, 8.9
```

RT-DETR import check:
```
$ python -c "from ultralytics import YOLO; from ultralytics import RTDETR"
YOLO import: OK
RTDETR import: OK -> <class 'ultralytics.models.rtdetr.model.RTDETR'>
```

## Interpreter/executable hazard (recorded 2026-09-16)

yolo on PATH is **not** the locked environment:

| executable | ultralytics |
|---|---|
| <OLD_USER_HOME>\miniconda3\Scripts\yolo.exe (first PATH hit, base env) | 8.4.133 |
| <OLD_USER_HOME>\AppData\Local\Programs\Python\Python312\Scripts\yolo.exe (system Python) | different build; invoking it reset the shared Ultralytics settings file |
| **<OLD_USER_HOME>\miniconda3\envs\pig_exp\Scripts\yolo.exe** (locked env, used by every experiment) | **8.4.135** |

Therefore every script in this extension invokes the pig_exp executable by **absolute path** and never
a bare yolo. All other tooling uses <OLD_USER_HOME>\miniconda3\envs\pig_exp\python.exe.
