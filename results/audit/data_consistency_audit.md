# SCI 投稿前数据一致性审计报告（2026-09-02）

> 目的：为论文定稿 v6 提供可溯源的数据证据链。所有数字均为真实评测输出，不参与任何模型/超参调整。

## 1. 多 seed test 审计（核查：表 3 与逐 seed 评测一致）

- 协议：各 run 的 best.pt（训练时 val 最优 epoch）在官方 test 集（250 张，5436 实例）上统一评测；
- 工具：ultralytics 8.4.135 `YOLO(...).val(split='test', imgsz=640, batch=8)`，与表 1 同口径；
- 脚本：`scripts/audit_multiseed_test.py`；原始输出：`results/multiseed/test_audit_20260902.json`；日志：`logs/audit_multiseed_test.log`。

| model | seed | P | R | mAP50 | mAP50-95 | 权重来源 |
|---|---|---|---|---|---|---|
| E0 | 42 | 0.9638 | 0.9313 | 0.9803 | 0.7747 | results/baseline/E0_baseline/weights/best.pt |
| E0 | 1 | 0.9579 | 0.9336 | 0.9790 | 0.7739 | results/multiseed/E0_seed1/weights/best.pt |
| E0 | 7 | 0.9734 | 0.9291 | 0.9761 | 0.7768 | results/multiseed/E0_seed7/weights/best.pt |
| E0 | 21 | 0.9583 | 0.9249 | 0.9712 | 0.7683 | results/multiseed/E0_seed21/weights/best.pt |
| E0 | 100 | 0.9571 | 0.9305 | 0.9779 | 0.7724 | results/multiseed/E0_seed100/weights/best.pt |
| E1 | 42 | 0.9644 | 0.9367 | 0.9753 | **0.7744** | results/ablation/E1_CA_fair/weights/best.pt（公平对比版，aligned 初始化；早期 E1_CA 为 config 从头训练版，mAP50-95=0.7451，不作为正式口径） |
| E1 | 1 | 0.9700 | 0.9323 | 0.9794 | 0.7774 | results/multiseed/E1_seed1/weights/best.pt |
| E1 | 7 | 0.9735 | 0.9310 | 0.9804 | 0.7790 | results/multiseed/E1_seed7/weights/best.pt |
| E3 | 42 | 0.9627 | 0.9266 | 0.9741 | 0.7755 | results/ablation/E3_SIoU/weights/best.pt |
| E3 | 1 | 0.9640 | 0.9311 | 0.9745 | 0.7748 | results/multiseed/E3_seed1/weights/best.pt |
| E3 | 7 | 0.9694 | 0.9297 | 0.9742 | 0.7728 | results/multiseed/E3_seed7/weights/best.pt |
| E5 | 42 | 0.9698 | 0.9382 | 0.9826 | 0.7806 | results/ablation/E5_CA_SIoU/weights/best.pt |
| E5 | 1 | 0.9671 | 0.9308 | 0.9792 | 0.7725 | results/multiseed/E5_seed1/weights/best.pt |
| E5 | 7 | 0.9635 | 0.9378 | 0.9759 | 0.7759 | results/multiseed/E5_seed7/weights/best.pt |
| E5 | 21 | 0.9707 | 0.9379 | 0.9756 | 0.7770 | results/multiseed/E5_seed21/weights/best.pt |
| E5 | 100 | 0.9675 | 0.9301 | 0.9749 | 0.7746 | results/multiseed/E5_seed100/weights/best.pt |

### 表 3 统计核对（按 4 位原始值，样本标准差）
- E0（n=5）：mean = 0.7732；std = 0.0032；跨度 0.7683–0.7768（0.0085）
- E1（n=3）：mean = 0.7769；std = 0.0023
- E3（n=3）：mean = 0.7744；std = 0.0014
- E5（n=5）：mean = 0.7761；std = 0.0030；跨度 0.7725–0.7806（0.0081）
- E5 − E0 均值差 = +0.0029

### 结论
E0/E3/E5 各 seed 与论文表 3 此前 3 位小数记录一致（差 ≤0.0004，源自四舍五入）；E1 的 seed42 正确口径为 E1_CA_fair（0.7744）。表 3 已按本审计 4 位值更新，全部可溯源。

## 2. final held-out 零泄漏审计（核查：204 张未参与 final 协议开发）

方法：对 7 个图像分区（basename 集合）求交集。原始输出：`results/final_test/audit_overlap_20260902.txt`。

| 分区 | 张数 |
|---|---|
| dev（官方开发池） | 2681 |
| 主实验 train | 2411 |
| 主实验 val | 270 |
| 主实验 test | 250 |
| ft train_ft | 2226 |
| ft val_ft | 251 |
| ft final_test | 204 |

| 交集 | 数量 | 含义 |
|---|---|---|
| final_test ∩ train_ft | **0** | final 协议训练集无 final_test 帧 |
| final_test ∩ val_ft | **0** | final 协议验证集无 final_test 帧 |
| final_test ∩ 主 test(250) | **0** | 与全部开发分析（消融/遮挡/错误/Grad-CAM，均基于主 test）无帧重叠 |
| final_test ∩ 主 train | 186 | 帧源自 dev → 曾属主实验 train（主模型见过同帧） |
| final_test ∩ 主 val | 18 | 帧源自 dev → 曾属主实验 val |
| dev ∩ 主 test | 0 | 结构保证 |

场景构成：final_test = danuma 132 / psota 41 / bergamini 21 / alameer 10；主 test = danuma 250（全部）。

### 结论与论文表述要求
- 对 **final 协议模型（E0_ft/E5_ft）**：final_test 在镜头级与帧级均完全未见（∩ train_ft/val_ft = 0）——评测有效；
- final_test 帧因取自 dev 池而曾属主实验 train/val——**不得**表述为"从未参与任何开发"，必须如实说明"最终评测对象为按 final 协议重训的模型"（v6 第 4.3 节已按此改写）；
- 重叠审计清单随 Supplementary 提供。

## 3. final held-out 评测溯源（锁死声明核对）

- 评测脚本：`scripts/final_test_eval.py`（固化 conf=0.25/IoU=0.5/imgsz=640/batch=1；记录权重 md5 与 git commit）；
- 数据配置：`dataset/pig_ft.yaml`（train_ft/val_ft/final_test）；
- 权重链：`results/final_test/E0_ft/weights/best.pt` ≡ `weights/E0_ft_best.pt`（md5 0162826A15007EC06426F313D24BCB81）；E5 同理（F7B37D7E11485A87A70F2513693ED2B0）；
- E0_ft 评测曾运行 3 次（E0_ft_eval / -2 / -3），三次 predictions.json md5 全部一致（CC4F073E5770F1582509C1C7A22E326C）——为脚本调试期重跑（image_id 映射 bug 修复），结果位级一致、未导致任何基于结果的调整；正式报告：`E0_ft_final_report.md`（git dbdf66b）；
- E5_ft 评测 1 次（E5_ft_eval，md5 0792CB3608EBF31124A9E55A4CB5E900）；正式报告：`E5_ft_final_report.md`；
- 最终对比：`results/final_test/FINAL_TEST_对比报告.md`。

## 4. 复现性补充（固定 seed 重复）
E0/E5 在 seed42 下各重复 2 次（E0_rep2/rep3、E5_rep2/rep3），训练结果与首轮完全一致（工程可复现性，非统计稳定性）。

## 5. 待办（与 v6 checklist 对应）
- [x] 表 3 溯源（本次审计）
- [x] final_test 零泄漏审计（本次审计）
- [x] E5 措辞收紧为 selected candidate（v6 4.2）
- [x] 4.3 节精确化 final_test 来源描述
- [ ] 全文英译、参考文献核对、图 1-2 绘制（投稿稿制作阶段）
