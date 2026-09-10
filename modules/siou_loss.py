# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# SIoU bounding-box regression loss (E3 / E5 / E6 in the manuscript).
#
# Provenance: this function is the SIoU implementation that was patched into
# `ultralytics/utils/loss.py` for the experiments. It is reproduced here verbatim so that the
# modified regression path is fully auditable; the surrounding training framework is the official
# Ultralytics implementation.
#
# Integration (as used in the experiments):
#   `ultralytics/utils/loss.py`, near the module imports:
#
#       import os
#       USE_SIOU = os.environ.get("YOLO_SIOU", "0") == "1"   # E3/E5/E6 switch
#
#   and inside `BboxLoss.forward`, the IoU term becomes:
#
#       if USE_SIOU:  # SIoU box regression loss
#           loss_iou = (siou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask]).unsqueeze(-1) * weight).sum() / target_scores_sum
#       else:
#           ... native YOLO26 regression loss ...
#
# Enabling SIoU changes only this regression path: classification loss, label assignment and the
# remaining training logic are untouched, and no trainable parameters are added.

import math

import torch


def siou_loss(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7, theta: float = 4.0) -> torch.Tensor:
    """SIoU (SCYLLA-IoU) loss per the original paper (Löw et al. 2022), xyxy boxes.

    Combines IoU cost, angle cost, distance cost and shape cost:
        L = 1 - IoU + (Delta + Omega) / 2
    Returns a per-sample loss tensor of the same leading shape as the inputs.
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.unbind(dim=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.unbind(dim=-1)

    # IoU cost
    inter_w = torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)
    inter_h = torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    inter = inter_w.clamp(min=0) * inter_h.clamp(min=0)
    union = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter + eps
    iou = inter / union

    # smallest enclosing box
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

    # center offsets
    s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
    s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
    sigma = torch.sqrt(s_cw**2 + s_ch**2 + eps)

    # angle cost
    sin_alpha = torch.abs(s_ch) / sigma
    sin_beta = torch.abs(s_cw) / sigma
    sin_alpha = torch.where(sin_alpha > 0.5, sin_beta, sin_alpha)  # use beta when alpha > pi/4
    angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)  # Lambda

    # distance cost
    gamma = 2.0 - angle_cost
    rho_x = (s_cw / (cw + eps)) ** 2
    rho_y = (s_ch / (ch + eps)) ** 2
    distance_cost = (1.0 - torch.exp(-gamma * rho_x)) + (1.0 - torch.exp(-gamma * rho_y))

    # shape cost
    w1 = b1_x2 - b1_x1
    h1 = b1_y2 - b1_y1
    w2 = b2_x2 - b2_x1
    h2 = b2_y2 - b2_y1
    omega_w = torch.abs(w1 - w2) / (torch.max(w1, w2) + eps)
    omega_h = torch.abs(h1 - h2) / (torch.max(h1, h2) + eps)
    shape_cost = (1.0 - torch.exp(-omega_w)) ** theta + (1.0 - torch.exp(-omega_h)) ** theta

    return 1.0 - iou + (distance_cost + shape_cost) * 0.5
