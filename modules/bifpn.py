# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPNAdd(nn.Module):
    """
    BiFPN-style learnable weighted feature fusion node.

    Fuses multiple same-shape feature maps with learnable non-negative
    weights (softmax-normalized), followed by a 3x3 conv + BN + SiLU.
    Used in the YOLO26 head to replace plain Concat at the P4/P5
    bottom-up fusion points.

    Args:
        channels (int): Number of channels of every input feature map.
        num_inputs (int): Number of inputs to fuse (default 2).
        eps (float): Small value for numerical stability.
    """

    def __init__(self, channels, num_inputs=2, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, inputs):
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        out = sum(w * x for w, x in zip(weights, inputs))
        out = self.conv(out)
        out = self.bn(out)
        out = self.act(out)
        return out
