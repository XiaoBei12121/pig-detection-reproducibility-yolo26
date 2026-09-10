# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn


class CoordAtt(nn.Module):
    """
    Coordinate Attention (CA) module for enhanced spatial position encoding.

    Placed after C2PSA in the YOLO26 backbone: x = C2PSA(x); x = CA(x).
    Height-wise and width-wise pooled features are fused with learnable 1x1
    convolutions to produce position-aware attention weights.

    Args:
        inp (int): Number of input channels.
        reduction (int): Channel reduction ratio for the bottleneck (default 32).
    """

    def __init__(self, inp, reduction=32):
        super().__init__()
        mip = max(8, inp // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w
