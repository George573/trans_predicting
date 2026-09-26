"""All convolution stages use independent multiscale paths."""

import torch
from torch import nn


class MultiscaleConv1d(nn.Module):
    def __init__(self, channels, paths, stride=1):
        super().__init__()
        self.paths = nn.ModuleList(
            nn.Conv1d(channels, c, k, padding=d * (k - 1) // 2, dilation=d, stride=stride)
            for c, k, d in paths
        )

    def forward(self, x):
        return torch.cat(
            [
                torch.nn.functional.gelu(path(x), approximate="none")
                for path in self.paths
            ],
            dim=1,
        )
