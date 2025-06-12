import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),  # 8 groups
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels)
        )

    def forward(self, x):
        return F.gelu(x + self.block(x))