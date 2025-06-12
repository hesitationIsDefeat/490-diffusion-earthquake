from torch import nn

from models.self_attention import SelfAttention


class BottleneckAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)  # Reduce H,W by half
        self.attn = SelfAttention(channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x_pooled = self.pool(x)
        x_attn = self.attn(x_pooled)
        return self.upsample(x_attn)