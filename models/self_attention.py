import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(in_channels)

        # QKV projection
        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.proj_out = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, HW, C]

        x_norm = self.norm(x_)  # Normalize before attention
        qkv = self.qkv(x_norm)  # [B, HW, 3C]
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Reshape for multi-head attention
        q = q.reshape(B, H * W, self.num_heads, C // self.num_heads).transpose(1, 2)  # [B, heads, HW, C//heads]
        k = k.reshape(B, H * W, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.reshape(B, H * W, self.num_heads, C // self.num_heads).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.softmax((q @ k.transpose(-2, -1)) / (C ** 0.5), dim=-1)
        attn_output = attn_weights @ v  # [B, heads, HW, C//heads]

        out = attn_output.transpose(1, 2).reshape(B, H * W, C)
        out = self.proj_out(out)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

        return x + out  # Residual connection
