import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.film import FiLM


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, cond_dim=4, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU()
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 2),
            nn.BatchNorm2d(base_channels * 2),
        )
        self.film1 = FiLM(base_channels * 2, cond_dim + time_emb_dim)

        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
        )
        self.film2 = FiLM(base_channels * 4, cond_dim + time_emb_dim)

        self.mid = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
        )
        self.film_mid = FiLM(base_channels * 4, cond_dim + time_emb_dim)

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
        )
        self.film_up2 = FiLM(base_channels * 2, cond_dim + time_emb_dim)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
        )
        self.film_up1 = FiLM(base_channels, cond_dim + time_emb_dim)

        self.final = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t, cond):
        t_emb = sinusoidal_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)
        cond_emb = torch.cat([cond, t_emb], dim=-1)

        h = self.init_conv(x)

        d1 = self.film1(self.down1(h), cond_emb)
        d2 = self.film2(self.down2(F.gelu(d1)), cond_emb)
        m = self.film_mid(self.mid(F.gelu(d2)), cond_emb)

        d2 = self._center_crop_to_match(d2, m)
        u2 = self.film_up2(self.up2(torch.cat([m, d2], dim=1)), cond_emb)

        d1 = self._center_crop_to_match(d1, u2)
        u1 = self.film_up1(self.up1(torch.cat([u2, d1], dim=1)), cond_emb)

        out = self.final(F.gelu(u1))

        # Center-crop output to match input size
        _, _, h_target, w_target = x.shape
        _, _, h_out, w_out = out.shape
        dh = (h_out - h_target) // 2
        dw = (w_out - w_target) // 2
        out = out[:, :, dh:dh + h_target, dw:dw + w_target]

        return out

    @staticmethod
    def _center_crop_to_match(source, target):
        _, _, h, w = source.shape
        _, _, ht, wt = target.shape
        dh = (h - ht) // 2
        dw = (w - wt) // 2
        return source[:, :, dh:dh+ht, dw:dw+wt]


def sinusoidal_embedding(timesteps, dim):
    device = timesteps.device
    half = dim // 2
    emb = torch.arange(half, device=device).float()
    emb = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * emb / (half - 1))
    args = timesteps[:, None] * emb[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb
