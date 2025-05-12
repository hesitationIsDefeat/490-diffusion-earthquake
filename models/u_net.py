import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.film import FiLM


class ConditionalUNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 base_channels=64,
                 cond_dim=4,
                 time_emb_dim=128):
        super().__init__()
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Down
        self.down1 = nn.Conv2d(base_channels, base_channels*2, 4, 2, 1)
        self.film1 = FiLM(base_channels*2, cond_dim+time_emb_dim)
        self.down2 = nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1)
        self.film2 = FiLM(base_channels*4, cond_dim+time_emb_dim)

        # Middle
        self.mid = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
        )
        self.film_mid = FiLM(base_channels*4, cond_dim+time_emb_dim)

        # Up
        self.up2 = nn.ConvTranspose2d(base_channels*8, base_channels*2, 4, 2, 1)
        self.film_up2 = FiLM(base_channels*2, cond_dim+time_emb_dim)
        self.up1 = nn.ConvTranspose2d(base_channels*4, base_channels, 4, 2, 1)
        self.film_up1 = FiLM(base_channels, cond_dim+time_emb_dim)

        # Final
        self.final = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t, cond):
        t_emb = sinusoidal_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        cond_emb = torch.cat([cond, t_emb], dim=-1)  # (B, cond_dim + time_emb_dim)

        h = self.init_conv(x)

        d1 = self.down1(h)
        d1 = F.gelu(self.film1(d1, cond_emb))

        d2 = self.down2(d1)
        d2 = F.gelu(self.film2(d2, cond_emb))

        m = self.mid(d2)
        m = F.gelu(self.film_mid(m, cond_emb))

        u2 = self.up2(torch.cat([m, d2], dim=1))
        u2 = F.gelu(self.film_up2(u2, cond_emb))

        u1 = self.up1(torch.cat([u2, d1], dim=1))
        u1 = F.gelu(self.film_up1(u1, cond_emb))

        return self.final(u1)

def sinusoidal_embedding(timesteps, dim):
    device = timesteps.device
    half = dim // 2
    emb = torch.arange(half, device=device).float()
    emb = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * emb / (half - 1))
    args = timesteps[:, None] * emb[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb