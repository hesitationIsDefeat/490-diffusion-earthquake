import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, in_channels, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, in_channels)
        self.beta  = nn.Linear(cond_dim, in_channels)
    def forward(self, x, cond):
        g = self.gamma(cond)[:, :, None, None]
        b = self.beta(cond) [:, :, None, None]
        return x * (1 + g) + b