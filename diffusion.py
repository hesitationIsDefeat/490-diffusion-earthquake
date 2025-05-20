import torch
import torch.nn.functional as F

class Diffusion:
    def __init__(self, device: torch.device, timesteps: int, beta_start: float, beta_end: float):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_prod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0, device=self.device)

        t = t.to(self.device)

        sqrt_alpha_prod = self.alpha_prod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = (1 - self.alpha_prod[t]).sqrt().view(-1, 1, 1, 1)

        return sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise

    def p_losses(self, model, x0: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x0, device=self.device)
        x_noisy = self.q_sample(x0, t, noise)
        noise_pred = model(x_noisy, t, cond)

        return F.mse_loss(noise_pred, noise)
