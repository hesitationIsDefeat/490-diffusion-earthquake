import torch
import torch.nn.functional as F
import math

class Diffusion:
    def __init__(self, device: torch.device, timesteps: int,
                 beta_start: float, beta_end: float,
                 schedule_type: str = 'linear'):

        self.timesteps = timesteps
        self.device = device

        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        elif schedule_type == 'cosine':
            s = 0.008
            t = torch.linspace(0, timesteps, steps=timesteps + 1, device=device)
            alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = 1 - alphas
            self.betas = torch.clip(self.betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.alphas = 1.0 - self.betas
        self.alpha_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_prod = torch.sqrt(self.alpha_prod)
        self.sqrt_one_minus_alpha_prod = torch.sqrt(1 - self.alpha_prod)

    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size):
        b = t.shape[0]
        return a.gather(0, t).reshape(b, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_prod = self.extract(self.sqrt_alpha_prod, t, x0.shape)
        sqrt_one_minus_alpha_prod = self.extract(self.sqrt_one_minus_alpha_prod, t, x0.shape)

        return sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise

    def p_losses(self, model, x0: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, loss_type='huber') -> torch.Tensor:
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        noise_pred = model(x_noisy, t, cond)

        if loss_type == 'l1':
            return F.l1_loss(noise_pred, noise)
        elif loss_type == 'huber':
            return F.smooth_l1_loss(noise_pred, noise)
        else:
            return F.mse_loss(noise_pred, noise)

