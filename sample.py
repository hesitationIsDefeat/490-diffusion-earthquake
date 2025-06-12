# sample.py

import torch


@torch.no_grad()
def sample(
    model,
    diffusion,
    cond,
    shape,
    sampler_type: str = 'ddpm',
    num_inference_steps: int = None
):
    model.eval()
    x = torch.randn(shape, device=diffusion.device, dtype=torch.float32)

    T = diffusion.timesteps
    inference_steps = T if num_inference_steps is None else min(num_inference_steps, T)
    step_ratio = T // inference_steps
    sample_timesteps = list(range(0, T, step_ratio))[::-1]

    #sample_timesteps = list(range(T - 1, -1, -1))

    if torch.isnan(x).any() or torch.isnan(cond).any():
        print("NaN detected in input!")

    for i, t in enumerate(sample_timesteps):
        tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        eps = model(x, tt, cond)

        beta_t = diffusion.betas[t]
        alpha_t = diffusion.alphas[t]
        alpha_bar_t = diffusion.alpha_prod[t]

        if t > 0:
            t_prev_idx = sample_timesteps[i + 1] if (i + 1) < len(sample_timesteps) else 0
            alpha_bar_t_prev = diffusion.alpha_prod[t_prev_idx]
        else:
            alpha_bar_t_prev = torch.tensor(1.0, device=x.device)

        if sampler_type == 'ddpm':
            coef_eps = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            mean = (1 / torch.sqrt(alpha_t)) * (x - coef_eps * eps)

            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean  # Final step: no noise added

        elif sampler_type == 'ddim':
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            dir_xt = torch.sqrt(1 - alpha_bar_t_prev) * eps
            x = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt

        elif sampler_type == 'ancestral':
            coef_eps = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            mean = (1 / torch.sqrt(alpha_t)) * (x - coef_eps * eps)

            if t > 0:
                sigma_t = torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t)
                noise = torch.randn_like(x)
                x = mean + sigma_t * noise
            else:
                x = mean

        else:
            raise ValueError(
                f"Unknown sampler type: '{sampler_type}'. Choose from ['ddpm', 'ddim', 'ancestral']."
            )

    model.train()
    return x
