import torch


@torch.no_grad()
def sample(model, diffusion, cond, shape):
    model.eval()
    x = torch.randn(shape, device=diffusion.device, dtype=torch.float32)

    for t in reversed(range(diffusion.timesteps)):
        tt = torch.full((shape[0],), t, device=diffusion.device, dtype=torch.long)
        eps = model(x, tt, cond)

        # Directly index and move to device if needed
        alpha_prod_t = diffusion.alpha_prod[t].to(x.device)
        alpha_prod_t_prev = (
            diffusion.alpha_prod[t - 1].to(x.device) if t > 0 else torch.tensor(1.0, device=x.device)
        )
        beta_t = diffusion.betas[t].to(x.device)

        # DDPM reverse process (Eq. 11)
        pred_x0 = (x - eps * (1 - alpha_prod_t).sqrt()) / alpha_prod_t.sqrt()

        coeff = (alpha_prod_t_prev.sqrt() * beta_t) / (1 - alpha_prod_t)
        x = pred_x0 + coeff.sqrt() * torch.randn_like(x)

    return x


