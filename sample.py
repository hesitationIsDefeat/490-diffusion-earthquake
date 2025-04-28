import torch


@torch.no_grad()
def sample(model, diffusion, cond, shape):
    model.eval()
    x = torch.randn(shape, device=diffusion.device)
    for t in reversed(range(diffusion.timesteps)):
        tt = torch.full((shape[0],), t, device=diffusion.device, dtype=torch.long)
        eps = model(x, tt, cond)
        beta = diffusion.betas[t]
        alpha = diffusion.alphas[t]
        alpha_prod = diffusion.alpha_prod[t]
        x = (1/alpha**0.5)*( x - ((1-alpha)/ (1-alpha_prod)**0.5)*eps )
        if t > 0:
            x += beta**0.5 * torch.randn_like(x)
    return x.clamp(0,1)
