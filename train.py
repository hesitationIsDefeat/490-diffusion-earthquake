import torch
from tqdm import tqdm

def train(model, diffusion, dataloader, epochs, lr):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5) # alter lr if loss doesn't change much
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for specs, conds in pbar:
            specs, conds = specs.to(diffusion.device), conds.to(diffusion.device)
            batch_size = specs.size(0)
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=diffusion.device)

            loss = diffusion.p_losses(model, specs, t, conds)
            optimizer.zero_grad()
            loss.backward()

            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

        scheduler.step()
