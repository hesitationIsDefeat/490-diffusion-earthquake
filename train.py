import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train(model, diffusion, dataloader, epochs, lr):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5) # alter lr if loss doesn't change much
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for specs, conds in pbar:
            specs, conds = specs.to(diffusion.device), conds.to(diffusion.device)
            batch_size = specs.size(0)
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=diffusion.device)

            loss = diffusion.p_losses(model, specs, t, conds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_epoch_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_epoch_loss)
