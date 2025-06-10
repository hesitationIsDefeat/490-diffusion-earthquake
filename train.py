# train.py

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split


def train(
    model,
    diffusion,
    dataloader,
    epochs,
    lr,
    model_save_path,
    patience: int = 10,
    min_delta: float = 0.001,
):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Split into train and validation datasets
    total_size = len(dataloader.dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataloader.dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=dataloader.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers,
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = model_save_path.replace(".pt", "_best.pt")

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} (Train)")

        for specs, conds in pbar:
            specs, conds = specs.to(diffusion.device), conds.to(diffusion.device)
            batch_size = specs.size(0)
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=diffusion.device)

            loss = diffusion.p_losses(model, specs, t, conds)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss_sum += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss_sum / len(train_loader)
        print(f"[Epoch {epoch + 1}] Avg Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} (Validation)")
            for specs, conds in val_pbar:
                specs, conds = specs.to(diffusion.device), conds.to(diffusion.device)
                batch_size = specs.size(0)
                t = torch.randint(0, diffusion.timesteps, (batch_size,), device=diffusion.device)

                loss = diffusion.p_losses(model, specs, t, conds)
                val_loss_sum += loss.item()
                val_pbar.set_postfix({"val_loss": loss.item()})

        avg_val_loss = val_loss_sum / len(val_loader)
        print(f"[Epoch {epoch + 1}] Avg Val Loss: {avg_val_loss:.4f}")

        scheduler.step()

        # Early Stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path} | Val Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s). Patience: {patience}")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered. Best Val Loss: {best_val_loss:.4f}")
                break

    torch.save(model.state_dict(), model_save_path)
    print(f"Final model saved to {model_save_path}")
