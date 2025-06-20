# train.py

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from test import evaluate_on_test_set
from utils.transform import create_data_splits, save_visual_sample, save_test_visuals


def train(
    model,
    diffusion,
    dataset,
    batch_size: int,
    epochs: int,
    lr: float,
    model_save_path: str,
    # TODO add patience: int
    num_workers: int = 4,
    min_delta: float = 0.0001,
):
    patience = epochs / 10
    train_idx, val_idx, test_idx = create_data_splits(dataset)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Save test indices for later use
    test_set = Subset(dataset, test_idx)
    torch.save(test_idx, model_save_path.replace('.pt', '_test_indices.pt'))

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = model_save_path.replace(".pt", "_best.pt")

    fixed_specs, fixed_conds = next(iter(val_loader))
    fixed_specs, fixed_conds = fixed_specs[0:1].to(diffusion.device), fixed_conds[0:1].to(diffusion.device)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

        if (epoch + 1) % 5 == 0:
            save_visual_sample(model, fixed_specs, fixed_conds, epoch + 1, "data/output/visual_logs/train")

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

    print("Evaluating best model on test set...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss = evaluate_on_test_set(model, diffusion, test_set, batch_size, num_workers)
    save_test_visuals(model, diffusion, test_set, save_dir="data/output/visual_logs/test", num_samples=20, color_mode="color")
    print(f"Test Loss: {test_loss:.4f}")

    # Save test results
    test_results = {
        'test_loss': test_loss,
        'best_val_loss': best_val_loss,
        'test_indices': test_idx
    }
    torch.save(test_results, model_save_path.replace('.pt', '_test_results.pt'))

    return test_results
