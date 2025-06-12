import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_on_test_set(model, diffusion, test_set, batch_size, num_workers=4):
    model.eval()
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loss_sum = 0
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing")
        for specs, conds in test_pbar:
            specs, conds = specs.to(diffusion.device), conds.to(diffusion.device)
            batch_size = specs.size(0)
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=diffusion.device)

            loss = diffusion.p_losses(model, specs, t, conds)
            test_loss_sum += loss.item()
            test_pbar.set_postfix({"test_loss": loss.item()})

    avg_test_loss = test_loss_sum / len(test_loader)
    return avg_test_loss