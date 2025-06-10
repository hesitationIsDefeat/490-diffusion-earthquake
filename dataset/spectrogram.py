import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

from utils.transform import normalize_conditions, load_normalization_stats


class SpectrogramDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        spec_dir: str,
        spec_ext: str,
        e_id_col: str,
        e_lat_col: str,
        e_lon_col: str,
        depth_col: str,
        mag_col: str,
        img_height: int,
        color_mode: str,
        cond_stats_path: str = "cond_stats.json",
        apply_augmentations: bool = True,
    ):
        # Load CSV metadata
        self.df = pd.read_csv(data_path)
        self.spec_dir = spec_dir
        self.spec_ext = spec_ext
        self.color_mode = color_mode
        self.img_height = img_height

        self.e_id_col = e_id_col
        self.cond_cols = [e_lat_col, e_lon_col, depth_col, mag_col]
        self.cond_stats = load_normalization_stats(cond_stats_path)

        self.apply_augmentations = apply_augmentations
        self.transform = self._build_transform() if apply_augmentations else None

    def _build_transform(self):
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(
            self.spec_dir,
            str(row[self.e_id_col]),
            self.color_mode,
            f"raw.{self.spec_ext}"
        )

        img = Image.open(img_path).convert('RGB' if self.color_mode == 'color' else 'L')
        if self.transform:
            img = self.transform(img)

        # Normalize image to [-1, 1]
        spec = np.asarray(img, dtype=np.float32) / 255.0
        spec = spec * 2.0 - 1.0

        # Normalize conditioning variables
        cond = np.array([row[col] for col in self.cond_cols], dtype=np.float32)
        cond = normalize_conditions(cond, self.cond_stats)

        # Convert to PyTorch tensors
        if self.color_mode == 'color':
            spec_tensor = torch.from_numpy(spec).permute(2, 0, 1)  # HWC -> CHW
        else:
            spec_tensor = torch.from_numpy(spec).unsqueeze(0)  # HW -> CHW

        cond_tensor = torch.from_numpy(cond)

        return spec_tensor.float(), cond_tensor.float()
