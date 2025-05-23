import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from utils.transform import normalize_conditions, compute_normalization_stats, resize_image_by_height


class SpectrogramDataset(Dataset):
    def __init__(self, data_path: str, spec_dir: str, spec_ext: str,
                 e_id_col: str, e_lat_col: str, e_lon_col: str,
                 depth_col: str, mag_col: str,
                 img_height: int, color_mode: str):
        self.df = pd.read_csv(data_path)
        self.spec_dir = spec_dir
        self.spec_ext = spec_ext
        self.e_id_col = e_id_col
        self.e_lat_col = e_lat_col
        self.e_lon_col = e_lon_col
        self.depth_col = depth_col
        self.mag_col = mag_col
        self.img_height = img_height
        self.color_mode = color_mode

        self.cond_cols = [e_lat_col, e_lon_col, depth_col, mag_col]
        self.cond_stats = compute_normalization_stats(self.df, self.cond_cols)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = f"{os.path.join(self.spec_dir, str(row[self.e_id_col]), self.color_mode, 'raw')}.{self.spec_ext}"
        img = Image.open(img_path).convert('RGB' if self.color_mode == 'color' else 'L')


        resized = resize_image_by_height(img, self.img_height)
        spec = np.array(resized, dtype=np.float32) / 255.0  # normalize to [0,1]

        cond = np.array([
            row[self.e_lat_col],
            row[self.e_lon_col],
            row[self.depth_col],
            row[self.mag_col]
        ], dtype=np.float32)

        cond = normalize_conditions(cond, self.cond_stats)
        # print(cond)

        import torch
        spec_t = torch.from_numpy(spec).permute(2,0,1) if self.color_mode == 'color' \
                 else torch.from_numpy(spec).unsqueeze(0)
        cond_t = torch.from_numpy(cond)

        return spec_t, cond_t
