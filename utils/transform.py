import numpy as np
import pandas as pd
from scipy.signal import stft
from skimage.transform import resize
from typing import Tuple
from matplotlib import cm, pyplot as plt
from io import BytesIO
from PIL import Image
import torch
import os
import json
from sklearn.model_selection import train_test_split


def compute_spectrogram_raw(
        waveform: np.ndarray,
        samp_fs: float,
        output_height: int,
        output_width: int,  # Add fixed width
        global_db_min: float,  # From dataset stats
        global_db_max: float,  # From dataset stats
        nperseg: int = 256,
        noverlap: int = 128,
        freq_range: Tuple[float, float] = (0, 17),
        color_mode: str = 'color'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # STFT computation
    f, t, Zxx = stft(waveform, fs=samp_fs, nperseg=nperseg, noverlap=noverlap, window='hann')

    # Magnitude to dB with global scaling
    magnitude = np.abs(Zxx)
    db = 20 * np.log10(magnitude + 1e-9)

    # Frequency cropping
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_cropped = f[freq_mask]
    db_cropped = db[freq_mask, :]

    # Global normalization
    db_clipped = np.clip(db_cropped, global_db_min, global_db_max)
    norm = (db_clipped - global_db_min) / (global_db_max - global_db_min)

    # Flip and resize
    flipped_norm = np.flipud(norm)
    resized = resize(flipped_norm, (output_height, output_width),
                     anti_aliasing=True, preserve_range=True)

    # Color processing
    if color_mode == 'grayscale':
        img = (resized * 255).astype(np.uint8)
        img = img[..., np.newaxis]  # (H, W, 1)
    else: # color_mode == 'color':
        cmap_func = cm.get_cmap('inferno')
        img = (cmap_func(resized)[:, :, :3] * 255).astype(np.uint8)

    return img, t, f_cropped[::-1], resized

def compute_spectrogram_labeled(image: np.ndarray,
                               times: np.ndarray,
                               freqs: np.ndarray,
                               norm_data: np.ndarray,
                               color_mode: str = 'color') -> np.ndarray:
    # Setup figure
    cmap_func = cm.get_cmap('gray' if color_mode == 'grayscale' else 'inferno')

    # Add labels using matplotlib
    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    extent = (
        float(times[0]), float(times[-1]),
        float(freqs[-1]), float(freqs[0])
    )
    ax.imshow(norm_data, aspect='auto', extent=extent, cmap=cmap_func)
    plt.tight_layout()

    # Save figure to numpy array
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB")
    labeled_img = np.array(pil_img)
    buf.close()

    return labeled_img


def compute_global_db_stats(csv_path):
    df = pd.read_csv(csv_path)
    all_db = []

    for i, row in df.iloc[:].iterrows():
        waveform = row.iloc[14:].values.astype(np.float32)  # Adjust index
        f, t, Zxx = stft(waveform, fs=100, nperseg=256)
        magnitude = np.abs(Zxx)
        db = 20 * np.log10(magnitude + 1e-9)
        all_db.append(db)

    all_db = np.concatenate(all_db)
    return np.percentile(all_db, 1), np.percentile(all_db, 99)  # 1st and 99th percentiles

def compute_normalization_stats(
        df: pd.DataFrame,
        columns: list[str],
        method: str = "zscore",
        save_path: str = "cond_stats.json"
) -> np.ndarray:
    # Only compute stats from training set and always save/load from the same file
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            stats_list = json.load(f)
        return np.array(stats_list, dtype=np.float32)

    stats_array = []
    for col in columns:
        if method == "zscore":
            stats_array.append([float(df[col].mean()), float(df[col].std())])
        elif method == "minmax":
            stats_array.append([float(df[col].min()), float(df[col].max())])
        else:
            raise ValueError("Unknown method. Use 'zscore' or 'minmax'.")

    # Save as list to make it JSON serializable
    with open(save_path, "w") as f:
        json.dump(stats_array, f, indent=4)

    return np.array(stats_array, dtype=np.float32)

def load_normalization_stats(stats_path: str):
    # Utility to load stats for both training and sampling
    with open(stats_path, "r") as f:
        stats_list = json.load(f)
    return np.array(stats_list, dtype=np.float32)

def normalize_conditions(
        values: np.ndarray,
        stats: np.ndarray,
        method: str = "zscore"
) -> np.ndarray:
    if method == "zscore":
        mean = stats[:, 0]
        std = stats[:, 1]
        return (values - mean) / (std + 1e-8)
    elif method == "minmax":
        min_ = stats[:, 0]
        max_ = stats[:, 1]
        return (values - min_) / (max_ - min_ + 1e-8)
    else:
        raise ValueError("Unknown method")

def prepare_condition_array(conds: list[float], stats_path="cond_stats.json", method="zscore"):
    # Always load stats from file for consistency
    cond_stats = load_normalization_stats(stats_path)
    cond_vals = np.array(conds, dtype=np.float32)
    norm_vals = normalize_conditions(cond_vals, cond_stats, method=method)
    return torch.from_numpy(norm_vals).float()

# def resize_image_by_height(image, target_height):
#     # Determine image type and extract original width/height
#     if isinstance(image, Image.Image):
#         width, height = image.size
#         image_np = np.array(image)
#     elif isinstance(image, np.ndarray):
#         if image.ndim == 2:
#             height, width = image.shape
#         elif image.ndim == 3:
#             height, width, _ = image.shape
#         else:
#             raise ValueError("Unsupported image shape: must be 2D or 3D array.")
#         image_np = image
#     else:
#         raise TypeError("Input must be a PIL.Image or a numpy.ndarray.")
#
#     # Calculate new width based on aspect ratio
#     aspect_ratio = width / height
#     new_width = int(round(target_height * aspect_ratio))
#
#     # Resize with skimage
#     resized_image = resize(image_np, (target_height, new_width), anti_aliasing=True)
#
#     return resized_image

def create_data_splits(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0)

    # Get indices for all data points
    indices = list(range(len(dataset)))

    # First split: train and temp (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed
    )

    # Second split: val and test from temp
    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_size,
        random_state=seed
    )

    return train_idx, val_idx, test_idx