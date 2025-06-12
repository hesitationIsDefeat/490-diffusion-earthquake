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
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from io import BytesIO
    from PIL import Image
    import numpy as np

    # Use same colormap
    cmap_name = 'gray' if color_mode == 'grayscale' else 'inferno'
    cmap_func = cm.get_cmap(cmap_name)

    # Determine image dimensions
    height, width = image.shape[:2]

    # Create figure exactly sized to the image
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    # Set extent for proper labeling
    extent = (
        float(times[0]), float(times[-1]),
        float(freqs[-1]), float(freqs[0])  # freq flipped vertically
    )

    # Render with same color normalization
    im = ax.imshow(norm_data, aspect='auto', extent=extent, cmap=cmap_func, vmin=0.0, vmax=1.0)

    # Optional: label axes
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    # Optional: remove white borders
    plt.tight_layout(pad=0)

    # Save as image to memory buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Convert to NumPy RGB image
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


import numpy as np
from PIL import Image


def concat_images_with_padding(img_array: np.ndarray, img_path: str, padding: int = 10,
                               pad_color=(255, 255, 255)) -> np.ndarray:
    # Load second image from disk
    img1 = Image.open(img_path).convert("RGB")
    img1_np = np.array(img1)

    # Ensure both images have 3 channels
    if img_array.ndim == 2:  # Grayscale (H, W)
        img2_np = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 1:  # (H, W, 1)
        img2_np = np.repeat(img_array, 3, axis=2)
    else:
        img2_np = img_array

    # Resize the second image to match the height of the first
    h1 = img1_np.shape[0]
    h2 = img2_np.shape[0]

    if h1 != h2:
        scale = h1 / h2
        new_w = int(img1_np.shape[1] * scale)
        img1 = img1.resize((new_w, h1), Image.ANTIALIAS)
        img1_np = np.array(img1)

    # Create the padding
    pad = np.full((h1, padding, 3), pad_color, dtype=np.uint8)

    # Concatenate all
    combined = np.concatenate([img1_np, pad, img2_np], axis=1)

    return combined
