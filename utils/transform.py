import numpy as np
from scipy.signal import stft
from skimage.transform import resize
from typing import Tuple
from matplotlib import cm

def compute_spectrogram(waveform: np.ndarray,
                        samp_fs: float,
                        output_size: Tuple[int, int],
                        nperseg: int = 256,
                        noverlap: int = 128,
                        freq_range: Tuple[float, float] = (2, 15),
                        eps: float = 1e-9,
                        dynamic_db_range: float = 60.0,
                        color_mode: str = 'color') -> np.ndarray:
    f, t, Zxx = stft(waveform, fs=samp_fs, nperseg=nperseg, noverlap=noverlap, window='hann')

    magnitude = np.abs(Zxx)
    db = 20 * np.log10(magnitude + eps)

    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    db_cropped = db[freq_mask, :]

    db_max = np.max(db_cropped)
    db_min = db_max - dynamic_db_range

    db_cropped = np.clip(db_cropped, db_min, db_max)
    norm = (db_cropped - db_min) / (db_max - db_min)

    resized = resize(norm, output_size, anti_aliasing=True)

    if color_mode == 'grayscale':
        img_uint8 = (resized * 255).astype(np.uint8)
        return img_uint8

    elif color_mode == 'color':
        cmap = cm.get_cmap('inferno')  # 'plasma', 'inferno', 'viridis'
        colored = cmap(resized)[:, :, :3]
        img_uint8 = (colored * 255).astype(np.uint8)
        return img_uint8

    else:
        raise ValueError(f"Invalid mode '{color_mode}'. Choose a valid type.")


