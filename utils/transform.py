import numpy as np
from scipy.signal import stft
from skimage.transform import resize
from typing import Tuple
from matplotlib import cm, pyplot as plt
from io import BytesIO
from PIL import Image

def compute_spectrogram(waveform: np.ndarray,
                        samp_fs: float,
                        output_height: int,
                        nperseg: int = 256,
                        noverlap: int = 128,
                        freq_range: Tuple[float, float] = (0, 17),
                        eps: float = 1e-9,
                        dynamic_db_range: float = 60.0,
                        color_mode: str = 'color',
                        with_labels: bool = True) -> np.ndarray:
    # Compute STFT
    f, t, Zxx = stft(waveform, fs=samp_fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    magnitude = np.abs(Zxx)
    db = 20 * np.log10(magnitude + eps)

    # Crop frequency range
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_cropped = f[freq_mask]
    db_cropped = db[freq_mask, :]

    # Normalize
    db_max = np.max(db_cropped)
    db_min = db_max - dynamic_db_range
    db_cropped = np.clip(db_cropped, db_min, db_max)
    norm = (db_cropped - db_min) / (db_max - db_min)

    # Flip only if with_labels is False
    if with_labels:
        final_norm = norm
        f_cropped = f_cropped[::-1]
    else:
        final_norm = np.flipud(norm)

    # Resize image
    orig_height, orig_width = final_norm.shape
    aspect_ratio = orig_width / orig_height
    target_width = int(round(output_height * aspect_ratio))
    resized = resize(final_norm, (output_height, target_width), anti_aliasing=True)

    # Color mapping
    if color_mode == 'grayscale':
        img = (resized * 255).astype(np.uint8)
        cmap_func = cm.get_cmap('gray')
    elif color_mode == 'color':
        cmap_func = cm.get_cmap('inferno')
        img = (cmap_func(resized)[:, :, :3] * 255).astype(np.uint8)
    else:
        raise ValueError(f"Invalid mode '{color_mode}'.")

    if not with_labels:
        return img

    # Add labels using matplotlib
    fig, ax = plt.subplots(figsize=(target_width / 100, output_height / 100), dpi=100)
    extent = (
        float(t[0]), float(t[-1]),
        float(f_cropped[0]), float(f_cropped[-1])
    )
    ax.imshow(resized, aspect='auto', extent=extent, origin='lower', cmap=cmap_func)
    plt.tight_layout()

    # Save figure to numpy array
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB")
    labeled_img = np.array(pil_img)
    buf.close()

    return labeled_img
