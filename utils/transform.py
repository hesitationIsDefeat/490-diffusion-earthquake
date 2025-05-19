import numpy as np
from scipy.signal import stft
from skimage.transform import resize
from typing import Tuple
from matplotlib import cm, pyplot as plt
from io import BytesIO
from PIL import Image

def compute_spectrogram_raw(waveform: np.ndarray,
                        samp_fs: float,
                        output_height: int,
                        nperseg: int = 256,
                        noverlap: int = 128,
                        freq_range: Tuple[float, float] = (0, 17),
                        eps: float = 1e-9,
                        dynamic_db_range: float = 60.0,
                        color_mode: str = 'color') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    # Flip vertically to match image coordinate system
    flipped_norm = np.flipud(norm)

    # Resize to maintain aspect ratio
    orig_height, orig_width = flipped_norm.shape
    aspect_ratio = orig_width / orig_height
    target_width = int(round(output_height * aspect_ratio))
    resized = resize(flipped_norm, (output_height, target_width), anti_aliasing=True)

    # Apply colormap
    if color_mode == 'grayscale':
        img = (resized * 255).astype(np.uint8)
    elif color_mode == 'color':
        cmap_func = cm.get_cmap('inferno')
        img = (cmap_func(resized)[:, :, :3] * 255).astype(np.uint8)
    else:
        raise ValueError(f"Invalid color_mode '{color_mode}'.")

    return img, t, f_cropped[::-1], resized  # Also return t, f for labeling; inverting freq for bottom to top labels

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
