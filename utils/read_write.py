import os

import imageio
import pandas as pd
import numpy as np
from typing import Tuple

def read_waveform_data(data_path: str,
                       last_md_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(data_path)
    time_col_start = df.columns.get_loc(last_md_col) + 1
    waveform_data = df.iloc[:, time_col_start:].values.astype(np.float32)
    return df, waveform_data

def save_spectrogram_image(img: np.ndarray,
                           e_id: str,
                           output_dir: str,
                           extension: str,
                           file_name: str,
                           color_mode: str) -> None:
    dir_name = f"{os.path.join(output_dir, str(e_id), color_mode)}"
    os.makedirs(dir_name, exist_ok=True)

    if color_mode == 'grayscale' and img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(2)


    filename = f"{os.path.join(dir_name, file_name)}.{extension}"
    imageio.imwrite(filename, img)