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
                           extension: str) -> None:
    filename = f"{output_dir}/{e_id}.{extension}"
    imageio.imwrite(filename, img)