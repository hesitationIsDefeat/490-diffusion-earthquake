import os
import argparse
import json
from tqdm import tqdm

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset.spectrogram import SpectrogramDataset
from diffusion import Diffusion
from models.u_net import ConditionalUNet
from sample import sample
from train import train
from utils.read_write import read_waveform_data, save_spectrogram_image
from utils.transform import compute_spectrogram_raw, compute_spectrogram_labeled, prepare_condition_array

TESTING_LIMIT: int = 5

def main_preprocess(data_path: str, output_path: str, spec_height: int, spec_ext: str,
                    e_id_col: str, samp_fs_col: str, last_md_col: str,
                    stft_color_mode: str, is_testing: bool):
    df, waveforms = read_waveform_data(data_path=data_path, last_md_col=last_md_col)
    os.makedirs(output_path, exist_ok=True)
    iteration_amount: int = TESTING_LIMIT if is_testing else len(df)
    for idx, row in tqdm(df.head(iteration_amount).iterrows(), total=iteration_amount, desc="Preprocess Progress"):
        e_id = row[e_id_col]
        fs = row[samp_fs_col]
        img, times, freq, norm = compute_spectrogram_raw(
            waveform=waveforms[idx],
            samp_fs=fs,
            color_mode=stft_color_mode,
            output_height=spec_height
        )
        labeled_img = compute_spectrogram_labeled(img, times, freq, norm, stft_color_mode)
        save_spectrogram_image(img=img, e_id=e_id, output_dir=output_path, extension=spec_ext, file_name='raw', color_mode=stft_color_mode)
        save_spectrogram_image(img=labeled_img, e_id=e_id, output_dir=output_path, extension=spec_ext, file_name='labeled', color_mode=stft_color_mode)



def main_train(data_path: str, spec_dir: str, spec_height: int, color_mode: str, spec_ext: str,
               batch_size: int, epochs: int, lr: float,
               model_save_path: str, device_name: str,
               e_id_col: str, lat_col: str, lon_col: str, dep_col: str, mag_col: str,
               diff_ts: int, diff_beta_start: float, diff_beta_end: float):
    dataset = SpectrogramDataset(
        data_path=data_path,
        spec_dir=spec_dir,
        spec_ext=spec_ext,
        img_height=spec_height,
        color_mode=color_mode,
        e_id_col=e_id_col,
        e_lat_col=lat_col,
        e_lon_col=lon_col,
        depth_col=dep_col,
        mag_col=mag_col
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=4)

    device = torch.device(device_name)

    model = ConditionalUNet(in_channels=(3 if color_mode == 'color' else 1)).to(device)
    diffusion = Diffusion(device=device, timesteps=diff_ts, beta_start=diff_beta_start, beta_end=diff_beta_end)

    train(model, diffusion, dataloader, epochs=epochs, lr=lr)

    torch.save(model.state_dict(), model_save_path)


def main_sample(data_path: str, model_path: str, spec_height: int, aspect_ratio_json_path: str, color_mode: str, spec_ext: str,
                num_samples: int, sample_save_dir: str, device_name: str,
                e_id_col: str, lat_col: str, lon_col: str, dep_col: str, mag_col: str,
                diff_ts: int, diff_beta_start: float, diff_beta_end: float):

    os.makedirs(sample_save_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    device = torch.device(device_name)

    model = ConditionalUNet(in_channels=(3 if color_mode == 'color' else 1)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    diffusion = Diffusion(device=device, timesteps=diff_ts, beta_start=diff_beta_start, beta_end=diff_beta_end)

    if not os.path.exists(aspect_ratio_json_path):
        raise FileNotFoundError(f"Aspect ratio JSON file not found at: {aspect_ratio_json_path}")

    with open(aspect_ratio_json_path, 'r') as f:
        ratio_data = json.load(f)

    aspect_ratio = ratio_data.get("aspect_ratio", None)
    if aspect_ratio is None:
        raise ValueError("JSON file does not contain 'aspect_ratio' key.")

    spec_width = int(round(spec_height * aspect_ratio))
    spec_dim = (spec_height, spec_width)  # (H, W)

    for idx, row in tqdm(df.head(num_samples).iterrows(), total=num_samples, desc="Sampling Progress"):
        cond = prepare_condition_array([
            row[lat_col],
            row[lon_col],
            row[dep_col],
            row[mag_col]]
        ).unsqueeze(0).to(device)

        shape = (1, 3, *spec_dim) if color_mode == 'color' else (1, 1, *spec_dim)
        x_gen = sample(model, diffusion, cond, shape)

        img = x_gen.squeeze(0).cpu()  # (C, H, W)
        if img.dim() == 3:
            img = img.permute(1, 2, 0)  # (H, W, C)

        img = x_gen.squeeze(0).cpu()

        # Fix dimensions based on color mode
        if color_mode == 'color':
            img = img.permute(1, 2, 0)  # (C, H, W) → (H, W, C)
        else:
            img = img.squeeze(0)  # (1, H, W) → (H, W)

        # Normalize and convert to uint8
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min + 1e-9)
        img = (img * 255).round().to(torch.uint8).numpy()

        save_spectrogram_image(
            img=img,
            e_id=row[e_id_col],
            output_dir=sample_save_dir,
            extension=spec_ext,
            file_name='sample_raw',
            color_mode=color_mode)

SPEC_DIM_HEIGHT: int = 256
SPEC_EXTENSION: str = "png"
IS_TESTING: bool = False
COLOR_MODE: str = "grayscale"
BATCH_SIZE: int = 32
LR: float = 1e-4
EPOCHS: int = 300

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess, Train, or Sample for diffusion STFT')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Preprocess
    p_pre = subparsers.add_parser('preprocess')
    p_pre.add_argument('--data-path', default="data/input/train/ew.csv")
    p_pre.add_argument('--output-path', default="data/output/spec/ew")
    p_pre.add_argument('--spec-dim-height', type=int, default=SPEC_DIM_HEIGHT)
    p_pre.add_argument('--event-id-col', default="EventID")
    p_pre.add_argument('--sampling-fs-col', default="SamplingRate")
    p_pre.add_argument('--last-metadata-col', default="NumFreqSteps")
    p_pre.add_argument('--spec-color-mode', choices=["grayscale", "color"], default=COLOR_MODE)
    p_pre.add_argument('--spec-ext', choices=["png"], default=SPEC_EXTENSION)
    p_pre.add_argument('--is-testing', type=bool, default=IS_TESTING)

    # Train
    p_tr = subparsers.add_parser('train')
    p_tr.add_argument('--data-path', default="data/input/train/ew.csv")
    p_tr.add_argument('--spec-dir', default="data/output/spec/ew")
    p_tr.add_argument('--spec-dim-height', type=int, default=SPEC_DIM_HEIGHT)
    p_tr.add_argument('--spec-color-mode', choices=["grayscale", "color"], default=COLOR_MODE)
    p_tr.add_argument('--spec-ext', choices=["png"], default=SPEC_EXTENSION)
    p_tr.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    p_tr.add_argument('--epochs', type=int, default=EPOCHS)
    p_tr.add_argument('--lr', type=float, default=LR)
    p_tr.add_argument('--model-save-path', default="models/ddpm.pth")
    p_tr.add_argument('--device', default="cuda")
    p_tr.add_argument('--event-id-col', default="EventID", help='Name of the event id column')
    p_tr.add_argument('--lat-col', default="EventLat", help='Name of the latitude column')
    p_tr.add_argument('--lon-col', default="EventLon", help='Name of the longitude column')
    p_tr.add_argument('--dep-col', default="Depth", help='Name of the depth column')
    p_tr.add_argument('--mag-col', default="Magnitude", help='Name of the magnitude column')
    p_tr.add_argument('--diff-ts', type=int, default=1000, help='')
    p_tr.add_argument('--diff-beta-start', type=float, default=1e-4, help='')
    p_tr.add_argument('--diff-beta-end', type=float, default=0.02, help='')

    # Sample
    p_sm = subparsers.add_parser('sample')
    p_sm.add_argument('--cond-csv', default="data/input/train/ew.csv")
    p_sm.add_argument('--model-path', default="models/ddpm.pth")
    p_sm.add_argument('--spec-dir', default="data/output/spec/ew")
    p_sm.add_argument('--spec-dim-height', type=int, default=SPEC_DIM_HEIGHT)
    p_sm.add_argument('--spec-aspect-ratio-path', type=str, default="spec_aspect_ratio.json")
    p_sm.add_argument('--spec-color-mode', choices=["grayscale", "color"], default=COLOR_MODE)
    p_sm.add_argument('--spec-ext', choices=["png"], default=SPEC_EXTENSION)
    p_sm.add_argument('--num-samples', type=int, default=5)
    p_sm.add_argument('--sample-save-dir', default="data/output/spec/ew")
    p_sm.add_argument('--device', default="cuda")
    p_sm.add_argument('--event-id-col', default="EventID", help='Name of the event id column')
    p_sm.add_argument('--lat-col', default="EventLat", help='Name of the latitude column')
    p_sm.add_argument('--lon-col', default="EventLon", help='Name of the longitude column')
    p_sm.add_argument('--dep-col', default="Depth", help='Name of the depth column')
    p_sm.add_argument('--mag-col', default="Magnitude", help='Name of the magnitude column')
    p_sm.add_argument('--diff-ts', type=int, default=1000, help='')
    p_sm.add_argument('--diff-beta-start', type=float, default=1e-4, help='')
    p_sm.add_argument('--diff-beta-end', type=float, default=0.02, help='')

    args = parser.parse_args()

    if args.command == 'preprocess':
        main_preprocess(
            data_path=args.data_path, output_path=args.output_path, spec_height=args.spec_dim_height, spec_ext=args.spec_ext,
            e_id_col=args.event_id_col, samp_fs_col=args.sampling_fs_col, last_md_col=args.last_metadata_col,
            stft_color_mode=args.spec_color_mode,
            is_testing=args.is_testing
        )
    elif args.command == 'train':
        main_train(
            data_path=args.data_path, spec_dir=args.spec_dir, spec_height=args.spec_dim_height, color_mode=args.spec_color_mode, spec_ext=args.spec_ext,
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
            model_save_path=args.model_save_path, device_name=args.device,
            e_id_col=args.event_id_col, lat_col=args.lat_col, lon_col=args.lon_col, dep_col=args.dep_col, mag_col=args.mag_col,
            diff_ts=args.diff_ts, diff_beta_start=args.diff_beta_start, diff_beta_end=args.diff_beta_end
        )
    elif args.command == 'sample':
        main_sample(
            data_path=args.cond_csv, model_path=args.model_path, spec_height=args.spec_dim_height, aspect_ratio_json_path=args.spec_aspect_ratio_path, color_mode=args.spec_color_mode, spec_ext=args.spec_ext,
            num_samples=args.num_samples, sample_save_dir=args.sample_save_dir, device_name=args.device,
            e_id_col=args.event_id_col ,lat_col=args.lat_col, lon_col=args.lon_col, dep_col=args.dep_col, mag_col=args.mag_col,
            diff_ts=args.diff_ts, diff_beta_start=args.diff_beta_start, diff_beta_end=args.diff_beta_end
        )
