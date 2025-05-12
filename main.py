import os
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset.spectrogram import SpectrogramDataset
from diffusion import Diffusion
from models.u_net import ConditionalUNet
from sample import sample
from train import train
from utils.read_write import read_waveform_data, save_spectrogram_image
from utils.transform import compute_spectrogram

TESTING_LIMIT: int = 5

def main_preprocess(data_path: str, output_path: str, spec_height: int, spec_ext: str,
                    e_id_col: str, samp_fs_col: str, last_md_col: str,
                    stft_color_mode: str, is_testing: bool):
    df, waveforms = read_waveform_data(data_path=data_path, last_md_col=last_md_col)
    os.makedirs(output_path, exist_ok=True)
    iteration_amount: int = TESTING_LIMIT if is_testing else len(df)
    for idx, row in df.head(iteration_amount).iterrows():
        e_id = row[e_id_col]
        fs = row[samp_fs_col]
        img = compute_spectrogram(
            waveform=waveforms[idx],
            samp_fs=fs,
            color_mode=stft_color_mode,
            output_height=spec_height
        )
        save_spectrogram_image(img=img, e_id=e_id, output_dir=output_path, extension=spec_ext)


def main_train(data_path: str, spec_dir: str, spec_dim: tuple[int, int], color_mode: str, spec_ext: str,
               batch_size: int, epochs: int, lr: float,
               model_save_path: str, device_name: str,
               e_id_col: str, lat_col: str, lon_col: str, dep_col: str, mag_col: str,
               diff_ts: int, diff_beta_start: float, diff_beta_end: float):
    dataset = SpectrogramDataset(
        data_path=data_path,
        spec_dir=spec_dir,
        spec_ext=spec_ext,
        img_size=spec_dim,
        color_mode=color_mode,
        e_id_col=e_id_col,
        e_lat_col=lat_col,
        e_lon_col=lon_col,
        depth_col=dep_col,
        mag_col=mag_col
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device(device_name)

    model = ConditionalUNet(in_channels=(3 if color_mode == 'color' else 1)).to(device)
    diffusion = Diffusion(device=device, timesteps=diff_ts, beta_start=diff_beta_start, beta_end=diff_beta_end)

    train(model, diffusion, dataloader, epochs=epochs, lr=lr)

    torch.save(model.state_dict(), model_save_path)


def main_sample(data_path: str, model_path: str, spec_dim: tuple[int, int], color_mode: str, spec_ext: str,
                num_samples: int, sample_save_dir: str, device_name: str,
                e_id_col: str, lat_col: str, lon_col: str, dep_col: str, mag_col: str,
                diff_ts: int, diff_beta_start: float, diff_beta_end: float):
    os.makedirs(sample_save_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    device = torch.device(device_name)

    model = ConditionalUNet(in_channels=(3 if color_mode == 'color' else 1)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    diffusion = Diffusion(device=device, timesteps=diff_ts, beta_start=diff_beta_start, beta_end=diff_beta_end)

    for idx, row in df.head(num_samples).iterrows():
        cond = torch.tensor([
            row[lat_col], row[lon_col], row[dep_col], row[mag_col]
        ], dtype=torch.float32, device=device_name).unsqueeze(0)

        shape = (1, 3, *spec_dim) if color_mode == 'color' else (1, 1, *spec_dim)
        x_gen = sample(model, diffusion, cond, shape)

        img = x_gen.squeeze(0).cpu()  # remove batch dimension
        if img.dim() == 3:  # (C, H, W)
            img = img.permute(1, 2, 0)  # (H, W, C)

        img = (img.numpy() * 255).astype(np.uint8)

        save_spectrogram_image(img=img, e_id=f"{row[e_id_col]}_sample", output_dir=sample_save_dir, extension=spec_ext)

SPEC_DIM_WIDTH: int = 256
SPEC_DIM_HEIGHT: int = 256
SPEC_EXTENSION: str = "png"
IS_TESTING: bool = True

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
    p_pre.add_argument('--spec-color-mode', choices=["grayscale", "color"], default="color")
    p_pre.add_argument('--spec-ext', choices=["png"], default=SPEC_EXTENSION)
    p_pre.add_argument('--is-testing', type=bool, default=IS_TESTING)

    # Train
    p_tr = subparsers.add_parser('train')
    p_tr.add_argument('--data-path', default="data/input/train/ew.csv")
    p_tr.add_argument('--spec-dir', default="data/output/spec/ew")
    p_tr.add_argument('--spec-dim-width', type=int, default=SPEC_DIM_WIDTH)
    p_tr.add_argument('--spec-dim-height', type=int, default=SPEC_DIM_HEIGHT)
    p_tr.add_argument('--spec-color-mode', choices=["grayscale", "color"], default="color")
    p_tr.add_argument('--spec-ext', choices=["png"], default=SPEC_EXTENSION)
    p_tr.add_argument('--batch-size', type=int, default=16)
    p_tr.add_argument('--epochs', type=int, default=1000)
    p_tr.add_argument('--lr', type=float, default=1e-4)
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
    p_sm.add_argument('--spec-dim-width', type=int, default=SPEC_DIM_WIDTH)
    p_sm.add_argument('--spec-dim-height', type=int, default=SPEC_DIM_HEIGHT)
    p_sm.add_argument('--spec-color-mode', choices=["grayscale", "color"], default="color")
    p_sm.add_argument('--spec-ext', choices=["png"], default=SPEC_EXTENSION)
    p_sm.add_argument('--num-samples', type=int, default=20)
    p_sm.add_argument('--sample-save-dir', default="data/output/samples")
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
            data_path=args.data_path, spec_dir=args.spec_dir, spec_dim=(args.spec_dim_width, args.spec_dim_height), color_mode=args.spec_color_mode, spec_ext=args.spec_ext,
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
            model_save_path=args.model_save_path, device_name=args.device,
            e_id_col=args.event_id_col, lat_col=args.lat_col, lon_col=args.lon_col, dep_col=args.dep_col, mag_col=args.mag_col,
            diff_ts=args.diff_ts, diff_beta_start=args.diff_beta_start, diff_beta_end=args.diff_beta_end
        )
    elif args.command == 'sample':
        main_sample(
            data_path=args.cond_csv, model_path=args.model_path, spec_dim=(args.spec_dim_width, args.spec_dim_height), color_mode=args.spec_color_mode, spec_ext=args.spec_ext,
            num_samples=args.num_samples, sample_save_dir=args.sample_save_dir, device_name=args.device,
            e_id_col=args.event_id_col ,lat_col=args.lat_col, lon_col=args.lon_col, dep_col=args.dep_col, mag_col=args.mag_col,
            diff_ts=args.diff_ts, diff_beta_start=args.diff_beta_start, diff_beta_end=args.diff_beta_end
        )
