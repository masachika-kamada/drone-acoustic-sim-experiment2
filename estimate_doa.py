import argparse
import os

import numpy as np

from lib.custom import create_doa_object, perform_fft_on_frames
from src.file_io import load_config, load_signal_from_wav
from src.visualization_tools import plot_music_spectra
from generate_acoustic_sim import Drone


def main(args, config):
    config_drone = config["drone"]
    fs = config["pra"]["room"]["fs"]
    drone = Drone(config_drone, fs=fs)

    signal_source = load_signal_from_wav(f"{args.config_dir}/simulation/source.wav", fs)
    signal_ncm_rev = load_signal_from_wav(f"{args.config_dir}/simulation/ncm_rev.wav", fs)

    X_source = perform_fft_on_frames(signal_source, args.window_size, args.hop_size)
    X_noise = perform_fft_on_frames(signal_ncm_rev, args.window_size, args.hop_size)

    print("X_source.shape", X_source.shape)
    print("X_noise.shape", X_noise.shape)

    for method in ["SEVD", "GEVD", "GSVD"]:
        output_dir = f"{args.config_dir}/{method}"
        os.makedirs(output_dir, exist_ok=True)
        doa = create_doa_object(
            method=method,
            source_noise_thresh=100,
            mic_positions=drone.mic_positions,
            fs=fs,
            nfft=args.window_size,
            num_src=len(config["voice"]["source"]),
            output_dir=output_dir,
        )
        frame_length = 100
        for f in range(0, X_source.shape[2], frame_length // 4):
            xs = X_source[:, :, f : f + frame_length]
            xn = X_noise[:, :, f : f + frame_length]
            doa.locate_sources(xs, xn, freq_range=args.freq_range, auto_identify=True)
        plot_music_spectra(doa, output_dir=output_dir)
        np.save(f"{output_dir}/decomposed_values.npy", np.array(doa.decomposed_values_strage))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate room acoustics based on YAML config.")
    parser.add_argument("--config_dir", type=str, required=True, help="Directory containing the config.yaml file")
    parser.add_argument("--window_size", default=512, type=int, help="Window size for FFT")
    parser.add_argument("--hop_size", default=128, type=int, help="Hop size for FFT")
    parser.add_argument("--freq_range", default=[300, 3500], type=int, nargs=2, help="Frequency range for DoA")
    parser.add_argument("--source_noise_thresh", default=100, type=int, help="Frequency range for DoA")
    args = parser.parse_args()

    if not os.path.exists(args.config_dir):
        raise FileNotFoundError(f"Config directory '{args.config_dir}' does not exist.")

    config = load_config(f"{args.config_dir}/config.yaml")

    main(args, config)
