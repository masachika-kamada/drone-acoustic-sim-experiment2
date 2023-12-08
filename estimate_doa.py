import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra

from lib.custom import create_doa_object, perform_fft_on_frames
from src.file_io import load_config, load_signal_from_wav
from src.visualization_tools import plot_music_spectrum
from generate_acoustic_sim import Drone


def main(config, output_dir):
    config_drone = config["drone"]
    drone = Drone(config_drone, fs=config["pra"]["room"]["fs"])

    signal_source = load_signal_from_wav(f"{output_dir}/source.wav", config["pra"]["room"]["fs"])
    signal_noise = load_signal_from_wav(f"{output_dir}/noise_template.wav", config["pra"]["room"]["fs"])

    fft_config = config["fft"]
    window_size = fft_config["window_size"]
    hop_size = fft_config["hop_size"]
    X_source = perform_fft_on_frames(signal_source, window_size, hop_size)
    X_noise = perform_fft_on_frames(signal_noise, window_size, hop_size)

    config_doa = config["doa"]
    doa = create_doa_object(
        method=config_doa["method"],
        source_noise_thresh=config_doa["source_noise_thresh"],
        mic_positions=drone.mic_positions,
        fs=config["pra"]["room"]["fs"],
        nfft=window_size,
        X_noise=X_noise,
        output_dir=output_dir,
    )
    doa.locate_sources(X_source, X_noise, freq_range=config_doa["freq_range"], auto_identify=True)
    plot_music_spectrum(doa, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate room acoustics based on YAML config.")
    parser.add_argument("--config_dir", type=str, required=True, help="Directory containing the config.yaml file")
    args = parser.parse_args()

    config_dir = f"experiments/{args.config_dir}"
    config = load_config(f"{config_dir}/config.yaml")
    output_dir = f"{config_dir}/output"
    os.makedirs(output_dir, exist_ok=True)

    main(config, output_dir)
