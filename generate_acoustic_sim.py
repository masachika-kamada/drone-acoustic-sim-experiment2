import argparse
import os

from src.class_room import Room
from src.class_sound import Ambient, Drone, Voice
from src.file_io import load_config, write_signal_to_wav
from src.snr import adjust_snr


def main(config, output_dir):
    config_pra = config["pra"]
    room = Room(config_pra)

    config_voice = config["voice"]
    voice = Voice(config_voice, fs=room.fs)

    config_drone = config["drone"]
    drone = Drone(config_drone, fs=room.fs)

    config_ambient = config.get("ambient", None)
    ambient = Ambient(config_ambient, fs=room.fs) if config_ambient is not None else None

    room.place_microphones(drone.mic_positions)

    adjust_snr(room, voice, drone, drone.snr, output_dir)
    if ambient is not None:
        adjust_snr(room, voice, ambient, ambient.snr, output_dir)

    room.place_source(voice=voice, drone=drone, ambient=ambient)

    start = int(room.fs * config["processing"]["start_time"])
    end = int(room.fs * config["processing"]["end_time"])

    for signal, name in zip(room.simulate(output_dir), ["source", "ncm_rev", "ncm_dir"]):
        signal = signal[:, start:end]
        # TODO: signalがint16でオーバーフローするのでnpzで保存する
        write_signal_to_wav(signal, f"{output_dir}/{name}.wav", room.fs)


def confirm_execution(output_dir):
    """既存のディレクトリが存在する場合にユーザーに確認を求める"""
    if os.path.exists(output_dir):
        response = input(f"The directory '{output_dir}' already exists. Do you want to continue? (y/n): ")
        return response.strip().lower() == 'y'
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate room acoustics based on YAML config.")
    parser.add_argument("--config_dir", type=str, required=True, help="Directory containing the config.yaml file")
    args = parser.parse_args()

    config_dir = f"{args.config_dir}"
    config = load_config(f"{config_dir}/config.yaml")
    output_dir = f"{config_dir}/simulation"
    if confirm_execution(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        main(config, output_dir)
    else:
        print("Execution canceled.")
