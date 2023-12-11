import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra

from src.file_io import load_config, load_signal_from_wav, write_signal_to_wav

from lib.room import custom_plot

pra.Room.plot = custom_plot


class AudioLoader:
    def __init__(self, config, fs=16000):
        self.data = []
        config_source = config["source"]
        for s in config_source:
            idx_start = int(fs * s["start_time"])
            signal = load_signal_from_wav(s["file_path"], fs)[idx_start:]
            position = s["position"]
            self.data.append((signal, position))


class Voice(AudioLoader):
    def __init__(self, config, fs):
        super().__init__(config, fs)


class Drone(AudioLoader):
    def __init__(self, config, fs):
        config_mic_positions = config["mic_positions"]
        self.mic_positions = self._create_mic_positions(config_mic_positions)
        config_propeller = config.get("propeller", {})
        self.offset = np.array(config_propeller.get("offset", [0, 0]))
        self.width = config_propeller.get("width", 0.1)
        self._adjust_source_positions(config, config_mic_positions["center"])
        super().__init__(config, fs)

    def _create_mic_positions(self, config):
        return pra.circular_2D_array(
            center=config["center"],
            M=config["M"],
            phi0=config["phi0"],
            radius=config["radius"]
        )

    def _adjust_source_positions(self, config, center):
        num_sources = len(config["source"])
        if num_sources == 0:
            return
        xs = np.linspace(-self.width / 2, self.width / 2, num_sources)
        for i, x in enumerate(xs):
            config["source"][i]["position"] = [x + center[0] + self.offset[0], center[1] + self.offset[1]]


class Room:
    def __init__(self, config):
        config_room = config["room"]
        room_dim = config_room["room_dim"]
        corners = np.array([
            [int(-room_dim[0] / 2), 0],
            [int(-room_dim[0] / 2), room_dim[1]],
            [int( room_dim[0] / 2), room_dim[1]],
            [int( room_dim[0] / 2), 0],
        ]).T
        corners = self._generate_floor(corners, config_room["floor"])
        self.fs = config_room["fs"]
        self.snr = config_room["snr"]
        self.max_order = config_room["max_order"]
        self.pra_material = config_room["floor"]["material"]

        materials = self._load_materials(len(corners[0]), self.pra_material)
        self.room_source = self._create_room(corners, self.max_order, materials)
        self.room_ncm_rev = self._create_room(corners, self.max_order, materials)
        materials = self._load_materials(len(corners[0]), None)
        self.room_ncm_no_rev = self._create_room(corners, 0, materials)

    def _generate_floor(self, corners, config):
        shape = config["shape"]
        if shape == "flat":
            return corners

        corners = corners.T  # [x, y]
        x_min = corners[0, 0]
        x_max = corners[-1, 0]
        y = corners[0, 1]

        if shape in ["triangle", "square"]:
            new_corners = []
            interval = config["interval"]
            height = config["height"]
            n_rough = int((x_max - x_min) / interval)
            interval = (x_max - x_min) / n_rough

            for i in range(1, n_rough):
                x = x_max - i * interval
                if shape == "triangle":
                    if i % 2 == 0:
                        new_corners.append([x, y])
                    else:
                        new_corners.append([x, y + height])
                elif shape == "square":
                    if i % 2 == 0:
                        new_corners.append([x, y + height])
                        new_corners.append([x, y])
                    else:
                        new_corners.append([x, y])
                        new_corners.append([x, y + height])
            new_corners = np.array(new_corners)

        elif shape == "random":
            seed = config["seed"]
            np.random.seed(seed)
            min_interval = config["min_interval"]
            max_interval = config["max_interval"]
            n_max = int((x_max - x_min) // min_interval - 1)
            x_rand = np.random.rand(n_max) * (max_interval - min_interval) + min_interval
            x_rand = x_max - np.cumsum(x_rand)
            x_rand = x_rand[x_rand >= x_min + min_interval]
            y_rand = np.random.rand(len(x_rand)) * max_interval * 0.5
            new_corners = np.vstack([x_rand, y_rand]).T

        new_corners = np.vstack([corners, new_corners])
        return new_corners.T


    def _load_materials(self, num_corners, floor_material):
        floor_material = self._create_materials(floor_material)
        no_wall_material = self._create_materials()
        materials = [floor_material] + [no_wall_material] * 3 + [floor_material] * (num_corners - 4)
        # listを逆順にする
        materials = materials[::-1]
        return materials

    def _create_materials(self, m=None):
        return pra.Material(energy_absorption=1.0) if m is None else pra.Material(m)

    def _create_room(self, corners, max_order, materials):
        return pra.Room.from_corners(corners, fs=self.fs, max_order=max_order, materials=materials)

    def place_microphones(self, mic_positions):
        self.room_source.add_microphone_array(pra.MicrophoneArray(mic_positions, self.fs))
        self.room_ncm_rev.add_microphone_array(pra.MicrophoneArray(mic_positions, self.fs))
        self.room_ncm_no_rev.add_microphone_array(pra.MicrophoneArray(mic_positions, self.fs))

    def place_source(self, voice, drone):
        for signal, position in voice.data:
            self.room_source.add_source(position, signal=signal)
        for signal, position in drone.data:
            self.room_source.add_source(position, signal=signal)
            self.room_ncm_rev.add_source(position, signal=signal)
            self.room_ncm_no_rev.add_source(position, signal=signal)

    def simulate(self, output_dir):
        t0 = time.time()
        self.room_source.simulate(snr=self.snr)
        print(f"Room simulation time (source): {time.time() - t0} [sec]")

        self.room_ncm_rev.simulate(snr=self.snr)
        self.room_ncm_no_rev.simulate(snr=self.snr)

        self.room_source.plot()
        plt.savefig(f"{output_dir}/room.png")
        plt.close()

        self.room_ncm_rev.plot()
        plt.savefig(f"{output_dir}/room_noise.png")
        plt.close()

        return (self.room_source.mic_array.signals,
                self.room_ncm_rev.mic_array.signals,
                self.room_ncm_no_rev.mic_array.signals)


def main(config, output_dir):
    config_pra = config["pra"]
    room = Room(config_pra)

    config_drone = config["drone"]
    drone = Drone(config_drone, fs=room.fs)

    config_voice = config["voice"]
    voice = Voice(config_voice, fs=room.fs)

    room.place_microphones(drone.mic_positions)
    room.place_source(voice, drone)

    start = int(room.fs * config["processing"]["start_time"])
    end = int(room.fs * config["processing"]["end_time"])

    for signal, name in zip(room.simulate(output_dir), ["source", "ncm_rev", "ncm_no_rev"]):
        signal = signal[:, start:end]
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