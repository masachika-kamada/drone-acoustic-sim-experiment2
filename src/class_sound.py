import numpy as np
import pyroomacoustics as pra

from src.file_io import load_signal_from_wav


class AudioLoader:
    def __init__(self, config, fs=16000):
        self.data = []
        config_source = config["source"]
        for s in config_source:
            idx_start = int(fs * s["start_time"])
            signal = load_signal_from_wav(s["file_path"], fs)[idx_start:]
            position = s["position"]
            self.data.append((signal, position))
        self.n_data = len(self.data)


class Voice(AudioLoader):
    def __init__(self, config, fs):
        super().__init__(config, fs)


class Ambient(AudioLoader):
    def __init__(self, config, fs):
        self.snr = config["snr"]
        super().__init__(config, fs)


class Drone(AudioLoader):
    def __init__(self, config, fs):
        self.snr = config["snr"]
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
