import numpy as np
import sys
sys.path.append("../")
from lib.doa import MUSIC, GevdMUSIC, GsvdMUSIC


def create_doa_object(method, source_noise_thresh, mic_positions, fs, nfft, X_noise, output_dir):
    common_params = {
        "L": mic_positions,
        "fs": fs,
        "nfft": nfft,
        "c": 343.0,
        "mode": "far",
        "azimuth": np.linspace(-np.pi, np.pi, 360),
        "source_noise_thresh": source_noise_thresh,
        "output_dir": output_dir,
    }
    if method == "MUSIC":
        doa = MUSIC(**common_params)
    elif method == "GEVD-MUSIC":
        doa = GevdMUSIC(**common_params, X_noise=X_noise)
    elif method == "GSVD-MUSIC":
        doa = GsvdMUSIC(**common_params, X_noise=X_noise)
    else:
        raise ValueError(f"Unknown method: {method}")
    return doa


def perform_fft_on_frames(signal, nfft, hop_size):
    num_frames = (signal.shape[1] - nfft) // hop_size + 1
    X = np.empty((signal.shape[0], nfft // 2 + 1, num_frames), dtype=complex)
    for t in range(num_frames):
        frame = signal[:, t * hop_size : t * hop_size + nfft]
        X[:, :, t] = np.fft.rfft(frame, n=nfft)
    return X