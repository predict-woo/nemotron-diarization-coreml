"""Log-mel front end of a streaming chunk, a NumPy port of `NemotronAsrStreamingFeatureExtractor` + processor.

Per chunk: preemphasis (0.97) -> optional 96-sample zero pad (last chunk, uncentered) -> STFT (n_fft 512, hop 160,
symmetric Hann 400 centered in the frame, `center` zero-padding 256 on both sides for the first chunk) -> power
spectrum -> 128 slaney mel filters -> log(x + 2^-24). Frames beyond the valid length are dropped.
"""

from pathlib import Path

import numpy as np

N_FFT, HOP, WIN, N_MELS = 512, 160, 400, 128
PREEMPH = 0.97
LOG_GUARD = np.float32(2.0**-24)
CONSTANTS_DIR = Path(__file__).resolve().parent / "constants"


def hann_window_symmetric(n: int = WIN) -> np.ndarray:
    import torch

    return torch.hann_window(n, periodic=False, dtype=torch.float32).numpy()


def load_constants(kind: str = "hf") -> tuple[np.ndarray, np.ndarray]:
    """(window[400], mel_filters[128, 257]) float32. "hf" = the HF processor's, "nemo" = the .nemo checkpoint's."""
    path = CONSTANTS_DIR / f"frontend_{kind}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing, run scripts/export_constants.py")
    data = np.load(path)
    return data["window"].astype(np.float32), data["mel_filters"].astype(np.float32)


class FeatureExtractor:
    def __init__(self, constants: str = "hf"):
        window, self.mel_filters = load_constants(constants)
        left = (N_FFT - WIN) // 2  # torch.stft centers a shorter window inside the n_fft frame
        self.frame_window = np.zeros(N_FFT, np.float32)
        self.frame_window[left : left + WIN] = window

    def __call__(self, audio: np.ndarray, center: bool, is_last: bool) -> np.ndarray:
        """Features (num_frames, 128) of one chunk, exactly the frames the HF processor keeps."""
        x = np.asarray(audio, np.float32)
        x = np.concatenate([x[:1], x[1:] - np.float32(PREEMPH) * x[:-1]])
        length = len(x)
        if is_last and not center:
            pad = N_FFT // 2 - HOP
            x = np.concatenate([x, np.zeros(pad, np.float32)])
            length += pad
        if center:
            x = np.pad(x, (N_FFT // 2, N_FFT // 2))
            num_valid = length // HOP
        else:
            num_valid = (length - N_FFT) // HOP + 1
        num_valid = max(num_valid, 0)
        if num_valid == 0:
            return np.zeros((0, N_MELS), np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP][:num_valid]
        # float32 FFT, as torch.stft; magnitude then squared, as the original
        spec = np.fft.rfft((frames * self.frame_window).astype(np.float32), axis=-1)
        magnitude = np.sqrt(spec.real**2 + spec.imag**2)
        power = magnitude * magnitude
        mel = power @ self.mel_filters.T
        return np.log(mel + LOG_GUARD).astype(np.float32)
