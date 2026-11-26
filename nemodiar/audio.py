from pathlib import Path

import numpy as np
import soundfile as sf

from . import SAMPLE_RATE


def load_audio(path: str | Path) -> np.ndarray:
    """Loads a 16 kHz mono file as float32. Test files are pre-resampled so resampling is not under test."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim != 1 or sr != SAMPLE_RATE:
        raise ValueError(f"{path}: expected 16 kHz mono, got sr={sr}, shape={audio.shape}")
    return audio
