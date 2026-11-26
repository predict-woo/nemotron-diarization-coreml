"""Serialization of streaming runs (reference or candidate) to compact .npz files.

A run is the per-frame speaker logits plus the speaker-cache trajectory, stored as global encoder-frame ids per
step (-1 = silence embedding, -2 = empty slot), which is enough to compare two runs' states exactly.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

PAD_ID = -2


@dataclass
class Run:
    logits: np.ndarray  # (num_mel_frames, num_speakers) float32
    cache_ids: np.ndarray  # (num_steps, max_cache) int32, cache content before each step, PAD_ID padded
    fifo_ids: np.ndarray  # (num_steps, max_fifo) int32, FIFO content before each step
    step_seconds: np.ndarray | None = None  # (num_steps,) wall time of each step

    @property
    def probs(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logits.astype(np.float64)))

    @property
    def num_steps(self) -> int:
        return len(self.cache_ids)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {} if self.step_seconds is None else {"step_seconds": self.step_seconds}
        np.savez_compressed(path, logits=self.logits, cache_ids=self.cache_ids, fifo_ids=self.fifo_ids, **extra)

    @classmethod
    def load(cls, path: Path) -> "Run":
        data = np.load(path)
        return cls(
            logits=data["logits"],
            cache_ids=data["cache_ids"],
            fifo_ids=data["fifo_ids"],
            step_seconds=data["step_seconds"] if "step_seconds" in data else None,
        )


def pad_ids(rows: list[np.ndarray]) -> np.ndarray:
    width = max([len(r) for r in rows] + [1])
    out = np.full((len(rows), width), PAD_ID, dtype=np.int32)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def load_run_dir(path: Path) -> Run:
    """A run written as .npy files (the Swift CLI's --out layout)."""
    step = path / "step_seconds.npy"
    return Run(
        logits=np.load(path / "logits.npy"),
        cache_ids=np.load(path / "cache_ids.npy"),
        fifo_ids=np.load(path / "fifo_ids.npy"),
        step_seconds=np.load(step) if step.exists() else None,
    )
