"""Frame-level diarization error rate (10 ms frames, no collar, overlapped speech scored).

DER = (missed + false alarm + confusion) / reference speech, every speaker of every frame counted, with the
optimal one-to-one speaker mapping (Hungarian) between reference and hypothesis.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class DER:
    der: float
    miss: float
    false_alarm: float
    confusion: float
    speech_frames: int

    def __str__(self):
        return f"DER {self.der * 100:5.2f}% (miss {self.miss * 100:.2f}, FA {self.false_alarm * 100:.2f}, conf {self.confusion * 100:.2f})"


def der(reference: np.ndarray, hypothesis: np.ndarray) -> DER:
    """reference, hypothesis: (frames, speakers) boolean activity; trimmed to the shorter length."""
    n = min(len(reference), len(hypothesis))
    ref, hyp = reference[:n].astype(bool), hypothesis[:n].astype(bool)
    overlap = ref.T.astype(np.int64) @ hyp.astype(np.int64)  # (ref spk, hyp spk) co-active frames
    rows, cols = linear_sum_assignment(-overlap)
    correct = np.zeros(n, np.int64)
    for r, c in zip(rows, cols):
        correct += ref[:, r] & hyp[:, c]
    n_ref, n_hyp = ref.sum(1), hyp.sum(1)
    total = max(int(n_ref.sum()), 1)
    miss = np.maximum(n_ref - n_hyp, 0).sum()
    fa = np.maximum(n_hyp - n_ref, 0).sum()
    conf = (np.minimum(n_ref, n_hyp) - correct).sum()
    return DER((miss + fa + conf) / total, miss / total, fa / total, conf / total, total)


def activity(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return probs > threshold


# Acceptance of a float16 (Neural Engine) pipeline, which cannot be bit-faithful: its speaker cache may keep
# different (equally good) frames than the original's. It must diarize as well as the original.
MAX_DER_INCREASE_FILE = 0.010  # DER vs ground truth, absolute, per recording
MAX_DER_INCREASE_TOTAL = 0.0025  # over all recordings
MAX_FLIPPED_FILE = 0.03  # share of 10 ms frames whose speaker set differs from the original's, per recording
MAX_FLIPPED_TOTAL = 0.005
