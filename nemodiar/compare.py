"""Metrics comparing a candidate streaming run against the reference run."""

from dataclasses import asdict, dataclass

import numpy as np

from .golden import Run

FRAME_S = 0.01


def speaker_segments(probs: np.ndarray, threshold: float = 0.5) -> list[tuple[float, float, int]]:
    """Same as `Nemotron3DiarizationProcessor.extract_speaker_dict`: (start_s, end_s, speaker), sorted."""
    active = (probs > threshold).astype(np.int8)
    pad = np.zeros((1, active.shape[1]), np.int8)
    changes = np.diff(np.concatenate([pad, active, pad]), axis=0)
    segments = []
    for spk in range(active.shape[1]):
        starts = np.nonzero(changes[:, spk] == 1)[0]
        ends = np.nonzero(changes[:, spk] == -1)[0]
        segments += [(round(s * FRAME_S, 2), round(e * FRAME_S, 2), spk) for s, e in zip(starts, ends)]
    segments.sort(key=lambda seg: (seg[0], seg[2]))
    return segments


@dataclass
class Comparison:
    num_frames: int
    max_abs_prob: float  # max over frames and speakers of |p_ref - p_cand|
    mean_abs_prob: float
    p999_abs_prob: float  # 99.9th percentile of per-frame max |dp|
    max_abs_logit: float  # restricted to |logit_ref| < 15 where sigmoid is not saturated
    decision_mismatch_frames: int  # frames whose thresholded (0.5) speaker set differs
    segments_identical: bool
    num_segments_ref: int
    num_segments_cand: int
    state_identical: bool  # cache/FIFO frame ids identical at every step
    first_state_divergence: int  # step index, -1 if none
    num_state_divergent_steps: int

    def as_dict(self):
        return asdict(self)


def compare_runs(ref: Run, cand: Run) -> Comparison:
    if ref.logits.shape != cand.logits.shape:
        raise ValueError(f"shape mismatch: reference {ref.logits.shape}, candidate {cand.logits.shape}")
    p_ref, p_cand = ref.probs, cand.probs
    dp = np.abs(p_ref - p_cand)
    frame_dp = dp.max(axis=1)
    unsat = np.abs(ref.logits) < 15
    dlogit = np.abs(ref.logits.astype(np.float64) - cand.logits)[unsat]
    seg_ref, seg_cand = speaker_segments(p_ref), speaker_segments(p_cand)

    divergent = _divergent_steps(ref, cand)
    return Comparison(
        num_frames=len(p_ref),
        max_abs_prob=float(dp.max()),
        mean_abs_prob=float(dp.mean()),
        p999_abs_prob=float(np.quantile(frame_dp, 0.999)),
        max_abs_logit=float(dlogit.max()) if dlogit.size else 0.0,
        decision_mismatch_frames=int(((p_ref > 0.5) != (p_cand > 0.5)).any(axis=1).sum()),
        segments_identical=seg_ref == seg_cand,
        num_segments_ref=len(seg_ref),
        num_segments_cand=len(seg_cand),
        state_identical=len(divergent) == 0,
        first_state_divergence=int(divergent[0]) if len(divergent) else -1,
        num_state_divergent_steps=len(divergent),
    )


def _divergent_steps(ref: Run, cand: Run) -> np.ndarray:
    if ref.num_steps != cand.num_steps:
        raise ValueError(f"step count mismatch: reference {ref.num_steps}, candidate {cand.num_steps}")
    bad = np.zeros(ref.num_steps, bool)
    for a, b in ((ref.cache_ids, cand.cache_ids), (ref.fifo_ids, cand.fifo_ids)):
        width = max(a.shape[1], b.shape[1])
        a = np.pad(a, ((0, 0), (0, width - a.shape[1])), constant_values=-2)
        b = np.pad(b, ((0, 0), (0, width - b.shape[1])), constant_values=-2)
        bad |= (a != b).any(axis=1)
    return np.nonzero(bad)[0]


# What "identical to the original" means for a candidate run (see README):
#  - the speaker-cache / FIFO contents are the same frames at every step (exact, discrete),
#  - every 10 ms frame has the same thresholded speaker set, hence the same segments (exact, discrete),
#  - speaker probabilities agree within MAX_ABS_PROB: fp32 rounding noise, the size of the original's own
#    fp32-vs-fp64 difference.
MAX_ABS_PROB = 1e-4


def identity_failures(c: Comparison, max_abs_prob: float = MAX_ABS_PROB) -> list[str]:
    failures = []
    if not c.state_identical:
        failures.append(f"speaker cache diverged at step {c.first_state_divergence} "
                        f"({c.num_state_divergent_steps} steps differ)")
    if c.decision_mismatch_frames:
        failures.append(f"{c.decision_mismatch_frames} frames with a different speaker set")
    if not c.segments_identical:
        failures.append(f"segments differ ({c.num_segments_ref} vs {c.num_segments_cand})")
    if c.max_abs_prob > max_abs_prob:
        failures.append(f"max |dp| {c.max_abs_prob:.2e} > {max_abs_prob:.0e}")
    return failures
