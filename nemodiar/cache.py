"""NumPy port of `Nemotron3DiarizationSpeakerCache` (batch size 1): Arrival-Order Speaker Cache + FIFO queue.

Same arithmetic as the original, in float32. Every slot also carries the global encoder-frame id it came from
(-1 for the learned silence embedding), so runs can be compared state by state.

`topk` selects the top-k implementation used by cache compression:
  "torch"  - torch.topk on CPU, the original's own tie-breaking among exactly equal scores;
  "stable" - deterministic (score descending, then index ascending), portable to other runtimes.
Ties are common: confident single-speaker frames saturate the sigmoid and score exactly alike.
"""

import math
from dataclasses import dataclass

import numpy as np

SILENCE_ID = -1


@dataclass(frozen=True)
class CacheConfig:
    fifo_length: int = 264
    speaker_cache_update_period: int = 222
    speaker_cache_length: int = 264
    silence_frames_per_speaker: int = 1
    prediction_score_threshold: float = 0.25
    latest_frames_score_boost: float = 0.05
    strong_boost_rate: float = 0.75
    weak_boost_rate: float = 1.5
    min_positive_scores_rate: float = 0.5
    num_speakers: int = 8
    subsampling_factor: int = 8
    hidden_size: int = 512


def topk_indices(values: np.ndarray, k: int, impl: str) -> np.ndarray:
    """Indices of the k largest entries of each row of a 2-D array (unordered)."""
    if impl == "torch":
        import torch

        return torch.topk(torch.from_numpy(np.ascontiguousarray(values)), k, dim=1, sorted=False)[1].numpy()
    if impl == "stable":
        # argsort is stable: equal values keep ascending index order
        return np.argsort(-values, axis=1, kind="stable")[:, :k]
    if impl == "cpp":
        lib = _cpp_topk_lib()
        values = np.ascontiguousarray(values, np.float32)
        out = np.zeros((values.shape[0], k), np.int64)
        for row in range(values.shape[0]):
            lib.torch_topk_largest(values[row].ctypes.data, values.shape[1], k, out[row].ctypes.data)
        return out
    raise ValueError(impl)


_LIB = None


def _cpp_topk_lib():
    """native/torch_topk.cpp: torch's CPU top-k (std::nth_element), the algorithm the Swift runtime uses."""
    global _LIB
    if _LIB is None:
        import ctypes
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "native" / "libtorch_topk.dylib"
        _LIB = ctypes.CDLL(str(path))
        _LIB.torch_topk_largest.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    return _LIB


class SpeakerCache:
    def __init__(self, config: CacheConfig, silence_embeds: np.ndarray, topk: str = "torch"):
        self.cfg = config
        self.topk = topk
        self.silence_embeds = silence_embeds.astype(np.float32)
        budget = config.speaker_cache_length // config.num_speakers - config.silence_frames_per_speaker
        self.min_positive_scores = math.floor(budget * config.min_positive_scores_rate)
        self.num_strong_boosted = math.floor(budget * config.strong_boost_rate)
        self.num_weak_boosted = math.floor(budget * config.weak_boost_rate)

        h, s = config.hidden_size, config.num_speakers
        self.cache_embeds = np.zeros((0, h), np.float32)
        self.cache_probs = np.zeros((0, s), np.float32)
        self.cache_ids = np.zeros(0, np.int64)
        self.fifo_embeds = np.zeros((0, h), np.float32)
        self.fifo_ids = np.zeros(0, np.int64)
        self.is_compressed = False

    @property
    def num_cache(self) -> int:
        return len(self.cache_embeds)

    @property
    def num_fifo(self) -> int:
        return len(self.fifo_embeds)

    def embeds(self) -> np.ndarray:
        return np.concatenate([self.cache_embeds, self.fifo_embeds])

    def _num_popped(self, num_fifo: int) -> int:
        if num_fifo <= self.cfg.fifo_length:
            return 0
        return min(max(self.cfg.speaker_cache_update_period, num_fifo - self.cfg.fifo_length), num_fifo)

    def update(self, chunk_embeds: np.ndarray, chunk_ids: np.ndarray, probs: np.ndarray):
        """Pushes the scored chunk frames; `probs` (encoder rate) cover the whole step input [cache, fifo, chunk]."""
        nc, nf, n = self.num_cache, self.num_fifo, len(chunk_embeds)
        fifo_embeds = np.concatenate([self.fifo_embeds, chunk_embeds])
        fifo_ids = np.concatenate([self.fifo_ids, chunk_ids])
        num_popped = self._num_popped(len(fifo_embeds))
        if num_popped:
            fifo_probs = probs[nc : nc + nf + n]  # no look-back frames: fifo and chunk are contiguous
            stored = self.cache_probs if self.is_compressed else probs[:nc]
            cache_embeds = np.concatenate([self.cache_embeds, fifo_embeds[:num_popped]])
            cache_probs = np.concatenate([stored, fifo_probs[:num_popped]])
            cache_ids = np.concatenate([self.cache_ids, fifo_ids[:num_popped]])
            fifo_embeds, fifo_ids = fifo_embeds[num_popped:], fifo_ids[num_popped:]
            if len(cache_embeds) > self.cfg.speaker_cache_length:
                frame_idx = self.compression_indices(cache_probs)
                cache_embeds = np.concatenate([cache_embeds, self.silence_embeds[None]])[frame_idx]
                cache_probs = np.concatenate([cache_probs, np.zeros((1, cache_probs.shape[1]), np.float32)])[frame_idx]
                cache_ids = np.concatenate([cache_ids, [SILENCE_ID]])[frame_idx]
                self.is_compressed = True
            self.cache_embeds, self.cache_probs, self.cache_ids = cache_embeds, cache_probs, cache_ids
        self.fifo_embeds, self.fifo_ids = fifo_embeds, fifo_ids

    # --- compression -------------------------------------------------------------------------------------------

    def frame_scores(self, probs: np.ndarray) -> np.ndarray:
        thr = np.float32(self.cfg.prediction_score_threshold)
        log_probs = np.log(np.maximum(probs, thr))
        log_comp = np.log(np.maximum(np.float32(1.0) - probs, thr))
        sum_comp = log_comp[:, :1].copy()
        for j in range(1, log_comp.shape[1]):  # sequential, the summation order of torch.sum over 8 speakers
            sum_comp += log_comp[:, j : j + 1]
        scores = log_probs - log_comp + sum_comp - np.float32(math.log(0.5))
        is_speech = probs > 0.5
        scores = np.where(is_speech, scores, np.float32(-np.inf))
        is_positive = scores > 0
        has_enough = is_positive.sum(axis=0, keepdims=True) >= self.min_positive_scores
        return np.where(~is_positive & is_speech & has_enough, np.float32(-np.inf), scores).astype(np.float32)

    def boost(self, scores: np.ndarray, num_boosted: int, boost: float) -> np.ndarray:
        idx = topk_indices(scores.T, num_boosted, self.topk)  # per speaker, over frames
        scores = scores.copy()
        cols = np.arange(scores.shape[1])[:, None]
        scores[idx, np.broadcast_to(cols, idx.shape)] += np.float32(boost)
        return scores

    def compression_indices(self, probs: np.ndarray) -> np.ndarray:
        """Rows of [frames..., silence] kept by the compressed cache, in cache order."""
        num_frames, num_speakers = probs.shape
        scores = self.frame_scores(probs)
        scores[self.cfg.speaker_cache_length :] += np.float32(self.cfg.latest_frames_score_boost)
        scores = self.boost(scores, self.num_strong_boosted, -2.0 * math.log(0.5))
        scores = self.boost(scores, self.num_weak_boosted, -math.log(0.5))
        n_sil = self.cfg.silence_frames_per_speaker
        scores = np.concatenate([scores, np.full((n_sil, num_speakers), np.inf, np.float32)])

        num_scored = num_frames + n_sil
        sentinel = num_scored * num_speakers
        flat = scores.T.reshape(1, -1)  # speaker-major
        idx = topk_indices(flat, self.cfg.speaker_cache_length, self.topk)[0]
        idx = np.where(flat[0, idx] == -np.inf, sentinel, idx)
        idx = np.sort(idx)
        return np.where(idx == sentinel, num_frames, np.minimum(idx % num_scored, num_frames))
