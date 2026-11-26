"""Streaming diarization pipeline: front end + speaker cache on the host, encoder step on a pluggable backend.

Per step, exactly as the original streaming forward:
  features -> feature-stacking embeddings of the chunk (+ look-ahead)
  sequence = [speaker cache, FIFO, chunk + look-ahead]   (positions restart at 0 every step)
  backend(sequence) -> speaker logits at the 10 ms rate for every position
  speaker cache / FIFO update with the encoder-rate probabilities; the chunk's own logits are emitted.
"""

import time
from typing import Protocol

import numpy as np

from .cache import CacheConfig, SpeakerCache
from .features import FeatureExtractor, load_constants
from .golden import Run, pad_ids
from .schedule import SUBSAMPLING, chunk_schedule


class EncoderBackend(Protocol):
    def embed(self, features: np.ndarray) -> np.ndarray:
        """(T, 128) log-mel -> (ceil(T / 8), 512) feature-stacking embeddings."""

    def encode(self, sequence: np.ndarray) -> np.ndarray:
        """(L, 512) step input -> (L * 8, 8) speaker logits."""


def stack_features(features: np.ndarray, subsampling: int = SUBSAMPLING) -> np.ndarray:
    pad = -len(features) % subsampling
    features = np.pad(features, ((0, pad), (0, 0)))
    return features.reshape(len(features) // subsampling, -1)


class NumpyEmbedder:
    """Feature stacking + bias-free projection on the host."""

    def __init__(self):
        from .features import CONSTANTS_DIR

        self.weight_t = np.ascontiguousarray(np.load(CONSTANTS_DIR / "host_weights.npz")["stacking_weight"].T)

    def __call__(self, features: np.ndarray) -> np.ndarray:
        return stack_features(features) @ self.weight_t


def pooled_probs(logits: np.ndarray, factor: int = SUBSAMPLING) -> np.ndarray:
    """Encoder-rate speaker probabilities: mean of the sigmoid over each group of `factor` frames.

    Sequential sum then division, the order of torch's avg_pool1d."""
    probs = (1.0 / (1.0 + np.exp(-logits.astype(np.float32)))).reshape(-1, factor, logits.shape[-1])
    total = probs[:, 0].copy()
    for j in range(1, factor):
        total += probs[:, j]
    return total / np.float32(factor)


class StreamingDiarizer:
    def __init__(self, backend: EncoderBackend, mode: str = "low_latency", frontend: str = "hf",
                 topk: str = "torch", cache_config: CacheConfig = CacheConfig()):
        from .features import CONSTANTS_DIR

        self.backend = backend
        self.mode = mode
        self.features = FeatureExtractor(frontend)
        self.topk = topk
        self.cache_config = cache_config
        self.silence_embeds = np.load(CONSTANTS_DIR / "host_weights.npz")["silence_embeds"]

    def run(self, audio: np.ndarray) -> Run:
        cache = SpeakerCache(self.cache_config, self.silence_embeds, topk=self.topk)
        logits_out, cache_ids, fifo_ids, step_seconds = [], [], [], []
        for chunk in chunk_schedule(len(audio), self.mode):
            t0 = time.perf_counter()
            offline_single = chunk.is_first and chunk.is_last
            feats = self.features(audio[chunk.audio_start : chunk.audio_end], center=chunk.is_first,
                                  is_last=chunk.is_last and not offline_single)
            embeds = self.backend.embed(feats)
            first_id = chunk.mel_start // SUBSAMPLING
            ids = np.arange(first_id, first_id + len(embeds))
            num_chunk = len(embeds) - chunk.num_lookahead

            cache_ids.append(cache.cache_ids.copy())
            fifo_ids.append(cache.fifo_ids.copy())
            offset = cache.num_cache + cache.num_fifo
            sequence = np.concatenate([cache.embeds(), embeds])
            logits = self.backend.encode(sequence)
            cache.update(embeds[:num_chunk], ids[:num_chunk], pooled_probs(logits))

            chunk_logits = logits[offset * SUBSAMPLING : (offset + num_chunk) * SUBSAMPLING]
            logits_out.append(chunk_logits[: len(feats)])
            step_seconds.append(time.perf_counter() - t0)
        return Run(
            logits=np.concatenate(logits_out).astype(np.float32),
            cache_ids=pad_ids(cache_ids),
            fifo_ids=pad_ids(fifo_ids),
            step_seconds=np.array(step_seconds),
        )


class TorchBackend:
    """The original HF modules as the encoder step (validates the host-side pipeline)."""

    def __init__(self, model=None):
        import torch

        if model is None:
            from .reference_hf import load_hf

            _, model = load_hf()
        self.model = model
        self.torch = torch

    def embed(self, features):
        with self.torch.inference_mode():
            x = self.torch.from_numpy(np.ascontiguousarray(features))[None]
            return self.model.model.audio_tower.embedder(x)[0].numpy()

    def encode(self, sequence):
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(sequence, dtype=np.float32))[None]
            position_ids = torch.arange(x.shape[1])[None]
            mask = torch.ones(1, x.shape[1], dtype=torch.bool)  # as the streaming forward passes it
            hidden = self.model.model(inputs_embeds=x, attention_mask=mask, position_ids=position_ids)
            return self.model.classifier(hidden.last_hidden_state)[0].numpy()
