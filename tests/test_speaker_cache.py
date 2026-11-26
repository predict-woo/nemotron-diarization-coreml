"""Speaker-cache compression of the NumPy port (mirrored by the Swift runtime) against the original's `_compress`."""

import math

import numpy as np
import pytest
import torch

from nemodiar.cache import CacheConfig, SpeakerCache


def tie_heavy_probs(rng, num_frames: int, num_speakers: int = 8) -> np.ndarray:
    """Probabilities with many saturated values (exactly 0 / 1 in float32), the source of exactly tied scores."""
    logits = rng.normal(0, 6, size=(num_frames, num_speakers))
    dominant = rng.integers(0, num_speakers, num_frames)
    logits[np.arange(num_frames), dominant] += rng.choice([0.0, 30.0], num_frames, p=[0.4, 0.6])
    logits[rng.random((num_frames, num_speakers)) < 0.4] = -40.0
    return (1 / (1 + np.exp(-logits))).astype(np.float32)


@pytest.mark.parametrize("topk", ["torch", "cpp"])
def test_compression_matches_original(hf, topk):
    from transformers.models.nemotron3_diarization.modeling_nemotron3_diarization import (
        Nemotron3DiarizationSpeakerCache,
    )

    _, model = hf
    original = Nemotron3DiarizationSpeakerCache(model.config.streaming_config)
    port = SpeakerCache(CacheConfig(), np.zeros(512, np.float32), topk=topk)
    rng = np.random.default_rng(0)
    for trial in range(40):
        num_frames = int(rng.integers(265, 600))
        probs = tie_heavy_probs(rng, num_frames)
        # embeddings that encode their own row index, so the gathered rows reveal the selection
        embeds = torch.zeros(1, num_frames, 512)
        embeds[0, :, 0] = torch.arange(num_frames, dtype=torch.float32)
        silence = torch.full((512,), -1.0)
        kept, kept_probs = original._compress(embeds, torch.from_numpy(probs)[None], silence)
        expected = kept[0, :, 0].numpy()
        expected = np.where(expected < 0, num_frames, expected).astype(np.int64)  # silence row -> num_frames
        got = port.compression_indices(probs)
        np.testing.assert_array_equal(got, expected, err_msg=f"trial {trial}")
        np.testing.assert_array_equal(np.concatenate([probs, np.zeros((1, 8), np.float32)])[got], kept_probs[0].numpy())


def test_boost_constants_match_original(hf):
    _, model = hf
    cfg = model.config.streaming_config
    port = SpeakerCache(CacheConfig(), np.zeros(512, np.float32))
    budget = cfg.speaker_cache_length // cfg.num_speakers - cfg.speaker_cache_silence_frames_per_speaker
    assert port.min_positive_scores == math.floor(budget * cfg.min_positive_scores_rate)
    assert port.num_strong_boosted == math.floor(budget * cfg.strong_boost_rate)
    assert port.num_weak_boosted == math.floor(budget * cfg.weak_boost_rate)
    assert CacheConfig().fifo_length == cfg.fifo_length
    assert CacheConfig().speaker_cache_update_period == cfg.speaker_cache_update_period
