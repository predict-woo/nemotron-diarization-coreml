"""The NumPy front end (ported to Swift) against the HF processor, chunk by chunk."""

import numpy as np
import pytest

from nemodiar.audio import load_audio
from nemodiar.features import FeatureExtractor
from nemodiar.schedule import chunk_schedule

from .conftest import audio_path


@pytest.mark.parametrize("name", ["en-Alice_woman", "librispeech_mr_quilter", "mandarin_voxcpm_zh"])
@pytest.mark.parametrize("mode", ["low_latency", "ultra_low_latency"])
def test_features_match_processor(hf, name, mode):
    processor, _ = hf
    processor.set_streaming_mode(mode)
    audio = load_audio(audio_path(name))
    fe = FeatureExtractor("hf")
    worst = 0.0
    for chunk in chunk_schedule(len(audio), mode):
        kwargs = dict(sampling_rate=16000, is_streaming=True, is_first_audio_chunk=chunk.is_first)
        if chunk.is_last:
            kwargs["is_last_audio_chunk"] = True
        ref = processor(audio[chunk.audio_start : chunk.audio_end], **kwargs)["input_features"][0].numpy()
        got = fe(audio[chunk.audio_start : chunk.audio_end], center=chunk.is_first, is_last=chunk.is_last)
        assert got.shape == ref.shape, f"chunk {chunk.index}: {got.shape} vs {ref.shape}"
        worst = max(worst, float(np.abs(got - ref).max()))
    processor.set_streaming_mode("low_latency")
    # float32 FFT rounding, visible only in near-silent bins through the log
    assert worst < 2e-3


def test_schedule_matches_model_card_generator(hf):
    processor, _ = hf
    n = 16000 * 7 + 123
    chunks = chunk_schedule(n, "low_latency")
    assert chunks[0].audio_end == processor.num_samples_first_audio_chunk
    mel = processor.num_mel_frames_per_step
    for c in chunks[1:-1]:
        assert c.audio_start == processor.audio_chunk_start(mel)
        assert c.audio_end - c.audio_start == processor.num_samples_per_audio_chunk
        mel += processor.num_mel_frames_per_step
    assert chunks[-1].is_last and chunks[-1].audio_end == n
