"""The original: HF Transformers Nemotron-3-Diarization run in streaming mode, plus speaker-cache tracking.

`run_reference` streams a recording chunk by chunk exactly as the model card does, with the model's own
`Nemotron3DiarizationSpeakerCache`. `TrackedSpeakerCache` subclasses it without changing its arithmetic, and
records which global encoder frame (or the silence embedding, id -1) sits in each cache/FIFO slot, so a run's whole
state trajectory can be compared exactly between implementations.
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from . import HF_MODEL_DIR, SAMPLE_RATE
from .schedule import SUBSAMPLING, chunk_schedule

SILENCE_ID = -1


def load_hf(dtype=torch.float32, attn_implementation: str = "sdpa"):
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor

    processor = AutoProcessor.from_pretrained(HF_MODEL_DIR)
    model = AutoModelForAudioFrameClassification.from_pretrained(
        HF_MODEL_DIR, dtype=dtype, attn_implementation=attn_implementation
    ).eval()
    return processor, model


def make_tracked_cache_class():
    from transformers.models.nemotron3_diarization.modeling_nemotron3_diarization import (
        Nemotron3DiarizationSpeakerCache,
    )

    class TrackedSpeakerCache(Nemotron3DiarizationSpeakerCache):
        """Model's own speaker cache; additionally tracks the global encoder-frame id of every slot."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cache_ids = np.zeros(0, dtype=np.int64)
            self.fifo_ids = np.zeros(0, dtype=np.int64)
            self.pending_chunk_ids: np.ndarray | None = None
            self._frame_indices: np.ndarray | None = None

        def update(self, chunk_input_embeds, chunk_logits, silence_embeds, num_chunk_frames, mask=None,
                   num_lookback_frames=0):
            num_cache, num_fifo = self.num_cache_frames, self.num_fifo_frames
            fifo_ids = np.concatenate([self.fifo_ids[:num_fifo], self.pending_chunk_ids[:num_chunk_frames]])
            num_popped = self._num_popped_frames(len(fifo_ids))
            self._frame_indices = None
            super().update(chunk_input_embeds, chunk_logits, silence_embeds, num_chunk_frames, mask,
                           num_lookback_frames)
            if num_popped:
                cache_ids = np.concatenate([self.cache_ids[:num_cache], fifo_ids[:num_popped]])
                if self._frame_indices is not None:
                    cache_ids = np.concatenate([cache_ids, [SILENCE_ID]])[self._frame_indices]
                self.cache_ids = cache_ids
                fifo_ids = fifo_ids[num_popped:]
            self.fifo_ids = fifo_ids
            assert len(self.cache_ids) == self.num_cache_frames and len(self.fifo_ids) == self.num_fifo_frames

        # verbatim copy of the parent `_compress`, recording `frame_indices` (batch 0)
        def _compress(self, embeds, probs, silence_embeds):
            batch_size, num_frames, num_speakers = probs.shape

            scores = self._get_frame_scores(probs)
            scores[:, self.speaker_cache_length :] += self.latest_frames_score_boost

            scores = self._boost_scores(scores, self.num_strong_boosted_frames, boost=-2.0 * math.log(0.5))
            scores = self._boost_scores(scores, self.num_weak_boosted_frames, boost=-math.log(0.5))
            scores = nn.functional.pad(scores, (0, 0, 0, self.num_silence_frames), value=float("inf"))
            embeds = torch.cat([embeds, silence_embeds.to(embeds).view(1, 1, -1).expand(batch_size, 1, -1)], dim=1)
            probs = nn.functional.pad(probs, (0, 0, 0, 1))

            num_scored_frames = num_frames + self.num_silence_frames
            sentinel = num_scored_frames * num_speakers
            flat_scores = scores.transpose(1, 2).reshape(batch_size, -1)

            topk_scores, topk_indices = torch.topk(flat_scores, self.speaker_cache_length, dim=1, sorted=False)
            topk_indices = topk_indices.masked_fill(topk_scores == float("-inf"), sentinel)
            topk_indices, _ = torch.sort(topk_indices, dim=1)
            frame_indices = torch.where(
                topk_indices == sentinel, num_frames, (topk_indices % num_scored_frames).clamp(max=num_frames)
            )
            self._frame_indices = frame_indices[0].cpu().numpy()

            batch_indices = torch.arange(batch_size, device=probs.device)[:, None]
            return (
                embeds[batch_indices.to(embeds.device), frame_indices.to(embeds.device)],
                probs[batch_indices, frame_indices],
            )

    return TrackedSpeakerCache


@dataclass
class StepRecord:
    mel_start: int
    num_input_frames: int  # encoder frames of the chunk input (chunk + look-ahead)
    num_lookahead: int
    cache_ids: np.ndarray  # before the step
    fifo_ids: np.ndarray  # before the step
    is_compressed_after: bool
    elapsed_s: float


@dataclass
class ReferenceResult:
    logits: np.ndarray  # (num_mel_frames, num_speakers), float32
    steps: list[StepRecord] = field(default_factory=list)
    embeds_by_id: dict[int, np.ndarray] = field(default_factory=dict)  # pre-encode embedding of each frame id
    silence_embeds: np.ndarray | None = None

    @property
    def probs(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logits.astype(np.float64)))


@torch.inference_mode()
def run_reference(audio: np.ndarray, processor, model, mode: str = "low_latency", check_state: bool = True,
                  progress: bool = False) -> ReferenceResult:
    """Streams `audio` through the HF model as in the model card, recording the speaker-cache trajectory."""
    processor.set_streaming_mode(mode)
    tracked_cls = make_tracked_cache_class()
    dtype = next(model.parameters()).dtype
    result = ReferenceResult(logits=np.zeros((0, 8), np.float32))
    result.silence_embeds = model.silence_embeds.detach().float().cpu().numpy()

    # hook the embedder to capture each chunk's pre-encode embeddings, keyed by global frame id
    captured = {}
    hook = model.model.audio_tower.embedder.register_forward_hook(lambda m, i, o: captured.__setitem__("e", o))

    cache = None
    logits = []
    schedule = chunk_schedule(len(audio), mode)
    try:
        for chunk in schedule:
            kwargs = dict(sampling_rate=SAMPLE_RATE, is_streaming=True, is_first_audio_chunk=chunk.is_first)
            if chunk.is_last:
                kwargs["is_last_audio_chunk"] = True
            if chunk.is_first and chunk.is_last:  # shorter than one chunk: plain offline call
                kwargs = dict(sampling_rate=SAMPLE_RATE)
            inputs = processor(audio[chunk.audio_start : chunk.audio_end], **kwargs).to(dtype=dtype)

            first_id = chunk.mel_start // SUBSAMPLING
            num_input = math.ceil(inputs["input_features"].shape[1] / SUBSAMPLING)
            if cache is None:
                cache = tracked_cls(model.config.streaming_config)
            cache.pending_chunk_ids = np.arange(first_id, first_id + num_input)
            cache_ids, fifo_ids = cache.cache_ids.copy(), cache.fifo_ids.copy()

            t0 = time.perf_counter()
            if chunk.is_first and chunk.is_last:
                out = model(**inputs)
            else:
                out = model(**inputs, speaker_cache=cache)
                cache = out.speaker_cache
            elapsed = time.perf_counter() - t0

            embeds = captured["e"][0].float().cpu().numpy()
            # the FIFO keeps a frame's embedding from the step that scores it, not from its look-ahead step
            num_scored = num_input - chunk.num_lookahead
            for j, frame_id in enumerate(range(first_id, first_id + num_input)):
                if j < num_scored:
                    result.embeds_by_id[frame_id] = embeds[j]
                else:
                    result.embeds_by_id.setdefault(frame_id, embeds[j])
            logits.append(out.logits[0].float().cpu().numpy())
            result.steps.append(
                StepRecord(chunk.mel_start, num_input, chunk.num_lookahead, cache_ids, fifo_ids,
                           bool(cache.is_compressed), elapsed)
            )
            if check_state and cache.is_initialized:
                _check_tracked_state(cache, result)
            if progress and chunk.index % 50 == 0:
                print(f"  step {chunk.index}/{len(schedule)} cache={cache.num_cache_frames} "
                      f"fifo={cache.num_fifo_frames} {elapsed * 1e3:.0f} ms")
    finally:
        hook.remove()
    result.logits = np.concatenate(logits, axis=0)
    return result


def _check_tracked_state(cache, result: ReferenceResult):
    """The tracked ids must reproduce the model's cache and FIFO contents exactly."""
    table = lambda ids: np.stack(
        [result.silence_embeds if i == SILENCE_ID else result.embeds_by_id[int(i)] for i in ids]
    ) if len(ids) else np.zeros((0, 512), np.float32)
    got_cache = cache.embeds[0, : cache.num_cache_frames].float().cpu().numpy()
    got_fifo = cache.fifo[0, : cache.num_fifo_frames].float().cpu().numpy()
    np.testing.assert_array_equal(got_cache, table(cache.cache_ids))
    np.testing.assert_array_equal(got_fifo, table(cache.fifo_ids))


def to_run(result: ReferenceResult):
    from .golden import Run, pad_ids

    return Run(
        logits=result.logits.astype(np.float32),
        cache_ids=pad_ids([s.cache_ids for s in result.steps]),
        fifo_ids=pad_ids([s.fifo_ids for s in result.steps]),
        step_seconds=np.array([s.elapsed_s for s in result.steps]),
    )
