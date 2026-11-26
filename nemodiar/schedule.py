"""Audio chunk schedule of a streaming session, identical to the generator of the model card / HF processor.

Every chunk but the last carries `chunk_length + right_context` encoder frames, the trailing `right_context`
ones being look-ahead only; consecutive chunks overlap by those look-ahead frames. The last chunk holds whatever
is left and is scored entirely.
"""

from dataclasses import dataclass

from . import STREAMING_MODES

HOP = 160
WIN = 400
N_FFT = 512
SUBSAMPLING = 8


@dataclass(frozen=True)
class AudioChunk:
    index: int
    audio_start: int  # first sample fed to the feature extractor (may precede the chunk's first frame)
    audio_end: int  # exclusive, None-like value len(audio) for the last chunk
    mel_start: int  # global index of the chunk's first mel frame
    is_first: bool
    is_last: bool
    num_lookahead: int  # look-ahead encoder frames, 0 for the last chunk


def num_samples_first_chunk(mode: str) -> int:
    chunk, rc = STREAMING_MODES[mode]
    return ((chunk + rc) * SUBSAMPLING - 1) * HOP + WIN // 2


def num_samples_per_chunk(mode: str) -> int:
    chunk, rc = STREAMING_MODES[mode]
    return (chunk + rc) * SUBSAMPLING * HOP + WIN


def audio_chunk_start(mel_frame_idx: int) -> int:
    return mel_frame_idx * HOP - N_FFT // 2


def chunk_schedule(num_samples: int, mode: str = "low_latency") -> list[AudioChunk]:
    chunk, rc = STREAMING_MODES[mode]
    first = num_samples_first_chunk(mode)
    if num_samples < first:
        # shorter than one chunk: a single centered chunk, every frame scored
        return [AudioChunk(0, 0, num_samples, 0, True, True, 0)]

    chunks = [AudioChunk(0, 0, first, 0, True, False, rc)]
    mel_idx = chunk * SUBSAMPLING
    start = audio_chunk_start(mel_idx)
    per_chunk = num_samples_per_chunk(mode)
    while (end := start + per_chunk) <= num_samples:
        chunks.append(AudioChunk(len(chunks), start, end, mel_idx, False, False, rc))
        mel_idx += chunk * SUBSAMPLING
        start = audio_chunk_start(mel_idx)
    chunks.append(AudioChunk(len(chunks), start, num_samples, mel_idx, False, True, 0))
    return chunks
