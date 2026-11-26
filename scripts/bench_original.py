"""Real-time factor of the original (HF Transformers, PyTorch CPU fp32): the model card's streaming loop, untraced.

    python scripts/bench_original.py obama stress_12_5spk_300s two_speakers
"""

import argparse
import time

import torch

from nemodiar import DATA_DIR, SAMPLE_RATE
from nemodiar.audio import load_audio
from nemodiar.reference_hf import load_hf
from nemodiar.schedule import chunk_schedule


@torch.inference_mode()
def stream(audio, processor, model, mode="low_latency"):
    processor.set_streaming_mode(mode)
    cache, logits = None, []
    for chunk in chunk_schedule(len(audio), mode):
        kwargs = dict(sampling_rate=SAMPLE_RATE, is_streaming=True, is_first_audio_chunk=chunk.is_first)
        if chunk.is_last:
            kwargs["is_last_audio_chunk"] = True
        inputs = processor(audio[chunk.audio_start : chunk.audio_end], **kwargs)
        out = model(**inputs, speaker_cache=cache)
        cache = out.speaker_cache
        logits.append(out.logits)
    return torch.cat(logits, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    processor, model = load_hf()
    total_audio = total_time = 0.0
    for name in args.names:
        audio = load_audio(DATA_DIR / f"{name}.wav")
        t0 = time.perf_counter()
        stream(audio, processor, model)
        dt = time.perf_counter() - t0
        dur = len(audio) / SAMPLE_RATE
        total_audio, total_time = total_audio + dur, total_time + dt
        print(f"{name:32s} {dur:7.1f}s  {dt:7.2f}s  RTFx {dur / dt:5.2f}", flush=True)
    print(f"overall RTFx {total_audio / total_time:.2f} ({total_audio:.1f}s audio in {total_time:.2f}s)")


if __name__ == "__main__":
    main()
