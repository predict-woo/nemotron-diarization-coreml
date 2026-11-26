"""Runs the original HF Transformers model (fp32, CPU) in streaming mode and stores golden runs.

    python scripts/make_golden.py                 # every file in data/16k
    python scripts/make_golden.py --mode ultra_low_latency two_speakers
"""

import argparse
import time

import torch

from nemodiar import DATA_DIR, GOLDEN_DIR, STREAMING_MODES
from nemodiar.audio import load_audio
from nemodiar.reference_hf import load_hf, run_reference, to_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="file stems in data/16k (default: all)")
    parser.add_argument("--mode", default="low_latency", choices=list(STREAMING_MODES))
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    args = parser.parse_args()

    torch.set_num_threads(8)
    names = args.names or sorted(p.stem for p in DATA_DIR.glob("*.wav"))
    processor, model = load_hf(dtype=getattr(torch, args.dtype))
    out_dir = GOLDEN_DIR / f"hf_{args.mode}_{args.dtype}"
    for name in names:
        audio = load_audio(DATA_DIR / f"{name}.wav")
        t0 = time.perf_counter()
        result = run_reference(audio, processor, model, mode=args.mode)
        dt = time.perf_counter() - t0
        to_run(result).save(out_dir / f"{name}.npz")
        num_compress = sum(s.is_compressed_after for s in result.steps)
        print(f"{name:40s} {len(audio) / 16000:7.1f}s  steps={len(result.steps):4d}  "
              f"compressed_steps={num_compress:4d}  RTFx={len(audio) / 16000 / dt:5.2f}")


if __name__ == "__main__":
    main()
