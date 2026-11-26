"""Streams recordings through a Python pipeline backend and saves each run as a .npy directory
(the layout of the Swift CLI's --out, readable by scripts/compare_runs.py and scripts/der_report.py).

    python scripts/run_backend.py --backend coreml --model build/encoder_ane_576.mlpackage --compute-units cpu_and_ne --out runs/ane_py
"""

import argparse
import time
from pathlib import Path

import numpy as np

from nemodiar import DATA_DIR
from nemodiar.audio import load_audio
from nemodiar.backends import make_backend
from nemodiar.pipeline import StreamingDiarizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--backend", default="coreml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--compute-units", default="cpu_and_gpu")
    parser.add_argument("--mode", default="low_latency")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    names = args.names or sorted(p.stem for p in DATA_DIR.glob("*.wav"))
    diarizer = StreamingDiarizer(make_backend(args.backend, model=args.model, compute_units=args.compute_units),
                                 mode=args.mode, topk="cpp")
    total_audio = total_time = 0.0
    for name in names:
        audio = load_audio(DATA_DIR / f"{name}.wav")
        t0 = time.perf_counter()
        run = diarizer.run(audio)
        dt = time.perf_counter() - t0
        d = args.out / name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "logits.npy", run.logits)
        np.save(d / "cache_ids.npy", run.cache_ids)
        np.save(d / "fifo_ids.npy", run.fifo_ids)
        np.save(d / "step_seconds.npy", run.step_seconds)
        total_audio, total_time = total_audio + len(audio) / 16000, total_time + dt
        print(f"{name:34s} RTFx {len(audio) / 16000 / dt:5.1f}", flush=True)
    print(f"overall RTFx {total_audio / total_time:.1f}")


if __name__ == "__main__":
    main()
