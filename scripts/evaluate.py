"""Runs a pipeline backend over the test set and compares every run against the golden (original) runs.

    python scripts/evaluate.py --backend torch
    python scripts/evaluate.py --backend coreml --model build/encoder_fp32.mlpackage --compute-units cpu_and_gpu
"""

import argparse
import json
import time

import numpy as np

from nemodiar import DATA_DIR, GOLDEN_DIR
from nemodiar.audio import load_audio
from nemodiar.backends import make_backend
from nemodiar.compare import compare_runs
from nemodiar.golden import Run
from nemodiar.pipeline import StreamingDiarizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--backend", default="torch")
    parser.add_argument("--model", default=None)
    parser.add_argument("--compute-units", default="cpu_and_gpu")
    parser.add_argument("--topk", default="torch", choices=["torch", "stable", "cpp"])
    parser.add_argument("--mode", default="low_latency")
    parser.add_argument("--golden", default=None, help="golden dir name (default hf_<mode>_float32)")
    parser.add_argument("--json", default=None, help="write the per-file metrics here")
    args = parser.parse_args()

    golden_dir = GOLDEN_DIR / (args.golden or f"hf_{args.mode}_float32")
    names = args.names or sorted(p.stem for p in golden_dir.glob("*.npz"))
    backend = make_backend(args.backend, model=args.model, compute_units=args.compute_units)
    diarizer = StreamingDiarizer(backend, mode=args.mode, topk=args.topk)

    header = (f"{'file':34s} {'dur':>6s} {'RTFx':>6s} {'max|dp|':>9s} {'mean|dp|':>9s} {'p99.9':>9s} "
              f"{'dec':>4s} {'segs':>5s} {'state':>6s}")
    print(header)
    results, total_audio, total_time = {}, 0.0, 0.0
    for name in names:
        audio = load_audio(DATA_DIR / f"{name}.wav")
        ref = Run.load(golden_dir / f"{name}.npz")
        t0 = time.perf_counter()
        cand = diarizer.run(audio)
        dt = time.perf_counter() - t0
        c = compare_runs(ref, cand)
        dur = len(audio) / 16000
        total_audio, total_time = total_audio + dur, total_time + dt
        state = "same" if c.state_identical else f"@{c.first_state_divergence}"
        print(f"{name[:34]:34s} {dur:6.1f} {dur / dt:6.1f} {c.max_abs_prob:9.2e} {c.mean_abs_prob:9.2e} "
              f"{c.p999_abs_prob:9.2e} {c.decision_mismatch_frames:4d} "
              f"{'same' if c.segments_identical else 'DIFF':>5s} {state:>6s}")
        results[name] = c.as_dict() | {"rtfx": dur / dt}
    print(f"overall RTFx (incl. Python host logic): {total_audio / total_time:.1f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
