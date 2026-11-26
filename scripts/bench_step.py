"""Latency of one encoder step per model / compute unit / step length."""

import argparse
import time

import numpy as np

from nemodiar.coreml_backend import CoreMLBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--units", default="cpu_and_gpu")
    parser.add_argument("--lengths", default="40,320,430,541")
    parser.add_argument("--iters", type=int, default=15)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    lengths = [int(x) for x in args.lengths.split(",")]
    print(f"{'model':42s} {'units':12s} {'load':>6s} " + " ".join(f"{'L=' + str(n):>9s}" for n in lengths))
    for model in args.models:
        for units in args.units.split(","):
            t0 = time.perf_counter()
            backend = CoreMLBackend(model, units)
            load = time.perf_counter() - t0
            cells = []
            for n in lengths:
                seq = rng.normal(size=(n, 512)).astype(np.float32)
                backend.encode(seq)
                backend.encode(seq)
                times = []
                for _ in range(args.iters):
                    t = time.perf_counter()
                    backend.encode(seq)
                    times.append(time.perf_counter() - t)
                cells.append(f"{np.median(times) * 1e3:7.1f}ms")
            print(f"{model[-42:]:42s} {units:12s} {load:5.1f}s " + " ".join(f"{c:>9s}" for c in cells))


if __name__ == "__main__":
    main()
