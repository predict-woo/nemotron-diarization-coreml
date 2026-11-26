"""Effect of CoreML optimization hints on load time and step latency (GPU, fp32)."""

import sys
import time

import coremltools as ct
import numpy as np


def feeds_for(model_path, n, layout):
    x = np.random.default_rng(n).normal(size=(1, n, 512)).astype(np.float32)
    if layout == "range":
        return {"sequence": x}
    L = 541
    seq = np.zeros((1, L, 512), np.float32); seq[0, :n] = x[0]
    kb = np.full((1, 1, 1, L), -1e4, np.float32); kb[..., :n] = 0
    return {"sequence": seq, "key_bias": kb}


def run(path, layout, hints, label):
    t0 = time.perf_counter()
    m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_GPU, optimization_hints=hints)
    load = time.perf_counter() - t0
    out = []
    lengths = [322, 430, 541] if layout == "range" else [541]
    for n in lengths:
        f = feeds_for(path, n, layout)
        t = time.perf_counter(); m.predict(f); first = time.perf_counter() - t
        m.predict(f)
        ts = []
        for _ in range(10):
            t = time.perf_counter(); m.predict(f); ts.append(time.perf_counter() - t)
        out.append(f"L={n}: {np.median(ts)*1e3:.1f}ms (first {first*1e3:.0f})")
    if layout == "range":  # alternate shapes: re-specialisation cost
        fs = [feeds_for(path, n, layout) for n in (325, 433, 520)]
        ts = []
        for i in range(9):
            t = time.perf_counter(); m.predict(fs[i % 3]); ts.append(time.perf_counter() - t)
        out.append(f"alternating new shapes: {' '.join(f'{x*1e3:.0f}' for x in ts)}")
    print(f"{label:44s} load {load:5.1f}s  " + "  ".join(out), flush=True)


if __name__ == "__main__":
    path, layout = sys.argv[1], sys.argv[2]
    RF, SS = ct.ReshapeFrequency, ct.SpecializationStrategy
    configs = {
        "default": None,
        "fastPrediction": {"specializationStrategy": SS.FastPrediction},
        "reshape frequent": {"reshapeFrequency": RF.Frequent},
        "reshape infrequent": {"reshapeFrequency": RF.Infrequent},
        "infrequent + fastPrediction": {"reshapeFrequency": RF.Infrequent, "specializationStrategy": SS.FastPrediction},
    }
    for name in sys.argv[3].split(","):
        run(path, layout, configs[name], f"{layout}: {name}")
