"""Step latency of fixed-length fused fp32 models across lengths (alignment effects)."""

import sys
import time

import coremltools as ct
import numpy as np
import torch

from nemodiar.export_model import EncoderStep, key_bias
from nemodiar.reference_hf import load_hf

_, hf = load_hf()
for L in [int(x) for x in sys.argv[1].split(",")]:
    step = EncoderStep.from_hf(hf, L, fused=True)
    x, kb = torch.randn(1, L, 512), key_bias(L - 5, L)
    with torch.inference_mode():
        traced = torch.jit.trace(step, (x, kb))
    m = ct.convert(traced, inputs=[ct.TensorType("sequence", shape=(1, L, 512), dtype=np.float32),
                                   ct.TensorType("key_bias", shape=(1, 1, 1, L), dtype=np.float32)],
                   outputs=[ct.TensorType("logits", dtype=np.float32)], convert_to="mlprogram",
                   compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.macOS15,
                   compute_units=ct.ComputeUnit.CPU_AND_GPU)
    feeds = {"sequence": x.numpy(), "key_bias": kb.numpy()}
    m.predict(feeds); m.predict(feeds)
    ts = []
    for _ in range(15):
        t = time.perf_counter(); m.predict(feeds); ts.append(time.perf_counter() - t)
    print(f"L={L:4d}  {np.median(ts) * 1e3:6.2f} ms   ({np.median(ts) * 1e3 / L * 1000:.1f} us/token)", flush=True)
