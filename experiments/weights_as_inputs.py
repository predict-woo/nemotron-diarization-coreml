"""Do weights passed as inputs (instead of constants) cost a per-call copy on the GPU?"""

import time

import coremltools as ct
import numpy as np
import torch
from torch import nn

L, H, F, N = 541, 512, 2048, 31


class ConstMLP(nn.Module):
    def __init__(self, w1, w2):
        super().__init__()
        self.w1 = nn.Parameter(w1); self.w2 = nn.Parameter(w2)

    def forward(self, x):
        for i in range(N):
            x = x + torch.matmul(nn.functional.gelu(torch.matmul(x, self.w1[i])), self.w2[i])
        return x


class InputMLP(nn.Module):
    def forward(self, x, w1, w2):
        for i in range(N):
            x = x + torch.matmul(nn.functional.gelu(torch.matmul(x, w1[i])), w2[i])
        return x


w1 = torch.randn(N, H, F) * 0.02
w2 = torch.randn(N, F, H) * 0.02
x = torch.randn(1, L, H)


def bench(m, feeds, name):
    m.predict(feeds); m.predict(feeds)
    ts = []
    for _ in range(10):
        t = time.perf_counter(); m.predict(feeds); ts.append(time.perf_counter() - t)
    print(f"{name:40s} {np.median(ts) * 1e3:6.1f} ms", flush=True)


with torch.inference_mode():
    tc = torch.jit.trace(ConstMLP(w1, w2).eval(), (x,))
    ti = torch.jit.trace(InputMLP().eval(), (x, w1, w2))
common = dict(convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
              minimum_deployment_target=ct.target.macOS15, compute_units=ct.ComputeUnit.CPU_AND_GPU,
              outputs=[ct.TensorType("y", dtype=np.float32)])
mc = ct.convert(tc, inputs=[ct.TensorType("x", shape=(1, L, H), dtype=np.float32)], **common)
mi = ct.convert(ti, inputs=[ct.TensorType("x", shape=(1, L, H), dtype=np.float32),
                            ct.TensorType("w1", shape=(N, H, F), dtype=np.float32),
                            ct.TensorType("w2", shape=(N, F, H), dtype=np.float32)], **common)
bench(mc, {"x": x.numpy()}, "weights as constants (260 MB)")
bench(mi, {"x": x.numpy(), "w1": w1.numpy(), "w2": w2.numpy()}, "weights as inputs (260 MB per call)")
