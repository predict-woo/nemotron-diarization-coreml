"""Timing ablations of the Neural Engine formulation at one bucket length (not exact: for locating ANE time)."""

import sys
import time

import coremltools as ct
import numpy as np
import torch
from torch import nn

from nemodiar.ane_model import ANEAttention, ANEEncoderStep, ane_inputs
from nemodiar.export_model import EncoderStep
from nemodiar.reference_hf import load_hf

L = int(sys.argv[2]) if len(sys.argv) > 2 else 544
_, hf = load_hf()


def bench(module, name):
    example = ane_inputs(torch.randn(L - 5, 512), L)
    with torch.inference_mode():
        traced = torch.jit.trace(module.eval(), example)
    fp16 = np.float16
    m = ct.convert(traced, inputs=[ct.TensorType("sequence", shape=(1, 512, 1, L), dtype=fp16),
                                   ct.TensorType("key_bias", shape=(1, L, 1, 1), dtype=fp16),
                                   ct.TensorType("valid", shape=(1, 1, 1, L), dtype=fp16)],
                   outputs=[ct.TensorType("logits", dtype=fp16)], convert_to="mlprogram",
                   compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15,
                   compute_units=ct.ComputeUnit.CPU_AND_NE)
    feeds = {k: v.numpy().astype(fp16) for k, v in zip(["sequence", "key_bias", "valid"], example)}
    m.predict(feeds); m.predict(feeds)
    ts = []
    for _ in range(15):
        t = time.perf_counter(); m.predict(feeds); ts.append(time.perf_counter() - t)
    print(f"{name:48s} {np.median(ts) * 1e3:6.1f} ms", flush=True)


def make():
    return ANEEncoderStep(EncoderStep.from_hf(hf, L, fused=True))


def patch(step, fn):
    for layer in step.layers:
        layer.attn.forward = fn.__get__(layer.attn)
    return step


def no_core(self, x, cos, sin_signed, key_bias):  # projections only
    return self.o(self.v(x) + self.q(x) + self.k(x))


def no_rope(self, x, cos, sin_signed, key_bias):
    q, k, v = self.q(x), self.k(x), self.v(x)
    d = self.head_dim
    outs = []
    for qh, kh, vh in zip(q.split(d, dim=1), k.transpose(1, 3).split(d, dim=3), v.split(d, dim=1)):
        w = (torch.einsum("bchq,bkhc->bkhq", qh, kh) + key_bias).softmax(dim=1)
        outs.append(torch.einsum("bkhq,bchk->bchq", w, vh))
    return self.o(torch.cat(outs, dim=1))


which = sys.argv[1].split(",")
if "full" in which:
    bench(make(), "full")
if "nocore" in which:
    bench(patch(make(), no_core), "attention core removed (projections only)")
if "norope" in which:
    bench(patch(make(), no_rope), "RoPE removed")
if "nomlp" in which:
    s = make()
    for layer in s.layers:
        layer.forward = (lambda self, x, c, si, kb: x + self.attn(self.norm1(x), c, si, kb)).__get__(layer)
    bench(s, "MLP removed")


# ---- exact reformulations -------------------------------------------------------------------------------------

def permuted(conv: nn.Conv2d, head_dim: int) -> torch.Tensor:
    """Weights whose output is the input projection with the two halves of every head swapped."""
    w = conv.weight.detach()
    idx = torch.arange(w.shape[0]).view(-1, 2, head_dim // 2).flip(1).reshape(-1)
    return w[idx]


class RopeWeights(nn.Module):
    """q|q_swapped and k|k_swapped from two convs with stacked weights: RoPE without split/concat."""

    def __init__(self, attn: ANEAttention, batched: bool):
        super().__init__()
        self.a, self.batched = attn, batched
        d = attn.head_dim
        self.qq = nn.Conv2d(512, 1024, 1, bias=False)
        self.kk = nn.Conv2d(512, 1024, 1, bias=False)
        with torch.no_grad():
            self.qq.weight.copy_(torch.cat([attn.q.weight, permuted(attn.q, d)]))
            self.kk.weight.copy_(torch.cat([attn.k.weight, permuted(attn.k, d)]))

    def forward(self, x, cos, sin_signed, key_bias):
        a, d, h = self.a, self.a.head_dim, self.a.heads
        qq, kk = self.qq(x), self.kk(x)
        q = qq[:, :512] * cos + qq[:, 512:] * sin_signed
        k = kk[:, :512] * cos + kk[:, 512:] * sin_signed
        v = a.v(x)
        S = x.shape[3]
        if self.batched:
            qh = q.reshape(1, h, d, S)
            kh = k.reshape(1, h, d, S)
            vh = v.reshape(1, h, d, S)
            w = torch.matmul(kh.transpose(2, 3), qh) + key_bias.reshape(1, 1, S, 1)  # (1, h, S_k, S_q)
            w = w.softmax(dim=2)
            out = torch.matmul(vh, w).reshape(1, h * d, 1, S)
            return a.o(out)
        outs = []
        for qh, kh, vh in zip(q.split(d, dim=1), k.transpose(1, 3).split(d, dim=3), v.split(d, dim=1)):
            w = (torch.einsum("bchq,bkhc->bkhq", qh, kh) + key_bias).softmax(dim=1)
            outs.append(torch.einsum("bkhq,bchk->bchq", w, vh))
        return a.o(torch.cat(outs, dim=1))


def with_rope_weights(batched):
    s = make()
    for layer in s.layers:
        layer.attn = RopeWeights(layer.attn, batched)
    return s


if "ropeweights" in which:
    bench(with_rope_weights(False), "RoPE via stacked weights (per-head einsum)")
if "batched" in which:
    bench(with_rope_weights(True), "RoPE via stacked weights + batched heads")
