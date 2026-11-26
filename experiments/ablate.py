"""Timing ablations of the encoder step on the GPU (not exact variants: only for locating time)."""

import sys
import time

import coremltools as ct
import numpy as np
import torch
from torch import nn

from nemodiar.export_model import EncoderStep, FusedAttention, key_bias
from nemodiar.reference_hf import load_hf

L = 541


def bench(module, name, masked=True, iters=12):
    x = torch.randn(1, L, 512)
    kb = key_bias(L - 20, L)
    args = (x, kb) if masked else (x,)
    with torch.inference_mode():
        traced = torch.jit.trace(module, args)
    inputs = [ct.TensorType("sequence", shape=(1, L, 512), dtype=np.float32)]
    if masked:
        inputs.append(ct.TensorType("key_bias", shape=(1, 1, 1, L), dtype=np.float32))
    m = ct.convert(traced, inputs=inputs, outputs=[ct.TensorType("logits", dtype=np.float32)],
                   convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
                   minimum_deployment_target=ct.target.macOS15, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    feeds = {"sequence": x.numpy()}
    if masked:
        feeds["key_bias"] = kb.numpy()
    m.predict(feeds); m.predict(feeds)
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); m.predict(feeds); ts.append(time.perf_counter() - t)
    print(f"{name:52s} {np.median(ts) * 1e3:6.1f} ms", flush=True)
    return m


class NoMaskStep(nn.Module):
    def __init__(self, s):
        super().__init__(); self.s = s
    def forward(self, x):
        return self.s(x)


def patch_attention(step, fn):
    for layer in step.layers:
        layer.self_attn.forward = fn.__get__(layer.self_attn)
    return step


def attn_no_softmax(self, x, cos, sin_signed, key_bias):
    h, d, b = self.heads, self.head_dim, self.batch
    qkv = self.qkv_proj(x).reshape(b, -1, 3 * h, d).transpose(1, 2)
    qk, v = qkv[:, : 2 * h], qkv[:, 2 * h :]
    q, k = qk[:, :h], qk[:, h:]
    scores = torch.matmul(q, k.transpose(2, 3))
    out = torch.matmul(scores, v)
    return self.o_proj(out.transpose(1, 2).reshape(b, -1, h * d))


def attn_no_rope(self, x, cos, sin_signed, key_bias):
    h, d, b = self.heads, self.head_dim, self.batch
    qkv = self.qkv_proj(x).reshape(b, -1, 3 * h, d).transpose(1, 2)
    qk, v = qkv[:, : 2 * h], qkv[:, 2 * h :]
    q, k = qk[:, :h], qk[:, h:]
    scores = torch.matmul(q, k.transpose(2, 3)) + key_bias
    out = torch.matmul(torch.softmax(scores, dim=-1), v)
    return self.o_proj(out.transpose(1, 2).reshape(b, -1, h * d))


def attn_manual_softmax(self, x, cos, sin_signed, key_bias):
    h, d, b = self.heads, self.head_dim, self.batch
    qkv = self.qkv_proj(x).reshape(b, -1, 3 * h, d).transpose(1, 2)
    qk, v = qkv[:, : 2 * h], qkv[:, 2 * h :]
    swapped = torch.flip(qk.reshape(b, 2 * h, -1, 2, d // 2), dims=[3]).reshape(b, 2 * h, -1, d)
    qk = qk * cos + swapped * sin_signed
    q, k = qk[:, :h], qk[:, h:]
    scores = torch.matmul(q, k.transpose(2, 3)) + key_bias
    e = torch.exp(scores - scores.amax(dim=-1, keepdim=True))
    out = torch.matmul(e, v) / e.sum(dim=-1, keepdim=True)  # normalise after AV: L x d instead of L x L
    return self.o_proj(out.transpose(1, 2).reshape(b, -1, h * d))


def attn_k_pretransposed(self, x, cos, sin_signed, key_bias):
    """K produced directly in (d, L) layout from a separate projection, so Q @ K needs no transpose."""
    h, d, b = self.heads, self.head_dim, self.batch
    qkv = self.qkv_proj(x).reshape(b, -1, 3 * h, d).transpose(1, 2)
    qk, v = qkv[:, : 2 * h], qkv[:, 2 * h :]
    swapped = torch.flip(qk.reshape(b, 2 * h, -1, 2, d // 2), dims=[3]).reshape(b, 2 * h, -1, d)
    qk = qk * cos + swapped * sin_signed
    q, k = qk[:, :h], qk[:, h:]
    kt = k.transpose(2, 3).contiguous()
    scores = torch.matmul(q, kt) + key_bias
    out = torch.matmul(torch.softmax(scores, dim=-1), v)
    return self.o_proj(out.transpose(1, 2).reshape(b, -1, h * d))


if __name__ == "__main__":
    which = sys.argv[1:] or ["base", "nomask", "nosoftmax", "norope", "manualsoftmax", "conv"]
    _, hf = load_hf()
    mk = lambda: EncoderStep.from_hf(hf, L, fused=True)
    if "base" in which:
        bench(mk(), "fused (current)")
    if "nomask" in which:
        bench(NoMaskStep(mk()).eval(), "no key mask (range-like, fixed shape)", masked=False)
    if "nosoftmax" in which:
        bench(patch_attention(mk(), attn_no_softmax), "no RoPE, no mask, no softmax")
    if "norope" in which:
        bench(patch_attention(mk(), attn_no_rope), "no RoPE")
    if "manualsoftmax" in which:
        bench(patch_attention(mk(), attn_manual_softmax), "manual softmax, normalise after AV")
    if "ktrans" in which:
        bench(patch_attention(mk(), attn_k_pretransposed), "K transposed explicitly")
