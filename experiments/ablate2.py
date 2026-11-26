"""GELU cost and conv-1x1 layout for the MLP (timing only)."""

import sys

import torch
from torch import nn

sys.path.insert(0, "experiments")
from ablate import L, bench  # noqa: E402

from nemodiar.export_model import EncoderStep  # noqa: E402
from nemodiar.reference_hf import load_hf  # noqa: E402

_, hf = load_hf()
mk = lambda: EncoderStep.from_hf(hf, L, fused=True)


class NoGelu(nn.Module):
    def forward(self, x):
        return x


def layer_no_gelu(step):
    import nemodiar.export_model as em
    orig = nn.functional.gelu
    for layer in step.layers:
        def fwd(self, x, cos, sin, key_bias):
            x = x + self.self_attn(self.layer_norm1(x), cos, sin, key_bias)
            return x + self.fc2(self.fc1(self.layer_norm2(x)))
        layer.forward = fwd.__get__(layer)
    return step


class ConvMLPLayer(nn.Module):
    """fc1/fc2 as 1x1 conv2d on (1, C, 1, L)."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.c1 = nn.Conv2d(512, 2048, 1)
        self.c2 = nn.Conv2d(2048, 512, 1)
        with torch.no_grad():
            self.c1.weight.copy_(layer.fc1.weight[:, :, None, None]); self.c1.bias.copy_(layer.fc1.bias)
            self.c2.weight.copy_(layer.fc2.weight[:, :, None, None]); self.c2.bias.copy_(layer.fc2.bias)

    def forward(self, x, cos, sin, key_bias):
        lay = self.layer
        x = x + lay.self_attn(lay.layer_norm1(x), cos, sin, key_bias)
        h = lay.layer_norm2(x).transpose(1, 2)[:, :, None, :]  # (1, 512, 1, L)
        h = self.c2(nn.functional.gelu(self.c1(h)))
        return x + h[:, :, 0, :].transpose(1, 2)


if "gelu" in sys.argv:
    bench(layer_no_gelu(mk()), "GELU removed")
if "conv" in sys.argv:
    s = mk()
    s.layers = nn.ModuleList([ConvMLPLayer(layer) for layer in s.layers])
    bench(s.eval(), "MLP as conv1x1 (NCHW)")
if "tanh" in sys.argv:
    for layer in (s := mk()).layers:
        def fwd(self, x, cos, sin, key_bias):
            x = x + self.self_attn(self.layer_norm1(x), cos, sin, key_bias)
            return x + self.fc2(nn.functional.gelu(self.fc1(self.layer_norm2(x)), approximate="tanh"))
        layer.forward = fwd.__get__(layer)
    bench(s, "GELU tanh approximation (not exact)")
