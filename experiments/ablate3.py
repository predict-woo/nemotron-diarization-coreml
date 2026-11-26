"""Cost of the head and of the last layer (timing only), bucket length 576."""

import sys

import torch
from torch import nn

sys.path.insert(0, "experiments")
import ablate  # noqa: E402

ablate.L = 576
from ablate import bench  # noqa: E402

from nemodiar.export_model import EncoderStep  # noqa: E402
from nemodiar.reference_hf import load_hf  # noqa: E402



class Body(nn.Module):
    def __init__(self, step, num_layers):
        super().__init__(); self.s = step; self.n = num_layers
    def forward(self, sequence, key_bias):
        s = self.s
        x = s.input_layer_norm(sequence)
        for layer in s.layers[: self.n]:
            x = layer(x, s.cos, s.sin_signed, key_bias)
        return s.layer_norm(x)


if __name__ == "__main__":
    _, hf = load_hf()
    step = EncoderStep.from_hf(hf, 576, fused=True)
    bench(step, "full step (31 layers + head), L=576")
    bench(Body(step, 31).eval(), "31 layers, no head")
    bench(Body(step, 30).eval(), "30 layers, no head")
