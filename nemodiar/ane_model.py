"""Neural Engine (ANE) formulation of the encoder step, after apple/ml-ane-transformers.

Layout (1, C, 1, S): channels first, the sequence on the last axis; linear layers are 1x1 convolutions; attention
runs per head with einsums; LayerNorm normalises over channels. The ANE computes in float16, so:
  - LayerNorm inputs are divided by `ln_prescale` (LayerNorm is scale-invariant: eps is divided by its square),
    because the residual stream reaches ~1000 and squared deviations would overflow float16;
  - the attention score scale is folded into the query weights (an exact power of two).

    sequence  (1, 512, 1, S)   step input [cache, FIFO, chunk + look-ahead], zero-padded to S
    key_bias  (1, S, 1, 1)     0 on valid positions, -1e4 on padding
    valid     (1, 1, 1, S)     1 on valid positions, 0 on padding
 -> logits    (1, 8, 8, S)     [speaker, sub-frame k, position s]: 10 ms frame s * 8 + k
"""

import torch
from torch import nn

from .export_model import EncoderStep, NEG_BIAS


class ChannelLayerNorm(nn.Module):
    def __init__(self, ln: nn.LayerNorm, prescale: float):
        super().__init__()
        self.weight = nn.Parameter(ln.weight.detach().clone().view(1, -1, 1, 1))
        self.bias = nn.Parameter(ln.bias.detach().clone().view(1, -1, 1, 1))
        self.prescale = prescale
        self.eps = ln.eps / prescale**2

    def forward(self, x):
        x = x * (1.0 / self.prescale)
        zero_mean = x - x.mean(dim=1, keepdim=True)
        variance = (zero_mean * zero_mean).mean(dim=1, keepdim=True)
        return zero_mean * torch.rsqrt(variance + self.eps) * self.weight + self.bias


def conv_from_linear(linear: nn.Linear, scale: float = 1.0) -> nn.Conv2d:
    conv = nn.Conv2d(linear.in_features, linear.out_features, 1, bias=linear.bias is not None)
    with torch.no_grad():
        conv.weight.copy_(linear.weight[:, :, None, None] * scale)
        if linear.bias is not None:
            conv.bias.copy_(linear.bias * scale)
    return conv


class ANEAttention(nn.Module):
    def __init__(self, attn, hidden: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, hidden // heads
        scale = self.head_dim**-0.5
        w = attn.qkv_proj.weight  # fused [q * scale; k; v] (scale already folded by FusedAttention)
        self.q = nn.Conv2d(hidden, hidden, 1, bias=False)
        self.k = nn.Conv2d(hidden, hidden, 1, bias=False)
        self.v = nn.Conv2d(hidden, hidden, 1, bias=False)
        with torch.no_grad():
            self.q.weight.copy_(w[:hidden, :, None, None])
            self.k.weight.copy_(w[hidden : 2 * hidden, :, None, None])
            self.v.weight.copy_(w[2 * hidden :, :, None, None])
        self.o = conv_from_linear(attn.o_proj)
        del scale

    def rope(self, x, cos, sin_signed):
        # rotate_half within each head: swap the two 32-channel halves (sign folded into sin_signed)
        half = self.head_dim // 2
        parts = x.split(half, dim=1)
        swapped = torch.cat([parts[i ^ 1] for i in range(len(parts))], dim=1)
        return x * cos + swapped * sin_signed

    def forward(self, x, cos, sin_signed, key_bias):
        q = self.rope(self.q(x), cos, sin_signed)
        k = self.rope(self.k(x), cos, sin_signed)
        v = self.v(x)
        d = self.head_dim
        qs = q.split(d, dim=1)  # (1, d, 1, S) per head
        ks = k.transpose(1, 3).split(d, dim=3)  # (1, S, 1, d) per head
        vs = v.split(d, dim=1)
        outs = []
        for qh, kh, vh in zip(qs, ks, vs):
            w = torch.einsum("bchq,bkhc->bkhq", qh, kh) + key_bias  # (1, S_k, 1, S_q)
            w = w.softmax(dim=1)
            outs.append(torch.einsum("bkhq,bchk->bchq", w, vh))  # (1, d, 1, S_q)
        return self.o(torch.cat(outs, dim=1))


class ANELayer(nn.Module):
    def __init__(self, layer, hidden: int, heads: int, ln_prescale: float):
        super().__init__()
        self.norm1 = ChannelLayerNorm(layer.layer_norm1, ln_prescale)
        self.attn = ANEAttention(layer.self_attn, hidden, heads)
        self.norm2 = ChannelLayerNorm(layer.layer_norm2, ln_prescale)
        self.fc1 = conv_from_linear(layer.fc1)
        self.fc2 = conv_from_linear(layer.fc2)

    def forward(self, x, cos, sin_signed, key_bias):
        x = x + self.attn(self.norm1(x), cos, sin_signed, key_bias)
        return x + self.fc2(nn.functional.gelu(self.fc1(self.norm2(x))))


class ANEEncoderStep(nn.Module):
    def __init__(self, step: EncoderStep, ln_prescale: float = 32.0):
        super().__init__()
        assert step.fused, "built from the fused EncoderStep (query scale folded)"
        self.length = step.max_len
        hidden = step.input_layer_norm.weight.shape[0]
        heads = step.layers[0].self_attn.heads
        self.input_norm = ChannelLayerNorm(step.input_layer_norm, 1.0)
        self.layers = nn.ModuleList([ANELayer(l, hidden, heads, ln_prescale) for l in step.layers])
        self.final_norm = ChannelLayerNorm(step.layer_norm, ln_prescale)
        self.proj = conv_from_linear(step.proj)
        self.upsample, self.head_hidden = step.upsample, step.head_hidden
        up = step.upsampler
        self.up = nn.Conv2d(up.in_channels, up.out_channels, (1, 3), padding=(0, 1))
        with torch.no_grad():
            self.up.weight.copy_(up.weight[:, :, None, :])
            self.up.bias.copy_(up.bias)
        self.dense = conv_from_linear(step.dense)
        self.out = conv_from_linear(step.out_proj)
        # RoPE tables in channel layout, tiled over heads: (1, hidden, 1, S)
        cos = step.cos[0, 0, : self.length].T  # (head_dim, S)
        sin = step.sin_signed[0, 0, : self.length].T
        self.register_buffer("cos", cos.repeat(heads, 1)[None, :, None, :].contiguous(), persistent=False)
        self.register_buffer("sin_signed", sin.repeat(heads, 1)[None, :, None, :].contiguous(), persistent=False)

    def forward(self, sequence, key_bias, valid):
        x = self.input_norm(sequence)
        for layer in self.layers:
            x = layer(x, self.cos, self.sin_signed, key_bias)
        h = self.proj(self.final_norm(x)) * valid  # (1, 192, 1, S); padding zeroed for the 3-tap conv
        up = self.up(h)  # (1, 8 * 192, 1, S), channel = k * 192 + j
        up = up.reshape(1, self.upsample, self.head_hidden, self.length).transpose(1, 2)  # (1, 192, 8, S)
        return self.out(torch.relu(self.dense(torch.relu(up))))  # (1, 8, 8, S)


def ane_inputs(sequence: torch.Tensor, length: int):
    """(n, 512) step input -> (sequence, key_bias, valid) in the ANE layout, zero-padded to `length`."""
    n = sequence.shape[0]
    seq = torch.zeros(1, sequence.shape[1], 1, length)
    seq[0, :, 0, :n] = sequence.T
    key_bias = torch.full((1, length, 1, 1), NEG_BIAS)
    key_bias[:, :n] = 0.0
    valid = torch.zeros(1, 1, 1, length)
    valid[..., :n] = 1.0
    return seq, key_bias, valid


def logits_from_ane(out: torch.Tensor, n: int) -> torch.Tensor:
    """(1, 8, 8, S) ANE output -> (8n, 8) logits of the first n positions."""
    return out[0, :, :, :n].permute(2, 1, 0).reshape(n * out.shape[2], out.shape[1])
