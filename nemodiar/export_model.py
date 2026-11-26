"""Conversion-friendly re-statement of the encoder step, loaded with the original weights.

EncoderStep(sequence[1, L, 512], key_bias[1, 1, 1, L]) -> logits[1, L * 8, 8]

`sequence` is the step input [speaker cache, FIFO, chunk + look-ahead] zero-padded to the fixed length L, and
`key_bias` is 0 on valid positions and a large negative value on padding, so that padded keys get zero attention
weight: valid positions compute exactly the original math. Valid tokens come first, so their RoPE positions are
0..n-1 as in the original, and the per-position layers (norms, MLP, head) never mix positions, except the
upsampler's 3-tap convolution, whose output at the last valid position reads one padded zero position, as the
original's zero padding of the convolution does.
"""

import torch
from torch import nn

NEG_BIAS = -1e4  # finite so the same graph is safe in float16


class Attention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, hidden // heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=True)

    def forward(self, x, cos, sin, key_bias):
        # batch 1; the length is the only free dimension, written as -1 so tracing emits no shape arithmetic
        h, d = self.heads, self.head_dim
        q = self.q_proj(x).reshape(1, -1, h, d).transpose(1, 2)
        k = self.k_proj(x).reshape(1, -1, h, d).transpose(1, 2)
        v = self.v_proj(x).reshape(1, -1, h, d).transpose(1, 2)
        q = q * cos + rotate_half(q, d) * sin
        k = k * cos + rotate_half(k, d) * sin
        scores = torch.matmul(q, k.transpose(2, 3)) * (d**-0.5)
        if key_bias is not None:
            scores = scores + key_bias
        out = torch.matmul(torch.softmax(scores, dim=-1), v)
        return self.o_proj(out.transpose(1, 2).reshape(1, -1, h * d))


class FusedAttention(nn.Module):
    """Same math as `Attention`, restated with fewer GPU ops; every rewrite is exact in floating point.

    - q, k and v come from one 512 -> 1536 projection, split after a single head transpose;
    - the 1/sqrt(64) = 2^-3 score scale is folded into the query weights (power-of-two scaling is exact);
    - RoPE: rotate_half(x) * sin == swap_halves(x) * [-sin, sin] (a sign flip is exact), applied to q and k at once.
    """

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, hidden // heads
        self.qkv_proj = nn.Linear(hidden, 3 * hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=True)
        self.batch = 1  # static batch size, set by EncoderStep.fuse

    def load_from(self, attn: "Attention"):
        scale = self.head_dim**-0.5
        assert scale == 2.0 ** round(torch.log2(torch.tensor(scale)).item()), "scale folding needs a power of two"
        with torch.no_grad():
            self.qkv_proj.weight.copy_(torch.cat([attn.q_proj.weight * scale, attn.k_proj.weight, attn.v_proj.weight]))
            self.o_proj.load_state_dict(attn.o_proj.state_dict())

    def forward(self, x, cos, sin_signed, key_bias):
        h, d, b = self.heads, self.head_dim, self.batch
        qkv = self.qkv_proj(x).reshape(b, -1, 3 * h, d).transpose(1, 2)  # (B, 3h, L, d)
        qk, v = qkv[:, : 2 * h], qkv[:, 2 * h :]
        swapped = torch.flip(qk.reshape(b, 2 * h, -1, 2, d // 2), dims=[3]).reshape(b, 2 * h, -1, d)
        qk = qk * cos + swapped * sin_signed
        q, k = qk[:, :h], qk[:, h:]
        scores = torch.matmul(q, k.transpose(2, 3))
        if key_bias is not None:
            scores = scores + key_bias
        out = torch.matmul(torch.softmax(scores, dim=-1), v)
        return self.o_proj(out.transpose(1, 2).reshape(b, -1, h * d))


def rotate_half(x, dim: int):
    half = dim // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Layer(nn.Module):
    def __init__(self, hidden: int, heads: int, intermediate: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.self_attn: nn.Module = Attention(hidden, heads)
        self.layer_norm2 = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, intermediate)
        self.fc2 = nn.Linear(intermediate, hidden)

    def forward(self, x, cos, sin, key_bias):
        x = x + self.self_attn(self.layer_norm1(x), cos, sin, key_bias)
        h = self.fc2(nn.functional.gelu(self.fc1(self.layer_norm2(x))))
        return x + h


class EncoderStep(nn.Module):
    def __init__(self, max_len: int, hidden=512, heads=8, intermediate=2048, layers=31, head_hidden=192,
                 num_speakers=8, upsample=8, rope_theta=10000.0, dynamic: bool = False):
        super().__init__()
        self.max_len, self.upsample, self.head_hidden = max_len, upsample, head_hidden
        self.dynamic = dynamic  # length taken from the input (flexible-shape models) instead of max_len
        self.input_layer_norm = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList([Layer(hidden, heads, intermediate) for _ in range(layers)])
        self.layer_norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, head_hidden)
        self.upsampler = nn.Conv1d(head_hidden, head_hidden * upsample, kernel_size=3, padding=1)
        self.dense = nn.Linear(head_hidden, head_hidden)
        self.out_proj = nn.Linear(head_hidden, num_speakers)

        head_dim = hidden // heads
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        freqs = torch.arange(max_len, dtype=torch.float)[:, None] * inv_freq[None]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None], persistent=False)  # (1, 1, L, head_dim)
        self.register_buffer("sin", emb.sin()[None, None], persistent=False)
        sign = torch.cat([-torch.ones(head_dim // 2), torch.ones(head_dim // 2)])
        self.register_buffer("sin_signed", (emb.sin() * sign)[None, None], persistent=False)
        self.fused = False
        self.batch = 1

    def fuse(self, batch: int = 1) -> "EncoderStep":
        """Swaps every attention for `FusedAttention` (same weights, fewer GPU ops)."""
        self.batch = batch
        for layer in self.layers:
            fused = FusedAttention(layer.self_attn.heads * layer.self_attn.head_dim, layer.self_attn.heads)
            fused.load_from(layer.self_attn)
            fused.batch = batch
            layer.self_attn = fused
        self.fused = True
        return self.eval()

    def forward(self, sequence, key_bias=None):
        cos, sin = self.cos, self.sin_signed if self.fused else self.sin
        if self.dynamic:
            cos, sin = cos[:, :, : sequence.shape[1]], sin[:, :, : sequence.shape[1]]
        x = self.input_layer_norm(sequence)
        for layer in self.layers:
            x = layer(x, cos, sin, key_bias)
        h = self.proj(self.layer_norm(x))  # (1, L, 192)
        if key_bias is not None:
            # the original convolution zero-pads; padding positions of the input are zeroed for the same effect
            h = h * (key_bias[:, 0, 0, :, None] == 0).to(h.dtype)
        up = self.upsampler(h.transpose(1, 2)).transpose(1, 2)  # (1, L, 8 * 192)
        up = up.reshape(self.batch, -1, self.head_hidden)
        return self.out_proj(torch.relu(self.dense(torch.relu(up))))

    @classmethod
    def from_hf(cls, hf_model, max_len: int, dynamic: bool = False, fused: bool = False,
                batch: int = 1) -> "EncoderStep":
        cfg = hf_model.config
        ac, hc = cfg.audio_config, cfg.head_config
        assert ac.num_key_value_heads == ac.num_attention_heads
        assert ac.rope_parameters.get("partial_rotary_factor", 1.0) == 1.0 and ac.rope_parameters["rope_type"] == "default"
        assert ac.hidden_act == "gelu"
        step = cls(max_len, ac.hidden_size, ac.num_attention_heads, ac.intermediate_size, ac.num_hidden_layers,
                   hc.hidden_size, hc.num_speakers, hc.subsampling_factor, ac.rope_parameters["rope_theta"],
                   dynamic=dynamic)
        src = hf_model.state_dict()
        dst = {}
        for k, v in src.items():
            if k.startswith("model.audio_tower.layers."):
                k2 = k[len("model.audio_tower.") :].replace("mlp.", "")
                dst[k2] = v
            elif k.startswith("model.audio_tower.input_layer_norm.") or k.startswith("model.audio_tower.layer_norm."):
                dst[k[len("model.audio_tower.") :]] = v
            elif k.startswith("model.proj."):
                dst[k[len("model.") :]] = v
            elif k.startswith("model.upsampler.conv."):
                dst["upsampler." + k[len("model.upsampler.conv.") :]] = v
            elif k.startswith("classifier."):
                dst[k[len("classifier.") :]] = v
        missing, unexpected = step.load_state_dict(dst, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        assert batch == 1 or fused, "batched steps need the fused attention"
        return step.fuse(batch) if fused else step.eval()


def key_bias(num_valid: int, max_len: int) -> torch.Tensor:
    bias = torch.full((1, 1, 1, max_len), NEG_BIAS)
    bias[..., :num_valid] = 0.0
    return bias


def max_sequence_length(fifo_length=264, cache_length=264, chunk=9, right_context=4) -> int:
    return cache_length + fifo_length + chunk + right_context

