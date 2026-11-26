"""The conversion graph (`EncoderStep`) against the original HF modules, before any CoreML conversion."""

import numpy as np
import pytest
import torch

from nemodiar.export_model import EncoderStep, key_bias


def original_logits(model, sequence: torch.Tensor) -> torch.Tensor:
    n = sequence.shape[1]
    hidden = model.model(inputs_embeds=sequence, attention_mask=torch.ones(1, n, dtype=torch.bool),
                         position_ids=torch.arange(n)[None])
    return model.classifier(hidden.last_hidden_state)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("num_valid", [13, 320, 541])
@torch.inference_mode()
def test_padded_step_matches_original(hf, fused, num_valid):
    _, model = hf
    max_len = 541
    step = EncoderStep.from_hf(model, max_len, fused=fused)
    torch.manual_seed(num_valid)
    sequence = torch.randn(1, num_valid, 512)
    padded = torch.zeros(1, max_len, 512)
    padded[:, :num_valid] = sequence
    padded[:, num_valid:] = torch.randn(1, max_len - num_valid, 512)  # padding content must not leak in
    got = step(padded, key_bias(num_valid, max_len))[:, : num_valid * 8]
    np.testing.assert_allclose(got.numpy(), original_logits(model, sequence).numpy(), atol=2e-4, rtol=0)


@torch.inference_mode()
def test_fused_attention_is_bit_identical(hf):
    _, model = hf
    plain, fused = EncoderStep.from_hf(model, 300), EncoderStep.from_hf(model, 300, fused=True)
    x, bias = torch.randn(1, 300, 512), key_bias(250, 300)
    assert torch.equal(plain(x, bias), fused(x, bias))
