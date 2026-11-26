"""Encoder-step backends for `StreamingDiarizer`."""

import numpy as np

from .pipeline import NumpyEmbedder, TorchBackend


def make_backend(kind: str, model: str | None = None, compute_units: str = "cpu_and_gpu"):
    if kind == "torch":
        return TorchBackend()
    if kind == "torch_padded":
        return TorchPaddedBackend()
    if kind == "coreml":
        from .coreml_backend import CoreMLBackend

        return CoreMLBackend(model, compute_units=compute_units)
    raise ValueError(kind)


class TorchPaddedBackend:
    """The conversion module (`EncoderStep`) in PyTorch, fixed length + key mask: validates the export graph."""

    def __init__(self, max_len: int | None = None):
        import torch

        from .export_model import EncoderStep, max_sequence_length
        from .reference_hf import load_hf

        _, hf = load_hf()
        self.max_len = max_len or max_sequence_length()
        self.step = EncoderStep.from_hf(hf, self.max_len)
        self.embedder = NumpyEmbedder()
        self.torch = torch

    def embed(self, features):
        return self.embedder(features)

    def encode(self, sequence):
        from .export_model import key_bias

        torch = self.torch
        n = len(sequence)
        x = np.zeros((1, self.max_len, sequence.shape[1]), np.float32)
        x[0, :n] = sequence
        with torch.inference_mode():
            logits = self.step(torch.from_numpy(x), key_bias(n, self.max_len))
        return logits[0, : n * 8].numpy()
