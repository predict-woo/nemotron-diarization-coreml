"""Teacher-forced error of fp16 encoder variants vs fp32 on real step inputs (same inputs for every variant)."""

import sys

import coremltools as ct
import numpy as np
import torch

from nemodiar.audio import load_audio
from nemodiar.export_model import EncoderStep, key_bias
from nemodiar.pipeline import StreamingDiarizer, TorchBackend
from nemodiar.reference_hf import load_hf

L = 544
_, hf = load_hf()
seqs = []


class Capture(TorchBackend):
    def encode(self, s):
        if len(s) > 400 and len(seqs) < 12 and np.random.default_rng(len(s)).random() < 0.3:
            seqs.append(s.copy())
        return super().encode(s)


StreamingDiarizer(Capture(hf)).run(load_audio("data/16k/diarization_example.wav")[: 16000 * 95])
ref_step = EncoderStep.from_hf(hf, L, fused=True)


def fp32_logits(s):
    x = torch.zeros(1, L, 512); x[0, : len(s)] = torch.from_numpy(s)
    with torch.inference_mode():
        return ref_step(x, key_bias(len(s), L))[0, : len(s) * 8].numpy()


def ane_logits(model, s):
    n = len(s)
    seq = np.zeros((1, 512, 1, L), np.float16); seq[0, :, 0, :n] = s.T
    kb = np.full((1, L, 1, 1), -1e4, np.float16); kb[:, :n] = 0
    va = np.zeros((1, 1, 1, L), np.float16); va[..., :n] = 1
    out = model.predict({"sequence": seq, "key_bias": kb, "valid": va})["logits"]
    return out[0, :, :, :n].transpose(2, 1, 0).reshape(n * 8, 8).astype(np.float32)


def report(name, model):
    errs_l, errs_p = [], []
    for s in seqs:
        a, b = fp32_logits(s), ane_logits(model, s)
        errs_l.append(np.abs(a - b)[np.abs(a) < 15].max())
        pa, pb = 1 / (1 + np.exp(-a)), 1 / (1 + np.exp(-b))
        errs_p.append(np.abs(pa - pb).max())
    print(f"{name:40s} max|dlogit| {np.max(errs_l):.3e} (median {np.median(errs_l):.3e})  "
          f"max|dp| {np.max(errs_p):.3e} (median {np.median(errs_p):.3e})", flush=True)


if __name__ == "__main__":
    path = sys.argv[1]
    for units, name in [("CPU_AND_NE", "ANE fp16"), ("CPU_AND_GPU", "GPU fp16 (exact functions)")]:
        report(f"{name} [{path.split('/')[-1]}]",
               ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, units), function_name=f"len_{L}"))
