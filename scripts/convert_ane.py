"""Converts the ANE formulation of the encoder step (nemodiar/ane_model.py) to a float16 ML Program.

Multifunction model with one fixed-length function per bucket (`len_<L>`), like scripts/convert.py, but for the
Neural Engine: inputs `sequence` (1, 512, 1, L), `key_bias` (1, L, 1, 1), `valid` (1, 1, 1, L), output `logits`
(1, 8, 8, L), all float16.

    python scripts/convert_ane.py --buckets 576 --out build/encoder_ane_576.mlpackage
"""

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from nemodiar.ane_model import ANEEncoderStep, ane_inputs
from nemodiar.export_model import EncoderStep, max_sequence_length
from nemodiar.reference_hf import load_hf


def convert_bucket(hf, length: int, ln_prescale: float) -> ct.models.MLModel:
    step = ANEEncoderStep(EncoderStep.from_hf(hf, length, fused=True), ln_prescale=ln_prescale).eval()
    example = ane_inputs(torch.randn(length - 5, 512), length)
    with torch.inference_mode():
        traced = torch.jit.trace(step, example)
    fp16 = np.float16
    return ct.convert(
        traced,
        inputs=[ct.TensorType("sequence", shape=(1, 512, 1, length), dtype=fp16),
                ct.TensorType("key_bias", shape=(1, length, 1, 1), dtype=fp16),
                ct.TensorType("valid", shape=(1, 1, 1, length), dtype=fp16)],
        outputs=[ct.TensorType("logits", dtype=fp16)],
        convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buckets", default="576")
    parser.add_argument("--ln-prescale", type=float, default=32.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    buckets = sorted({int(b) for b in args.buckets.split(",")})
    if buckets[-1] < max_sequence_length():
        raise SystemExit(f"the largest bucket must hold the longest step ({max_sequence_length()})")
    _, hf = load_hf()
    t0 = time.time()
    if args.out.exists():
        shutil.rmtree(args.out)
    tmp = Path(tempfile.mkdtemp())
    desc = ct.utils.MultiFunctionDescriptor()
    for length in buckets:
        path = tmp / f"len_{length}.mlpackage"
        convert_bucket(hf, length, args.ln_prescale).save(str(path))
        desc.add_function(str(path), src_function_name="main", target_function_name=f"len_{length}")
    desc.default_function_name = f"len_{buckets[-1]}"
    ct.utils.save_multifunction(desc, str(args.out))
    shutil.rmtree(tmp)
    print(f"converted in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
