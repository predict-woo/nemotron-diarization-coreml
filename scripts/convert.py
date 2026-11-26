"""Converts the encoder step to a CoreML ML Program.

Default: a multifunction model with one fixed-length function per length bucket (`len_<L>`, weights shared on
disk). Each takes `sequence` (1, L, 512), the step input zero-padded to L, and `key_bias` (1, 1, 1, L), and
returns `logits` (1, 8L, 8); the runtime picks the smallest bucket holding the step. Multiples of 64 run at the
GPU's best per-position cost; each bucket holds its own GPU copy of the weights (~400 MB).

    python scripts/convert.py --out build/encoder_fp32_buckets.mlpackage                    # 192,384,448,512,576
    python scripts/convert.py --buckets 576 --out build/encoder_fp32_576.mlpackage           # compact: one bucket
    python scripts/convert.py --shapes range --out build/encoder_fp32_fused_range.mlpackage  # flexible length
    python scripts/convert.py --precision fp16 --buckets 576 --out ...                        # not exact
"""

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from nemodiar.export_model import EncoderStep, key_bias, max_sequence_length
from nemodiar.reference_hf import load_hf

DEFAULT_BUCKETS = "192,384,448,512,576"


class NoMask(torch.nn.Module):
    def __init__(self, step):
        super().__init__()
        self.step = step

    def forward(self, sequence):
        return self.step(sequence)


def precision(name: str):
    return ct.precision.FLOAT32 if name == "fp32" else ct.precision.FLOAT16


def convert_fixed(hf, length: int, prec: str, fused: bool) -> ct.models.MLModel:
    step = EncoderStep.from_hf(hf, length, fused=fused)
    with torch.inference_mode():
        traced = torch.jit.trace(step, (torch.randn(1, length, 512), key_bias(length - 5, length)))
    return ct.convert(
        traced,
        inputs=[ct.TensorType("sequence", shape=(1, length, 512), dtype=np.float32),
                ct.TensorType("key_bias", shape=(1, 1, 1, length), dtype=np.float32)],
        outputs=[ct.TensorType("logits", dtype=np.float32)],
        convert_to="mlprogram", compute_precision=precision(prec), minimum_deployment_target=ct.target.macOS15,
    )


def convert_range(hf, max_len: int, prec: str, fused: bool) -> ct.models.MLModel:
    step = NoMask(EncoderStep.from_hf(hf, max_len, dynamic=True, fused=fused)).eval()
    with torch.inference_mode():
        traced = torch.jit.trace(step, (torch.randn(1, max_len - 50, 512),))
    return ct.convert(
        traced,
        inputs=[ct.TensorType("sequence", shape=(1, ct.RangeDim(1, max_len, max_len), 512), dtype=np.float32)],
        outputs=[ct.TensorType("logits", dtype=np.float32)],
        convert_to="mlprogram", compute_precision=precision(prec), minimum_deployment_target=ct.target.macOS15,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--shapes", default="buckets", choices=["buckets", "range"])
    parser.add_argument("--buckets", default=DEFAULT_BUCKETS, help="comma-separated bucket lengths")
    parser.add_argument("--max-len", type=int, default=max_sequence_length(), help="range models only")
    parser.add_argument("--unfused", action="store_true", help="plain attention graph (fused is exact too)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    _, hf = load_hf()
    fused = not args.unfused
    t0 = time.time()
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.shapes == "range":
        model = convert_range(hf, args.max_len, args.precision, fused)
        model.short_description = f"Nemotron-3-Diarization encoder step ({args.precision}, length 1..{args.max_len})"
        model.save(str(args.out))
    else:
        buckets = sorted({int(b) for b in args.buckets.split(",")})
        if buckets[-1] < max_sequence_length():
            raise SystemExit(f"the largest bucket must hold the longest step ({max_sequence_length()})")
        tmp = Path(tempfile.mkdtemp())
        desc = ct.utils.MultiFunctionDescriptor()
        for length in buckets:
            path = tmp / f"len_{length}.mlpackage"
            convert_fixed(hf, length, args.precision, fused).save(str(path))
            desc.add_function(str(path), src_function_name="main", target_function_name=f"len_{length}")
        desc.default_function_name = f"len_{buckets[-1]}"
        ct.utils.save_multifunction(desc, str(args.out))
        shutil.rmtree(tmp)
    print(f"converted in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
