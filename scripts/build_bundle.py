"""Assembles the deployable model bundle used by the Swift runtime.

dist/Nemotron3Diarization/
  Encoder.mlmodelc        compiled CoreML encoder step (GPU fp32 buckets, or the Neural Engine fp16 formulation)
  window.f32              400 float32, symmetric Hann window (HF processor constants by default)
  mel_filters.f32         128 x 257 float32, slaney mel filter bank
  stacking_weight.f32     512 x 1024 float32, feature-stacking projection (row-major, out x in)
  silence_embeds.f32      512 float32, learned silence embedding of the speaker cache
  EncoderExact.mlmodelc   optional fp32 model for the steps that update the speaker cache (Neural Engine bundles)
  config.json             streaming and model-layout parameters

    python scripts/build_bundle.py --model build/encoder_fp32_buckets.mlpackage
"""

import argparse
import json
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np

from nemodiar import ROOT, STREAMING_MODES
from nemodiar.cache import CacheConfig
from nemodiar.coreml_backend import model_layout
from nemodiar.features import CONSTANTS_DIR, load_constants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "build" / "encoder_fp32_buckets.mlpackage")
    parser.add_argument("--frontend", default="hf", choices=["hf", "nemo"])
    parser.add_argument("--exact-model", type=Path, default=None,
                        help="fp32 model for the steps that update the speaker cache (Neural Engine bundles)")
    parser.add_argument("--exact-units", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "Nemotron3Diarization")
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    ct.utils.compile_model(str(args.model), destination_path=str(args.out / "Encoder.mlmodelc"))

    layout, buckets = model_layout(str(args.model))
    max_len = buckets[-1]
    exact = None
    if args.exact_model is not None:
        ct.utils.compile_model(str(args.exact_model), destination_path=str(args.out / "EncoderExact.mlmodelc"))
        exact_layout, exact_buckets = model_layout(str(args.exact_model))
        assert exact_layout == "buckets", "the exact model is a GPU-layout fp32 bucket model (scripts/convert.py)"
        exact = {"model_layout": exact_layout, "buckets": exact_buckets, "compute_units": args.exact_units}

    window, mel = load_constants(args.frontend)
    host = np.load(CONSTANTS_DIR / "host_weights.npz")
    window.astype("<f4").tofile(args.out / "window.f32")
    mel.astype("<f4").tofile(args.out / "mel_filters.f32")
    host["stacking_weight"].astype("<f4").tofile(args.out / "stacking_weight.f32")
    host["silence_embeds"].astype("<f4").tofile(args.out / "silence_embeds.f32")

    cache = CacheConfig()
    config = {
        "model_layout": layout,
        "buckets": buckets,
        "max_sequence_length": int(max_len),
        "exact": exact,
        "hidden_size": cache.hidden_size,
        "num_speakers": cache.num_speakers,
        "subsampling_factor": cache.subsampling_factor,
        "frontend": args.frontend,
        "sample_rate": 16000, "n_fft": 512, "hop_length": 160, "win_length": 400, "num_mel_bins": 128,
        "preemphasis": 0.97,
        "streaming_modes": {k: list(v) for k, v in STREAMING_MODES.items()},
        "speaker_cache": {
            "fifo_length": cache.fifo_length,
            "speaker_cache_update_period": cache.speaker_cache_update_period,
            "speaker_cache_length": cache.speaker_cache_length,
            "silence_frames_per_speaker": cache.silence_frames_per_speaker,
            "prediction_score_threshold": cache.prediction_score_threshold,
            "latest_frames_score_boost": cache.latest_frames_score_boost,
            "strong_boost_rate": cache.strong_boost_rate,
            "weak_boost_rate": cache.weak_boost_rate,
            "min_positive_scores_rate": cache.min_positive_scores_rate,
        },
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1e6
    extra = f", exact pop steps on {exact['compute_units']} {exact['buckets']}" if exact else ""
    print(f"bundle -> {args.out} ({layout} {buckets}{extra}, {size:.0f} MB)")


if __name__ == "__main__":
    main()
