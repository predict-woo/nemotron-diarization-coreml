"""Exports the front-end constants and the host-side weights of the pipeline.

nemodiar/constants/frontend_hf.npz    window + mel filters of the HF processor (float32, the default)
nemodiar/constants/frontend_nemo.npz  window + mel filters stored in the .nemo checkpoint (bf16-rounded values)
nemodiar/constants/host_weights.npz   feature-stacking projection (1024->512, no bias) and silence embedding
"""

import tarfile
from pathlib import Path

import numpy as np
import torch

from nemodiar import HF_MODEL_DIR
from nemodiar.features import CONSTANTS_DIR


def main():
    from transformers import AutoProcessor
    from safetensors.numpy import load_file

    CONSTANTS_DIR.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(HF_MODEL_DIR)
    fe = proc.feature_extractor
    assert fe.mel_filters_dtype == "float32"
    window = torch.hann_window(fe.win_length, periodic=False, dtype=torch.float32).numpy()
    np.savez(CONSTANTS_DIR / "frontend_hf.npz", window=window, mel_filters=fe.mel_filters.numpy().astype(np.float32))

    with tarfile.open(HF_MODEL_DIR / "Nemotron-3-Diarization.nemo") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("model_weights.ckpt"))
        state = torch.load(tar.extractfile(member), map_location="cpu", weights_only=True)
    np.savez(
        CONSTANTS_DIR / "frontend_nemo.npz",
        window=state["preprocessor.featurizer.window"].float().numpy(),
        mel_filters=state["preprocessor.featurizer.fb"].float().numpy()[0],
    )

    weights = load_file(str(HF_MODEL_DIR / "model.safetensors"))
    np.savez(
        CONSTANTS_DIR / "host_weights.npz",
        stacking_weight=weights["model.audio_tower.embedder.projection.weight"].astype(np.float32),
        silence_embeds=weights["silence_embeds"].astype(np.float32),
    )
    for f in sorted(CONSTANTS_DIR.glob("*.npz")):
        print(f, {k: v.shape for k, v in np.load(f).items()})


if __name__ == "__main__":
    main()
