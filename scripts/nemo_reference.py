"""NeMo (training framework) reference: streaming diarization with the model-card "Low latency" setting.

Run with the NeMo venv:  .venv-nemo/bin/python scripts/nemo_reference.py data/16k/*.wav
Writes golden/nemo/<name>.npz with the 10 ms speaker logits and probabilities.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
NEMO_PATH = ROOT / "hf" / "Nemotron-3-Diarization" / "Nemotron-3-Diarization.nemo"

MODES = {  # spkcache_len, fifo_len, chunk_len, chunk_right_context, spkcache_update_period
    "low_latency": (264, 264, 9, 4, 222),
    "very_low_latency": (264, 264, 6, 2, 222),
    "ultra_low_latency": (264, 264, 3, 1, 222),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--mode", default="low_latency", choices=list(MODES))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "golden" / "nemo")
    args = parser.parse_args()

    from nemo.collections.asr.models import SortformerEncLabelModel

    torch.set_num_threads(8)
    model = SortformerEncLabelModel.restore_from(restore_path=str(NEMO_PATH), map_location="cpu", strict=True)
    model.eval()
    sm = model.sortformer_modules
    sm.spkcache_len, sm.fifo_len, sm.chunk_len, sm.chunk_right_context, sm.spkcache_update_period = MODES[args.mode]
    model._check_streaming_parameters()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.files:
        audio, sr = sf.read(path, dtype="float32")
        assert sr == 16000 and audio.ndim == 1
        signal = torch.from_numpy(audio)[None]
        length = torch.tensor([len(audio)])
        t0 = time.perf_counter()
        with torch.inference_mode():
            preds, logits, _ = model.forward(audio_signal=signal, audio_signal_length=length, return_logits=True)
        dt = time.perf_counter() - t0
        np.savez(args.out_dir / f"{path.stem}.npz", logits=logits[0].numpy(), probs=preds[0].numpy())
        print(f"{path.stem}: {preds.shape[1]} frames, {dt:.1f}s, RTFx {len(audio) / 16000 / dt:.1f}")


if __name__ == "__main__":
    main()
