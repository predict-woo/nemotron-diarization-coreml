"""Nemotron-3-Diarization: reference runners, CoreML pipeline and equivalence tooling."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_MODEL_DIR = ROOT / "hf" / "Nemotron-3-Diarization"
DATA_DIR = ROOT / "data" / "16k"
GOLDEN_DIR = ROOT / "golden"
SAMPLE_RATE = 16000

# Streaming modes of the processor, (chunk_length, chunk_right_context) in 80 ms encoder frames.
STREAMING_MODES = {
    "low_latency": (9, 4),
    "very_low_latency": (6, 2),
    "ultra_low_latency": (3, 1),
}
