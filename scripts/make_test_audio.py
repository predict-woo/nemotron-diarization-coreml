"""Builds synthetic multi-speaker test recordings from the single-speaker clips in data/16k.

The real clips exercise the model on natural audio; these synthetic "meetings" add many speakers, turn-taking,
overlaps and silences so that the streaming speaker cache fills up, compresses repeatedly and has to keep up to
eight speakers apart.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
SOURCES = [
    "en-Alice_woman",
    "en-Carter_man",
    "en-Frank_man",
    "librispeech_mr_quilter",
    "fleurs_en_us_test_0346",
    "macron",
    "mandarin_voxcpm_zh",
    "obama",
]


def load(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    assert sr == SR, path
    return audio / (np.abs(audio).max() + 1e-6) * 0.5


FRAME = 160  # 10 ms label frames
VAD_RMS = 0.01  # speech threshold on the placed segment (clips peak at 0.5 before a 0.6-1.0 gain)


def speech_frames(segment: np.ndarray, offset: int, num_frames: int) -> np.ndarray:
    """10 ms frames (global grid) where a placed segment is speech: RMS over the frame above VAD_RMS, pauses
    up to 0.3 s bridged, blips under 0.1 s dropped."""
    active = np.zeros(num_frames, bool)
    first, last = offset // FRAME, min((offset + len(segment) - 1) // FRAME, num_frames - 1)
    for f in range(first, last + 1):
        a, b = max(f * FRAME - offset, 0), min((f + 1) * FRAME - offset, len(segment))
        if b > a and np.sqrt(np.mean(segment[a:b] ** 2)) > VAD_RMS:
            active[f] = True
    return smooth(active, fill=30, drop=10)


def smooth(active: np.ndarray, fill: int, drop: int) -> np.ndarray:
    out = active.copy()
    idx = np.nonzero(out)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= fill + 1:
            out[a:b] = True
    edges = np.diff(np.concatenate([[0], out.astype(np.int8), [0]]))
    for s, e in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
        if e - s < drop:
            out[s:e] = False
    return out


def make_meeting(data_dir: Path, seed: int, duration_s: float, num_speakers: int):
    """(audio, labels): labels are (num 10 ms frames, num_speakers) speech activity of each speaker."""
    rng = np.random.default_rng(seed)
    clips = [load(data_dir / f"{name}.wav") for name in SOURCES[:num_speakers]]
    total = int(duration_s * SR)
    out = np.zeros(total + 20 * SR, dtype=np.float32)
    num_frames = total // FRAME
    labels = np.zeros((num_frames, num_speakers), bool)
    cursor = int(0.5 * SR)
    while cursor < total:
        speaker = int(rng.integers(len(clips)))
        clip = clips[speaker]
        seg_len = int(rng.uniform(1.5, 8.0) * SR)
        seg_len = min(seg_len, len(clip) - 1)
        start = int(rng.integers(0, len(clip) - seg_len))
        segment = clip[start : start + seg_len] * rng.uniform(0.6, 1.0)
        out[cursor : cursor + seg_len] += segment
        labels[:, speaker] |= speech_frames(segment, cursor, num_frames)
        # next turn: overlap (negative gap), back-to-back, or a pause
        gap = rng.choice([-0.8, -0.3, 0.0, 0.2, 0.6, 1.5, 3.0], p=[0.1, 0.1, 0.2, 0.25, 0.2, 0.1, 0.05])
        cursor += seg_len + int(gap * SR)
    out = out[:total]
    out += rng.normal(0, 1e-3, size=out.shape).astype(np.float32)  # light noise floor
    return np.clip(out, -1.0, 1.0), labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/16k"))
    parser.add_argument("--stress", action="store_true", help="also six 5-minute meetings with 2-8 speakers")
    args = parser.parse_args()
    specs = [
        ("synth_meeting_4spk_120s", 1, 120.0, 4),
        ("synth_meeting_8spk_300s", 2, 300.0, 8),
    ]
    if args.stress:
        specs += [(f"stress_{seed}_{n}spk_300s", seed, 300.0, n) for seed, n in zip(range(10, 16), [2, 3, 5, 6, 7, 8])]
    label_dir = args.data_dir.parent / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    for name, seed, duration, num_speakers in specs:
        audio, labels = make_meeting(args.data_dir, seed, duration, num_speakers)
        sf.write(args.data_dir / f"{name}.wav", audio, SR, subtype="FLOAT")
        np.savez_compressed(label_dir / f"{name}.npz", activity=labels)
        print(f"wrote {name}.wav ({len(audio) / SR:.1f}s) and labels ({labels.any(1).mean():.0%} speech, "
              f"{(labels.sum(1) > 1).mean():.0%} overlap)")


if __name__ == "__main__":
    main()
