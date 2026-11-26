"""The Neural Engine bundle (float16, the default `dist/Nemotron3Diarization`).

float16 cannot reproduce the original bit for bit, so this bundle is held to diarization quality:
  - every operation of every model function runs on the Neural Engine;
  - DER against ground truth (synthetic meetings) stays within nemodiar.der.MAX_DER_INCREASE_* of the original's;
  - few 10 ms frames get a different speaker set than the original's (nemodiar.der.MAX_FLIPPED_*);
  - the runtime is at least 4x real time.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from nemodiar import DATA_DIR
from nemodiar.der import (MAX_DER_INCREASE_FILE, MAX_DER_INCREASE_TOTAL, MAX_FLIPPED_FILE, MAX_FLIPPED_TOTAL,
                          activity, der)
from nemodiar.golden import Run, load_run_dir

from .conftest import GOLDEN, audio_path, golden_names

LABELS = DATA_DIR.parent / "labels"


@pytest.fixture(scope="module")
def bundle(request) -> Path:
    path = Path(request.config.getoption("--bundle"))
    if not path.exists():
        pytest.skip(f"{path} missing: run scripts/convert_ane.py and scripts/build_bundle.py")
    return path


@pytest.fixture(scope="module")
def ane_runs(request, swift_cli, bundle, tmp_path_factory):
    names = golden_names(request.config)
    out = tmp_path_factory.mktemp("ane")
    proc = subprocess.run([str(swift_cli), "--bundle", str(bundle), "--out", str(out),
                           *[str(audio_path(n)) for n in names]], capture_output=True, text=True, check=True)
    return names, out, proc.stdout


def test_every_operation_runs_on_the_neural_engine(swift_cli, bundle):
    """CoreML's compute plan of every bucket function: all non-constant operations on the Neural Engine."""
    report = subprocess.run([str(swift_cli), "--bundle", str(bundle), "--compute-plan"], capture_output=True,
                            text=True, check=True).stdout.strip().splitlines()
    assert report, "no functions reported"
    for line in report:
        name, counts = line.split(": ")
        devices = dict(item.split("=") for item in counts.split())
        assert set(devices) == {"NeuralEngine"}, f"{name}: {devices}"


@pytest.mark.slow
def test_ane_diarizes_as_well_as_the_original(ane_runs):
    names, out, _ = ane_runs
    flips = frames = 0
    der_ref = der_ane = speech = 0.0
    for name in names:
        original = activity(Run.load(GOLDEN / f"{name}.npz").probs)
        ane = activity(load_run_dir(out / name).probs)
        flipped = (original != ane).any(axis=1).sum()
        assert flipped / len(original) <= MAX_FLIPPED_FILE, f"{name}: {flipped} of {len(original)} frames differ"
        flips, frames = flips + flipped, frames + len(original)
        if (LABELS / f"{name}.npz").exists():
            truth = np.load(LABELS / f"{name}.npz")["activity"]
            d_ref, d_ane = der(truth, original), der(truth, ane)
            assert d_ane.der - d_ref.der <= MAX_DER_INCREASE_FILE, f"{name}: DER {d_ane} vs original {d_ref}"
            der_ref += d_ref.der * d_ref.speech_frames
            der_ane += d_ane.der * d_ane.speech_frames
            speech += d_ref.speech_frames
    assert flips / frames <= MAX_FLIPPED_TOTAL, f"{flips} of {frames} frames differ"
    if speech:
        assert (der_ane - der_ref) / speech <= MAX_DER_INCREASE_TOTAL


@pytest.mark.slow
def test_ane_runtime_is_at_least_4x_real_time(ane_runs):
    _, _, stdout = ane_runs
    rtfx = float(next(l for l in stdout.splitlines() if l.startswith("overall RTFx")).split()[2])
    assert rtfx >= 4.0, stdout
