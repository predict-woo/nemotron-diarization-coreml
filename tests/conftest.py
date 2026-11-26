import os
import subprocess
from pathlib import Path

import pytest

from nemodiar import DATA_DIR, GOLDEN_DIR, ROOT

GOLDEN = GOLDEN_DIR / "hf_low_latency_float32"
# quick subset: a short single speaker, a 60 s two-speaker conversation and a 120 s synthetic 4-speaker meeting
QUICK = ["en-Alice_woman", "two_speakers", "synth_meeting_4spk_120s"]
SDKROOT = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"


def pytest_addoption(parser):
    parser.addoption("--full", action="store_true", help="end-to-end tests on every golden file, not the quick subset")
    parser.addoption("--coreml-model", default=str(ROOT / "build" / "encoder_fp32_buckets.mlpackage"))
    parser.addoption("--bundle", default=str(ROOT / "dist" / "Nemotron3Diarization"),
                     help="the Neural Engine bundle (fp16): judged on diarization quality")
    parser.addoption("--gpu-bundle", default=str(ROOT / "dist" / "Nemotron3Diarization-gpu"),
                     help="the exact GPU bundle (fp32): judged on identity with the original")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end runs over recordings")


def golden_names(config) -> list[str]:
    names = sorted(p.stem for p in GOLDEN.glob("*.npz"))
    if not names:
        pytest.skip(f"no golden runs in {GOLDEN}: run scripts/make_golden.py")
    names = names if config.getoption("--full") else [n for n in QUICK if n in names]
    missing = [n for n in names if not audio_path(n).exists()]
    if missing:
        pytest.skip(f"test audio missing ({missing[0]}...): run scripts/fetch_test_audio.sh and "
                    "scripts/make_test_audio.py --stress")
    return names


def pytest_generate_tests(metafunc):
    if "golden_name" in metafunc.fixturenames:
        metafunc.parametrize("golden_name", golden_names(metafunc.config))


@pytest.fixture(scope="session")
def hf():
    from nemodiar.reference_hf import load_hf

    return load_hf()


@pytest.fixture(scope="session")
def swift_cli():
    """Release build of the Swift CLI (built on demand; Package.swift is at the repository root)."""
    env = dict(os.environ, SDKROOT=os.environ.get("SDKROOT", SDKROOT))
    subprocess.run(["swift", "build", "-c", "release"], cwd=ROOT, env=env, check=True, capture_output=True)
    return ROOT / ".build" / "release" / "nemotron-diarize"


def audio_path(name: str) -> Path:
    return DATA_DIR / f"{name}.wav"
