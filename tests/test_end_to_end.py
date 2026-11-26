"""End-to-end equivalence of the exact (fp32) pipelines with the original (golden HF Transformers runs).

Every candidate streams the same 16 kHz recordings with the Low latency configuration and must be identical to
the original in the sense of `nemodiar.compare.identity_failures`: the host pipeline with the original encoder,
the CoreML pipeline in Python, and the Swift runtime with the GPU bundle (live, whole-file, other modes).
The Neural Engine (fp16) bundle is judged on diarization quality instead, in test_ane.py.
Quick subset by default, `--full` for every golden file.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from nemodiar.audio import load_audio
from nemodiar.compare import compare_runs, identity_failures
from nemodiar.golden import Run, load_run_dir
from nemodiar.pipeline import StreamingDiarizer, TorchBackend

from .conftest import GOLDEN, audio_path, total_rtfx


@pytest.fixture(scope="module")
def torch_diarizer(hf):
    return StreamingDiarizer(TorchBackend(hf[1]), topk="cpp")


@pytest.fixture(scope="module")
def coreml_diarizer(request):
    from nemodiar.coreml_backend import CoreMLBackend

    path = Path(request.config.getoption("--coreml-model"))
    if not path.exists():
        pytest.skip(f"{path} missing: run scripts/convert.py")
    return StreamingDiarizer(CoreMLBackend(str(path), "cpu_and_gpu"), topk="cpp")


def check(name: str, run: Run):
    comparison = compare_runs(Run.load(GOLDEN / f"{name}.npz"), run)
    failures = identity_failures(comparison)
    assert not failures, f"{name}: " + "; ".join(failures)


@pytest.mark.slow
def test_host_pipeline_with_original_encoder(torch_diarizer, golden_name):
    """Host side alone (front end, embedder, speaker cache, chunking) with the original PyTorch encoder."""
    check(golden_name, torch_diarizer.run(load_audio(audio_path(golden_name))))


@pytest.mark.slow
def test_coreml_pipeline(coreml_diarizer, golden_name):
    check(golden_name, coreml_diarizer.run(load_audio(audio_path(golden_name))))


def run_swift(request, swift_cli, out: Path, *extra: str, bundle_option: str = "--gpu-bundle"):
    bundle = Path(request.config.getoption(bundle_option))
    if not bundle.exists():
        pytest.skip(f"{bundle} missing: run scripts/build_bundle.py")
    from .conftest import golden_names

    names = golden_names(request.config)
    proc = subprocess.run([str(swift_cli), "file", "--bundle", str(bundle), "--format", "stats", "--out", str(out),
                           *extra, *[str(audio_path(n)) for n in names]], capture_output=True, text=True, check=True)
    return out, proc.stdout


@pytest.fixture(scope="module")
def swift_runs(request, swift_cli, tmp_path_factory):
    """Live streaming: audio pushed in 80 ms blocks."""
    return run_swift(request, swift_cli, tmp_path_factory.mktemp("swift"))


@pytest.fixture(scope="module")
def swift_offline_runs(request, swift_cli, tmp_path_factory):
    """Whole-file mode: the same steps spread over GPU streams and the CPU."""
    return run_swift(request, swift_cli, tmp_path_factory.mktemp("swift_offline"), "--offline")


@pytest.mark.slow
def test_swift_runtime(swift_runs, golden_name):
    out, _ = swift_runs
    check(golden_name, load_run_dir(out / golden_name))


@pytest.mark.slow
def test_swift_offline_runtime(swift_offline_runs, golden_name):
    out, _ = swift_offline_runs
    check(golden_name, load_run_dir(out / golden_name))


@pytest.mark.slow
def test_swift_runtime_is_at_least_4x_real_time(swift_runs):
    _, stdout = swift_runs
    assert total_rtfx(stdout) >= 4.0, stdout


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["very_low_latency", "ultra_low_latency"])
def test_swift_runtime_other_modes(request, swift_cli, tmp_path, mode):
    """The same bundle in the other model-card streaming modes (goldens: make_golden.py --mode ...)."""
    golden = GOLDEN.parent / f"hf_{mode}_float32"
    names = sorted(p.stem for p in golden.glob("*.npz"))
    if not names:
        pytest.skip(f"no goldens in {golden}")
    bundle = request.config.getoption("--gpu-bundle")
    subprocess.run([str(swift_cli), "file", "--bundle", bundle, "--latency", mode, "--out", str(tmp_path),
                    *[str(audio_path(n)) for n in names]], capture_output=True, text=True, check=True)
    for name in names:
        comparison = compare_runs(Run.load(golden / f"{name}.npz"), load_run_dir(tmp_path / name))
        failures = identity_failures(comparison)
        assert not failures, f"{mode} {name}: " + "; ".join(failures)


@pytest.mark.slow
def test_swift_output_independent_of_push_block_size(request, swift_cli, tmp_path):
    """Live streaming in 7 ms blocks and one push of the whole file run the same steps.

    The results are normally bit-identical; under memory pressure the GPU may pick other kernels between processes,
    so the check is the same speaker-cache trajectory and fp32-rounding agreement."""
    bundle = request.config.getoption("--gpu-bundle")
    runs = []
    for block_ms in (7, 10**8):
        out = tmp_path / str(block_ms)
        subprocess.run([str(swift_cli), "file", "--bundle", bundle, "--block-ms", str(block_ms), "--out", str(out),
                        str(audio_path("two_speakers"))], capture_output=True, check=True)
        runs.append(load_run_dir(out / "two_speakers"))
    assert runs[0].logits.shape == runs[1].logits.shape
    assert np.array_equal(runs[0].cache_ids, runs[1].cache_ids)
    assert np.array_equal(runs[0].fifo_ids, runs[1].fifo_ids)
    assert np.abs(runs[0].logits - runs[1].logits).max() < 1e-4
