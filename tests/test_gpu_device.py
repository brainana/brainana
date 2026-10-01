"""Device policy: resolve_device(), cuda_hidden() and the thread default.

Four GPU bugs in a row were silent (CPU instead of GPU, the wrong GPU, a CUDA
driver probe in CPU mode). These pin the policy documented in
nhp_mri_prep/utils/gpu_device.py without needing a GPU: torch.cuda is
monkeypatched, and a hidden-CUDA test fails if the driver is touched at all.
"""

import pytest
import torch

from fastsurfer_nn.utils import gpu_utils
from fastsurfer_nn.utils.threads import get_num_threads
from nhp_mri_prep.config.config_validation import validate_gpu_device
from nhp_mri_prep.utils import gpu_device

# The implementation (what monkeypatching must target) and the public name used by
# the pipeline must stay one and the same function.
assert gpu_device.resolve_device is gpu_utils.resolve_device


@pytest.fixture(autouse=True)
def standalone(monkeypatch):
    """Outside a pipeline task unless a test sets BRAINANA_DEVICE; fresh report."""
    monkeypatch.delenv("BRAINANA_DEVICE", raising=False)
    monkeypatch.setattr(gpu_utils, "_resolutions", [])
    monkeypatch.setattr(gpu_utils, "_fallbacks", [])


@pytest.fixture
def no_driver(monkeypatch):
    """Fail the test if anything asks the CUDA driver."""

    def boom(*_a, **_k):
        raise AssertionError("torch.cuda was probed while CUDA is hidden")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    monkeypatch.setattr(torch.cuda, "device_count", boom)


@pytest.fixture
def gpus(monkeypatch):
    """Pretend CUDA works with `n` GPUs."""

    def _set(n):
        monkeypatch.setattr(gpu_utils, "_cuda_is_usable", lambda: n > 0)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: n > 0)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: n)

    return _set


@pytest.mark.parametrize("value", ["", "  ", "-1", "none", "NoDevFiles"])
def test_cuda_hidden_values(monkeypatch, value):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    assert gpu_utils.cuda_hidden()


@pytest.mark.parametrize("value", [None, "0", "1", "2,3"])
def test_cuda_not_hidden_values(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    assert not gpu_utils.cuda_hidden()


@pytest.mark.parametrize("spec", ["auto", None, "cuda", "gpu", "cpu", -1, "-1"])
def test_hidden_cuda_resolves_cpu_without_driver(monkeypatch, no_driver, spec):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert gpu_device.resolve_device(spec) == torch.device("cpu")


@pytest.mark.parametrize("spec", [0, "1", "cuda:0"])
def test_hidden_cuda_explicit_gpu_follows_scheduler(monkeypatch, no_driver, spec):
    """CPU-mode task (pipeline hid the GPUs): an explicit index runs on CPU."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert gpu_device.resolve_device(spec) == torch.device("cpu")


@pytest.mark.parametrize("spec", ["auto", "cuda", "gpu"])
def test_auto_picks_gpu_when_usable(monkeypatch, gpus, spec):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    gpus(1)
    assert gpu_device.resolve_device(spec) == torch.device("cuda:0")


def test_auto_without_gpu_is_cpu_never_mps(monkeypatch, gpus):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    gpus(0)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert gpu_device.resolve_device("auto") == torch.device("cpu")


@pytest.mark.parametrize("spec", [0, 1, "1", "cuda:1"])
def test_token_gpu_index_maps_to_cuda0(monkeypatch, gpus, spec):
    """A Nextflow GPU token exposes one GPU; any index means that GPU."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    gpus(1)
    assert gpu_device.resolve_device(spec) == torch.device("cuda:0")


def test_explicit_index_validated_against_device_count(monkeypatch, gpus):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    gpus(2)
    assert gpu_device.resolve_device("cuda:1") == torch.device("cuda:1")
    with pytest.raises(RuntimeError, match="only 2 GPU"):
        gpu_device.resolve_device(2)


def test_explicit_gpu_without_cuda_raises(monkeypatch, gpus):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    gpus(0)
    with pytest.raises(RuntimeError, match="not usable"):
        gpu_device.resolve_device(0)


def test_explicit_mps_unavailable_raises(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS"):
        gpu_device.resolve_device("mps")


@pytest.mark.parametrize("spec", ["cuda:x", "-2", "gpu1"])
def test_unknown_spec_raises(spec):
    with pytest.raises(ValueError):
        gpu_device.resolve_device(spec)


def test_least_busy_ignores_nvidia_smi_under_visible_subset(monkeypatch, gpus):
    """nvidia-smi indices do not map onto a CUDA_VISIBLE_DEVICES subset."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    gpus(2)
    assert gpu_utils._get_least_busy_gpu() == 0
    assert gpu_utils.get_least_busy_gpu() == 0


def test_setup_device_delegates_to_policy(monkeypatch, no_driver):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert gpu_utils.setup_device("cuda") == torch.device("cpu")


@pytest.mark.parametrize("value", ["auto", "cpu", -1, "-1", 0, "1", "cuda:1", None])
def test_validate_gpu_device_accepts(value):
    validate_gpu_device(value)


@pytest.mark.parametrize("value", ["cuda:x", "-2", True, "mps", "gpu1"])
def test_validate_gpu_device_rejects(value):
    with pytest.raises(ValueError):
        validate_gpu_device(value)


def test_threads_follow_omp_num_threads(monkeypatch):
    """Nextflow exports OMP_NUM_THREADS=task.cpus; torch must not exceed it."""
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert get_num_threads() == 2


def test_threads_default_without_omp(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert 1 <= get_num_threads() <= 8


@pytest.mark.parametrize("spec", ["auto", 1, "cuda:1", "cpu", -1])
def test_pipeline_cpu_task_is_cpu_whatever_the_spec(monkeypatch, no_driver, spec):
    monkeypatch.setenv("BRAINANA_DEVICE", "cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert gpu_device.resolve_device(spec) == torch.device("cpu")


@pytest.mark.parametrize("spec", ["auto", 0, 1, "cuda:1"])
def test_pipeline_gpu_slot_is_cuda0(monkeypatch, gpus, spec):
    monkeypatch.setenv("BRAINANA_DEVICE", "cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    gpus(1)
    assert gpu_device.resolve_device(spec) == torch.device("cuda:0")


def test_pipeline_gpu_slot_with_broken_cuda_falls_back(monkeypatch, gpus):
    monkeypatch.setenv("BRAINANA_DEVICE", "cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    gpus(0)
    assert gpu_device.resolve_device("auto") == torch.device("cpu")


def test_bad_spec_rejected_even_inside_pipeline(monkeypatch):
    monkeypatch.setenv("BRAINANA_DEVICE", "cpu")
    with pytest.raises(ValueError):
        gpu_device.resolve_device("cuda:x")


def test_find_device_delegates_and_keeps_value_error(monkeypatch, gpus):
    from fastsurfer_nn.utils.common import find_device

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    gpus(0)
    assert find_device("auto") == torch.device("cpu")
    with pytest.raises(ValueError, match="--device cpu"):
        find_device("cuda:0")


def test_device_report_records_resolution(monkeypatch, no_driver):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert gpu_device.device_report() is None
    gpu_device.resolve_device("auto")
    report = gpu_device.device_report()
    assert report["device"] == "cpu"
    assert report["resolutions"][0]["reason"] == "CUDA hidden (CPU mode)"


def _oom():
    return torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")


def test_is_cuda_oom_sees_through_wrappers():
    try:
        try:
            raise _oom()
        except Exception as inner:
            raise RuntimeError("segmentation failed") from inner
    except RuntimeError as outer:
        assert gpu_device.is_cuda_oom(outer)
    assert not gpu_device.is_cuda_oom(RuntimeError("shape mismatch"))


def test_cpu_fallback_after_cuda_oom(monkeypatch):
    calls = []

    def step(device):
        calls.append(device.type)
        if device.type == "cuda":
            raise RuntimeError("inference failed") from _oom()
        return "ok"

    assert gpu_device.run_with_cpu_fallback(step, torch.device("cuda:0"), "test step") == "ok"
    assert calls == ["cuda", "cpu"]
    report = gpu_device.device_report()
    assert report["device"] == "cpu"
    assert report["cpu_fallbacks"][0]["step"] == "test step"


def test_no_fallback_for_other_errors_or_cpu(monkeypatch):
    def fails(device):
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        gpu_device.run_with_cpu_fallback(fails, torch.device("cuda:0"), "x")

    def cpu_oom(device):
        raise _oom()

    with pytest.raises(torch.cuda.OutOfMemoryError):
        gpu_device.run_with_cpu_fallback(cpu_oom, torch.device("cpu"), "x")


def test_explicit_cpu_honoured_inside_gpu_slot(monkeypatch, gpus):
    """The OOM retry asks for cpu from inside a GPU-slot task."""
    monkeypatch.setenv("BRAINANA_DEVICE", "cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    gpus(1)
    assert gpu_device.resolve_device("cpu") == torch.device("cpu")
    assert gpu_device.resolve_device(-1) == torch.device("cpu")
