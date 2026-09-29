"""
Centralized GPU device selection for brainana.

The only implementation of device policy in the code base. It lives in
fastsurfer_nn because nhp_mri_prep imports fastsurfer_nn at package import time, so
only this direction is free of import cycles; nhp_mri_prep.utils.gpu_device (the public name the pipeline
uses), fastsurfer_nn.utils.common.find_device and nhp_skullstrip_nn.utils.gpu
re-export or delegate to it.

Canonical spec: "auto" | "cuda" | "gpu" | -1 / "-1" / "cpu" | 0..N / "N" |
"cuda:N" | "mps" (explicit only)

Policy:
    - Under the Nextflow pipeline the task's environment decides (the scheduler
      has already applied general.gpu_device): BRAINANA_DEVICE=cpu means CPU;
      BRAINANA_DEVICE=cuda means the single GPU slot the task holds (cuda:0),
      or CPU with a warning if CUDA turns out to be unusable. An explicit "cpu"
      / -1 is always honoured (e.g. the CPU retry after a CUDA out-of-memory).
    - Otherwise (standalone, Lite) the spec decides:
        - "auto" (and "cuda"/"gpu") picks a usable CUDA GPU, else CPU. It never
          picks Apple MPS: that path is not scheduled or validated.
        - An explicit GPU request (index, "cuda:N", "mps") that the machine
          cannot serve raises instead of silently running on CPU.
        - When every GPU is hidden (CUDA_VISIBLE_DEVICES="") an explicit GPU
          index resolves to CPU with a warning.
        - When exactly one GPU is visible, any GPU index resolves to cuda:0.

Every resolution is logged as ``[Device] ...`` and recorded; save_metadata()
(nhp_mri_prep.utils.nextflow) writes the record into the step's metadata JSON, together with
any CPU fallback after a CUDA out-of-memory error (run_with_cpu_fallback).
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import torch

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# What this process resolved and fell back to; read by device_report().
_resolutions: List[Dict[str, str]] = []
_fallbacks: List[Dict[str, str]] = []


def cuda_hidden() -> bool:
    """True when CUDA_VISIBLE_DEVICES deliberately hides every GPU.

    The pipeline exports CUDA_VISIBLE_DEVICES="" for every task that holds no GPU
    slot. Callers must then skip torch.cuda probes entirely: torch.cuda.is_available()
    still loads and initializes the CUDA driver, which is unnecessary in CPU mode and
    can abort the process outright on some drivers (WSL2 GPU passthrough).
    NoDevFiles is what some schedulers (e.g. SGE/SLURM GPU plugins) export for jobs
    granted no GPU.
    """
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    return value is not None and value.strip().lower() in ("", "-1", "none", "nodevfiles")


def cuda_available() -> bool:
    """torch.cuda.is_available(), without touching the driver when CUDA is hidden."""
    return not cuda_hidden() and torch.cuda.is_available()


def _cuda_is_usable() -> bool:
    """Return True only when CUDA can actually execute a tensor op."""
    if not cuda_available():
        return False
    try:
        # Probe real kernel execution, not just device enumeration.
        _ = torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False


def _visible_gpu_entries() -> list[str] | None:
    """CUDA_VISIBLE_DEVICES as a list, or None when unset (all GPUs visible)."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def get_least_busy_gpu() -> int:
    """Get the CUDA index of the GPU with the least memory in use.

    When CUDA_VISIBLE_DEVICES is set, returns 0: nvidia-smi reports every
    physical GPU, so its indices do not map onto the visible subset.
    """
    if not cuda_available():
        return 0
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1 or _visible_gpu_entries() is not None:
        return 0

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            memory_usages = [
                int(line.strip())
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
            # nvidia-smi numbers GPUs in PCI order; CUDA only agrees when
            # CUDA_DEVICE_ORDER=PCI_BUS_ID (set by the pipeline and the image).
            if (
                len(memory_usages) == gpu_count
                and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
            ):
                return memory_usages.index(min(memory_usages))
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass
    return 0


# Pre-unification private name, still used by tests and older callers.
_get_least_busy_gpu = get_least_busy_gpu


def _explicit_gpu(index: int, spec) -> torch.device:
    """Resolve an explicit GPU index, enforcing the policy in the module docstring."""
    if cuda_hidden():
        logger.warning(
            f"[Device] gpu_device={spec!r} requested, but this task has no visible GPU "
            "(CUDA_VISIBLE_DEVICES is empty: CPU mode); using cpu"
        )
        return torch.device("cpu")
    if not _cuda_is_usable():
        raise RuntimeError(
            f"gpu_device={spec!r} requests a CUDA GPU, but CUDA is not usable here "
            "(no GPU, driver problem, or a CPU-only PyTorch). Set general.gpu_device "
            "to 'auto' or -1 to run on CPU."
        )
    visible = _visible_gpu_entries()
    if visible is not None and len(visible) == 1:
        if index != 0:
            logger.info(
                f"[Device] gpu_device={spec!r}: CUDA_VISIBLE_DEVICES={visible[0]} "
                "exposes one GPU, using it as cuda:0"
            )
        return torch.device("cuda:0")
    count = torch.cuda.device_count()
    if index >= count:
        raise RuntimeError(
            f"gpu_device={spec!r} but only {count} GPU(s) are visible (valid: 0..{count - 1})"
        )
    return torch.device(f"cuda:{index}")


def _record(spec, device: torch.device, reason: str) -> torch.device:
    logger.info(f"[Device] gpu_device={spec!r} -> {device} ({reason})")
    _resolutions.append(
        {"requested": str(spec), "device": str(device), "reason": reason}
    )
    return device


def _from_pipeline(spec, assigned: str) -> torch.device:
    """Honour the device the Nextflow scheduler assigned to this task."""
    if assigned == "cpu":
        return _record(spec, torch.device("cpu"), "pipeline: CPU task")
    # "cuda": this task holds a GPU slot, exposed as the only visible GPU.
    if _cuda_is_usable():
        return _record(spec, torch.device("cuda:0"), "pipeline: GPU slot")
    logger.warning(
        "[Device] the pipeline assigned this task a GPU, but CUDA is not usable "
        "(driver or PyTorch problem); running on cpu"
    )
    return _record(spec, torch.device("cpu"), "pipeline: GPU slot, CUDA unusable")


def resolve_device(spec: Union[str, int, torch.device, None] = "auto") -> torch.device:
    """Resolve a device spec to a torch.device.

    Args:
        spec: Device specification (see module docstring for the policy):
            - "auto" / "cuda" / "gpu" / None: a usable CUDA GPU, else CPU
            - -1, "-1" or "cpu": CPU
            - 0, 1, ... / "0", "1", ... / "cuda:N": that GPU (explicit)
            - "mps": Apple MPS (explicit only)
            - torch.device: returned as-is (already resolved by the caller)

    Returns:
        torch.device object

    Raises:
        RuntimeError: an explicit GPU/MPS request the machine cannot serve.
        ValueError: an unrecognised spec.
    """
    if isinstance(spec, torch.device):
        return spec

    s = "auto" if spec is None else str(spec).strip().lower()
    digits = s[len("cuda:"):] if s.startswith("cuda:") else s
    if not (s in ("auto", "cuda", "gpu", "", "cpu", "-1", "mps") or digits.isdigit()):
        raise ValueError(
            f"Unrecognised gpu_device {spec!r}: use auto, cpu, -1, a GPU index or cuda:N"
        )

    assigned = os.environ.get("BRAINANA_DEVICE", "").strip().lower()
    if assigned in ("cpu", "cuda") and s not in ("cpu", "-1"):
        return _from_pipeline(spec, assigned)

    if s in ("auto", "cuda", "gpu", ""):
        if _cuda_is_usable():
            return _record(spec, torch.device(f"cuda:{get_least_busy_gpu()}"), "CUDA usable")
        reason = "CUDA hidden (CPU mode)" if cuda_hidden() else "no usable CUDA GPU"
        return _record(spec, torch.device("cpu"), reason)

    if s in ("cpu", "-1"):
        return _record(spec, torch.device("cpu"), "requested")

    if s == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return _record(spec, torch.device("mps"), "requested")
        raise RuntimeError("gpu_device='mps' but Apple MPS is not available")

    return _record(spec, _explicit_gpu(int(digits), spec), "requested")


def is_cuda_oom(exc: BaseException) -> bool:
    """True when ``exc`` (or anything it wraps) is a CUDA out-of-memory error."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, torch.cuda.OutOfMemoryError) or "CUDA out of memory" in str(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def run_with_cpu_fallback(
    fn: Callable[[torch.device], _T],
    device: torch.device,
    what: str,
    log: Optional[logging.Logger] = None,
) -> _T:
    """Run ``fn(device)``; after a CUDA out-of-memory error, run ``fn(cpu)`` once.

    The fallback is logged as a warning and recorded for the step's metadata JSON.
    Any other error, or an OOM on a non-CUDA device, propagates unchanged.
    """
    log = log or logger
    try:
        return fn(device)
    except Exception as exc:
        if device.type != "cuda" or not is_cuda_oom(exc):
            raise
        log.warning(
            f"[Device] {what}: CUDA out of memory on {device}; retrying once on cpu "
            "(slower, same result)"
        )
        _fallbacks.append({"step": what, "from": str(device), "to": "cpu", "reason": "CUDA out of memory"})
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        return fn(torch.device("cpu"))


def device_report() -> Optional[Dict[str, Any]]:
    """What this process ran on, for the step's metadata JSON (None if nothing ran)."""
    if not _resolutions and not _fallbacks:
        return None
    report: Dict[str, Any] = {
        "device": _fallbacks[-1]["to"] if _fallbacks else _resolutions[-1]["device"],
        "resolutions": list(_resolutions),
    }
    if _fallbacks:
        report["cpu_fallbacks"] = list(_fallbacks)
    for key in ("BRAINANA_DEVICE", "CUDA_VISIBLE_DEVICES"):
        if key in os.environ:
            report[key] = os.environ[key]
    return report


# Backwards-compatible aliases for migration
def get_device() -> torch.device:
    """Get the best available device (alias for resolve_device('auto'))."""
    return resolve_device("auto")


def setup_device(device_id: Union[int, str] = "auto") -> torch.device:
    """Setup device for model training/inference (alias for resolve_device)."""
    return resolve_device(device_id)
