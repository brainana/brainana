"""
Centralized GPU device selection for brainana.

Single source of truth for device resolution. Respects CUDA_VISIBLE_DEVICES
when set (e.g. by Nextflow); otherwise uses nvidia-smi / PyTorch for
least-busy GPU selection.

Canonical spec: "auto" | -1 (CPU) | 0..N (GPU index) | "cuda:N" | "cpu"
"""

from __future__ import annotations

import os
import subprocess
from typing import Union

import torch

from fastsurfer_nn.utils.gpu_utils import cuda_available


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


def _get_least_busy_gpu() -> int:
    """Get the GPU index with the least memory usage.

    When CUDA_VISIBLE_DEVICES is set, returns 0 (only visible GPU).
    Otherwise uses nvidia-smi or PyTorch fallback.
    """
    if not _cuda_is_usable():
        return 0

    gpu_count = torch.cuda.device_count()
    if gpu_count == 1:
        return 0

    # When CUDA_VISIBLE_DEVICES restricts visibility, use cuda:0 (the only visible GPU)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and cuda_visible.strip():
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
            if memory_usages:
                return memory_usages.index(min(memory_usages))
    except (subprocess.SubprocessError, FileNotFoundError, ValueError, IndexError):
        pass

    # Fallback: PyTorch memory allocation
    min_memory = float("inf")
    best_gpu = 0
    for i in range(gpu_count):
        mem = torch.cuda.memory_allocated(i)
        if mem < min_memory:
            min_memory = mem
            best_gpu = i
    return best_gpu


def resolve_device(spec: Union[str, int, torch.device] = "auto") -> torch.device:
    """Resolve a device spec to a torch.device.

    Args:
        spec: Device specification:
            - "auto": least-busy GPU or CPU; when CUDA_VISIBLE_DEVICES is set,
              uses cuda:0 (the only visible GPU)
            - -1 or "cpu": CPU
            - 0, 1, ...: GPU index (cuda:N)
            - "cuda:0", "cuda:1", ...: explicit cuda device
            - torch.device: returned as-is (after validation)

    Returns:
        torch.device object
    """
    if isinstance(spec, torch.device):
        return spec

    s = str(spec).strip().lower()
    if s == "auto" or spec is None:
        if _cuda_is_usable():
            idx = _get_least_busy_gpu()
            return torch.device(f"cuda:{idx}")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if s == "cpu" or spec == -1:
        return torch.device("cpu")

    if s.startswith("cuda:"):
        return torch.device(s)

    if s.startswith("mps"):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # Treat as GPU index
    try:
        idx = int(spec)
        if idx < 0:
            return torch.device("cpu")
        return torch.device(f"cuda:{idx}")
    except (TypeError, ValueError):
        return torch.device("cpu")


# Backwards-compatible aliases for migration
def get_device() -> torch.device:
    """Get the best available device (alias for resolve_device('auto'))."""
    return resolve_device("auto")


def setup_device(device_id: Union[int, str] = "auto") -> torch.device:
    """Setup device for model training/inference (alias for resolve_device)."""
    return resolve_device(device_id)
