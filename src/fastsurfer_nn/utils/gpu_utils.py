"""
GPU utilities for automatic device selection.
"""

import os
import torch
import subprocess
from typing import Union


def cuda_hidden() -> bool:
    """True when CUDA_VISIBLE_DEVICES deliberately hides every GPU.

    The pipeline exports CUDA_VISIBLE_DEVICES="" for CPU-mode tasks. Callers
    must then skip torch.cuda probes entirely: torch.cuda.is_available() still
    loads and initializes the CUDA driver, which is unnecessary in CPU mode and
    can abort the process outright on some drivers (WSL2 GPU passthrough).
    """
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    return value is not None and value.strip().lower() in ("", "-1", "none")


def cuda_available() -> bool:
    """torch.cuda.is_available(), without touching the driver when CUDA is hidden."""
    return not cuda_hidden() and torch.cuda.is_available()


def get_device():
    """Get the best available device (least busy GPU or CPU)."""
    if cuda_available():
        return torch.device(f"cuda:{get_least_busy_gpu()}")
    return torch.device("cpu")


def get_least_busy_gpu():
    """Get the GPU with the least memory usage based on system-wide memory usage."""
    if not cuda_available():
        return 0

    gpu_count = torch.cuda.device_count()
    if gpu_count == 1:
        return 0

    try:
        # Use nvidia-smi to get actual system-wide memory usage and total memory
        result_used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        result_total = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result_used.returncode == 0 and result_total.returncode == 0:
            memory_usages = []
            memory_totals = []
            for line in result_used.stdout.strip().split("\n"):
                if line.strip():
                    memory_usages.append(int(line.strip()))
            for line in result_total.stdout.strip().split("\n"):
                if line.strip():
                    memory_totals.append(int(line.strip()))

            # Find GPU with minimum memory usage (maximum unused memory)
            if (
                memory_usages
                and memory_totals
                and len(memory_usages) == len(memory_totals)
            ):
                best_gpu = memory_usages.index(min(memory_usages))
                memory_unused = memory_totals[best_gpu] - memory_usages[best_gpu]
                print(f"[GPU Selection] GPU memory usage: {memory_usages} MB")
                print(
                    f"[GPU Selection] Selected GPU {best_gpu} with {memory_unused} MB unused"
                )
                return best_gpu

    except (subprocess.SubprocessError, FileNotFoundError, ValueError, IndexError) as e:
        print(
            f"[GPU Selection] nvidia-smi failed ({e}), falling back to PyTorch memory check"
        )
        pass

    # Fallback: Check memory usage for each GPU using PyTorch
    min_memory = float("inf")
    best_gpu = 0

    for i in range(gpu_count):
        torch.cuda.set_device(i)
        memory_used = torch.cuda.memory_allocated(i)
        if memory_used < min_memory:
            min_memory = memory_used
            best_gpu = i

    print(f"[GPU Selection] Selected GPU {best_gpu} (PyTorch fallback)")
    return best_gpu


def setup_device(device_id: Union[int, str] = "auto") -> torch.device:
    """Setup device for model training/inference.

    Args:
        device_id: Device specification ('auto', -1 for CPU, specific GPU index, or full device string)

    Returns:
        torch.device object
    """
    if device_id == "auto":
        return get_device()
    elif device_id == -1 or device_id == "cpu":
        return torch.device("cpu")
    elif isinstance(device_id, str) and device_id.startswith("cuda:"):
        # If it's already a full cuda device string, use it directly
        return torch.device(device_id)
    elif isinstance(device_id, torch.device):
        # If it's already a torch.device object, return it directly
        return device_id
    else:
        # Otherwise, treat it as a GPU index
        return torch.device(f"cuda:{device_id}")
