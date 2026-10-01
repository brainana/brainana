"""
Centralized GPU device selection for brainana (public name).

Re-exports fastsurfer_nn.utils.gpu_utils, the single implementation of the device
policy (see its module docstring). It is implemented there because nhp_mri_prep
imports fastsurfer_nn at package import time; the reverse import would be circular.
"""

from fastsurfer_nn.utils.gpu_utils import (  # noqa: F401
    _cuda_is_usable,
    _get_least_busy_gpu,
    cuda_available,
    cuda_hidden,
    device_report,
    get_device,
    get_least_busy_gpu,
    is_cuda_oom,
    resolve_device,
    run_with_cpu_fallback,
    setup_device,
)

__all__ = [
    "cuda_available",
    "cuda_hidden",
    "device_report",
    "get_device",
    "get_least_busy_gpu",
    "is_cuda_oom",
    "resolve_device",
    "run_with_cpu_fallback",
    "setup_device",
]
