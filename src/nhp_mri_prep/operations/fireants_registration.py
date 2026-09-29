"""
FireANTs registration for nhp_mri_prep (syn = affine + deformable).

Used for syn (xfm_type="syn"): runs FireANTs AffineRegistration (affine init) followed by
GreedyRegistration (deformable warp). Standalone rigid/affine-only transforms are handled by
ANTs, not here. Runs on GPU when available; FireANTs 1.5.0 also ships a pure-PyTorch CPU
baseline optimizer (selected automatically on CPU), so the SAME algorithm runs without a GPU
— gated by _use_fireants() and the `registration.fireants_allow_cpu` config flag.

Same interface and output contract as ants_cpu_register. Optional dependency:
fireants, torch, scipy. When unavailable, use ants_cpu_register.
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Union, Optional, Dict, Any, List

import numpy as np
import torch
import logging
from pathlib import Path

from fireants.io import BatchedImages, FakeBatchedImages, Image
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration

from fireants.utils.globals import MIN_IMG_SIZE

from .validation import validate_input_file, ensure_working_directory
from ..utils.gpu_device import resolve_device
from ..utils.mri import pad_image_to_min_size, crop_image_to_original
from ..utils.logger import quiet_external_output

try:
    from scipy.io import loadmat, savemat
except ImportError:
    loadmat = savemat = None


@dataclass
class FireANTsRegistrationParams:
    """FireANTs registration parameters (hardcoded defaults)."""

    scales: List[int] = field(default_factory=lambda: [4, 2, 1])
    iterations_affine: List[int] = field(default_factory=lambda: [200, 100, 50])
    iterations_deformable: List[int] = field(default_factory=lambda: [200, 100, 50])
    optimizer_affine: str = "Adam"
    optimizer_deformable: str = "Adam"
    lr_affine: float = 3e-3
    lr_deformable: float = 0.5
    cc_kernel_size: int = 5
    smooth_grad_sigma: float = 1.0
    deformation_type: str = "compositive"


DEFAULT_FIREANTS_PARAMS = FireANTsRegistrationParams()


def _compute_safe_scales(
    fixed_shape: tuple,
    moving_shape: tuple,
    default_scales: List[int],
    default_iters: List[int],
    logger: logging.Logger,
) -> tuple:
    """Adapt multi-scale pyramid to image dimensions.

    FireANTs' ``downsample_fft`` assumes the target size is *smaller* than
    the source.  When an image dimension ``d`` is small (common for low-res /
    small-FOV functional data), ``max(d // scale, MIN_IMG_SIZE)`` can exceed
    ``d``, turning a downsample into an upsample.  The FFT crop + padding-
    removal then produces a zero-length tensor and ``ifftn`` raises
    ``RuntimeError: Invalid number of data points (0) specified``.

    This function filters the requested scale list so that only scales where
    *every* spatial dimension of *both* images satisfies
    ``d // scale >= MIN_IMG_SIZE`` are kept.  Scale 1 is kept unconditionally
    (no downsampling), but images with dims < MIN_IMG_SIZE must be
    zero-padded beforehand — see the padding logic in fireants_registration().

    Returns:
        (safe_scales, safe_iters) with matching lengths.
    """
    min_dim = min(min(fixed_shape), min(moving_shape))

    safe_scales: List[int] = []
    safe_iters: List[int] = []
    for scale, iters in zip(default_scales, default_iters):
        if scale <= 1:
            # Scale 1: no downsampling performed, always safe
            safe_scales.append(scale)
            safe_iters.append(iters)
        elif min_dim // scale >= MIN_IMG_SIZE:
            safe_scales.append(scale)
            safe_iters.append(iters)
        else:
            logger.info(
                f"FireANTs: skipping scale {scale} "
                f"(min spatial dim {min_dim}, "
                f"min_dim/scale={min_dim / scale:.1f} < MIN_IMG_SIZE={MIN_IMG_SIZE})"
            )

    if not safe_scales:
        # All coarse levels were dropped — register at full resolution only
        safe_scales = [1]
        safe_iters = [default_iters[-1]]
        logger.warning(
            f"FireANTs: no multi-scale levels feasible for min spatial dim {min_dim}; "
            "using single-scale (scale=1) registration"
        )

    if safe_scales != default_scales:
        logger.info(
            f"FireANTs: adapted scales {default_scales} -> {safe_scales} "
            f"(iters {default_iters} -> {safe_iters}) "
            f"for image dims fixed={list(fixed_shape)}, moving={list(moving_shape)}"
        )

    return safe_scales, safe_iters


def _run_affine_registration(
    registration: AffineRegistration,
    device: str,
    logger: logging.Logger,
    reg_type: str = "affine",
) -> float:
    start = time.perf_counter()
    registration.optimize()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device=device)
    elapsed = time.perf_counter() - start
    logger.info(f"{reg_type.capitalize()} registration: {elapsed:.1f}s")
    return elapsed


def _run_greedy_registration(
    registration: GreedyRegistration,
    device: str,
    logger: logging.Logger,
    direction: str = "forward",
) -> float:
    start = time.perf_counter()
    registration.optimize()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device=device)
    elapsed = time.perf_counter() - start
    logger.info(f"{direction.capitalize()} deformable registration: {elapsed:.1f}s")
    return elapsed


def _save_registered_image(
    registration: Union[AffineRegistration, GreedyRegistration],
    batch_fixed: BatchedImages,
    batch_moving: BatchedImages,
    output_path: Union[str, Path],
    logger: logging.Logger,
) -> None:
    moved = registration.evaluate(batch_fixed, batch_moving)
    moved_batch = FakeBatchedImages(moved, batch_fixed)
    moved_batch.write_image(str(output_path))
    logger.info(f"Wrote registered image: {output_path}")


def _get_output_paths(output_path_prefix: str, xfm_type: str) -> Dict[str, str]:
    output_dir = Path(output_path_prefix).parent
    base_name = Path(output_path_prefix).name
    paths = {"registered": str(output_dir / f"{base_name}_registered.nii.gz")}
    if xfm_type == "syn":
        paths["forward_transform"] = str(output_dir / f"{base_name}_warp.nii.gz")
        paths["inverse_transform"] = str(
            output_dir / f"{base_name}_inverse_warp.nii.gz"
        )
    else:
        suffix = "rigid" if xfm_type == "rigid" else "affine"
        paths["forward_transform"] = str(output_dir / f"{base_name}_{suffix}.mat")
        paths["inverse_transform"] = str(
            output_dir / f"{base_name}_inverse_{suffix}.mat"
        )
    return paths


def _invert_affine_mat(
    affine_mat_path: Union[str, Path],
    inverse_mat_path: Union[str, Path],
    logger: logging.Logger,
) -> None:
    """Compute inverse of an ITK-style affine .mat and write to inverse_mat_path."""
    if loadmat is None or savemat is None:
        raise RuntimeError(
            "scipy is required to invert affine .mat (pip install scipy)"
        )
    data = loadmat(str(affine_mat_path))
    key = [
        k for k in data if k.startswith("AffineTransform_") and not k.startswith("__")
    ]
    if not key:
        raise ValueError(f"No AffineTransform_* key found in {affine_mat_path}")
    key = key[0]
    params = data[key].flatten().astype(np.float64)
    original_shape = data[key].shape
    original_dtype = data[key].dtype
    total_params = params.size
    if total_params == 12:
        dims = 3
    elif total_params == 6:
        dims = 2
    else:
        match = re.search(r"_(\d+)_\d+", key)
        if match:
            dims = int(match.group(1))
            expected_params = dims * dims + dims
            if total_params != expected_params:
                raise ValueError(
                    f"Parameter count mismatch: got {total_params}, expected {expected_params} for {dims}D transform"
                )
        else:
            raise ValueError(
                f"Cannot infer dimensions from parameter count {total_params}. Expected 6 (2D) or 12 (3D)."
            )
    matrix_size = dims * dims
    A_flat = params[:matrix_size]
    t = params[matrix_size:]
    A = A_flat.reshape(dims, dims)
    A_inv = np.linalg.inv(A)
    t_inv = -A_inv @ t
    A_inv_flat = A_inv.flatten()
    params_inv = np.concatenate([A_inv_flat, t_inv])
    params_inv = params_inv.astype(original_dtype)
    mat_inv = params_inv.reshape(-1, 1) if original_shape[1] == 1 else params_inv
    output_data = {}
    for k, v in data.items():
        if k.startswith("__"):
            continue
        if k == key:
            output_data[k] = mat_inv
        else:
            output_data[k] = v
    if "fixed" not in output_data:
        output_data["fixed"] = np.zeros((dims, 1), dtype=np.float32)
    saved = False
    for fmt in ["4", "5"]:
        try:
            savemat(str(inverse_mat_path), output_data, format=fmt, oned_as="column")
            saved = True
            break
        except (ValueError, NotImplementedError):
            continue
    if not saved:
        savemat(str(inverse_mat_path), output_data, oned_as="column")
    logger.info(f"Wrote inverse affine: {inverse_mat_path}")


def fireants_registration(
    fixedf: Union[str, Path],
    movingf: Union[str, Path],
    working_dir: Union[str, Path],
    output_prefix: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    xfm_type: Optional[str] = "syn",
    compute_inverse: Optional[bool] = True,
    device: Optional[Any] = None,
) -> Dict[str, Optional[str]]:
    """Run FireANTs (GPU) registration with same output format as ants_cpu_register.

    Args:
        fixedf: Fixed (template) image path
        movingf: Moving image path
        working_dir: Working directory
        output_prefix: Prefix for output files (default: basename of moving without .nii.gz)
        config: Configuration dictionary (optional; only used to get xfm_type if not provided)
        logger: Logger instance (optional)
        xfm_type: One of 'rigid', 'affine', 'syn'.
        device: torch.device to run on, as resolved by the ants_register dispatcher.
            None resolves config general.gpu_device here.

    Returns:
        Dictionary with keys: output_path_prefix, imagef_registered, forward_transform, inverse_transform.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if output_prefix is None:
        output_prefix = (
            os.path.basename(movingf).replace(".nii.gz", "").replace(".nii", "")
        )
    if xfm_type is None and config is not None:
        reg_config = config.get("registration", {})
        xfm_type = reg_config.get("xfm_type", "syn")
    if xfm_type is None:
        xfm_type = "syn"
    if xfm_type != "syn":
        raise ValueError(
            f"FireANTs only supports xfm_type='syn'. For rigid/affine use ants_register (dispatcher). "
            f"Got: {xfm_type}"
        )

    work_dir = ensure_working_directory(working_dir, logger)
    output_path_prefix = os.path.join(str(work_dir), output_prefix)
    logger.info(f"Data: output prefix - {output_prefix}")

    fixed_path = validate_input_file(fixedf, logger)
    moving_path = validate_input_file(movingf, logger)

    outputs: Dict[str, Optional[str]] = {
        "output_path_prefix": output_path_prefix,
        "imagef_registered": None,
        "forward_transform": None,
        "inverse_transform": None,
    }
    output_paths = _get_output_paths(output_path_prefix, xfm_type)

    # ------------------------------------------------------------------
    # Pad images to MIN_IMG_SIZE if needed.
    #
    # FireANTs' GreedyRegistration.optimize() clamps the internal warp-
    # field grid to MIN_IMG_SIZE per dimension, but at scale=1 it does
    # NOT resize the fixed image to match.  This causes a shape mismatch
    # in the loss when any fixed-image spatial dim < MIN_IMG_SIZE.
    # Since either image can be the "fixed" (forward vs inverse), both
    # must satisfy the constraint.  Padding is transparent to the caller:
    # all outputs are cropped back to the original grids before return.
    # ------------------------------------------------------------------
    original_fixed_path = fixed_path
    original_moving_path = moving_path

    padded_fixed_f = os.path.join(str(work_dir), "_padded_fixed.nii.gz")
    fixed_pad_left = pad_image_to_min_size(
        fixed_path, MIN_IMG_SIZE, padded_fixed_f, logger
    )
    if fixed_pad_left is not None:
        fixed_path = Path(padded_fixed_f)

    padded_moving_f = os.path.join(str(work_dir), "_padded_moving.nii.gz")
    moving_pad_left = pad_image_to_min_size(
        moving_path, MIN_IMG_SIZE, padded_moving_f, logger
    )
    if moving_pad_left is not None:
        moving_path = Path(padded_moving_f)

    if device is None:
        gpu_spec = "auto"
        if config is not None:
            gpu_spec = config.get("general", {}).get("gpu_device", "auto")
        device = resolve_device(gpu_spec)
    device = str(device)  # "cuda:0" or "cpu"
    if device == "cpu":
        logger.warning("Using CPU for FireANTs registration.")

    # Force FireANTs image tensors onto the resolved device.
    # Without this, Image.load_file defaults to "cuda" (typically cuda:0),
    # which can ignore our auto-selected GPU (e.g., cuda:1) and cause OOM.
    fixed_image = Image.load_file(str(fixed_path), device=device)
    moving_image = Image.load_file(str(moving_path), device=device)
    batch_fixed = BatchedImages([fixed_image])
    batch_moving = BatchedImages([moving_image])
    params = DEFAULT_FIREANTS_PARAMS

    # Adapt multi-scale pyramid to actual image dimensions.
    # Low-res / small-FOV functional images can be smaller than
    # MIN_IMG_SIZE * scale, causing downsample_fft to fail.
    fixed_shape = tuple(batch_fixed().shape[2:])
    moving_shape = tuple(batch_moving().shape[2:])
    logger.info(
        f"FireANTs: fixed shape={list(fixed_shape)}, moving shape={list(moving_shape)}"
    )

    scales_affine, iters_affine = _compute_safe_scales(
        fixed_shape,
        moving_shape,
        params.scales,
        params.iterations_affine,
        logger,
    )
    scales_deformable, iters_deformable = _compute_safe_scales(
        fixed_shape,
        moving_shape,
        params.scales,
        params.iterations_deformable,
        logger,
    )

    # syn here = FireANTs affine (init) + greedy (deformable). The preceding rigid conform
    # (sitk/flirt) already coarsely aligned the input; this affine refines it before the warp.
    logger.info("Running forward registration (fixed → moving)...")
    with quiet_external_output(logger):
        affine = AffineRegistration(
            scales_affine,
            iters_affine,
            batch_fixed,
            batch_moving,
            optimizer=params.optimizer_affine,
            optimizer_lr=params.lr_affine,
            cc_kernel_size=params.cc_kernel_size,
        )
        _run_affine_registration(affine, device, logger, reg_type="affine")
        reg = GreedyRegistration(
            scales=scales_deformable,
            iterations=iters_deformable,
            fixed_images=batch_fixed,
            moving_images=batch_moving,
            cc_kernel_size=params.cc_kernel_size,
            deformation_type=params.deformation_type,
            smooth_grad_sigma=params.smooth_grad_sigma,
            optimizer=params.optimizer_deformable,
            optimizer_lr=params.lr_deformable,
            init_affine=affine.get_affine_matrix().detach(),
        )
        _run_greedy_registration(reg, device, logger, direction="forward")
        forward_warp = output_paths["forward_transform"]
        reg.save_as_ants_transforms(forward_warp)
        logger.info(f"Saved forward warp: {forward_warp}")

        if compute_inverse:
            logger.info(
                "Running inverse registration (moving → fixed) with full pipeline (affine + warp)..."
            )
            affine_inv = AffineRegistration(
                scales_affine,
                iters_affine,
                batch_moving,
                batch_fixed,
                optimizer=params.optimizer_affine,
                optimizer_lr=params.lr_affine,
                cc_kernel_size=params.cc_kernel_size,
            )
            _run_affine_registration(affine_inv, device, logger, reg_type="affine")
            reg_inv = GreedyRegistration(
                scales=scales_deformable,
                iterations=iters_deformable,
                fixed_images=batch_moving,
                moving_images=batch_fixed,
                cc_kernel_size=params.cc_kernel_size,
                deformation_type=params.deformation_type,
                smooth_grad_sigma=params.smooth_grad_sigma,
                optimizer=params.optimizer_deformable,
                optimizer_lr=params.lr_deformable,
                init_affine=affine_inv.get_affine_matrix().detach(),
            )
            _run_greedy_registration(reg_inv, device, logger, direction="inverse")
            inverse_warp = output_paths["inverse_transform"]
            reg_inv.save_as_ants_transforms(inverse_warp)
            logger.info(f"Saved inverse warp: {inverse_warp}")
        _save_registered_image(
            reg, batch_fixed, batch_moving, output_paths["registered"], logger
        )

    # ------------------------------------------------------------------
    # Crop outputs back to original input grids if padding was applied.
    # Forward warp + registered image live in fixed-image space;
    # inverse warp lives in moving-image space.
    # Only NIfTI files need cropping (.mat affine transforms are grid-
    # independent).
    # ------------------------------------------------------------------
    if fixed_pad_left is not None:
        for p in [
            output_paths.get("registered"),
            output_paths.get("forward_transform"),
        ]:
            if p and p.endswith(".nii.gz") and os.path.exists(p):
                crop_image_to_original(
                    p, str(original_fixed_path), fixed_pad_left, p, logger
                )
    if moving_pad_left is not None and compute_inverse:
        p = output_paths.get("inverse_transform")
        if p and p.endswith(".nii.gz") and os.path.exists(p):
            crop_image_to_original(
                p, str(original_moving_path), moving_pad_left, p, logger
            )

    # Clean up temporary padded files
    for tmp in [padded_fixed_f, padded_moving_f]:
        if os.path.exists(tmp):
            os.remove(tmp)

    for key, path in [
        ("imagef_registered", output_paths["registered"]),
        ("forward_transform", output_paths["forward_transform"]),
        (
            "inverse_transform",
            output_paths["inverse_transform"] if compute_inverse else None,
        ),
    ]:
        if path is not None and os.path.exists(path):
            outputs[key] = path
            logger.info(f"Output: {key} created - {path}")
        elif path is not None:
            logger.warning(f"Data: expected {key} not found - {path}")

    logger.info(
        f"Step: registration completed with {len([k for k, v in outputs.items() if v is not None])} output files - {list(outputs.keys())}"
    )
    return outputs
