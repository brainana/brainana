"""
Registration functions for nhp_mri_prep.

This module contains functions for image registration using ANTs and FLIRT,
including command composition and transformation application.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path


try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

from .validation import (
    validate_input_file,
    ensure_working_directory,
    validate_output_file,
)
from ..utils import run_command, calculate_func_tmean
from ..utils.bids import get_filename_stem
from ..config import get_config

# Default registration step parameters (hardcoded)
REGISTRATION_STEP_DEFAULTS = {
    "translation": {
        "gradient_step": "[0.1]",
        "metric": ["MI[fixed,moving,1,32,regular,0.25]"],
        "shrink": "8x4x2x1",
        "convergence": "[1000x500x250x100,1e-6,10]",
        "smooth": "3x2x1x0vox",
    },
    "rigid": {
        "gradient_step": "[0.1]",
        "metric": ["MI[fixed,moving,1,32,regular,0.25]"],
        "shrink": "8x4x2x1",
        "convergence": "[1000x500x250x100,1e-6,10]",
        "smooth": "3x2x1x0vox",
    },
    "affine": {
        "gradient_step": "[0.1]",
        "metric": ["MI[fixed,moving,1,32,regular,0.25]"],
        "shrink": "8x4x2x1",
        "convergence": "[1000x500x250x100,1e-6,10]",
        "smooth": "3x2x1x0vox",
    },
    "syn": {
        "gradient_step": "[0.1,3,0]",
        "metric": [
            "mattes[fixed,moving,0.5,32,regular,0.3]",
            "cc[fixed,moving,0.5,4,regular,0.3]",
        ],
        "shrink": "4x2x1",
        "convergence": "[500x300x100,1e-8,10]",
        "smooth": "1x0.5x0vox",
    },
}


# %%
def compose_transform_cmd(
    transform: str,
    config: Dict[str, Any],
    fixedf: Union[str, Path],
    movingf: Union[str, Path],
) -> List[str]:
    """Compose ANTs transform command arguments.

    Args:
        transform: Transform type
        config: Transform configuration
        fixedf: Fixed image path
        movingf: Moving image path

    Returns:
        List of command arguments
    """
    cmd = [
        "--transform",
        f"{transform}{config.get('gradient_step')}",
        "--convergence",
        f"{config.get('convergence')}",
        "--shrink-factors",
        f"{config.get('shrink')}",
        "--smoothing-sigmas",
        f"{config.get('smooth')}",
    ]

    # Add metrics
    metrics = config.get("metric", [])
    if metrics:
        for metric in metrics:
            metric_str = metric.replace("fixed", str(fixedf)).replace(
                "moving", str(movingf)
            )
            cmd.extend(["--metric", metric_str])

    return cmd


def compose_ants_registration_cmd(
    fixedf: Union[str, Path],
    movingf: Union[str, Path],
    output_path_prefix: str,
    config: Dict[str, Any],
    xfm_type: str = "affine",
) -> List[str]:
    """Compose ANTs registration command arguments.

    Args:
        fixedf: Fixed (template) image path
        movingf: Moving image path
        config: Registration configuration
        xfm_type: Type of transformation to perform. Options: 'translation', 'rigid', 'affine', 'syn'

    Returns:
        List of command arguments
    """
    cmd = [
        "antsRegistration",
        "--dimensionality",
        "3",
        "--float",
        "0",
        "--interpolation",
        config.get("interpolation", "Linear"),
        "--use-histogram-matching",
        "0",
        "--initial-moving-transform",
        f"[{str(fixedf)},{str(movingf)},1]",
        "--winsorize-image-intensities",
        "[0.005,0.995]",
        "--output",
        f"[{output_path_prefix}_,{output_path_prefix}_registered.nii.gz]",
        "--write-composite-transform",
        "1",
    ]

    # Define the order of transformation stages
    stages = ["translation", "rigid", "affine", "syn"]

    # Find the index of the xfm_type
    try:
        xfm_index = stages.index(xfm_type)
    except ValueError:
        raise ValueError(f"Invalid xfm_type value: {xfm_type}. Must be one of {stages}")

    # Enable initialize-transforms-per-stage when multiple linear stages are used
    # This allows each linear stage to initialize from the previous stage's transform
    # (e.g., Translation -> Rigid -> Affine)
    if xfm_index >= 1:  # More than just translation (rigid, affine, or syn)
        cmd.extend(["--initialize-transforms-per-stage", "1"])
        # disable collapse-output-transforms when initialize-transforms-per-stage is enabled
        cmd.extend(["--collapse-output-transforms", "0"])
    else:
        cmd.extend(["--collapse-output-transforms", "1"])

    # Add transformation stages up to and including the xfm_type
    for i, stage_name in enumerate(stages):
        if i > xfm_index:
            break
        # Use hardcoded defaults for registration step parameters
        stage_config = REGISTRATION_STEP_DEFAULTS.get(stage_name)
        if stage_config is None:
            raise ValueError(f"Unknown registration stage: {stage_name}")
        stage_cmd = compose_transform_cmd(stage_name, stage_config, fixedf, movingf)
        cmd.extend(stage_cmd)

    return cmd


def ants_cpu_register(
    fixedf: Union[str, Path],
    movingf: Union[str, Path],
    working_dir: Union[str, Path],
    output_prefix: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    xfm_type: Optional[str] = "syn",
    compute_inverse: Optional[bool] = True,
) -> Dict[str, str]:
    """Run ANTs registration (CPU, subprocess antsRegistration).

    Args:
        fixedf: Fixed (template) image path
        movingf: Moving image path
        working_dir: Working directory
        output_prefix: Prefix for output files
        config: Configuration dictionary (optional, will use default if not provided)
        logger: Logger instance (optional, will create if not provided)
        xfm_type: Type of transformation to perform. Options: 'translation', 'rigid', 'affine', 'syn'

    Returns:
        Dictionary containing paths to output files

    Raises:
        FileNotFoundError: If input files don't exist
        RuntimeError: If registration fails
        ValueError: If configuration is invalid
    """
    # Use provided config/logger or get defaults
    if config is None:
        config = get_config().to_dict()
    if logger is None:
        logger = logging.getLogger(__name__)
    if output_prefix is None:
        output_prefix = get_filename_stem(Path(movingf))

    # Validate inputs and setup
    work_dir = ensure_working_directory(working_dir, logger)
    output_path_prefix = os.path.join(str(work_dir), output_prefix)
    logger.info(f"Data: output prefix - {output_prefix}")

    # Get registration configuration with proper nesting access
    reg_config = config.get("registration")
    if not reg_config:
        raise ValueError("Registration configuration not found in config")

    # Validate input files
    fixed_path = validate_input_file(fixedf, logger)
    moving_path = validate_input_file(movingf, logger)

    # Initialize outputs dictionary
    outputs = {
        "output_path_prefix": output_path_prefix,
        "imagef_registered": None,
        "forward_transform": None,
        "inverse_transform": None,
    }

    # Get thread count from environment variable (set by Nextflow)
    num_threads = int(os.environ.get("OMP_NUM_THREADS", 8))

    # Set up ITK thread environment variables for subprocess
    from ..utils.system import set_numerical_threads

    env = set_numerical_threads(num_threads, include_itk=True, return_dict=True)
    # Merge with current environment to preserve other variables
    env = {**os.environ, **env}

    # Build ANTs registration command
    command_ants = compose_ants_registration_cmd(
        fixed_path, moving_path, output_path_prefix, reg_config, xfm_type=xfm_type
    )
    # Normalize verbose to integer (0, 1, or 2) and convert to ANTs format (0 or 1)
    from ..utils.logger import normalize_verbose

    verbose = normalize_verbose(config.get("general", {}).get("verbose", 1))
    command_ants.extend(["--verbose", str(1 if verbose >= 1 else 0)])

    # Execute ANTs registration with thread-limited environment
    try:
        logger.info(
            f"Step: executing ANTs registration command with {len(command_ants)} arguments"
        )
        logger.info(
            f"System: using {num_threads} threads for ITK operations (capped at 32)"
        )
        returncode, stdout, stderr = run_command(
            command_ants, env=env, step_logger=logger
        )

        if returncode != 0:
            raise RuntimeError(
                f"antsRegistration failed (exit code {returncode}): {stderr}"
            )

        logger.info("Step: ANTs registration completed successfully")

    except Exception as e:
        logger.error(f"Step: ANTs registration failed - {str(e)}")
        raise

    # Collect and validate output files (outputs dict already initialized above)

    # Main registered image
    registered_image = os.path.join(
        os.path.dirname(output_path_prefix),
        os.path.basename(output_path_prefix) + "_registered.nii.gz",
    )
    if os.path.exists(registered_image):
        validate_output_file(registered_image, logger)
        outputs["imagef_registered"] = registered_image
        logger.info(f"Output: registered image created - {registered_image}")
    else:
        logger.warning(
            f"Data: expected registered image not found - {registered_image}"
        )

    # Transform files
    forward_transform = os.path.join(
        os.path.dirname(output_path_prefix),
        os.path.basename(output_path_prefix) + "_Composite.h5",
    )
    if os.path.exists(forward_transform):
        outputs["forward_transform"] = forward_transform
        logger.info(f"Output: forward transform created - {forward_transform}")

    inverse_transform = os.path.join(
        os.path.dirname(output_path_prefix),
        os.path.basename(output_path_prefix) + "_InverseComposite.h5",
    )
    if compute_inverse and os.path.exists(inverse_transform):
        outputs["inverse_transform"] = inverse_transform
        logger.info(f"Output: inverse transform created - {inverse_transform}")

    logger.info(
        f"Step: registration completed with {len(outputs)} output files - {list(outputs.keys())}"
    )
    return outputs


def _use_fireants(logger: logging.Logger, allow_cpu: bool = True, device=None) -> bool:
    """Return True when FireANTs can run on ``device`` — GPU, or CPU when ``allow_cpu``.

    ``device`` is the torch.device the registration will use (resolved once by the
    caller from general.gpu_device). The probe runs on that device only, so a task
    that resolved to CPU never touches the CUDA driver here. None resolves "auto".

    FireANTs 1.5.0 selects its optimizer per device: the compiled CUDA fused Adam
    on a CUDA tensor, and a device-agnostic PyTorch baseline
    (``adam_update_fused_baseline``) on CPU (see fireants optimizers/adam.py). The
    rest of the greedy/affine path is likewise CPU-safe (non-FFT downsampling and the
    non-fused interpolator/CC kernels are auto-selected on CPU). So FireANTs can run
    on CPU — benchmarked faster and equal/better quality vs antspyx SyN — not "GPU only".

    Gate logic:
    1. FireANTs importable?
    2. A usable GPU exists? → probe the fused Adam on a CUDA tensor (catches
       driver/runtime mismatches where basic CUDA works but the compiled ext fails);
       ``torch.cuda.synchronize()`` surfaces async errors immediately.
    3. else, if ``allow_cpu`` → probe the baseline Adam on a CPU tensor. If False,
       skip FireANTs and let the dispatcher fall back to ANTs/antspyx.
    """
    try:
        from .fireants_registration import fireants_registration  # noqa: F401
    except ImportError as e:
        logger.debug(
            f"REGISTRATION: FireANTs skipped — not installed or dependency missing: {e}"
        )
        return False

    from ..utils.gpu_device import resolve_device
    import torch

    if device is None:
        device = resolve_device("auto")

    if device.type == "cuda":
        try:
            from fireants.registration.optimizers.adam import adam_update_fused

            t = torch.zeros(4, device=device, dtype=torch.float32)
            adam_update_fused(t, t.clone(), t.clone(), 1.0, 1.0, 1e-8)
            torch.cuda.synchronize(device)
        except Exception as e:
            logger.debug(
                f"REGISTRATION: FireANTs skipped — GPU fused ops probe failed: {e}"
            )
            return False
        return True

    if device.type != "cpu":
        logger.info(
            f"REGISTRATION: FireANTs skipped — device {device} is not supported (CUDA or CPU only)"
        )
        return False

    # CPU device (no usable GPU, or CPU mode).
    if not allow_cpu:
        logger.debug(
            "REGISTRATION: FireANTs skipped — no usable GPU (fireants_allow_cpu=False)"
        )
        return False
    try:
        from fireants.registration.optimizers.adam import adam_update_fused_baseline

        t = torch.zeros(4, dtype=torch.float32)  # CPU tensor
        adam_update_fused_baseline(t, t.clone(), t.clone(), 1.0, 1.0, 1e-8)
    except Exception as e:
        logger.debug(f"REGISTRATION: FireANTs CPU baseline probe failed: {e}")
        return False
    logger.info("REGISTRATION: CPU device — FireANTs will run on CPU (baseline optimizer)")
    return True


# Conform scanner→template FLIRT defaults (used by conform_to_template).
FLIRT_CONFORM_CONFIG_ANAT: Dict[str, Any] = {
    "registration": {
        "flirt": {
            "cost": "corratio",
            "searchcost": "corratio",
            "searchrx": (-180, 180),
            "searchry": (-180, 180),
            "searchrz": (-180, 180),
            "coarsesearch": 40,
            "finesearch": 15,
        }
    }
}

FLIRT_CONFORM_CONFIG_FUNC: Dict[str, Any] = {
    "registration": {
        "flirt": {
            "cost": "mutualinfo",
            "searchcost": "mutualinfo",
            "searchrx": (-180, 180),
            "searchry": (-180, 180),
            "searchrz": (-180, 180),
            "coarsesearch": 30,
            "finesearch": 10,
        }
    }
}


def flirt_config_for_modality(modality: str) -> Dict[str, Any]:
    """Return FLIRT config for conform_to_template (anat vs func)."""
    if modality == "func":
        return FLIRT_CONFORM_CONFIG_FUNC
    return FLIRT_CONFORM_CONFIG_ANAT


def flirt_register(
    fixedf: Union[str, Path],
    movingf: Union[str, Path],
    working_dir: Union[str, Path],
    output_prefix: str,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    dof: Optional[int] = 6,
) -> Dict[str, str]:
    """Run FLIRT rigid registration.

    Args:
        fixedf: Fixed (reference) image path
        movingf: Moving image path
        working_dir: Working directory
        output_prefix: Prefix for output files
        config: Configuration dictionary (optional, will use default if not provided)
        logger: Logger instance (optional, will create if not provided)
        dof: Degrees of freedom for registration (default: 6 for rigid)

    Returns:
        Dictionary containing paths to output files:
        - 'forward_transform': Forward transformation matrix (.mat file)
        - 'inverse_transform': Inverse transformation matrix (.mat file)

    Raises:
        FileNotFoundError: If input files don't exist
        RuntimeError: If registration fails
        ValueError: If configuration is invalid
    """
    # Use provided config/logger or get defaults
    if config is None:
        config = get_config().to_dict()
    if logger is None:
        logger = logging.getLogger(__name__)

    # Validate inputs and setup
    work_dir = ensure_working_directory(working_dir, logger)

    # Validate input files
    fixed_path = validate_input_file(fixedf, logger)
    moving_path = validate_input_file(movingf, logger)

    # Build output file paths
    forward_transform = work_dir / f"{output_prefix}.mat"
    inverse_transform = work_dir / f"{output_prefix}_inverse.mat"

    # Get optional FLIRT parameters from config
    reg_config = config.get("registration", {})
    flirt_config = reg_config.get("flirt", {})

    # Helper function to extract simple parameter from config
    def get_param(config_key: str, default: Any) -> str:
        """Extract parameter from config and convert to string."""
        return str(flirt_config.get(config_key, default))

    # Helper function to extract search range from config
    def get_search_range(
        config_key: str, default_min: int = -180, default_max: int = 180
    ) -> tuple[str, str]:
        """Extract min/max search range from config, supporting list/tuple or dict formats."""
        value = flirt_config.get(config_key)
        if not value:
            return str(default_min), str(default_max)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return str(value[0]), str(value[1])
        elif isinstance(value, dict):
            return str(value.get("min", default_min)), str(
                value.get("max", default_max)
            )
        return str(default_min), str(default_max)

    # Extract all FLIRT parameters from config
    cost = get_param("cost", "mutualinfo")
    searchcost = get_param("searchcost", "mutualinfo")
    searchrx_min, searchrx_max = get_search_range("searchrx")
    searchry_min, searchry_max = get_search_range("searchry")
    searchrz_min, searchrz_max = get_search_range("searchrz")
    coarsesearch = get_param("coarsesearch", 30)
    finesearch = get_param("finesearch", 10)

    # Build FLIRT registration command
    cmd = [
        "flirt",
        "-in",
        str(moving_path),
        "-ref",
        str(fixed_path),
        "-dof",
        str(dof),
        "-cost",
        cost,
        "-searchcost",
        searchcost,
        "-searchrx",
        searchrx_min,
        searchrx_max,
        "-searchry",
        searchry_min,
        searchry_max,
        "-searchrz",
        searchrz_min,
        searchrz_max,
        "-coarsesearch",
        coarsesearch,
        "-finesearch",
        finesearch,
        "-omat",
        str(forward_transform),
    ]

    # Add optional FLIRT parameters if specified in config
    bins = flirt_config.get("bins")
    if bins is not None:
        cmd.extend(["-bins", str(bins)])

    logger.debug(f"Command: {' '.join(cmd)}")

    # Execute FLIRT registration (matching preprocessing.py pattern)
    try:
        logger.info("Step: FLIRT rigid registration")
        returncode, stdout, stderr = run_command(cmd, step_logger=logger)
        if returncode != 0:
            raise RuntimeError(
                f"flirt registration failed (exit code {returncode}): {stderr}"
            )

        # Validate transformation matrix exists
        validate_output_file(forward_transform, logger)
        logger.info(f"Transformation matrix saved to: {forward_transform}")

    except Exception as e:
        logger.error(f"Error during FLIRT registration: {e}")
        raise RuntimeError(f"FLIRT registration failed: {e}")

    # Initialize outputs dictionary
    outputs = {
        "forward_transform": str(forward_transform),
    }

    # Compute inverse transform using convert_xfm
    try:
        logger.info("Step: computing inverse transformation matrix")
        cmd_inverse = [
            "convert_xfm",
            "-omat",
            str(inverse_transform),
            "-inverse",
            str(forward_transform),
        ]
        logger.debug(f"Command: {' '.join(cmd_inverse)}")
        returncode, stdout, stderr = run_command(cmd_inverse, step_logger=logger)

        if returncode != 0:
            raise RuntimeError(f"convert_xfm failed (exit code {returncode}): {stderr}")

        if os.path.exists(inverse_transform):
            outputs["inverse_transform"] = str(inverse_transform)
            logger.info(f"Output: inverse transform created - {inverse_transform}")
        else:
            logger.warning(
                f"Data: expected inverse transform not found - {inverse_transform}"
            )

    except Exception as e:
        logger.warning(f"Step: inverse transform computation failed - {str(e)}")
        # Don't raise - inverse transform is optional

    return outputs


def flirt_apply_transforms(
    movingf: Union[str, Path],
    outputf_name: Union[str, Path],
    reff: Union[str, Path],
    working_dir: Union[str, Path],
    transformf: Union[str, Path],
    logger: Optional[logging.Logger] = None,
    interpolation: Optional[str] = "trilinear",
    generate_tmean: Optional[bool] = True,
) -> Dict[str, str]:
    """Apply FLIRT transformation to an image.

    Args:
        movingf: Moving (source) image to transform
        outputf_name: Name for output file
        reff: Reference image for output space
        working_dir: Working directory for outputs
        transformf: FLIRT transformation matrix file (.mat)
        logger: Logger instance
        interpolation: Interpolation method (default: 'trilinear')
        generate_tmean: Whether to generate temporal mean (for functional data)

    Returns:
        Dictionary with output file paths:
        - 'imagef_registered': Transformed output image
        - 'imagef_registered_tmean': Temporal mean (if generate_tmean=True and applicable)

    Raises:
        FileNotFoundError: If input files don't exist
        RuntimeError: If transformation application fails
    """
    # Validate inputs
    if logger is None:
        logger = logging.getLogger(__name__)

    movingf = validate_input_file(movingf, logger)
    reff = validate_input_file(reff, logger)
    transformf = validate_input_file(transformf, logger)
    work_dir = ensure_working_directory(working_dir, logger)

    outputf_name = work_dir / outputf_name

    # Build FLIRT apply transform command (matching preprocessing.py pattern)
    cmd = [
        "flirt",
        "-in",
        str(movingf),
        "-ref",
        str(reff),
        "-out",
        str(outputf_name),
        "-applyxfm",
        "-init",
        str(transformf),
        "-interp",
        interpolation,
    ]

    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        logger.info("Step: applying affine transformation")
        returncode, stdout, stderr = run_command(cmd, step_logger=logger)
        if returncode != 0:
            raise RuntimeError(
                f"flirt applyxfm failed (exit code {returncode}): {stderr}"
            )

        validate_output_file(outputf_name, logger)
        logger.info(f"Transformed image saved to: {outputf_name}")

        outputs = {"imagef_registered": str(outputf_name)}

        # Generate temporal mean if requested
        if generate_tmean:
            output_stem = get_filename_stem(Path(outputf_name))
            tmean_filename = f"{output_stem}_tmean.nii.gz"
            tmean_file = work_dir / tmean_filename
            calculate_func_tmean(str(outputf_name), str(tmean_file), logger)
            outputs["imagef_registered_tmean"] = str(tmean_file)
            logger.info(f"Output: tmean generated - {tmean_file}")

        return outputs

    except Exception as e:
        logger.error(f"Error during affine transformation application: {e}")
        raise RuntimeError(f"Transform application failed: {e}")


def ants_register(
    fixedf: Union[str, Path],
    movingf: Union[str, Path],
    working_dir: Union[str, Path],
    output_prefix: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    xfm_type: Optional[str] = "syn",
    compute_inverse: Optional[bool] = True,
    enable_fireants: bool = True,
    fireants_allow_cpu: bool = True,
) -> Dict[str, Any]:
    """Run ANTs-style registration: FireANTs when available, otherwise ANTs CPU.

    Dispatcher: when FireANTs is installed and enabled for syn transforms, runs
    fireants_registration; otherwise runs ants_cpu_register (subprocess antsRegistration).

    Args:
        fixedf: Fixed (template) image path
        movingf: Moving image path
        working_dir: Working directory
        output_prefix: Prefix for output files
        config: Configuration dictionary (optional)
        logger: Logger instance (optional)
        xfm_type: Type of transformation. Options: 'translation', 'rigid', 'affine', 'syn'
        compute_inverse: If True (default), compute and include inverse transform in outputs.
        enable_fireants: If True (default), allow FireANTs for syn registration.
        fireants_allow_cpu: If True (default), allow FireANTs to run its syn (affine+greedy)
            on CPU via the PyTorch baseline optimizer when no GPU is present (same algorithm
            as the GPU path; benchmarked faster and equal/better quality than antspyx SyN).
            Set False to fall back to ANTs/antspyx on CPU instead.

    Returns:
        Dictionary with output_path_prefix, imagef_registered, forward_transform, inverse_transform
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    # FireANTs is only used for syn (affine+greedy); rigid/affine-only transforms use ANTs.
    fireants_device = None
    fireants_available = False
    if enable_fireants and xfm_type == "syn":
        from ..utils.gpu_device import resolve_device

        gpu_spec = ((config or {}).get("general") or {}).get("gpu_device", "auto")
        try:
            fireants_device = resolve_device(gpu_spec)
        except (RuntimeError, ValueError) as e:
            logger.warning(f"REGISTRATION: FireANTs skipped — {e}")
        else:
            fireants_available = _use_fireants(
                logger, allow_cpu=fireants_allow_cpu, device=fireants_device
            )
    use_fireants = enable_fireants and xfm_type == "syn" and fireants_available
    if use_fireants:
        logger.info("REGISTRATION: using FireANTs for syn")
        from .fireants_registration import fireants_registration

        from ..utils.gpu_device import run_with_cpu_fallback

        def _fireants_on(dev):
            return fireants_registration(
                fixedf=fixedf,
                movingf=movingf,
                working_dir=working_dir,
                output_prefix=output_prefix,
                config=config,
                logger=logger,
                xfm_type=xfm_type,
                compute_inverse=compute_inverse,
                device=dev,
            )

        try:
            # A CUDA out-of-memory error reruns FireANTs once on the CPU (same
            # algorithm) when fireants_allow_cpu; any other error falls back to ANTs.
            if fireants_allow_cpu:
                result = run_with_cpu_fallback(
                    _fireants_on, fireants_device, "FireANTs registration", logger
                )
            else:
                result = _fireants_on(fireants_device)
            logger.info("REGISTRATION: completed with FireANTs")
            if isinstance(result, dict):
                result["engine"] = "fireants"
            return result
        except Exception as e:
            import traceback

            logger.warning(
                f"FireANTs registration failed — falling back to ANTs (CPU).\n"
                f"Exception: {type(e).__name__}: {e}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
    reason = (
        "FireANTs disabled by configuration"
        if not enable_fireants
        else f"FireANTs only applies to syn, got xfm_type={xfm_type!r}"
        if xfm_type != "syn"
        else "fallback after FireANTs error"
        if use_fireants
        else "FireANTs unavailable (not installed, or no GPU with fireants_allow_cpu=False)"
    )
    # CPU backend: ANTs CLI if available, else antspyx (Python), else error.
    from .antspyx_ops import cli_available, antspyx_available, antspyx_register

    reg_kwargs = dict(
        fixedf=fixedf,
        movingf=movingf,
        working_dir=working_dir,
        output_prefix=output_prefix,
        config=config,
        logger=logger,
        xfm_type=xfm_type,
        compute_inverse=compute_inverse,
    )
    if cli_available("antsRegistration"):
        logger.info(f"REGISTRATION: using ANTs (CPU) — {reason}")
        result = ants_cpu_register(**reg_kwargs)
        logger.info("REGISTRATION: completed with ANTs (CPU)")
        engine = "ants"
    elif antspyx_available():
        logger.info(
            f"Workflow: antsRegistration CLI not found — using antspyx backend ({reason})"
        )
        result = antspyx_register(**reg_kwargs)
        engine = "antspyx"
    else:
        raise RuntimeError(
            "Registration requires the ANTs CLI (antsRegistration) on PATH or the antspyx "
            "package (`pip install antspyx`); neither is available."
        )
    # Record the actual engine used (survives the FireANTs->CPU fallback above) so
    # transform sidecars can attribute GeneratedBy to the real backend.
    if isinstance(result, dict):
        result["engine"] = engine
    return result


def ants_apply_transforms(
    movingf: Union[str, Path],
    moving_type: int,
    interpolation: str,
    outputf_name: Union[str, Path],
    fixedf: Union[str, Path],
    working_dir: Union[str, Path],
    transformf: Union[list[Union[str, Path]], Union[str, Path]],
    logger: Optional[logging.Logger] = None,
    reff: Optional[Union[str, Path]] = None,
    generate_tmean: Optional[bool] = True,
) -> Dict[str, str]:
    """Apply ANTs transformations to functional data.

    Args:
        movingf: Moving (source) functional image
        moving_type: Type of moving image (0:scalar, 1:vector, 2:tensor, 3:time series)
        interpolation: Interpolation type
        outputf_name: Name for output file
        fixedf: Fixed (target) image
        working_dir: Working directory for outputs
        transformf: List of transformation files to apply. May be .nii.gz (displacement field,
            e.g. FireANTs), .h5 (ANTs composite), or .mat (affine). Paths are resolved so ITK
            picks the correct reader by extension.
        logger: Logger instance
        reff: Reference image for output space (optional)
        generate_tmean: Whether to generate temporal mean

    Returns:
        Dictionary with output file paths

    Raises:
        FileNotFoundError: If input files don't exist
        RuntimeError: If transformation application fails
    """
    # Validate inputs
    if logger is None:
        logger = logging.getLogger(__name__)

    movingf = validate_input_file(movingf, logger)
    fixedf = validate_input_file(fixedf, logger)
    if reff is not None:
        reff = validate_input_file(reff, logger)
    work_dir = ensure_working_directory(working_dir, logger)

    outputf_name = work_dir / outputf_name

    # Ensure transformf is a list
    if not isinstance(transformf, list):
        transformf = [transformf]

    # Backend: ANTs CLI if available, else antspyx (Python), else error.
    from .antspyx_ops import cli_available, antspyx_available, antspyx_apply_transforms

    if not cli_available("antsApplyTransforms"):
        if antspyx_available():
            logger.info(
                "Workflow: antsApplyTransforms CLI not found — using antspyx backend"
            )
            return antspyx_apply_transforms(
                movingf=movingf,
                moving_type=moving_type,
                interpolation=interpolation,
                outputf_name=Path(outputf_name).name,
                fixedf=fixedf,
                working_dir=work_dir,
                transformf=transformf,
                logger=logger,
                reff=reff,
                generate_tmean=generate_tmean,
            )
        raise RuntimeError(
            "antsApplyTransforms requires the ANTs CLI on PATH or the antspyx package "
            "(`pip install antspyx`); neither is available."
        )

    # Apply ANTs transformations
    cmd = [
        "antsApplyTransforms",
        "-d",
        "3",
        "-e",
        str(moving_type),
        "--input",
        str(movingf),
        "--output",
        str(outputf_name),
        "--interpolation",
        interpolation,
    ]

    if reff is not None:
        cmd.extend(["--reference-image", str(reff)])
    else:
        cmd.extend(["--reference-image", str(fixedf)])

    # Add transformation files in reverse order because ANTs applies transforms from right to left
    # i.e. the last transform in the command is applied first to the moving image
    # Use resolved path so ITK picks the correct reader by extension (.nii.gz vs .h5)
    for transform_file in reversed(transformf):
        resolved = Path(transform_file).resolve()
        cmd.extend(["--transform", str(resolved)])

    logger.debug(f"System: apply transforms command - {' '.join(cmd)}")

    # Get thread count from environment variable (set by Nextflow)
    num_threads = int(os.environ.get("OMP_NUM_THREADS", 8))

    # Set up ITK thread environment variables for subprocess
    from ..utils.system import set_numerical_threads

    env = set_numerical_threads(num_threads, include_itk=True, return_dict=True)
    # Merge with current environment to preserve other variables
    env = {**os.environ, **env}

    logger.debug(
        f"System: using {num_threads} threads for ITK operations (capped at 32)"
    )

    try:
        run_command(cmd, env=env, step_logger=logger)

        # Validate output
        validate_output_file(outputf_name, logger)

        outputs = {"imagef_registered": str(outputf_name)}
        logger.info(
            f"Step: transform application completed successfully - {outputf_name}"
        )

        # Generate temporal mean if requested
        if generate_tmean:
            # Extract just the filename from outputf_name (which already includes work_dir)
            tmean_filename = Path(outputf_name).name.split(".nii")[0] + "_tmean.nii.gz"
            tmean_file = work_dir / tmean_filename
            calculate_func_tmean(str(outputf_name), str(tmean_file), logger)
            outputs["imagef_registered_tmean"] = str(tmean_file)
            logger.info(f"Output: tmean generated - {tmean_file}")

        return outputs

    except Exception as e:
        logger.error(f"Step: transform application failed - {e}")
        raise RuntimeError(f"Transform application failed: {e}")
