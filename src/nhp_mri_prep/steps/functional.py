"""
Standalone functional processing step functions.

These functions wrap the existing operations to provide standardized
inputs/outputs for Nextflow integration.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from .types import StepInput, StepOutput
from ..operations.preprocessing import (
    slice_timing_correction,
    motion_correction,
    despike,
    bias_correction,
    conform_to_template,
    apply_skullstripping,
)
from ..operations.registration import (
    ants_register,
    ants_apply_transforms,
    flirt_apply_transforms,
    flirt_register,
)
from ..utils import get_image_resolution, calculate_func_tmean, get_image_shape
from ..utils import run_command
import numpy as np
import nibabel as nib
import pandas as pd

logger = logging.getLogger(__name__)


def _generate_pass_through_motion_params(
    input_file: Path, working_dir: Path, logger: logging.Logger
) -> Dict[str, Path]:
    """Generate pass-through outputs (tmean and zero-filled motion params) when motion correction is skipped.

    Args:
        input_file: Input functional file
        working_dir: Working directory for outputs
        logger: Logger instance

    Returns:
        Dictionary with 'tmean' and optionally 'motion_params' file paths
    """
    work_dir = Path(working_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Generate tmean from input file
    tmean_path = work_dir / "func_motion_corrected_tmean.nii.gz"
    calculate_func_tmean(str(input_file), str(tmean_path), logger)

    # Create zero-filled motion params TSV
    additional_files = {"tmean": tmean_path}
    shape = get_image_shape(str(input_file), logger)
    if len(shape) >= 4:
        nvols = shape[3]
        motion_params_df = pd.DataFrame(
            {
                "rot_x": [0.0] * nvols,
                "rot_y": [0.0] * nvols,
                "rot_z": [0.0] * nvols,
                "trans_x": [0.0] * nvols,
                "trans_y": [0.0] * nvols,
                "trans_z": [0.0] * nvols,
            }
        )
        motion_params_path = work_dir / "func_motion_corrected.tsv"
        motion_params_df.to_csv(motion_params_path, sep="\t", index=False)
        additional_files["motion_params"] = motion_params_path
        logger.info(f"Step: created zero-filled motion params TSV with {nvols} volumes")
    else:
        logger.warning(
            "Step: could not determine number of volumes for motion params TSV"
        )

    return additional_files


def func_slice_timing_correction(input: StepInput) -> StepOutput:
    """
    Perform slice timing correction on functional image.

    Args:
        input: StepInput with input_file, working_dir, config, metadata

    Returns:
        StepOutput with slice-timed file and tmean
    """
    if not input.config.get("func.slice_timing_correction.enabled", True):
        logger.info("Step: slice timing correction skipped (disabled in configuration)")
        return StepOutput(
            output_file=input.input_file,
            metadata={"step": "slice_timing_correction", "skipped": True},
        )

    # Call operation
    result = slice_timing_correction(
        imagef=str(input.input_file),
        working_dir=str(input.working_dir),
        output_name=input.output_name or "func_slice_timed.nii.gz",
        config=input.config,
        logger=logger,
        generate_tmean=True,
    )

    output_file = (
        Path(result["imagef_slice_time_corrected"])
        if result.get("imagef_slice_time_corrected")
        else input.input_file
    )
    tmean_file = (
        Path(result["imagef_slice_time_corrected_tmean"])
        if result.get("imagef_slice_time_corrected_tmean")
        else None
    )

    additional_files = {}
    if tmean_file:
        additional_files["tmean"] = tmean_file

    return StepOutput(
        output_file=output_file,
        metadata={"step": "slice_timing_correction", "modality": "func"},
        additional_files=additional_files,
    )


def func_motion_correction(input: StepInput) -> StepOutput:
    """
    Perform motion correction on functional image.

    Args:
        input: StepInput with input_file, working_dir, config, metadata

    Returns:
        StepOutput with motion-corrected file, tmean, and motion parameters
    """
    if not input.config.get("func.motion_correction.enabled", True):
        logger.info("Step: motion correction skipped (disabled in configuration)")
        additional_files = _generate_pass_through_motion_params(
            input.input_file, input.working_dir, logger
        )
        return StepOutput(
            output_file=input.input_file,
            metadata={"step": "motion_correction", "skipped": True},
            additional_files=additional_files,
        )

    # Call operation
    result = motion_correction(
        imagef=str(input.input_file),
        working_dir=str(input.working_dir),
        output_name=input.output_name or "func_motion_corrected.nii.gz",
        config=input.config,
        logger=logger,
        generate_tmean=True,
    )

    output_file = (
        Path(result["imagef_motion_corrected"])
        if result.get("imagef_motion_corrected")
        else input.input_file
    )

    # If motion correction was skipped (< 15 volumes), generate pass-through outputs
    if not result.get("imagef_motion_corrected") and not result.get(
        "imagef_motion_corrected_tmean"
    ):
        logger.info("Step: motion correction skipped - generating pass-through outputs")
        additional_files = _generate_pass_through_motion_params(
            input.input_file, input.working_dir, logger
        )
    else:
        # Normal execution - use outputs from motion_correction
        additional_files = {}
        if result.get("imagef_motion_corrected_tmean"):
            additional_files["tmean"] = Path(result["imagef_motion_corrected_tmean"])
        if result.get("motion_parameters"):
            additional_files["motion_params"] = Path(result["motion_parameters"])

    return StepOutput(
        output_file=output_file,
        metadata={
            "step": "motion_correction",
            "modality": "func",
            "dof": input.config.get("func", {})
            .get("motion_correction", {})
            .get("dof", 6),
            "skipped": not bool(result.get("imagef_motion_corrected")),
        },
        additional_files=additional_files,
    )


def func_compute_confounds(
    input: StepInput,
    motion_par_file: Optional[Path] = None,
    brain_mask_file: Optional[Path] = None,
    seg_file: Optional[Path] = None,
    seg_lut_file: Optional[Path] = None,
) -> StepOutput:
    """
    Compute fMRIPrep-compatible confound regressors for a preprocessed BOLD run.

    Produces a ``*_desc-confounds_timeseries.tsv`` + JSON sidecar (regressors only — the BOLD image
    is not modified). Tissue mean signals (csf / white_matter / csf_wm) are produced only when an
    anatomical segmentation (``seg_file`` + ``seg_lut_file``) is available; otherwise motion-only
    confounds are written.

    Args:
        input: StepInput with input_file = preprocessed BOLD (T1w space), working_dir, config, metadata
        motion_par_file: FSL mcflirt ``.par`` motion parameters (6-column)
        brain_mask_file: BOLD-space brain mask (for DVARS / global signal)
        seg_file: Anatomical segmentation in the same space as the BOLD (T1w space)
        seg_lut_file: Atlas LUT TSV (label index -> structure name) for tissue classification

    Returns:
        StepOutput with output_file = confounds TSV and additional_files['confounds_json'].
    """
    from ..operations.confounds import (
        compute_confounds,
        FD_OUTLIER_THRESHOLD_MM,
        FD_RADIUS_MM,
        STD_DVARS_OUTLIER_THRESHOLD,
    )

    input.working_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(input.working_dir / "desc-confounds_timeseries")

    # Treat any non-steady-state volumes the user asked despike to ignore as a lower bound.
    n_dummy_min = (
        input.config.get("func", {}).get("despike", {}).get("ignore_first_volumes", 0)
        or 0
    )

    # Outlier thresholds and the FD rotation radius are configurable (func.confounds); defaults are
    # macaque-scaled. The radius must be forwarded explicitly -- FD and rmsd are only interpretable
    # against the radius they were computed with, so leaving it to the operation default would pin
    # every non-macaque NHP to a 27 mm head.
    conf = input.config.get("func", {}).get("confounds", {}) or {}
    fd_thresh = conf.get("fd_outlier_threshold_mm", FD_OUTLIER_THRESHOLD_MM)
    dvars_thresh = conf.get("std_dvars_outlier_threshold", STD_DVARS_OUTLIER_THRESHOLD)
    fd_radius = conf.get("fd_radius_mm", FD_RADIUS_MM)

    result = compute_confounds(
        bold_file=str(input.input_file),
        motion_par_file=str(motion_par_file) if motion_par_file else None,
        working_dir=str(input.working_dir),
        output_prefix=output_prefix,
        brain_mask_file=str(brain_mask_file) if brain_mask_file else None,
        seg_file=str(seg_file) if seg_file else None,
        seg_lut_file=str(seg_lut_file) if seg_lut_file else None,
        n_dummy_min=int(n_dummy_min),
        fd_outlier_threshold=float(fd_thresh),
        std_dvars_outlier_threshold=float(dvars_thresh),
        radius_mm=float(fd_radius),
        logger=logger,
    )

    return StepOutput(
        output_file=Path(result["confounds_tsv"]),
        metadata={
            "step": "compute_confounds",
            "modality": "func",
            "has_segmentation": bool(seg_file and seg_lut_file),
        },
        additional_files={"confounds_json": Path(result["confounds_json"])},
    )


def func_despike(input: StepInput) -> StepOutput:
    """
    Perform despiking on functional image.

    Args:
        input: StepInput with input_file, working_dir, config, metadata

    Returns:
        StepOutput with despiked file and tmean
    """
    if not input.config.get("func.despike.enabled", True):
        logger.info("Step: despiking skipped (disabled in configuration)")
        return StepOutput(
            output_file=input.input_file, metadata={"step": "despike", "skipped": True}
        )

    # Call operation
    result = despike(
        imagef=str(input.input_file),
        working_dir=str(input.working_dir),
        output_name=input.output_name or "func_despiked.nii.gz",
        config=input.config,
        logger=logger,
        generate_tmean=True,
    )

    output_file = (
        Path(result["imagef_despiked"])
        if result.get("imagef_despiked")
        else input.input_file
    )
    tmean_file = (
        Path(result["imagef_despiked_tmean"])
        if result.get("imagef_despiked_tmean")
        else None
    )

    additional_files = {}
    if tmean_file:
        additional_files["tmean"] = tmean_file
    if result.get("spikiness_map"):
        additional_files["spikiness_map"] = Path(result["spikiness_map"])

    return StepOutput(
        output_file=output_file,
        metadata={"step": "despike", "modality": "func"},
        additional_files=additional_files,
    )


def func_bias_correction(input: StepInput) -> StepOutput:
    """
    Perform bias field correction on functional tmean image.

    Note: This operates on the tmean image, not the full 4D BOLD.

    Args:
        input: StepInput with input_file (should be tmean), working_dir, config, metadata

    Returns:
        StepOutput with bias-corrected tmean file
    """
    if not input.config.get("func.bias_correction.enabled", True):
        logger.info("Step: bias correction skipped (disabled in configuration)")
        return StepOutput(
            output_file=input.input_file,
            metadata={"step": "bias_correction", "skipped": True},
        )

    # Call operation
    result = bias_correction(
        imagef=str(input.input_file),
        working_dir=str(input.working_dir),
        modal="func",
        output_name=input.output_name or "func_bias_corrected.nii.gz",
        config=input.config,
        logger=logger,
    )

    output_file = Path(result["imagef_bias_corrected"])

    return StepOutput(
        output_file=output_file,
        metadata={
            "step": "bias_correction",
            "modality": "func",
            "algorithm": input.config.get("func", {})
            .get("bias_correction", {})
            .get("algorithm", "N4BiasFieldCorrection"),
        },
    )


def func_conform(
    input: StepInput, target_file: Path, bold_4d_file: Optional[Path] = None
) -> StepOutput:
    """
    Conform functional tmean image to target space, and optionally apply transform to full 4D BOLD.

    Args:
        input: StepInput with input_file (tmean), working_dir, config, metadata
        target_file: Target file for conforming (anatomical or template)
        bold_4d_file: Optional path to full 4D BOLD timeseries to also conform

    Returns:
        StepOutput with conformed tmean file, conformed 4D BOLD (if provided), and transform files
    """
    if not input.config.get("func.conform.enabled", True):
        logger.info("Step: conform skipped (disabled in configuration)")
        return StepOutput(
            output_file=input.input_file, metadata={"step": "conform", "skipped": True}
        )

    # Check if skullstripping is disabled
    skip_skullstripping = not input.config.get("func.skullstripping.enabled", True)

    # Step 1: Conform the tmean (used for registration)
    result = conform_to_template(
        imagef=str(input.input_file),
        template_file=str(target_file),
        working_dir=str(input.working_dir),
        output_name=input.output_name or "func_tmean_conformed.nii.gz",
        logger=logger,
        modal="func",
        skip_skullstripping=skip_skullstripping,
    )

    output_file = (
        Path(result["imagef_conformed"])
        if result.get("imagef_conformed")
        else input.input_file
    )
    additional_files = {}

    if result.get("forward_xfm"):
        additional_files["forward_transform"] = Path(result["forward_xfm"])
    if result.get("inverse_xfm"):
        additional_files["inverse_transform"] = Path(result["inverse_xfm"])
    if result.get("template_f"):
        additional_files["template_resampled"] = Path(result["template_f"])

    # Step 2: Apply the same transform to the full 4D BOLD timeseries if provided
    if bold_4d_file and bold_4d_file.exists() and result.get("forward_xfm"):
        logger.info("Step: applying conform transform to full 4D BOLD timeseries")
        from ..operations.registration import flirt_apply_transforms

        # Use the conformed tmean as reference for resampling
        # Pass only filename, not full path, since flirt_apply_transforms will prepend work_dir
        bold_4d_conformed_filename = "func_bold_conformed.nii.gz"

        # Apply transform to 4D BOLD
        bold_result = flirt_apply_transforms(
            movingf=str(bold_4d_file),
            outputf_name=bold_4d_conformed_filename,
            reff=str(output_file),  # Use conformed tmean as reference
            working_dir=str(input.working_dir),
            transformf=str(result["forward_xfm"]),
            logger=logger,
            interpolation="trilinear",
            generate_tmean=False,  # Don't generate tmean, we already have it
        )

        if bold_result.get("imagef_registered"):
            additional_files["bold_4d_conformed"] = Path(
                bold_result["imagef_registered"]
            )
            logger.info(
                f"Output: conformed 4D BOLD saved - {bold_result['imagef_registered']}"
            )

    return StepOutput(
        output_file=output_file,
        metadata={
            "step": "conform",
            "modality": "func",
            "target_file": str(target_file),
            "bold_4d_conformed": str(bold_4d_file) if bold_4d_file else None,
            # Real backend that produced the transform (for the xfm sidecar GeneratedBy).
            "engine": result.get("engine"),
        },
        additional_files=additional_files,
    )


def func_skullstripping(input: StepInput) -> StepOutput:
    """
    Perform skull stripping on functional tmean image.

    Args:
        input: StepInput with input_file (tmean), working_dir, config, metadata

    Returns:
        StepOutput with skull-stripped tmean file and brain mask
    """
    if not input.config.get("func.skullstripping.enabled", True):
        logger.info("Step: skull stripping skipped (disabled in configuration)")
        return StepOutput(
            output_file=input.input_file,
            metadata={"step": "skullstripping", "skipped": True},
        )

    # Call operation
    result = apply_skullstripping(
        imagef=str(input.input_file),
        modal="func",
        working_dir=str(input.working_dir),
        output_name=input.output_name or "func_brain.nii.gz",
        config=input.config,
        logger=logger,
    )

    output_file = Path(result["imagef_skullstripped"])
    additional_files = {}

    if result.get("brain_mask"):
        additional_files["brain_mask"] = Path(result["brain_mask"])

    return StepOutput(
        output_file=output_file,
        metadata={
            "step": "skullstripping",
            "modality": "func",
            "method": "nhp_skullstrip_nn",
        },
        additional_files=additional_files,
    )


def func_apply_mask(
    input: StepInput, brain_mask: Path, generate_tmean: bool = True
) -> StepOutput:
    """
    Apply mask to image.

    Args:
        input: StepInput with input_file (4d or 3d), working_dir, config, metadata
        brain_mask: Path to brain mask
        generate_tmean: Whether to generate tmean

    Returns:
        StepOutput with masked image and tmean (if generated)
    """
    # Ensure working directory exists
    input.working_dir.mkdir(parents=True, exist_ok=True)

    if not brain_mask.exists():
        raise FileNotFoundError(f"Brain mask not found: {brain_mask}")

    # Apply mask
    from ..operations.preprocessing import apply_mask

    output_name = input.output_name or "func_brain.nii.gz"
    result = apply_mask(
        imagef=str(input.input_file),
        maskf=str(brain_mask),
        working_dir=str(input.working_dir),
        output_name=output_name,
        logger=logger,
        generate_tmean=generate_tmean,
    )

    output_file = Path(result["imagef_masked"])  # type: ignore[arg-type]
    additional_files = {"brain_mask": brain_mask}
    if generate_tmean and result.get("imagef_masked_tmean"):
        additional_files["tmean"] = Path(result["imagef_masked_tmean"])

    return StepOutput(
        output_file=output_file,
        metadata={
            "step": "apply_brain_mask",
            "modality": "func",
        },
        additional_files=additional_files,
    )


def func_registration(
    input: StepInput, target_file: Path, target_type: str  # 'anat' or 'template'
) -> StepOutput:
    """
    Register functional tmean image to target (anatomical or template).

    Args:
        input: StepInput with input_file (tmean), working_dir, config, metadata
        target_file: Target file for registration
        target_type: Type of target ('anat' or 'template')

    Returns:
        StepOutput with registered tmean file and transform files
    """
    if not input.config.get("registration.enabled", True):
        logger.info("Step: registration skipped (disabled in configuration)")
        return StepOutput(
            output_file=input.input_file,
            metadata={"step": "registration", "skipped": True},
        )

    # Ensure working directory exists
    input.working_dir.mkdir(parents=True, exist_ok=True)

    # Get registration behavior from config
    reg_config = input.config.get("registration", {})
    config_key = f"func2{target_type}_xfm_type"
    xfm_type = reg_config.get(config_key, "syn")
    enable_fireants = reg_config.get("enable_fireants", True)
    fireants_allow_cpu = reg_config.get("fireants_allow_cpu", True)

    # Resample target to functional resolution if requested
    fixedf = str(target_file)
    if input.config.get("registration.keep_func_resolution", True):
        func_res = np.round(
            get_image_resolution(str(input.input_file), logger=logger), 1
        )
        reff = input.working_dir / "target_res-func_for_registration.nii.gz"
        cmd_resample = [
            "3dresample",
            "-input",
            str(target_file),
            "-prefix",
            str(reff),
            "-rmode",
            "Cu",
            "-dxyz",
            str(func_res[0]),
            str(func_res[1]),
            str(func_res[2]),
        ]
        run_command(cmd_resample, step_logger=logger)
        fixedf = str(reff)
        logger.info("Output: target resampled to func resolution for registration")

    # Call operation
    result = ants_register(
        movingf=str(input.input_file),
        fixedf=fixedf,
        working_dir=str(input.working_dir),
        output_prefix=f"func2{target_type}_tmean",
        config=input.config,
        logger=logger,
        xfm_type=xfm_type,
        enable_fireants=enable_fireants,
        fireants_allow_cpu=fireants_allow_cpu,
    )

    output_file = Path(result["imagef_registered"])
    additional_files = {}

    if result.get("forward_transform"):
        additional_files["forward_transform"] = Path(result["forward_transform"])
    if result.get("inverse_transform"):
        additional_files["inverse_transform"] = Path(result["inverse_transform"])

    return StepOutput(
        output_file=output_file,
        metadata={
            "step": "registration",
            "modality": "func",
            "target_type": target_type,
            "xfm_type": xfm_type,
            "fireants_requested": enable_fireants,
            # Real backend used (fireants/ants/antspyx) after any runtime fallback.
            "engine": result.get("engine"),
        },
        additional_files=additional_files,
    )


def func_apply_transforms(
    input: StepInput,
    transform_files: List[Path],
    reference_file: Path,
    interpolation: Optional[str] = None,
) -> StepOutput:
    """
    Apply transforms to full 4D functional BOLD image.

    Args:
        input: StepInput with input_file (full 4D BOLD), working_dir, config, metadata
        transform_files: List of transform files to apply
        reference_file: Reference file for resampling
        interpolation: Interpolation method (default from config)

    Returns:
        StepOutput with registered 4D BOLD file and tmean
    """
    if interpolation is None:
        interpolation = input.config.get("registration", {}).get(
            "interpolation", "BSpline"
        )

    # Convert transform files to strings
    transform_strs = [str(tf) for tf in transform_files]

    # Ensure working directory exists
    input.working_dir.mkdir(parents=True, exist_ok=True)

    # Call operation
    result = ants_apply_transforms(
        movingf=str(input.input_file),
        moving_type=3,  # 4D image (BOLD)
        interpolation=interpolation,
        outputf_name=input.output_name or "func_registered.nii.gz",
        fixedf=str(reference_file),
        transformf=transform_strs,
        reff=str(reference_file),
        working_dir=str(input.working_dir),
        generate_tmean=True,
        logger=logger,
    )

    output_file = Path(result["imagef_registered"])
    additional_files = {}

    if result.get("imagef_registered_tmean"):
        additional_files["tmean"] = Path(result["imagef_registered_tmean"])

    return StepOutput(
        output_file=output_file,
        metadata={
            "step": "apply_transforms",
            "modality": "func",
            "interpolation": interpolation,
            "num_transforms": len(transform_files),
        },
        additional_files=additional_files,
    )


def func_within_ses_coreg(
    input: StepInput,
    reference_tmean: Path,
    reference_run: str,
    current_run: str,
    bold_file: Path,
    method: Optional[str] = "ants",
) -> StepOutput:
    """
    Coregister a functional run to a reference run within the same session.

    This is used when multiple runs exist in a session. Later runs are
    coregistered to the first run using tmean images, then the transform
    is applied to the BOLD file.

    Args:
        input: StepInput with input_file (tmean to be coregistered), working_dir, config, metadata
        reference_tmean: Path to reference tmean (from first run)
        reference_run: Run identifier for reference (e.g., "01")
        current_run: Run identifier for current run (e.g., "02")
        bold_file: Path to BOLD file to be coregistered
        method: Registration method: "ants" (default) or "flirt". ANTs uses rigid (FireANTs/ANTs);
            FLIRT uses FSL FLIRT with rigid (dof=6).

    Returns:
        StepOutput with coregistered tmean, coregistered BOLD, and transform files
    """
    if method not in ("ants", "flirt"):
        raise ValueError(f"method must be 'ants' or 'flirt', got: {method!r}")

    prefix = f"run{current_run}_to_run{reference_run}_coreg"
    tmean_out = f"run{current_run}_to_run{reference_run}_tmean_coreg.nii.gz"
    bold_out = f"run{current_run}_to_run{reference_run}_bold_coreg.nii.gz"

    reg_config = input.config.get("registration", {})
    enable_fireants = reg_config.get("enable_fireants", True)

    if method == "ants":
        # Step 1: ANTs rigid registration (FireANTs when GPU available, else ANTs CPU)
        try:
            registration_result = ants_register(
                fixedf=str(reference_tmean),
                movingf=str(input.input_file),
                working_dir=str(input.working_dir),
                output_prefix=prefix,
                config=input.config,
                logger=logger,
                xfm_type="rigid",
                enable_fireants=enable_fireants,
            )
        except Exception as e:
            logger.error(
                f"Error during ANTs registration for run {current_run} to run {reference_run}: {e}"
            )
            raise RuntimeError(
                f"Within-session coregistration failed during ANTs registration: {e}"
            ) from e

        # ants_register already produces the registered tmean (imagef_registered)
        tmean_coregistered = Path(registration_result["imagef_registered"])
        xfm_forward_f = Path(registration_result["forward_transform"])
        xfm_inverse_f = (
            Path(registration_result["inverse_transform"])
            if registration_result.get("inverse_transform")
            else None
        )

        # Step 2: Apply same transform to BOLD (4D time series)
        bold_coregistered = None
        if bold_file and bold_file.exists():
            try:
                apply_result_bold = ants_apply_transforms(
                    movingf=str(bold_file),
                    moving_type=3,  # 4D image (BOLD)
                    interpolation=input.config.get("registration", {}).get(
                        "interpolation", "BSpline"
                    ),
                    outputf_name=bold_out,
                    fixedf=str(reference_tmean),
                    working_dir=str(input.working_dir),
                    transformf=[str(xfm_forward_f)],
                    reff=str(reference_tmean),
                    logger=logger,
                    generate_tmean=False,
                )
                bold_coregistered = Path(apply_result_bold["imagef_registered"])
            except Exception as e:
                logger.error(f"Error during BOLD transformation application: {e}")
                raise RuntimeError(
                    f"Within-session coregistration failed when applying transformation to BOLD: {e}"
                ) from e
    else:
        # method == "flirt"
        flirt_config = {
            "registration": {
                "flirt": {
                    "cost": "mutualinfo",
                    "searchcost": "mutualinfo",
                    "coarsesearch": 40,
                    "finesearch": 15,
                    "searchrx": [-60, 60],
                    "searchry": [-60, 60],
                    "searchrz": [-60, 60],
                }
            }
        }
        try:
            registration_result = flirt_register(
                fixedf=str(reference_tmean),
                movingf=str(input.input_file),
                working_dir=str(input.working_dir),
                output_prefix=prefix,
                config=flirt_config,
                logger=logger,
                dof=6,
            )
            xfm_forward_f = Path(registration_result["forward_transform"])
            xfm_inverse_f = (
                Path(registration_result["inverse_transform"])
                if registration_result.get("inverse_transform")
                else None
            )
        except Exception as e:
            logger.error(
                f"Error during FLIRT registration for run {current_run} to run {reference_run}: {e}"
            )
            raise RuntimeError(
                f"Within-session coregistration failed during FLIRT registration: {e}"
            ) from e

        try:
            apply_result = flirt_apply_transforms(
                movingf=str(input.input_file),
                outputf_name=tmean_out,
                reff=str(reference_tmean),
                working_dir=str(input.working_dir),
                transformf=str(xfm_forward_f),
                logger=logger,
                interpolation="trilinear",
                generate_tmean=False,
            )
            tmean_coregistered = Path(apply_result["imagef_registered"])
        except Exception as e:
            logger.error(f"Error during tmean transformation application: {e}")
            raise RuntimeError(
                f"Within-session coregistration failed when applying transformation to tmean: {e}"
            ) from e

        bold_coregistered = None
        if bold_file and bold_file.exists():
            try:
                apply_result_bold = flirt_apply_transforms(
                    movingf=str(bold_file),
                    outputf_name=bold_out,
                    reff=str(reference_tmean),
                    working_dir=str(input.working_dir),
                    transformf=str(xfm_forward_f),
                    logger=logger,
                    interpolation="trilinear",
                    generate_tmean=False,
                )
                bold_coregistered = Path(apply_result_bold["imagef_registered"])
            except Exception as e:
                logger.error(f"Error during BOLD transformation application: {e}")
                raise RuntimeError(
                    f"Within-session coregistration failed when applying transformation to BOLD: {e}"
                ) from e

    additional_files = {}
    if xfm_forward_f.exists():
        additional_files["forward_transform"] = xfm_forward_f
    if xfm_inverse_f is not None and xfm_inverse_f.exists():
        additional_files["inverse_transform"] = xfm_inverse_f
    if bold_coregistered and bold_coregistered.exists():
        additional_files["bold_coregistered"] = bold_coregistered

    return StepOutput(
        output_file=tmean_coregistered,
        metadata={
            "step": "within_ses_coreg",
            "modality": "func",
            "reference_run": reference_run,
            "current_run": current_run,
            "xfm_type": "rigid",
            "method": method,
            "fireants_requested": enable_fireants,
        },
        additional_files=additional_files,
    )


def func_average_tmean(
    tmean_files: List[Path], working_dir: Path, config: Dict[str, Any]
) -> StepOutput:
    """
    Average multiple tmean files (typically after within-session coregistration).

    Args:
        tmean_files: List of Path objects to tmean files to average
        working_dir: Working directory for output
        config: Configuration dictionary

    Returns:
        StepOutput with averaged tmean file
    """
    if len(tmean_files) == 0:
        raise ValueError("No tmean files provided for averaging")

    if len(tmean_files) == 1:
        # Single file, just return it (no averaging needed)
        logger.info("Only one tmean file provided, skipping averaging")
        return StepOutput(
            output_file=tmean_files[0],
            metadata={"step": "average_tmean", "num_files": 1, "skipped": True},
        )

    logger.info(f"Averaging {len(tmean_files)} tmean files")

    # Incremental mean calculation to avoid loading all images into memory at once
    # Process images one by one: accumulate sum, then divide by count
    sum_data = None
    reference_img = None
    valid_count = 0

    for tmean_file in tmean_files:
        if not tmean_file.exists():
            logger.warning(f"Tmean file does not exist: {tmean_file}, skipping")
            continue
        try:
            img = nib.load(str(tmean_file))
            img_data = img.get_fdata()

            # Initialize sum with first valid image
            if sum_data is None:
                sum_data = img_data.astype(
                    np.float64
                )  # Use float64 for accumulation precision
                reference_img = img  # Save first image for header/affine
            else:
                # Accumulate sum incrementally
                sum_data += img_data

            valid_count += 1
        except Exception as e:
            logger.warning(f"Failed to load tmean file {tmean_file}: {e}, skipping")
            continue

    if valid_count == 0:
        raise ValueError("No valid tmean files could be loaded for averaging")

    if valid_count == 1:
        logger.info("Only one valid tmean file after filtering, skipping averaging")
        return StepOutput(
            output_file=tmean_files[0],
            metadata={"step": "average_tmean", "num_files": 1, "skipped": True},
        )

    # Calculate mean by dividing accumulated sum by count
    mean_data = (sum_data / valid_count).astype(np.float32)

    # Create averaged image using reference header
    averaged_img = nib.Nifti1Image(
        mean_data.astype(np.float32),
        affine=reference_img.affine,
        header=reference_img.header,
    )

    # Generate output filename
    output_name = "func_tmean_averaged.nii.gz"
    averaged_path = working_dir / output_name

    # Save averaged image
    nib.save(averaged_img, str(averaged_path))

    logger.info(f"Averaged tmean saved: {averaged_path}")

    return StepOutput(
        output_file=averaged_path,
        metadata={
            "step": "average_tmean",
            "num_files": valid_count,
            "input_files": [str(f) for f in tmean_files],
        },
    )
