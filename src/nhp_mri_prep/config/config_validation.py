"""
Configuration validation for nhp_mri_prep.

This module provides functions to validate configuration parameters
and preprocessing settings.
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional
from .config_io import get_default_config, _deep_merge
from ..utils.mri import get_opposite_orientation


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Validated and normalized configuration

    Raises:
        ValueError: If configuration is invalid
    """
    # Start with default config
    validated_config = get_default_config().copy()

    # Deep merge with provided config
    validated_config = _deep_merge(validated_config, config)

    # Validate individual sections with new nested structure
    func_config = validated_config.get("func", {})
    anat_config = validated_config.get("anat", {})

    validate_func_config(func_config)
    validate_anat_config(anat_config)
    validate_slice_timing_config(func_config.get("slice_timing_correction", {}))
    validate_motion_correction_config(func_config.get("motion_correction", {}))
    validate_despike_config(func_config.get("despike", {}))
    validate_confounds_config(func_config.get("confounds", {}))
    validate_skullstripping_config(func_config.get("skullstripping", {}))
    validate_skullstripping_config(anat_config.get("skullstripping_segmentation", {}))
    validate_surface_reconstruction_config(
        anat_config.get("surface_reconstruction", {})
    )
    validate_bias_correction_config(func_config.get("bias_correction", {}))
    validate_bias_correction_config(anat_config.get("bias_correction", {}))
    validate_registration_config(validated_config.get("registration", {}))
    validate_quality_control_config(validated_config.get("quality_control", {}))
    validate_orientation_mismatch_correction_config(
        validated_config.get("orientation_mismatch_correction", {})
    )

    return validated_config


def validate_func_config(config: Dict[str, Any]) -> None:
    """Validate top-level functional configuration parameters.

    Args:
        config: Functional configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate coreg_runs_within_session
    if "coreg_runs_within_session" in config:
        coreg_enabled = config["coreg_runs_within_session"]
        if not isinstance(coreg_enabled, bool):
            raise ValueError(
                f"Configuration error in func: "
                f"coreg_runs_within_session must be a boolean, got {type(coreg_enabled).__name__}. "
                f"Please fix this in your configuration file."
            )


def validate_anat_config(config: Dict[str, Any]) -> None:
    """Validate top-level anatomical configuration parameters.

    Args:
        config: Anatomical configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    # synthesis_level selects cross-session ("subject") vs within-session
    # ("session") T1w/T2w synthesis; no other values are supported.
    if "synthesis_level" in config:
        level = config["synthesis_level"]
        if level not in ["subject", "session"]:
            raise ValueError(
                f"Configuration error in anat: "
                f"synthesis_level must be 'subject' or 'session', got {level!r}. "
                f"Please fix this in your configuration file."
            )


def validate_slice_timing_config(config: Dict[str, Any]) -> None:
    """Validate slice timing configuration.

    Args:
        config: Slice timing configuration

    Raises:
        ValueError: If configuration is invalid
    """
    # Only validate if slice timing is enabled
    if not config.get("enabled", False):
        return

    if "repetition_time" in config:
        tr = config["repetition_time"]
        # Skip validation if repetition_time is None (metadata missing)
        if tr is None:
            return
        if not isinstance(tr, (int, float)) or tr <= 0:
            raise ValueError(
                f"Configuration error in func.slice_timing_correction: "
                f"repetition_time must be a positive number, got {tr}. "
                f"Please fix this in your configuration file."
            )

    if "tzero" in config:
        tzero = config["tzero"]
        # Skip validation if tzero is None (metadata missing)
        if tzero is None:
            return
        if not isinstance(tzero, (int, float)):
            raise ValueError(
                f"Configuration error in func.slice_timing_correction: "
                f"tzero must be a number, got {tzero}. "
                f"Please fix this in your configuration file."
            )
        # Only check tzero < tr if both are present and valid
        if "repetition_time" in config and config["repetition_time"] is not None:
            if tzero > config["repetition_time"]:
                raise ValueError(
                    f"Configuration error in func.slice_timing_correction: "
                    f"tzero must be less than repetition time, got {tzero} > {config['repetition_time']}. "
                    f"Please fix this in your configuration file."
                )


def validate_motion_correction_config(config: Dict[str, Any]) -> None:
    """Validate motion correction configuration.

    Args:
        config: Motion correction configuration

    Raises:
        ValueError: If configuration is invalid
    """
    if "dof" in config:
        dof = config["dof"]
        if not isinstance(dof, int) or dof not in [6, 9, 12]:
            raise ValueError(
                f"Configuration error in func.motion_correction: "
                f"dof must be 6, 9, or 12, got {dof}. "
                f"Please fix this in your configuration file."
            )

    if "cost" in config:
        cost = config["cost"]
        valid_costs = [
            "mutualinfo",
            "woods",
            "corratio",
            "normcorr",
            "normmi",
            "leastsq",
        ]
        if cost not in valid_costs:
            raise ValueError(
                f"Configuration error in func.motion_correction: "
                f"cost must be one of {valid_costs}, got {cost}. "
                f"Please fix this in your configuration file."
            )

    if "ref_vol" in config:
        ref_vol = config["ref_vol"]
        if not (isinstance(ref_vol, int) or ref_vol in ["Tmean", "mid"]):
            raise ValueError(
                f"Configuration error in func.motion_correction: "
                f"ref_vol must be an integer or 'Tmean'/'mid', got {ref_vol}. "
                f"Please fix this in your configuration file."
            )


def validate_despike_config(config: Dict[str, Any]) -> None:
    """Validate despike configuration.

    Args:
        config: Despike configuration

    Raises:
        ValueError: If configuration is invalid
    """
    if "c1" in config:
        c1 = config["c1"]
        if not isinstance(c1, (int, float)) or c1 <= 0:
            raise ValueError(
                f"Configuration error in func.despike: "
                f"c1 must be a positive number, got {c1}. "
                f"Please fix this in your configuration file."
            )

    if "c2" in config:
        c2 = config["c2"]
        if not isinstance(c2, (int, float)) or c2 <= 0:
            raise ValueError(
                f"Configuration error in func.despike: "
                f"c2 must be a positive number, got {c2}. "
                f"Please fix this in your configuration file."
            )

    if "c1" in config and "c2" in config:
        if config["c1"] >= config["c2"]:
            raise ValueError(
                f"Configuration error in func.despike: "
                f"c1 must be less than c2, got c1={config['c1']} >= c2={config['c2']}. "
                f"Please fix this in your configuration file."
            )


def validate_confounds_config(config: Dict[str, Any]) -> None:
    """Validate confounds configuration.

    Args:
        config: Confounds configuration

    Raises:
        ValueError: If configuration is invalid
    """
    if "enabled" in config:
        enabled = config["enabled"]
        if not isinstance(enabled, bool):
            raise ValueError(
                f"Configuration error in func.confounds: "
                f"enabled must be a boolean, got {type(enabled).__name__}. "
                f"Please fix this in your configuration file."
            )

    # Outlier thresholds and the FD rotation radius. Without this check a string or a negative value
    # passes validation and only fails later inside a Nextflow process, far from the cause.
    for key in (
        "fd_outlier_threshold_mm",
        "std_dvars_outlier_threshold",
        "fd_radius_mm",
    ):
        if key not in config:
            continue
        value = config[key]
        # bool is a subclass of int, so check it first or `fd_radius_mm: true` reads as 1.0.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Configuration error in func.confounds: "
                f"{key} must be a number, got {type(value).__name__}. "
                f"Please fix this in your configuration file."
            )
        if value <= 0:
            raise ValueError(
                f"Configuration error in func.confounds: "
                f"{key} must be greater than 0, got {value}. "
                f"Please fix this in your configuration file."
            )


def validate_skullstripping_config(config: Dict[str, Any]) -> None:
    """Validate skullstripping configuration.

    Args:
        config: Skullstripping configuration

    Raises:
        ValueError: If configuration is invalid
    """
    if not config.get("enabled", False):
        return

    method = config.get("method", "fastSurferCNN")

    if method not in ["bet", "fastSurferCNN", "macacaMRINN"]:
        raise ValueError(
            f"Configuration error in skullstripping: "
            f"Invalid method: {method}. Must be 'bet', 'fastSurferCNN', or 'macacaMRINN'. "
            f"Please fix this in your configuration file."
        )

    if method == "bet":
        bet_cfg = config.get("bet", {})
        fractional_intensity = bet_cfg.get("fractional_intensity", 0.3)
        if not isinstance(fractional_intensity, (int, float)) or not (
            0 < fractional_intensity < 1
        ):
            raise ValueError(
                f"Configuration error in skullstripping.bet: "
                f"fractional_intensity must be between 0 and 1, got: {fractional_intensity}. "
                f"Please fix this in your configuration file."
            )

    elif method == "fastSurferCNN":
        fscnn_cfg = config.get("fastSurferCNN", {})

        # Validate gpu_device
        gpu_device = fscnn_cfg.get("gpu_device", "auto")
        if (
            not (isinstance(gpu_device, int) and gpu_device >= -1)
            and gpu_device != "auto"
        ):
            raise ValueError(
                f"Configuration error in skullstripping.fastSurferCNN: "
                f"gpu_device must be integer >= -1 (where -1 means CPU) or 'auto' for automatic selection, got: {gpu_device}. "
                f"Please fix this in your configuration file."
            )

        # Validate batch_size
        batch_size = fscnn_cfg.get("batch_size", 1)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(
                f"Configuration error in skullstripping.fastSurferCNN: "
                f"batch_size must be a positive integer, got: {batch_size}. "
                f"Please fix this in your configuration file."
            )

        # Validate boolean options
        for bool_param in ["use_mixed_model", "enable_crop_2round", "fix_V1_WM"]:
            if bool_param in fscnn_cfg:
                value = fscnn_cfg.get(bool_param)
                if not isinstance(value, bool):
                    raise ValueError(
                        f"Configuration error in skullstripping.fastSurferCNN: "
                        f"{bool_param} must be a boolean, got: {value}. "
                        f"Please fix this in your configuration file."
                    )

        # Validate plane weights (optional, can be None or float)
        for weight_name in [
            "plane_weight_coronal",
            "plane_weight_axial",
            "plane_weight_sagittal",
        ]:
            weight = fscnn_cfg.get(weight_name)
            if weight is not None:
                if not isinstance(weight, (int, float)) or not (0 <= weight <= 1):
                    raise ValueError(
                        f"Configuration error in skullstripping.fastSurferCNN: "
                        f"{weight_name} must be a float between 0 and 1, or None, got: {weight}. "
                        f"Please fix this in your configuration file."
                    )

    elif method == "macacaMRINN":
        mrin_cfg = config.get("macacaMRINN", {})

        # Validate gpu_device (only runtime parameter needed)
        # Other parameters are loaded from model checkpoint automatically
        gpu_device = mrin_cfg.get("gpu_device", "auto")
        if (
            not (isinstance(gpu_device, int) and gpu_device >= -1)
            and gpu_device != "auto"
        ):
            raise ValueError(
                f"Configuration error in skullstripping.macacaMRINN: "
                f"gpu_device must be integer >= -1 (where -1 means CPU) or 'auto' for automatic selection, got: {gpu_device}. "
                f"Please fix this in your configuration file."
            )


def validate_bias_correction_config(config: Dict[str, Any]) -> None:
    """Validate bias correction configuration.

    Args:
        config: Bias correction configuration

    Raises:
        ValueError: If configuration is invalid
    """
    if "shrink_factor" in config:
        shrink = config["shrink_factor"]
        if not isinstance(shrink, (int, float)) or shrink <= 0:
            raise ValueError(
                f"Configuration error in bias_correction: "
                f"shrink_factor must be a positive number, got {shrink}. "
                f"Please fix this in your configuration file."
            )

    if "bspline_fitting" in config:
        # Preferred form is a plain positive int (N4 B-spline mesh resolution, e.g. 100).
        # A legacy ANTs-format string (e.g. "[ 100 ]") is still accepted for backward
        # compatibility and passed through verbatim by the operation layer.
        bspline = config["bspline_fitting"]
        if not (
            (isinstance(bspline, int) and not isinstance(bspline, bool) and bspline > 0)
            or isinstance(bspline, str)
        ):
            raise ValueError(
                f"Configuration error in bias_correction: "
                f"bspline_fitting must be a positive integer (e.g. 100), got {bspline!r}. "
                f"Please fix this in your configuration file."
            )


def validate_registration_config(config: Dict[str, Any]) -> None:
    """Validate registration configuration.

    Args:
        config: Registration configuration

    Raises:
        ValueError: If configuration is invalid
    """
    valid_interpolations = [
        "Linear",
        "NearestNeighbor",
        "MultiLabel",
        "Gaussian",
        "BSpline",
        "CosineWindowedSinc",
        "WelchWindowedSinc",
        "HammingWindowedSinc",
        "LanczosWindowedSinc",
        "GenericLabel",
    ]

    # Transform type for each registration direction must be one of the
    # supported ANTs/FireANTs stages.
    valid_xfm_types = ["translation", "rigid", "affine", "syn"]
    for xfm_key in [
        "anat2template_xfm_type",
        "func2anat_xfm_type",
        "func2template_xfm_type",
    ]:
        if xfm_key in config:
            xfm_value = config[xfm_key]
            if xfm_value not in valid_xfm_types:
                raise ValueError(
                    f"Configuration error in registration: "
                    f"{xfm_key} must be one of {valid_xfm_types}, got {xfm_value!r}. "
                    f"Please fix this in your configuration file."
                )

    if "keep_func_resolution" in config and not isinstance(
        config["keep_func_resolution"], bool
    ):
        raise ValueError(
            f"Configuration error in registration: "
            f"keep_func_resolution must be a boolean, got "
            f"{type(config['keep_func_resolution']).__name__}. "
            f"Please fix this in your configuration file."
        )

    if "interpolation" in config:
        interp = config["interpolation"]
        if interp not in valid_interpolations:
            raise ValueError(
                f"Configuration error in registration: "
                f"interpolation must be one of {valid_interpolations}, got {interp}. "
                f"Please fix this in your configuration file."
            )

    if "enable_fireants" in config and not isinstance(config["enable_fireants"], bool):
        raise ValueError(
            f"Configuration error in registration: "
            f"enable_fireants must be a boolean, got {type(config['enable_fireants']).__name__}. "
            f"Please fix this in your configuration file."
        )

    if "fireants_allow_cpu" in config and not isinstance(
        config["fireants_allow_cpu"], bool
    ):
        raise ValueError(
            f"Configuration error in registration: "
            f"fireants_allow_cpu must be a boolean, got {type(config['fireants_allow_cpu']).__name__}. "
            f"Please fix this in your configuration file."
        )

    # Validate transform stages (if present)
    for stage in ["translation", "rigid", "affine", "syn"]:
        if stage in config:
            validate_transform_stage_config(config[stage], stage)


def validate_transform_stage_config(config: Dict[str, Any], stage_name: str) -> None:
    """Validate individual transform stage configuration.

    Args:
        config: Transform stage configuration
        stage_name: Name of the transform stage

    Raises:
        ValueError: If configuration is invalid
    """
    if "gradient_step" in config:
        grad_step = config["gradient_step"]
        # Allow both string format (e.g., "[0.1]") and list format
        if isinstance(grad_step, str):
            if not (grad_step.startswith("[") and grad_step.endswith("]")):
                raise ValueError(
                    f"Configuration error in registration.{stage_name}: "
                    f"gradient_step string must be in format '[value]' or '[value1,value2]', got: {grad_step!r}. "
                    f"Please fix this in your configuration file."
                )
        elif isinstance(grad_step, list):
            if not all(isinstance(x, (str, int, float)) for x in grad_step):
                raise ValueError(
                    f"Configuration error in registration.{stage_name}: "
                    f"gradient_step list must contain strings or numbers. "
                    f"Please fix this in your configuration file."
                )
        else:
            raise ValueError(
                f"Configuration error in registration.{stage_name}: "
                f"gradient_step must be a string (e.g., '[0.1]') or list, got: {type(grad_step).__name__}. "
                f"Please fix this in your configuration file."
            )

    if "metric" in config:
        metrics = config["metric"]
        if not isinstance(metrics, list) or not all(
            isinstance(x, str) for x in metrics
        ):
            raise ValueError(
                f"Configuration error in registration.{stage_name}: "
                f"metric must be a list of strings, got: {type(metrics).__name__}. "
                f"Please fix this in your configuration file."
            )


def validate_quality_control_config(config: Dict[str, Any]) -> None:
    """Validate quality control configuration.

    Args:
        config: Quality control configuration

    Raises:
        ValueError: If configuration is invalid
    """
    # Only validate if quality_control section exists (it's optional and may be removed)
    if not config:
        return

    if "snap_views" in config:
        views = config["snap_views"]
        if not isinstance(views, list) or not all(v in ["x", "y", "z"] for v in views):
            raise ValueError(
                f"Configuration error in quality_control: "
                f"snap_views must be a list containing 'x', 'y', and/or 'z', got: {views}. "
                f"Please fix this in your configuration file."
            )

    if "snap_slices" in config:
        slices = config["snap_slices"]
        if not isinstance(slices, int) or slices <= 0:
            raise ValueError(
                f"Configuration error in quality_control: "
                f"snap_slices must be a positive integer, got {slices}. "
                f"Please fix this in your configuration file."
            )


def validate_surface_reconstruction_config(config: Dict[str, Any]) -> None:
    """Validate surface reconstruction configuration.

    Args:
        config: Surface reconstruction configuration dictionary

    Raises:
        ValueError: If any surface reconstruction parameters are invalid
    """
    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration error in anat.surface_reconstruction: "
            f"config must be a dictionary, got: {type(config).__name__}. "
            f"Please fix this in your configuration file."
        )

    # Validate enabled flag
    if "enabled" in config:
        if not isinstance(config["enabled"], bool):
            raise ValueError(
                f"Configuration error in anat.surface_reconstruction: "
                f"enabled must be boolean, got: {type(config['enabled']).__name__}. "
                f"Please fix this in your configuration file."
            )

    # Validate use_t1wt2wcombined flag
    if "use_t1wt2wcombined" in config:
        if not isinstance(config["use_t1wt2wcombined"], bool):
            raise ValueError(
                f"Configuration error in anat.surface_reconstruction: "
                f"use_t1wt2wcombined must be boolean, got: {type(config['use_t1wt2wcombined']).__name__}. "
                f"Please fix this in your configuration file."
            )


def validate_paths(
    input_file: Union[str, Path],
    output_dir: Union[str, Path],
    template_file: Optional[Union[str, Path]] = None,
) -> None:
    """Validate input and output paths.

    Args:
        input_file: Path to input file
        output_dir: Path to output directory
        template_file: Optional path to template file

    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If paths are invalid
    """
    # Validate input file
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    # Check if it's a valid neuroimaging file
    if not str(input_path).endswith((".nii", ".nii.gz")):
        raise ValueError(f"Input file must be a NIFTI file: {input_path}")

    # Validate output directory
    output_path = Path(output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ValueError(f"Cannot create output directory: {e}")

    # Validate template file if provided
    if template_file is not None:
        template_path = Path(template_file)
        if not template_path.exists():
            raise FileNotFoundError(f"Template file does not exist: {template_path}")

        if not template_path.is_file():
            raise ValueError(f"Template path is not a file: {template_path}")

        if not str(template_path).endswith((".nii", ".nii.gz")):
            raise ValueError(f"Template file must be a NIFTI file: {template_path}")


def validate_orientation_mismatch_correction_config(config: Dict[str, Any]) -> None:
    """Validate orientation mismatch correction configuration.

    Validates that orientation correction parameters are valid and consistent:
    - real_P_is_actually_labeled_as and real_A_is_actually_labeled_as must be opposite directions
    - real_I_is_actually_labeled_as and real_S_is_actually_labeled_as must be opposite directions

    Args:
        config: Configuration dictionary containing orientation mismatch correction parameters

    Raises:
        ValueError: If configuration is invalid or inconsistent
        TypeError: If configuration values are not strings
    """
    valid_directions = ["P", "A", "I", "S", "R", "L"]

    def _validate_direction_pair(dir_label: str) -> None:
        """Validate a direction and its opposite are consistent."""
        dir_label_key = f"real_{dir_label}_is_actually_labeled_as"
        if dir_label_key not in config:
            return

        dir_real_value = config[dir_label_key]

        # Check if value is a string (needed before calling .capitalize())
        if not isinstance(dir_real_value, str):
            raise TypeError(
                f"Configuration error in orientation_mismatch_correction: "
                f"{dir_label_key} must be a string, got {type(dir_real_value).__name__}. "
                f"Please fix this in your configuration file."
            )

        dir_real = dir_real_value.capitalize()

        # Validate direction is one of the valid directions (also catches empty strings)
        if dir_real not in valid_directions:
            raise ValueError(
                f"Configuration error in orientation_mismatch_correction: "
                f"{dir_label_key} must be one of {valid_directions}, got {dir_real!r}. "
                f"Please fix this in your configuration file."
            )

        # Check that opposite direction key exists
        dir_label_opp = get_opposite_orientation(dir_label)
        dir_label_opp_key = f"real_{dir_label_opp}_is_actually_labeled_as"
        if dir_label_opp_key not in config:
            raise ValueError(
                f"Configuration error in orientation_mismatch_correction: "
                f"{dir_label_opp_key} is required when {dir_label_key} is specified. "
                f"Please add this parameter to your configuration file."
            )

        # Validate opposite direction value
        dir_real_opp_value = config[dir_label_opp_key]
        if not isinstance(dir_real_opp_value, str):
            raise TypeError(
                f"Configuration error in orientation_mismatch_correction: "
                f"{dir_label_opp_key} must be a string, got {type(dir_real_opp_value).__name__}. "
                f"Please fix this in your configuration file."
            )

        dir_real_opp_value_cap = dir_real_opp_value.capitalize()

        # Validate opposite direction is a valid direction
        if dir_real_opp_value_cap not in valid_directions:
            raise ValueError(
                f"Configuration error in orientation_mismatch_correction: "
                f"{dir_label_opp_key} must be one of {valid_directions}, got {dir_real_opp_value_cap!r}. "
                f"Please fix this in your configuration file."
            )
        dir_real_opp = get_opposite_orientation(dir_real)

        # Check that opposite directions are actually opposite
        if dir_real_opp_value_cap != dir_real_opp:
            raise ValueError(
                f"Configuration error in orientation_mismatch_correction: "
                f"{dir_label_key} and {dir_label_opp_key} must be opposite directions. "
                f"If {dir_label_key} is '{dir_real}', then {dir_label_opp_key} should be '{dir_real_opp}', "
                f"but got '{dir_real_opp_value_cap}'. Please fix this in your configuration file."
            )

    # Only validate if orientation correction is enabled
    if not config.get("enabled", False):
        return

    # Validate P/A pair (only need to check one, as the helper validates both)
    _validate_direction_pair("A")

    # Validate I/S pair (only need to check one, as the helper validates both)
    _validate_direction_pair("S")

    # Additional validation: Check for duplicate mappings and ensure valid permutation
    # Collect all mappings
    mappings = {}
    for dir_label in valid_directions:
        dir_label_key = f"real_{dir_label}_is_actually_labeled_as"
        if dir_label_key in config:
            dir_real_value = config[dir_label_key]
            if isinstance(dir_real_value, str):
                dir_real = dir_real_value.capitalize()
                if dir_real in valid_directions:
                    mappings[dir_label] = dir_real

    # Check for duplicate target directions (two sources mapping to same target)
    if mappings:
        target_counts = {}
        for source, target in mappings.items():
            if target in target_counts:
                target_counts[target].append(source)
            else:
                target_counts[target] = [source]

        duplicates = {
            target: sources
            for target, sources in target_counts.items()
            if len(sources) > 1
        }
        if duplicates:
            dup_str = ", ".join(
                f"{target} (from {', '.join(sources)})"
                for target, sources in duplicates.items()
            )
            raise ValueError(
                f"Configuration error in orientation_mismatch_correction: "
                f"Duplicate target directions found: {dup_str}. "
                f"Each target direction should be mapped from exactly one source direction. "
                f"Please fix this in your configuration file."
            )
