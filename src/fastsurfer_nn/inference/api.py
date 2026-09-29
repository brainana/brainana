# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
High-level API for brain segmentation.

This module provides the main user-facing functions for running
segmentation and skullstripping on brain images.
"""

import copy
import shutil
import traceback
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import torch

from fastsurfer_nn.data_loader import data_utils
from fastsurfer_nn.inference.predictor import RunModelOnData
from fastsurfer_nn.inference.predictor_utils import (
    crop_image_to_brain_mask,
    setup_atlas_from_checkpoints,
    should_apply_refinement,
    validate_checkpoints,
)
from fastsurfer_nn.postprocessing.postseg_utils import (
    create_hemisphere_masks,
    create_mask,
    relabel_small_islands,
)
from fastsurfer_nn.utils import logging
from fastsurfer_nn.utils.constants import (
    MASK_DILATION_SIZE_MM,
    ROUNDS_OF_MORPHOLOGICAL_OPERATIONS,
    TWO_PASS_BRAIN_RATIO_THRESHOLD,
    TWO_PASS_CROP_MARGIN,
)

LOGGER = logging.getLogger(__name__)


# %%
def _apply_two_pass_refinement(
    input_image: Path,
    output_dir: Path,
    file_ext: str,
    atlas_name: str | None,
    atlas_metadata: dict | None,
    ckpt_ax: Path | None,
    ckpt_sag: Path | None,
    ckpt_cor: Path | None,
    device: str,
    viewagg_device: str,
    threads: int | None,
    batch_size: int,
    plane_weight_coronal: float | None,
    plane_weight_axial: float | None,
    plane_weight_sagittal: float | None,
    fix_wm_islands: bool,
    create_hemimask: bool,
    output_data_format: Literal["mgz", "nifti"],
    logger: logging.Logger | None = None,
    save_debug_intermediates: bool = False,
) -> bool:
    """
    Apply two-pass refinement: crop image and run fresh segmentation.

    This function moves first-pass outputs to pass_1/ directory, crops the
    ORIGINAL input image using the brain mask, then runs a completely fresh
    segmentation on the cropped image. This avoids any state/affine issues
    from reusing the predictor.

    Parameters
    ----------
    input_image : Path
        Original input image path
    output_dir : Path
        Output directory
    file_ext : str
        File extension for output files
    atlas_name : str, None
        Name of the atlas (None for binary brain mask models)
    atlas_metadata : dict, optional
        Atlas metadata from checkpoint
    ckpt_ax, ckpt_cor, ckpt_sag : Path, optional
        Checkpoint paths for each plane
    device : str
        Device to run inference on
    viewagg_device : str
        Device to run view aggregation on
    threads : int or None
        Number of threads for CPU operations (None: get_num_threads())
    batch_size : int
        Batch size for inference
    plane_weight_coronal, plane_weight_axial, plane_weight_sagittal : float, optional
        Weights for multi-view prediction
    fix_wm_islands : bool
        Whether to apply WM island correction
    output_data_format : {"mgz", "nifti"}
        Output file format
    logger : logging.Logger, optional
        Logger instance to use for logging. If not provided, uses the module-level logger.
    save_debug_intermediates : bool, default=False
        If True, save intermediate files for debugging in the second pass.

    Returns
    -------
    bool
        True if refinement completed successfully, False if it failed
    """
    pass_1_dir = output_dir / "pass_1"
    pass_1_dir.mkdir(parents=True, exist_ok=True)

    # Use provided logger or fall back to module-level logger
    log = logger if logger is not None else LOGGER

    log.info("Applying two-pass refinement...")
    log.info(f"  Moving first-pass outputs to {pass_1_dir.name}/")

    # Define first-pass files
    first_pass_files = {
        "segmentation": output_dir / f"segmentation{file_ext}",
        "mask": output_dir / f"mask{file_ext}",
    }

    # Check for hemimask
    hemi_mask_path = output_dir / f"mask_hemi{file_ext}"
    if hemi_mask_path.exists():
        first_pass_files["hemimask"] = hemi_mask_path

    try:
        # Step 1: Move first-pass outputs to pass_1 directory
        for key, src_path in first_pass_files.items():
            if src_path.exists():
                dst_path = pass_1_dir / src_path.name
                shutil.move(str(src_path), str(dst_path))
                log.info(f"  Moved {key}: {src_path.name}")

        # Step 2: Crop ORIGINAL input image using first-pass mask
        mask_path = pass_1_dir / f"mask{file_ext}"
        cropped_input_path = output_dir / f"input_cropped{file_ext}"

        log.info(
            f"  Cropping original input to brain region (margin={TWO_PASS_CROP_MARGIN*100:.0f}%)..."
        )

        # Load original image to get its shape for logging
        orig_img = nib.load(input_image)

        cropped_img = crop_image_to_brain_mask(
            str(input_image),  # Crop the ORIGINAL input image
            str(mask_path),  # Using mask from first pass
            margin=TWO_PASS_CROP_MARGIN,
            save_path=cropped_input_path,
        )

        log.info(f"  ✓ Cropped: {orig_img.shape} → {cropped_img.shape}")
        log.info(
            f"  ✓ Space saved: {np.prod(orig_img.shape) / np.prod(cropped_img.shape):.1f}x reduction"
        )
        log.info(f"  ✓ Saved cropped input: {cropped_input_path.name}")

        # Step 3: Run FRESH segmentation on cropped image (starting from scratch)
        log.info("  Running 2nd pass with fresh predictor on cropped image...")
        log.info("  (This runs the full pipeline from scratch - no state reuse)")

        segmentation(
            input_image=cropped_input_path,
            output_dir=output_dir,  # Outputs go back to main directory
            atlas_name=atlas_name,
            atlas_metadata=atlas_metadata,
            ckpt_ax=ckpt_ax,
            ckpt_sag=ckpt_sag,
            ckpt_cor=ckpt_cor,
            device=device,
            viewagg_device=viewagg_device,
            threads=threads,
            batch_size=batch_size,
            plane_weight_coronal=plane_weight_coronal,
            plane_weight_axial=plane_weight_axial,
            plane_weight_sagittal=plane_weight_sagittal,
            fix_wm_islands=fix_wm_islands,
            create_hemimask=create_hemimask,
            output_data_format=output_data_format,
            enable_crop_2round=False,  # Don't recurse!
            logger=log,
            save_debug_intermediates=save_debug_intermediates,
        )

        log.info("  ✓ Two-pass refinement completed successfully")
        return True

    except Exception as e:
        log.error(f"  ✗ Two-pass refinement failed: {e}")
        log.debug(f"  Error details: {traceback.format_exc()}")
        log.warning("  Falling back to first-pass prediction")
        log.info(f"  First-pass outputs are available in {pass_1_dir.name}/")

        # Try to restore first-pass outputs to main directory
        try:
            log.info("  Restoring first-pass outputs to main directory...")
            for key, src_path in first_pass_files.items():
                dst_path = pass_1_dir / src_path.name
                if dst_path.exists():
                    shutil.copy(str(dst_path), str(src_path))
                    log.info(f"  Restored {key}: {src_path.name}")
        except Exception as restore_error:
            log.error(f"  Failed to restore first-pass outputs: {restore_error}")

        return False


def segmentation(
    input_image: str | Path,
    output_dir: str | Path,
    atlas_name: str | None = None,
    atlas_metadata: dict | None = None,
    ckpt_ax: Path | None = None,
    ckpt_sag: Path | None = None,
    ckpt_cor: Path | None = None,
    device: str = "auto",
    viewagg_device: str = "cpu",
    threads: int | None = None,
    batch_size: int = 1,
    plane_weight_coronal: float | None = None,
    plane_weight_axial: float | None = None,
    plane_weight_sagittal: float | None = None,
    fix_wm_islands: bool = True,
    label_island_min_volume_mm3: float = 0.0,
    create_hemimask: bool = True,
    output_data_format: Literal["mgz", "nifti"] = "nifti",
    enable_crop_2round: bool = False,
    logger: logging.Logger | None = None,
    save_debug_intermediates: bool = False,
) -> dict[str, Path]:
    """
    Run segmentation and save outputs (segmentation, mask, hemimask) to output directory.

    Supports both multi-class atlas segmentation and binary brain mask models.
    This is a high-level convenience function that implements the automatic
    "input space → model space → input space" workflow:
    1. Runs fastsurfer_nn segmentation on the input image (in model space)
    2. Resamples segmentation back to native input space (in-memory, pure Python)
    3. Creates brain mask from the resampled segmentation (with topological refinement)
    4. Creates hemisphere mask (multi-class only, requires LUT)
    5. Saves all outputs to the specified output directory (all in native space)

    All outputs are automatically in the same space as the input image.
    Uses pure Python resampling (no external tool dependencies).

    Both binary and multi-class models go through the same pipeline:
    - Prediction output is saved as "segmentation" (binary 0/1 or multi-class label IDs)
    - Brain mask is created via create_mask() which applies dilation, erosion, and
      largest component selection for topological refinement

    Parameters
    ----------
    input_image : str, Path
        Path to input image
    output_dir : str, Path
        Output directory where segmentation, mask, and hemimask will be saved
    atlas_name : str, optional
        Name of the atlas (e.g., "ARM2", "ARM3"). If None, will be auto-detected from checkpoints.
        For binary brain mask models (NUM_CLASSES=2), this should be None.
    atlas_metadata : dict, optional
        Atlas metadata extracted from checkpoint. If None, will be auto-detected from checkpoints.
    ckpt_ax, ckpt_cor, ckpt_sag : Path, optional
        Checkpoint paths for each plane. At least one must be provided.
    device : str, default="auto"
        Device to run inference on
    viewagg_device : str, default="cpu"
        Device to run view aggregation on
    threads : int, optional
        Number of threads for CPU operations. None uses get_num_threads(), which
        follows OMP_NUM_THREADS (the task's CPU allocation under Nextflow)
    batch_size : int, default=1
        Batch size for inference
    plane_weight_coronal, plane_weight_axial, plane_weight_sagittal : float, optional
        Weights for multi-view prediction
    fix_wm_islands : bool, default=True
        Whether to apply WM island correction (multi-class only, ignored for binary models)
    label_island_min_volume_mm3 : float, default=0.0
        Relabel detached label fragments smaller than this (mm^3) to their
        neighbours' label (multi-class only); 0 disables. See
        :func:`~fastsurfer_nn.postprocessing.postseg_utils.relabel_small_islands`.
    create_hemimask : bool, default=True
        If True, create hemisphere mask from segmentation (multi-class only, requires LUT).
        If False, skip hemimask creation to save processing time. Binary models always skip this.
    output_data_format : {"mgz", "nifti"}, default="nifti"
        Output file format. "mgz" saves as .mgz (MGH format), "nifti" saves as .nii.gz (NIfTI format).
        Resampling uses pure Python (in-memory), no external tools needed.
    enable_crop_2round : bool, default=False
        If True, enable two-pass refinement: after first pass, if brain occupies < 20% of FOV
        and image dimension > model height, crop the ORIGINAL input image to brain region
        and run a completely fresh segmentation on it. First-pass outputs are moved to
        output_dir/pass_1/, cropped input is saved as output_dir/input_cropped.{ext}, and
        final outputs (from second pass) are saved to main output_dir in cropped image's native space.
    logger : logging.Logger, optional
        Logger instance to use for logging. If not provided, uses the module-level logger.
    save_debug_intermediates : bool, default=False
        If True, save intermediate files for debugging:
        - conformed image (after conforming, before prediction)
        - prediction after each plane (before aggregation)
        - final aggregated prediction (before resampling)
        Files are saved to output_dir/debug_intermediates/

    Note
    ----
    Preprocessing parameters (vox_size, orientation, image_size)
    are automatically read from checkpoint metadata (required), ensuring consistency
    with the training configuration.

    Returns
    -------
    dict[str, Path]
        Dictionary with keys:
        - 'segmentation': Path to saved segmentation file (binary 0/1 or multi-class label IDs)
        - 'mask': Path to saved brain mask file (refined via create_mask)
        - 'hemimask': Path to saved hemisphere mask file (multi-class only, if created)
        - 'input_cropped': Path to cropped input (if two-pass refinement was applied)
    """
    input_image = Path(input_image)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use provided logger or fall back to module-level logger
    log = logger if logger is not None else LOGGER

    # Validate checkpoints
    validate_checkpoints(ckpt_ax, ckpt_cor, ckpt_sag)

    # Auto-detect atlas_name and atlas_metadata from checkpoints if not provided
    # Note: atlas_name can be None for binary models, so only check atlas_metadata
    if atlas_metadata is None:
        log.info("Auto-detecting atlas information from checkpoints...")
        detected_atlas_name, detected_atlas_metadata = setup_atlas_from_checkpoints(
            ckpt_ax=ckpt_ax,
            ckpt_cor=ckpt_cor,
            ckpt_sag=ckpt_sag,
        )
        atlas_name = atlas_name or detected_atlas_name
        atlas_metadata = detected_atlas_metadata

    # Determine binary vs multi-class mode and set all related flags
    # Binary models (NUM_CLASSES=2) don't have LUT, so certain features are disabled
    is_binary = atlas_metadata.get("is_binary_task", False)

    if is_binary:
        log.info("Binary brain mask model detected")
        # Binary models: disable WM island correction (requires LUT with hemisphere info)
        if fix_wm_islands:
            log.info(
                "  (fix_wm_islands=True was provided but disabled for binary models)"
            )
        fix_wm_islands = False
        create_hemi_mask = False  # Binary models don't have hemisphere labels
    else:
        # Multi-class models: use user-provided create_hemimask setting
        create_hemi_mask = create_hemimask
        if not create_hemimask:
            log.info(
                "  (create_hemimask=False: skipping hemisphere mask creation to save processing time)"
            )

    # Create debug directory if needed
    debug_dir = None
    if save_debug_intermediates:
        debug_dir = output_dir / "debug_intermediates"
        debug_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Debug mode enabled: intermediate files will be saved to {debug_dir}")

    # Initialize predictor
    # Preprocessing parameters (vox_size, orientation, image_size)
    # are automatically read from checkpoint metadata if available
    predictor = RunModelOnData(
        atlas_name=atlas_name,
        atlas_metadata=atlas_metadata,
        ckpt_ax=ckpt_ax,
        ckpt_sag=ckpt_sag,
        ckpt_cor=ckpt_cor,
        device=device,
        viewagg_device=viewagg_device,
        threads=threads,
        batch_size=batch_size,
        plane_weight_coronal=plane_weight_coronal,
        plane_weight_axial=plane_weight_axial,
        plane_weight_sagittal=plane_weight_sagittal,
        fix_wm_islands=fix_wm_islands,
        save_debug_intermediates=save_debug_intermediates,
        debug_dir=debug_dir,
    )

    # Run prediction (returns segmentation in model/conformed space)
    log.info(f"Running segmentation on {input_image}")
    pred_data = predictor.get_prediction(str(input_image))

    # Save final aggregated prediction before resampling (debug)
    if save_debug_intermediates and debug_dir is not None:
        debug_pred_path = debug_dir / "prediction_aggregated_before_resample.nii.gz"
        log.info(
            f"Saving aggregated prediction (before resampling) to {debug_pred_path.name}"
        )
        data_utils.save_image(
            predictor._conformed_img.header,
            predictor._conformed_img.affine,
            pred_data,
            debug_pred_path,
            dtype=np.int16,
        )

    # Debug: Log prediction statistics for binary models
    if is_binary:
        unique_vals, counts = np.unique(pred_data, return_counts=True)
        log.info(
            f"Binary model prediction statistics: unique values={unique_vals}, counts={counts}"
        )
        log.info(f"  Prediction shape: {pred_data.shape}, dtype: {pred_data.dtype}")
        log.info(f"  Prediction range: [{pred_data.min()}, {pred_data.max()}]")
        if len(unique_vals) > 2:
            log.warning(
                f"  WARNING: Binary model prediction has {len(unique_vals)} unique values (expected 2: 0 and 1)"
            )
        # Ensure binary predictions are integer type (0 or 1)
        if (
            pred_data.dtype != np.int16
            and pred_data.dtype != np.int32
            and pred_data.dtype != np.int64
        ):
            log.info(f"  Converting binary prediction from {pred_data.dtype} to int16")
            pred_data = pred_data.astype(np.int16)

    # Create masks from the final segmentation
    # Both binary and multi-class models go through the same create_mask() pipeline
    # for topological refinement (dilation, erosion, largest component selection)
    log.info("Creating brain mask from segmentation (with topological refinement)...")

    # Calculate mask dilation and erosion sizes based on image resolution
    # Get voxel size from conformed image (in model space where mask is created)
    zoom = predictor._conformed_img.header.get_zooms()[:3]
    resolution = np.mean(zoom)  # Average voxel size in mm
    mask_dilation_voxels = int(MASK_DILATION_SIZE_MM / resolution)
    mask_erosion_voxels = max(0, mask_dilation_voxels - 1)  # Ensure non-negative
    log.info(
        f"Mask parameters: dilation={mask_dilation_voxels} voxels, erosion={mask_erosion_voxels} voxels (resolution={resolution:.3f} mm)"
    )

    # do morphological operations
    brain_mask = create_mask(
        copy.deepcopy(pred_data),
        mask_dilation_voxels,
        mask_erosion_voxels,
        ROUNDS_OF_MORPHOLOGICAL_OPERATIONS,
        voxel_size=tuple(zoom),  # Pass voxel size for volume calculation
    )
    brain_mask = brain_mask.astype(np.uint8)

    # Ensure segmentation labels remain inside the morphology-refined brain mask.
    seg_nonzero_before_mask = int(np.count_nonzero(pred_data))
    pred_data[brain_mask == 0] = 0
    seg_nonzero_after_mask = int(np.count_nonzero(pred_data))
    log.info(
        "Applied morphology-refined brain mask to segmentation "
        f"(removed {seg_nonzero_before_mask - seg_nonzero_after_mask} labeled voxels)"
    )

    if not is_binary and label_island_min_volume_mm3 > 0:
        pred_data, n_islands, n_island_voxels = relabel_small_islands(
            pred_data, tuple(zoom), label_island_min_volume_mm3
        )
        log.info(
            f"Relabelled {n_islands} label fragments < {label_island_min_volume_mm3:g} mm^3 "
            f"({n_island_voxels} voxels) to their neighbours' label"
        )

    # Hemisphere mask creation and saving (multi-class only, requires LUT)
    hemi_mask = None
    if create_hemi_mask:
        log.info("Creating hemisphere mask...")
        try:
            hemi_mask = create_hemisphere_masks(
                brain_mask, pred_data, lut_path=predictor.lut_path
            )

        except Exception as e:
            log.warning(f"Could not create hemisphere mask: {e}")

    # Map output format to file extension
    format_to_ext = {"mgz": ".mgz", "nifti": ".nii.gz"}
    file_ext = format_to_ext[output_data_format]
    seg_path = output_dir / f"segmentation{file_ext}"
    mask_path = output_dir / f"mask{file_ext}"
    hemi_mask_path = None
    if hemi_mask is not None:
        hemi_mask_path = output_dir / f"mask_hemi{file_ext}"

    # Always resample to native space
    # Loop for pred_data, brain_mask, and hemi_mask to save
    data_to_save = [
        ("segmentation", pred_data, seg_path, np.int16),
        ("mask", brain_mask, mask_path, np.uint8),
    ]
    if hemi_mask is not None:
        data_to_save.append(("hemimask", hemi_mask, hemi_mask_path, np.uint8))

    if predictor._should_resample():
        log.info("Resampling to native space...")

        for name, data, path, dtype in data_to_save:
            log.info(f"Resampling {name}...")
            resampled = predictor._resample_to_native(data, interpolation="nearest")
            log.info(f"Successfully resampled {name} (shape: {resampled.shape})")

            data_utils.save_image(
                predictor._input_native_img.header,
                predictor._input_native_img.affine,
                resampled,
                path,
                dtype=dtype,
            )

    else:
        # Input was already conformed - save directly (still in native space)
        log.info("No resampling needed - input was already conformed")

        for name, data, path, dtype in data_to_save:
            log.info(f"Saving {name}...")
            data_utils.save_image(
                predictor._conformed_img.header,
                predictor._conformed_img.affine,
                data,
                path,
                dtype=dtype,
            )

    # Build result dictionary
    result = {
        "segmentation": seg_path,
        "mask": mask_path,
    }
    if hemi_mask_path is not None:
        result["hemimask"] = hemi_mask_path

    # Check if two-pass refinement is needed
    if enable_crop_2round:
        # Get model height from checkpoint config
        model_height = predictor.cfg_fin.MODEL.HEIGHT

        # Get reference image for header/affine (use original input image)
        reference_img = predictor._input_native_img

        # Check if refinement should be applied
        should_refine, brain_ratio, max_orig_dim = should_apply_refinement(
            brain_mask, reference_img, model_height
        )

        log.info("")
        log.info("=" * 60)
        log.info("Two-pass refinement decision:")
        log.info(f"  Brain occupancy: {brain_ratio*100:.1f}% of FOV")
        log.info(f"  Image dimensions: {reference_img.shape} (max: {max_orig_dim})")
        log.info(f"  Model height: {model_height}")
        log.info(
            f"  Threshold: brain < {TWO_PASS_BRAIN_RATIO_THRESHOLD*100:.0f}% AND dim > {model_height}"
        )

        # Log the decision clearly
        if should_refine:
            log.info(
                f"  → DECISION: WILL APPLY 2nd pass "
                f"(brain occupancy {brain_ratio*100:.1f}% < {TWO_PASS_BRAIN_RATIO_THRESHOLD*100:.0f}% "
                f"AND max dimension {max_orig_dim} > {model_height})"
            )
        else:
            reason = (
                f"brain occupancy {brain_ratio*100:.1f}% >= {TWO_PASS_BRAIN_RATIO_THRESHOLD*100:.0f}%"
                if brain_ratio >= TWO_PASS_BRAIN_RATIO_THRESHOLD
                else f"max dimension {max_orig_dim} <= {model_height}"
            )
            log.info(f"  → DECISION: WILL NOT apply 2nd pass ({reason})")
        log.info("=" * 60)
        log.info("")

        if should_refine:
            # Apply two-pass refinement (runs fresh segmentation on cropped image)
            refinement_applied = _apply_two_pass_refinement(
                input_image=input_image,
                output_dir=output_dir,
                file_ext=file_ext,
                atlas_name=atlas_name,
                atlas_metadata=atlas_metadata,
                ckpt_ax=ckpt_ax,
                ckpt_sag=ckpt_sag,
                ckpt_cor=ckpt_cor,
                device=device,
                viewagg_device=viewagg_device,
                threads=threads,
                batch_size=batch_size,
                plane_weight_coronal=plane_weight_coronal,
                plane_weight_axial=plane_weight_axial,
                plane_weight_sagittal=plane_weight_sagittal,
                fix_wm_islands=fix_wm_islands,
                create_hemimask=create_hemimask,
                output_data_format=output_data_format,
                logger=log,
                save_debug_intermediates=save_debug_intermediates,
            )

            # If refinement succeeded, update result paths to reflect second-pass outputs
            # (which are already saved by the fresh segmentation call)
            if refinement_applied:
                log.info("  ✓ Second-pass outputs are now in main directory")
                # Update result dict to include cropped input
                result["input_cropped"] = output_dir / f"input_cropped{file_ext}"

    # Explicit cleanup to free GPU memory
    del predictor
    # is_initialized(): only a run that actually used CUDA has a cache to clear,
    # and it does not load the driver in CPU mode.
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        log.info("GPU memory cache cleared after inference")

    return result
