# Copyright 2024 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
Segmentation interface for fastsurfer_nn.

This module provides a simple interface for multi-class atlas segmentation using fastsurfer_nn,
which generates segmentation, brain mask, and hemisphere mask outputs.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union, Literal

import torch

from fastsurfer_nn.inference.api import segmentation
from fastsurfer_nn.utils.gpu_utils import resolve_device, run_with_cpu_fallback
from fastsurfer_nn.inference.predictor_utils import setup_atlas_from_checkpoints
from fastsurfer_nn.utils.checkpoint import extract_atlas_metadata, is_binary_checkpoint
from fastsurfer_nn.utils.constants import (
    REPO_ROOT,
    PRETRAINED_MODEL_DIR,
    TEMPLATE_ZOO_ATLAS,
    TEMPLATE_ZOO_TEMPLATES,
)

logger = logging.getLogger(__name__)


def _resolve_template_path(filename: str) -> Path:
    """Resolve a template filename under template_zoo/template/ (first match)."""
    matches = list(TEMPLATE_ZOO_TEMPLATES.glob(f"**/{filename}"))
    if not matches:
        raise FileNotFoundError(
            f"Template file not found: {filename} under {TEMPLATE_ZOO_TEMPLATES}"
        )
    return matches[0]


def _resolve_atlas_path(filename: str) -> Path:
    """Resolve an atlas filename under template_zoo/atlas/ (first match)."""
    matches = list(TEMPLATE_ZOO_ATLAS.glob(f"**/{filename}"))
    if not matches:
        raise FileNotFoundError(
            f"Atlas file not found: {filename} under {TEMPLATE_ZOO_ATLAS}"
        )
    return matches[0]


def _extract_atlas_from_checkpoint(ckpt_path: Path) -> Optional[str]:
    """
    Extract atlas name from a checkpoint file.

    Parameters
    ----------
    ckpt_path : Path
        Path to checkpoint file

    Returns
    -------
    str, None
        Atlas name if found, None otherwise
    """
    try:
        metadata = extract_atlas_metadata(ckpt_path)
        if metadata:
            return metadata.get("atlas_name")
    except Exception as e:
        logger.debug(f"Could not extract atlas from checkpoint {ckpt_path}: {e}")
    return None


def apply_roi_wm_fix(
    seg_results: Dict[str, Any],
    input_image: Union[str, Path],
    atlas_name: Optional[str],
    roi_name: str = "V1",
    wm_thr: float = 0.5,
    registration_threads: Optional[int] = None,
    device_str: str = "cpu",
    enable_fireants: bool = True,
    enable_crop_2round: bool = False,
    logger: Optional[logging.Logger] = None,
    world_to_nmt2sym=None,
) -> None:
    """Add the template's thin ROI white matter to a finished segmentation, in place.

    ``seg_results`` holds the ``segmentation``, ``mask`` and ``hemimask`` paths
    of one CNN pass. Split out of :func:`run_segmentation` so a caller running
    several passes can apply it once, to the pass it keeps.
    ``world_to_nmt2sym`` (4x4, RAS) places the subject in NMT2Sym world for the
    template registration when it is not there already (see ``fix_roi_wm``).
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    # Skip ROI WM fixing for brainmask atlas
    if atlas_name == "brainmask":
        logger.info(
            f"Skipping {roi_name} white matter fixing: not applicable for brainmask atlas"
        )
    else:
        if enable_crop_2round and "input_cropped" in seg_results:
            logger.info(
                "2-pass refinement was applied - applying ROI WM fixing to final pass results only"
            )
        elif enable_crop_2round:
            logger.info(
                "2-pass criteria not met (single pass) - applying ROI WM fixing to results"
            )
        logger.info(f"Applying {roi_name} white matter fixing...")

        # Check required files exist
        if "segmentation" not in seg_results:
            raise ValueError(
                f"{roi_name} WM fixing requires segmentation file, but segmentation was not generated. "
                "This may occur with binary brain mask models."
            )
        if "hemimask" not in seg_results:
            raise ValueError(
                f"{roi_name} WM fixing requires hemisphere mask file, but hemimask was not generated."
            )

        seg_file = Path(seg_results["segmentation"])
        mask_file = Path(seg_results["mask"])
        hemi_mask_file = Path(seg_results["hemimask"])

        # Determine LUT path from checkpoint (same logic as predictor)
        if atlas_name is None:
            raise ValueError(
                "Cannot determine LUT path: atlas_name is None. "
                f"{roi_name} WM fixing requires a multi-class model with atlas_name in checkpoint metadata."
            )

        # Use the same logic as RunModelOnData to determine LUT path from atlas_name
        fastsurfercnn_dir = REPO_ROOT / "src" / "fastsurfer_nn"
        lut_path = (
            fastsurfercnn_dir
            / "atlas"
            / f"atlas-{atlas_name}"
            / f"{atlas_name}_ColorLUT.tsv"
        )

        if not lut_path.exists():
            raise FileNotFoundError(
                f"LUT file not found at {lut_path}. "
                f"This is determined from checkpoint atlas_name='{atlas_name}'. "
                "Please ensure the atlas is installed correctly."
            )
        logger.info(f"LUT path (from checkpoint): {lut_path}")

        # Resolve template file paths from template_zoo/template/ and template_zoo/atlas/
        tpl_T1w_f = _resolve_template_path(
            "tpl-NMT2Sym_res-05_T1w_brain.nii.gz"
        )
        tpl_roi_wm_f = _resolve_template_path(
            f"tpl-NMT2Sym_res-05_T1w_WM_{roi_name}.nii.gz"
        )
        tpl_seg_f = _resolve_atlas_path(
            f"atlas-{atlas_name}_space-NMT2Sym_res-05.nii.gz"
        )

        # Validate template files exist
        if not tpl_seg_f.exists():
            raise FileNotFoundError(
                f"Template segmentation file not found: {tpl_seg_f}"
            )
        if not tpl_T1w_f.exists():
            raise FileNotFoundError(f"Template T1w file not found: {tpl_T1w_f}")
        if not tpl_roi_wm_f.exists():
            raise FileNotFoundError(
                f"Template WM file not found: {tpl_roi_wm_f}"
            )

        # Import fix_roi_wm
        try:
            from fastsurfer_nn.postprocessing.fix_roi_wm import (
                fix_roi_wm,
                _get_stem_without_extension,
            )
        except ImportError as e:
            logger.error(f"Failed to import fix_roi_wm: {e}")
            raise ImportError(
                "Cannot import fix_roi_wm. Please ensure fastsurfer_nn.postprocessing.fix_roi_wm is available."
            ) from e

        # Determine which T1w to use (original or cropped if 2-round was applied)
        t1w_file = Path(input_image)
        if "input_cropped" in seg_results:
            # If 2-round cropping was applied, use the cropped input
            t1w_file = Path(seg_results["input_cropped"])
            logger.info(
                f"Using cropped input for {roi_name} WM fixing: {t1w_file}"
            )

        # Call fix_roi_wm
        try:
            fix_roi_wm(
                seg_f=str(seg_file),
                t1w_f=str(t1w_file),
                mask_f=str(mask_file),
                hemi_mask_f=str(hemi_mask_file),
                lut_path=str(lut_path),
                tpl_seg_f=str(tpl_seg_f),
                tpl_t1w_f=str(tpl_T1w_f),
                tpl_roi_wm_f=str(tpl_roi_wm_f),
                roi_name=roi_name,
                wm_thr=wm_thr,
                backup_original=True,
                logger=logger,
                registration_threads=registration_threads,
                gpu_device=device_str,
                enable_fireants=enable_fireants,
                world_to_nmt2sym=world_to_nmt2sym,
            )
            logger.info(
                f"{roi_name} white matter fixing completed successfully"
            )
        except Exception as e:
            logger.warning(
                f"{roi_name} WM fixing failed: {e}. "
                f"Falling back to unfixed segmentation."
            )
            logger.debug(
                f"{roi_name} WM fixing exception details:", exc_info=True
            )
            backup_f = seg_file.with_name(
                _get_stem_without_extension(seg_file) + "_orig.nii.gz"
            )
            if backup_f.exists():
                shutil.copy(str(backup_f), str(seg_file))
                logger.info(
                    f"Restored segmentation from pre-fix backup: {backup_f}"
                )



def run_segmentation(
    input_image: Union[str, Path],
    modal: str,
    output_dir: Union[str, Path],
    device_id: Union[int, str] = "auto",
    output_data_format: Literal["mgz", "nifti"] = "nifti",
    enable_crop_2round: bool = False,
    plane_weight_coronal: Optional[float] = None,
    plane_weight_axial: Optional[float] = None,
    plane_weight_sagittal: Optional[float] = None,
    use_mixed_model: bool = False,
    fix_wm_islands: bool = True,
    label_island_min_volume_mm3: float = 0.0,
    create_hemimask: bool = True,
    fix_roi_wm: bool = False,
    roi_name: str = "V1",
    wm_thr: float = 0.5,
    save_debug_intermediates: bool = False,
    registration_threads: Optional[int] = None,
    enable_fireants: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    """
    Perform multi-class atlas segmentation using fastsurfer_nn.

    This function performs segmentation and generates brain mask, segmentation, and hemisphere mask.

    The function implements an efficient "input space → model space → input space" workflow:
    1. Runs fastsurfer_nn segmentation on the input image (in model space)
    2. Resamples segmentation back to native input space (in-memory, single operation)
    3. Creates brain mask and hemisphere mask from the resampled segmentation
    4. Saves all outputs to the output directory (all in native input space)

    All outputs are automatically in the same space as the input image, providing
    a seamless user experience without manual resampling.

    Args:
        input_image: Path to the input image (T1w)
        modal: 'anat' (modality - only anatomical images are supported)
        output_dir: Directory to save all output files (segmentation, mask, hemimask)
        device_id: GPU device to use ('auto', -1 for CPU, or specific GPU index)
        logger: Logger instance (optional)
        output_data_format: {"mgz", "nifti"}, default="nifti"
            Output file format. "mgz" saves as .mgz (MGH format), "nifti" saves as .nii.gz (NIfTI format).
        enable_crop_2round: bool, default=False
            If True, enable two-pass refinement: after first pass, if brain occupies < 20% of FOV
            and image dimension > model height, crop image to brain region and run second pass.
            First-pass outputs are moved to output_dir/pass_1/, and cropped input is saved as
            output_dir/input_cropped.{ext}. Final outputs are in cropped image's native space.
        plane_weight_coronal: float, optional
            Weight for coronal plane in multi-view prediction.
        plane_weight_axial: float, optional
            Weight for axial plane in multi-view prediction.
        plane_weight_sagittal: float, optional
            Weight for sagittal plane in multi-view prediction.
        use_mixed_model: bool, default=False
            If True, use a single mixed-plane model checkpoint for all 3 planes instead of
            separate plane-specific checkpoints. The mixed checkpoint should be named
            {modal}_seg-{atlas}_mixed.pkl (e.g., T1w_seg-ARM2_mixed.pkl).
            When using mixed model, the same checkpoint is used for all 3 planes, and
            each plane is evaluated separately with the specified plane weights.
        fix_wm_islands: bool, default=True
            If True, apply WM island correction after segmentation. This fixes mislabeled
            disconnected WM regions by flipping them to the correct hemisphere based on
            spatial proximity. Only applies to multi-class models (ignored for binary models).
        label_island_min_volume_mm3: float, default=0.0
            Relabel detached label fragments smaller than this volume (mm^3) to their
            neighbours' label; 0 disables. Multi-class models only.
            Requires extended ColorLUT with region and hemisphere columns.
        create_hemimask: bool, default=True
            If True, create hemisphere mask from segmentation (multi-class only, requires LUT).
            If False, skip hemimask creation to save processing time. Binary models always skip this.
        fix_roi_wm: bool, default=False
            If True, apply ROI white matter fixing using template registration after segmentation.
            This fixes missing thin WM in the specified ROI by registering template ROI WM to individual space.
            Template files are resolved from template_zoo/template/ and template_zoo/atlas/. LUT path is automatically determined
            from checkpoint metadata (atlas_name).
        roi_name: str, default='V1'
            ROI name for WM fixing. Default is 'V1'.
        wm_thr: float, default=0.5
            Threshold for WM probability map in ROI WM fixing.
        registration_threads: int, optional
            Number of threads to use for ANTs registration when fix_roi_wm=True.
            The ANTs CLI itself follows OMP_NUM_THREADS (the task's CPU allocation).
        enable_fireants: bool, default=True
            registration.enable_fireants for the ROI WM fix's template registration.

        Note: Preprocessing parameters (vox_size, orientation, image_size)
        are automatically read from checkpoint metadata (required), ensuring consistency
        with the training configuration.

    Returns:
        Dictionary with output file paths:
        - 'brain_mask': Path to the generated brain mask (binary, values 0 or 1)
        - 'segmentation': Path to the segmentation file (optional)
        - 'hemimask': Path to the hemisphere mask file (optional)
        - 'input_image': Path to the input image
        - 'atlas_name': Name of atlas used (optional)

    Raises:
        FileNotFoundError: If input image doesn't exist
        RuntimeError: If segmentation fails
        ValueError: If modality is not 'anat' or no checkpoints found
    """

    # Setup logger if not provided
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Segmentation (fastsurfer_nn): starting for {modal} modality")

    # Validate inputs
    input_image = Path(input_image)
    if not input_image.exists():
        logger.error(
            f"Segmentation (fastsurfer_nn): input image not found: {input_image}"
        )
        raise FileNotFoundError(f"Input image not found: {input_image}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if modal != "anat":
        logger.error(
            f"Segmentation (fastsurfer_nn): invalid modality={modal}, must be 'anat'"
        )
        raise ValueError(f"Invalid modality: {modal}. Must be 'anat'")

    # Use PRETRAINED_MODEL_DIR from constants
    pretrained_dir = PRETRAINED_MODEL_DIR

    # Get checkpoint template for this modality
    # Hardcoded checkpoint mapping: {modality}_seg-{atlas}_planexxx.pkl
    # Replace 'planexxx' with actual plane name (axial, coronal, sagittal) or 'mixed'
    checkpoint_map = {
        "anat": "T1w_seg-ARM2_planexxx.pkl",
    }
    ckpt_template = checkpoint_map[modal]

    if use_mixed_model:
        # Mixed-plane model: look for single mixed checkpoint
        ckpt_name = ckpt_template.replace("planexxx", "mixed")
        mixed_ckpt_path = pretrained_dir / ckpt_name

        if not mixed_ckpt_path.exists():
            raise ValueError(
                f"Mixed-plane checkpoint not found for {modal} modality. "
                f"Expected file: {ckpt_name} in {pretrained_dir}"
            )

        logger.info(f"Checkpoint: using mixed-plane model {ckpt_name}")
        # Use the same mixed checkpoint for all 3 planes
        checkpoints = {
            "axial": mixed_ckpt_path,
            "coronal": mixed_ckpt_path,
            "sagittal": mixed_ckpt_path,
        }
    else:
        # Separate plane models: look for individual plane checkpoints
        # Resolve checkpoint paths by replacing 'planexxx' with actual plane names
        checkpoints = {}
        for plane in ["axial", "coronal", "sagittal"]:
            ckpt_name = ckpt_template.replace("planexxx", plane)
            ckpt_path = pretrained_dir / ckpt_name
            if ckpt_path.exists():
                checkpoints[plane] = ckpt_path
                logger.debug(f"Checkpoint: found {ckpt_name}")
            else:
                checkpoints[plane] = None
                logger.warning(f"Checkpoint: not found {ckpt_path}")

        # Validate that at least one checkpoint is found
        found_planes = [
            plane for plane, ckpt in checkpoints.items() if ckpt is not None
        ]
        if not found_planes:
            raise ValueError(
                f"No checkpoints found for {modal} modality. "
                f"Expected files like: {ckpt_template.replace('planexxx', '{plane}')} "
                f"in {pretrained_dir}"
            )

        logger.info(f"Checkpoint: found for planes={', '.join(found_planes)}")

    # Extract atlas metadata from checkpoints
    try:
        atlas_name, atlas_metadata = setup_atlas_from_checkpoints(
            ckpt_ax=checkpoints.get("axial"),
            ckpt_cor=checkpoints.get("coronal"),
            ckpt_sag=checkpoints.get("sagittal"),
        )
        # Check if this is a binary model
        is_binary = atlas_metadata.get("is_binary_task", False)
        if is_binary:
            logger.info(
                f"Model: binary brain mask task detected (atlas_name={atlas_name}, num_classes={atlas_metadata.get('num_classes', 2)})"
            )
        else:
            logger.info(
                f"Atlas: using {atlas_name} (num_classes={atlas_metadata.get('num_classes', 'unknown')})"
            )
    except Exception as e:
        logger.warning(f"Atlas: could not extract metadata: {e}")
        # Fallback: determine binary from checkpoint
        first_ckpt = next(ckpt for ckpt in checkpoints.values() if ckpt is not None)

        is_binary, num_classes = is_binary_checkpoint(first_ckpt)

        if is_binary is True:
            # Binary model - doesn't need an atlas
            num_classes = num_classes or 2
            atlas_name = None
            atlas_metadata = {
                "is_binary_task": True,
                "atlas_name": None,  # Binary models don't need atlas
                "num_classes": num_classes,
                "plane": "",  # Not specific to a single plane (multi-plane model)
            }
            logger.info(
                f"Model: detected binary model (num_classes={num_classes}, no atlas required)"
            )
        else:
            # Multi-class model or cannot determine - try to extract or use default
            atlas_name = _extract_atlas_from_checkpoint(first_ckpt)
            atlas_metadata = None
            logger.warning(
                f"Atlas: using fallback {atlas_name} (metadata will be auto-detected)"
            )

    # One device policy for every step (auto / cpu / -1 / index / cuda:N); a quoted
    # "0" or "-1" from YAML used to reach torch.device() verbatim and crash.
    device_str = str(resolve_device(device_id))

    try:
        # Run segmentation using the high-level API
        # This will create segmentation, mask, and hemimask
        logger.info(f"Segmentation: running on {input_image}")
        # A CUDA out-of-memory error reruns the network once on the CPU (logged,
        # and recorded in the step's metadata JSON) instead of failing the step.
        seg_results = run_with_cpu_fallback(
            lambda dev: segmentation(
                input_image=input_image,
                output_dir=output_dir,
                atlas_name=atlas_name,
                atlas_metadata=atlas_metadata,
                ckpt_ax=checkpoints.get("axial"),
                ckpt_cor=checkpoints.get("coronal"),
                ckpt_sag=checkpoints.get("sagittal"),
                device=str(dev),
                viewagg_device=str(dev),
                plane_weight_coronal=plane_weight_coronal,
                plane_weight_axial=plane_weight_axial,
                plane_weight_sagittal=plane_weight_sagittal,
                fix_wm_islands=fix_wm_islands,
                label_island_min_volume_mm3=label_island_min_volume_mm3,
                create_hemimask=create_hemimask,
                output_data_format=output_data_format,
                enable_crop_2round=enable_crop_2round,
                logger=logger,
                save_debug_intermediates=save_debug_intermediates,
            ),
            torch.device(device_str),
            "segmentation network",
            logger,
        )

        logger.info("Segmentation (fastsurfer_nn): completed successfully")
        logger.info(f"Output: files saved to {output_dir}")
        logger.info(f"Output: segmentation={seg_results.get('segmentation')}")
        logger.info(f"Output: brain_mask={seg_results.get('mask')}")
        if "hemimask" in seg_results:
            logger.info(f"Output: hemisphere_mask={seg_results.get('hemimask')}")
        if "input_cropped" in seg_results:
            logger.info(f"Output: input_cropped={seg_results.get('input_cropped')}")

        # Save LUT alongside segmentation when available (multi-class with atlas)
        if "segmentation" in seg_results and atlas_name is not None:
            fastsurfercnn_dir = REPO_ROOT / "src" / "fastsurfer_nn"
            lut_path = (
                fastsurfercnn_dir
                / "atlas"
                / f"atlas-{atlas_name}"
                / f"{atlas_name}_ColorLUT.tsv"
            )
            if lut_path.exists():
                output_f = str(seg_results["segmentation"])
                lut_out = output_f.replace(".nii.gz", "_lut.tsv").replace(
                    ".mgz", "_lut.tsv"
                )
                shutil.copy2(lut_path, lut_out)
                logger.info(f"Output: LUT saved to {lut_out}")
            else:
                logger.debug(f"LUT not found at {lut_path}, skipping LUT save")

        # Apply ROI WM fixing if enabled
        if fix_roi_wm:
            apply_roi_wm_fix(
                seg_results,
                input_image,
                atlas_name,
                roi_name=roi_name,
                wm_thr=wm_thr,
                registration_threads=registration_threads,
                device_str=device_str,
                enable_fireants=enable_fireants,
                enable_crop_2round=enable_crop_2round,
                logger=logger,
            )

        # Return all output file paths
        result = {
            "brain_mask": str(seg_results["mask"]),
            "input_image": str(input_image),
            "atlas_name": atlas_name,
        }
        if "segmentation" in seg_results:
            result["segmentation"] = str(seg_results["segmentation"])
        if "hemimask" in seg_results:
            result["hemimask"] = str(seg_results["hemimask"])
        if "input_cropped" in seg_results:
            result["input_cropped"] = str(seg_results["input_cropped"])

        return result

    except Exception as e:
        logger.error(f"Segmentation (fastsurfer_nn): failed: {str(e)}")
        raise RuntimeError(f"Segmentation (fastsurfer_nn) failed: {str(e)}") from e
