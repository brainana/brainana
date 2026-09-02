"""
Quality Control Snapshot Generation

This module creates visual snapshots for assessing preprocessing quality,
including registration overlays, motion parameter plots, and brain extraction results.
All snapshots are generated using clean, publication-ready visualizations.
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Dict, Union, List, Optional
from PIL import Image
import tempfile
import shutil
import subprocess

try:
    from fastsurfer_surfrecon.io.surface import convert_fs_surface_to_gifti
except Exception:
    convert_fs_surface_to_gifti = None

from .mri_plotting import (
    create_overlay_grid_3xN,
    create_motion_plot,
    create_grid_mri_image,
    outlier_frame_indices,
    save_timeseries_qc_figure,
    MOTION_OUTLIER_PREFIX,
    NONSTEADY_OUTLIER_PREFIX,
    _crop_white_space,
    _create_colorbar,
    _create_label_image,
    _find_content_width,
    MARGIN_PERCENT,
    SURFACE_SPACING,
    SURFACE_PLOT_SIZE,
    SURFACE_PLOT_ZOOM,
    PLOT_SURFACE_DPI,
    PLOT_VOL_DPI,
    PLOT_CBAR_DPI,
    CBAR_SPACING,
    CBAR_GRADIENT_WIDTH_RATIO,
    CBAR_TARGET_WIDTH_RATIO,
)
from ..utils.system import check_dependency

# Try to import surfplot
SURFPLOT_AVAILABLE = False
try:
    from surfplot import Plot

    SURFPLOT_AVAILABLE = True
except (ImportError, ValueError, Exception):
    SURFPLOT_AVAILABLE = False


# Outline colour for the processing-FOV box on the full-FOV conform snapshot.
# Not white and not CSS "lavender" (#E6E6FA, ~95% lightness): against a gray colormap
# both read as bright tissue. This is saturated enough to be unmistakably an annotation,
# and far enough in hue from the green/yellow "summer" template contours.
CONFORM_FOV_BOX_COLOR = "#B39DDB"


# %%
def create_conform_qc(
    conformed_file: str,
    template_file: str,
    save_f: Union[str, Path],
    modality: str = "anat",
    num_slices: int = 6,
    logger: Optional[logging.Logger] = None,
    full_fov_file: Optional[Union[str, Path]] = None,
    full_fov_template_file: Optional[Union[str, Path]] = None,
    fov_box_file: Optional[Union[str, Path]] = None,
    full_fov_save_f: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Dict[str, str]:
    """
    Generate conform quality control overlays.

    Renders one figure by default: the conformed image with the template as contours.

    When the full-FOV conform outputs and ``full_fov_save_f`` are supplied (anatomical
    only), a *second* figure is rendered first, on the uncropped grid, with the
    processing FOV drawn as a lavender box. The two are the same conform at two zoom
    levels — the standard figure is that box, enlarged. The pair is what makes the crop
    legible: a recording chamber or head-post outside the box is otherwise invisible
    here, because the cropped image is all this figure ever showed. With none of them
    supplied only the standard figure is produced, which is what the functional path does.

    Args:
        conformed_file: Path to conformed image (underlay)
        template_file: Path to resampled template image (overlay/contours)
        save_f: Full path for output file (e.g., 'figures/sub-01_desc-conform_T1w.png')
        modality: Imaging modality ("anat" or "func")
        num_slices: Number of slices per orientation
        logger: Logger instance
        full_fov_file: Optional conformed image on the enlarged, uncropped grid
        full_fov_template_file: Optional resampled template on that same grid
        fov_box_file: Optional binary mask of the target FOV within that grid
        full_fov_save_f: Output path for the extra full-FOV figure; without it only the
            standard figure is rendered even when the three inputs above are present

    Returns:
        Dictionary with snapshot file paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    output_path = Path(save_f)
    logger.info(f"QC: creating {modality} conform overlay")

    try:
        # Validate inputs
        for file_path, name in [
            (conformed_file, "conformed"),
            (template_file, "template"),
        ]:
            if not os.path.exists(file_path):
                logger.error(
                    f"QC: {name} file not found - {os.path.basename(file_path)}"
                )
                return {}

        # The full-FOV figure needs all three inputs plus its own output path; they
        # stand or fall together rather than mixing grids.
        full_fov_inputs = [full_fov_file, full_fov_template_file, fov_box_file]
        use_full_fov = (
            all(f is not None and os.path.exists(f) for f in full_fov_inputs)
            and full_fov_save_f is not None
        )
        if any(f is not None for f in full_fov_inputs) and not use_full_fov:
            logger.warning(
                "QC: incomplete full-FOV inputs for the conform overlay; "
                "rendering only the processing-FOV figure"
            )

        # No-expansion and fallback both copy the conformed image under the full-FOV
        # name, so the two grids are identical. The FOV box then covers the whole grid,
        # no outline can be drawn, and figure 1 would be a pixel-for-pixel duplicate of
        # figure 2 captioned "lavender box marks ...". Header read only, no voxel data.
        if use_full_fov:
            standard_shape = nib.load(str(conformed_file)).shape[:3]
            if nib.load(str(full_fov_file)).shape[:3] == standard_shape:
                logger.info(
                    "QC: the full-FOV grid matches the processing FOV (no expansion was "
                    "needed, or sizing fell back); rendering only the processing-FOV figure"
                )
                use_full_fov = False

        def _render(underlay, overlay, outline, dest):
            """Conformed image as underlay, template as contours, optional FOV box."""
            fig = create_grid_mri_image(
                underlay_data=underlay,
                overlay_data=overlay,
                num_cols=num_slices,
                perspectives=["axial"],
                title="",
                alpha=0.7,
                underlay_cmap="gray",
                overlay_cmap="summer",
                num_contour_levels=3,
                show_title=False,
                outline_data=outline,
                outline_color=CONFORM_FOV_BOX_COLOR,
                outline_linewidth=1.2,
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(dest, dpi=PLOT_VOL_DPI, bbox_inches="tight", facecolor="black")
            plt.close(fig)

        outputs = {}

        # Figure 1 (when available): the uncropped conform, with the processing FOV boxed.
        # Guarded on its own: the enlarged grid is the memory-hungry one (get_fdata plus
        # two full-volume percentile copies), and figure 2 is the figure the report and
        # the recorded qc_files entry actually depend on, so it must not share the blast
        # radius of an optional diagnostic.
        if use_full_fov:
            full_fov_path = Path(full_fov_save_f)
            try:
                _render(
                    str(full_fov_file),
                    str(full_fov_template_file),
                    str(fov_box_file),
                    full_fov_path,
                )
                outputs[f"{modality}_conform_fullfov_overlay"] = str(full_fov_path)
                logger.info(
                    f"QC: conform full-FOV overlay saved - {os.path.basename(full_fov_path)}"
                )
            except Exception as e:
                logger.warning(
                    f"QC: full-FOV conform overlay failed ({e}); "
                    f"rendering only the processing-FOV figure"
                )

        # Figure 2: the processing FOV itself — the boxed region of figure 1, enlarged.
        _render(conformed_file, template_file, None, output_path)
        outputs[f"{modality}_conform_overlay"] = str(output_path)

        logger.info(f"QC: conform overlay saved - {os.path.basename(output_path)}")
        return outputs

    except Exception as e:
        logger.error(f"QC: failed to generate conform overlay - {e}")
        return {}


def create_motion_correction_qc(
    motion_params: str,
    save_f: Union[str, Path],
    logger: Optional[logging.Logger] = None,
    confounds_file: Optional[str] = None,
    **kwargs,
) -> Dict[str, str]:
    """
    Generate motion correction quality control plots.

    Args:
        motion_params: Path to motion parameters file (.tsv or .par format)
        save_f: Full path for output file (e.g., 'figures/sub-01_desc-motion_bold.png')
        input_file: Path to input file for BIDS-compliant naming (used for fallback naming)
        logger: Logger instance
        confounds_file: Optional ``*_desc-confounds_timeseries.tsv``; when given, its
            non-steady-state and motion-outlier columns are shaded as frame bands, matching the
            confounds QC figure stacked below this one in the report
        **kwargs: Additional arguments

    Returns:
        Dictionary with snapshot file paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    output_path = Path(save_f)
    logger.info(f"QC: creating motion correction overlay - {output_path}")

    try:
        if not os.path.exists(motion_params):
            logger.error(f"Data: motion parameters file not found - {motion_params}")
            return {}

        # Load and analyze motion parameters
        # Handle both old .par format (no headers) and new .tsv format (with headers)
        if motion_params.endswith(".tsv"):
            motion_df = pd.read_csv(motion_params, sep="\t")
            # Ensure we have the expected columns in the right order
            expected_cols = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
            if all(col in motion_df.columns for col in expected_cols):
                motion_data = motion_df[expected_cols].values
            else:
                logger.warning(
                    f"Data: expected columns {expected_cols} not found in {motion_params} - using all columns"
                )
                motion_data = motion_df.values
        else:
            # Legacy .par format
            motion_data = np.loadtxt(motion_params)

        if motion_data.ndim == 1:
            motion_data = motion_data.reshape(1, -1)

        # Skip QC for pass-through (all-zero motion params, e.g. when MC skipped for <15 vols)
        if np.allclose(motion_data, 0.0):
            logger.info("QC: skipping motion plot - all-zero params (pass-through)")
            return {}

        # Flagged-frame shading is a nicety: a missing or malformed confounds TSV must cost the
        # bands, never the figure.
        nonsteady_frames = None
        motion_outlier_frames = None
        if confounds_file:
            try:
                confounds_df = pd.read_csv(str(confounds_file), sep="\t")
                if len(confounds_df) != motion_data.shape[0]:
                    logger.warning(
                        f"QC: skipping motion frame shading - confounds has {len(confounds_df)} "
                        f"frames but motion params have {motion_data.shape[0]}"
                    )
                else:
                    nonsteady_frames = outlier_frame_indices(
                        confounds_df, NONSTEADY_OUTLIER_PREFIX
                    )
                    motion_outlier_frames = outlier_frame_indices(
                        confounds_df, MOTION_OUTLIER_PREFIX
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"QC: skipping motion frame shading - {e}")

        # Create motion plot (6 rigid-body params only; no Euclidean-norm twin axis —
        # framewise displacement now lives in the separate confounds snapshot).
        fig = create_motion_plot(
            motion_data,
            title="",
            nonsteady_frames=nonsteady_frames,
            motion_outlier_frames=motion_outlier_frames,
        )

        # Ensure the parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_timeseries_qc_figure(fig, output_path, dpi=PLOT_VOL_DPI)
        plt.close(fig)

        logger.info(f"Output: motion QC plot saved - {output_path}")
        return {"motion_plot": str(output_path)}

    except Exception as e:
        logger.error(f"QC: failed to generate motion overlay - {e}")
        return {}


def create_skullstripping_qc(
    underlay_file: str,
    mask_file: str,
    save_f: Union[str, Path],
    modality: str = "anat",
    num_slices: int = 6,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Dict[str, str]:
    """
    Generate skullstripping quality control overlays.

    Args:
        underlay_file: Path to underlay image (e.g., T1w brain image)
        mask_file: Path to brain mask file
        save_f: Full path for output file (e.g., 'figures/sub-01_desc-skullstrip_T1w.png')
        modality: Imaging modality ("anat" or "func")
        num_slices: Number of slices per orientation
        logger: Logger instance

    Returns:
        Dictionary with snapshot file paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    output_path = Path(save_f)
    logger.info(f"QC: creating {modality} skullstripping overlay")

    try:
        # Validate inputs
        for file_path, name in [(underlay_file, "underlay"), (mask_file, "mask")]:
            if not os.path.exists(file_path):
                logger.error(
                    f"QC: {name} file not found - {os.path.basename(file_path)}"
                )
                return {}

        # Create brain extraction overlay (underlay as background, mask contours)
        # Pass file paths directly - let visualization function handle loading
        fig = create_overlay_grid_3xN(
            underlay_file,
            mask_file,
            num_cols=num_slices,
            title="",
            alpha=0.7,
            underlay_cmap="gray",
            overlay_cmap="Reds",
            num_contour_levels=1,
            show_title=False,
        )

        # Ensure the parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path, dpi=PLOT_VOL_DPI, bbox_inches="tight", facecolor="black"
        )
        plt.close(fig)

        logger.info(
            f"QC: skullstripping overlay saved - {os.path.basename(output_path)}"
        )
        return {f"{modality}_skullstripping_overlay": str(output_path)}

    except Exception as e:
        logger.error(f"QC: skullstripping overlay generation failed - {e}")
        return {}


def create_registration_qc(
    image_file: str,
    template_file: str,
    save_f: Union[str, Path],
    modality: str,
    num_slices: int = 6,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Dict[str, str]:
    """
    Generate registration quality control overlays.

    Args:
        image_file: Path to registered image (underlay)
        template_file: Path to template image (overlay/contours)
        save_f: Full path for output file (e.g., 'figures/sub-01_desc-func2anat_bold.png')
        modality: Registration type ("anat2template", "func2template", "func2anat")
        num_slices: Number of slices per orientation
        logger: Logger instance

    Returns:
        Dictionary with snapshot file paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    output_path = Path(save_f)
    logger.info(f"QC: creating {modality} registration overlay")

    try:
        # Validate inputs
        for file_path, name in [(image_file, "image"), (template_file, "template")]:
            if not os.path.exists(file_path):
                logger.error(
                    f"QC: {name} file not found - {os.path.basename(file_path)}"
                )
                return {}

        # Create registration overlay (image as underlay, template as contours)
        # Pass file paths directly - let visualization function handle loading and value scaling
        fig = create_overlay_grid_3xN(
            image_file,
            template_file,
            num_cols=num_slices,
            title="",
            alpha=0.7,
            underlay_cmap="gray",
            overlay_cmap="summer",
            num_contour_levels=6,
            show_title=False,
        )

        # Ensure the parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path, dpi=PLOT_VOL_DPI, bbox_inches="tight", facecolor="black"
        )
        plt.close(fig)

        logger.info(f"QC: registration overlay saved - {os.path.basename(output_path)}")
        return {f"{modality}_registration_overlay": str(output_path)}

    except Exception as e:
        logger.error(f"QC: failed to generate registration overlay - {e}")
        return {}


def create_bias_correction_qc(
    image_original: str,
    image_corrected: str,
    save_f: Union[str, Path],
    modality: str = "anat",
    num_slices: int = 6,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Dict[str, str]:
    """
    Generate bias correction quality control overlays.

    Args:
        image_original: Path to original image (before bias correction)
        image_corrected: Path to corrected image (after bias correction)
        save_f: Full path for output file (e.g., 'figures/sub-01_desc-biascorrect_T1w.png')
        modality: Imaging modality ("anat" or "func")
        num_slices: Number of slices per orientation
        logger: Logger instance

    Returns:
        Dictionary with snapshot file paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    output_path = Path(save_f)
    logger.info(f"QC: creating {modality} bias correction comparison")

    try:
        # Validate inputs
        for file_path, name in [
            (image_original, "original"),
            (image_corrected, "corrected"),
        ]:
            if not os.path.exists(file_path):
                logger.error(
                    f"QC: {name} file not found - {os.path.basename(file_path)}"
                )
                return {}

        # Create before/after comparison with two 3xN grids stacked vertically
        # Pass file paths directly - let visualization function handle loading and voxel sizes

        # Ensure the parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        _create_before_after_comparison(
            image_original,
            image_corrected,
            num_cols=num_slices,
            perspectives=["axial"],
            save_f=output_path,
            logger=logger,
        )

        logger.info(
            f"QC: bias correction comparison saved - {os.path.basename(output_path)}"
        )
        return {f"{modality}_bias_correction_comparison": str(output_path)}

    except Exception as e:
        logger.error(f"QC: bias correction comparison generation failed - {e}")
        return {}


def create_t1wt2w_combined_qc(
    image_before: str,
    image_combined: str,
    save_f: Union[str, Path],
    modality: str = "anat",
    num_slices: int = 6,
    mask_file: Optional[Union[str, Path]] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Dict[str, str]:
    """
    Generate T1wT2wCombined quality control overlays.

    Shows before/after comparison: T1w after bias correction vs T1wT2wCombined image.
    Optionally applies a brain mask to both images before visualization.

    Args:
        image_before: Path to T1w image after bias correction (before)
        image_combined: Path to T1wT2wCombined image (after)
        save_f: Full path for output file (e.g., 'figures/sub-01_desc-T1wT2wCombined_T1w.png')
        modality: Imaging modality ("anat" or "func")
        num_slices: Number of slices per orientation
        mask_file: Optional path to brain mask file (if provided, mask will be applied to both images)
        logger: Logger instance

    Returns:
        Dictionary with snapshot file paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    output_path = Path(save_f)
    logger.info(f"QC: creating {modality} T1wT2wCombined comparison")

    try:
        # Validate inputs
        for file_path, name in [
            (image_before, "T1w before"),
            (image_combined, "T1wT2wCombined"),
        ]:
            if not os.path.exists(file_path):
                logger.error(
                    f"QC: {name} file not found - {os.path.basename(file_path)}"
                )
                return {}

        # Apply mask if provided
        if mask_file is not None:
            if not os.path.exists(str(mask_file)):
                logger.warning(
                    f"QC: mask file not found - {os.path.basename(mask_file)}, proceeding without mask"
                )
            else:
                logger.info(
                    f"QC: applying mask to both images - {os.path.basename(mask_file)}"
                )
                mask_img = nib.load(str(mask_file))
                mask_data = mask_img.get_fdata().astype(bool)

                # Load images
                before_img = nib.load(str(image_before))
                before_data = before_img.get_fdata()
                if before_data.ndim == 4:
                    before_data = np.nanmean(before_data, axis=-1)

                combined_img = nib.load(str(image_combined))
                combined_data = combined_img.get_fdata()
                if combined_data.ndim == 4:
                    combined_data = np.nanmean(combined_data, axis=-1)

                # Ensure mask and images have same shape
                if mask_data.shape != before_data.shape:
                    logger.warning(
                        f"QC: mask shape {mask_data.shape} doesn't match image shape {before_data.shape}, skipping mask application"
                    )
                else:
                    # Apply mask to both images (set voxels outside mask to 0)
                    before_data[~mask_data] = 0
                    combined_data[~mask_data] = -1
                    # save to temporary nifti files
                    before_img = nib.Nifti1Image(
                        before_data, before_img.affine, before_img.header
                    )
                    combined_img = nib.Nifti1Image(
                        combined_data, combined_img.affine, combined_img.header
                    )
                    image_before_masked = output_path.parent / "before_masked.nii.gz"
                    image_combined_masked = (
                        output_path.parent / "combined_masked.nii.gz"
                    )
                    before_img.to_filename(str(image_before_masked))
                    combined_img.to_filename(str(image_combined_masked))
                    # update image_before and image_combined to the masked images
                    image_before = image_before_masked
                    image_combined = image_combined_masked

        # Ensure the parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create before/after comparison with two 3xN grids stacked vertically
        # Pass numpy arrays directly (already masked if mask was provided)
        _create_before_after_comparison(
            image_before,
            image_combined,
            num_cols=num_slices,
            perspectives=["axial"],
            before_after_labels=["T1w", "T1wT2wCombined"],
            save_f=output_path,
            logger=logger,
        )

        logger.info(
            f"QC: T1wT2wCombined comparison saved - {os.path.basename(output_path)}"
        )
        return {f"{modality}_t1wt2w_combined_comparison": str(output_path)}

    except Exception as e:
        logger.error(f"QC: T1wT2wCombined comparison generation failed - {e}")
        return {}


def create_atlas_segmentation_qc(
    underlay_file: str,
    atlas_file: str,
    save_f: Union[str, Path],
    modality: str = "anat",
    num_slices: int = 6,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Dict[str, str]:
    """
    Generate atlas segmentation quality control overlays with multi-label support.

    Args:
        underlay_file: Path to underlay image (e.g., T1w brain image with skull)
        atlas_file: Path to atlas segmentation file (multi-label)
        save_f: Full path for output file (e.g., 'figures/sub-01_desc-atlasSegmentation_T1w.png')
        modality: Imaging modality ("anat" or "func")
        num_slices: Number of slices per orientation
        logger: Logger instance

    Returns:
        Dictionary with snapshot file paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    output_path = Path(save_f)
    logger.info(f"QC: creating {modality} atlas segmentation overlay")

    try:
        # Validate inputs
        for file_path, name in [(underlay_file, "underlay"), (atlas_file, "atlas")]:
            if not os.path.exists(file_path):
                logger.error(
                    f"QC: {name} file not found - {os.path.basename(file_path)}"
                )
                return {}

        # Default colors for multi-label segmentation
        overlay_colors = [
            "limegreen",
            "red",
            "blue",
            "yellow",
            "magenta",
            "cyan",
            "orange",
            "pink",
            "purple",
            "brown",
            "gray",
            "olive",
            "navy",
            "teal",
            "coral",
            "gold",
        ]

        # Create atlas segmentation overlay with discrete multi-label contours
        # Use axial slices only for better visualization of multi-label segmentation
        fig = create_grid_mri_image(
            underlay_data=underlay_file,
            overlay_data=atlas_file,
            num_cols=num_slices,
            perspectives=["axial"],  # Only axial slices
            contour_type="discrete",
            overlay_colors=overlay_colors,
            show_legend=False,
            alpha=0.7,
            figsize_per_col=(5, 5),  # Larger subplots for better visibility
            col_margin=1,  # Extract extra slices on each side but only display middle num_slices
        )

        # Ensure the parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path, dpi=PLOT_VOL_DPI, bbox_inches="tight", facecolor="black"
        )
        plt.close(fig)

        logger.info(
            f"QC: atlas segmentation overlay saved - {os.path.basename(output_path)}"
        )
        return {f"{modality}_atlas_segmentation_overlay": str(output_path)}

    except Exception as e:
        logger.error(f"QC: atlas segmentation overlay generation failed - {e}")
        return {}


def _tsnr_add_right_colorbar(
    fig,
    vmin: float,
    vmax: float,
    cmap_name: str,
    *,
    map_right: float = 0.8,
    bar_height_ratio: float = 0.5,
    bar_width_to_height: float = 0.06,
    gap_ratio: float = 0.04,
    text_color: str = "white",
) -> None:
    """Right-side colorbar for tSNR matplotlib figures."""
    from matplotlib import cm, colors

    fig.subplots_adjust(right=map_right)
    fig_width_in, fig_height_in = fig.get_size_inches()

    bar_height = bar_height_ratio
    bar_width = (
        bar_height * bar_width_to_height * (fig_height_in / fig_width_in)
        if fig_width_in > 0
        else 0.015
    )
    # Keep the colorbar fully inside the canvas so save-time cropping cannot clip it.
    right_padding = 0.04
    cbar_x = min(map_right + gap_ratio, 1.0 - right_padding - bar_width)
    cbar_x = max(cbar_x, 0.0)
    cbar_y = (1.0 - bar_height) / 2.0

    cax = fig.add_axes([cbar_x, cbar_y, bar_width, bar_height])
    cax.set_in_layout(True)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap_name)
    scalar_map.set_array([])
    colorbar = fig.colorbar(scalar_map, cax=cax, label="tSNR")
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.ax.yaxis.tick_left()
    colorbar.ax.yaxis.label.set_color(text_color)
    colorbar.ax.tick_params(colors=text_color)
    for spine in colorbar.ax.spines.values():
        spine.set_edgecolor(text_color)


def _compute_display_range(
    values: np.ndarray,
    *,
    low_pct: float,
    high_pct: float,
    fallback_min: float = 0.0,
    fallback_max: float = 1.0,
) -> tuple[float, float]:
    """Compute a robust percentile range with safe fallback and non-zero width."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return fallback_min, fallback_max

    vmin, vmax = np.percentile(finite_values, [low_pct, high_pct])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def _is_valid_surface_map(path: Optional[Path]) -> bool:
    """Check whether a candidate surface metric file looks usable."""
    return (
        path is not None
        and path.exists()
        and ".dummy" not in str(path).lower()
        and path.stat().st_size > 0
    )


def _save_tsnr_volume_panel(
    session_tsnr_vol: Path,
    output_dir: Path,
    cmap_name: str,
) -> Path:
    """Render and save the axial tSNR volume panel to a temp PNG."""
    volume_voxels = np.asarray(
        nib.load(str(session_tsnr_vol)).get_fdata(), dtype=np.float64
    )
    volume_vmin = 0.0
    _, volume_vmax = _compute_display_range(volume_voxels, low_pct=0, high_pct=98)

    figure_volume = create_grid_mri_image(
        underlay_data=session_tsnr_vol,
        overlay_data=None,
        num_cols=6,
        perspectives=["axial"],
        title="",
        alpha=0.7,
        underlay_cmap=cmap_name,
        show_title=False,
        underlay_vmin=volume_vmin,
        underlay_vmax=volume_vmax,
    )
    _tsnr_add_right_colorbar(
        figure_volume, volume_vmin, volume_vmax, cmap_name, text_color="white"
    )

    volume_png_fd, volume_png_path_str = tempfile.mkstemp(
        suffix="_tsnr_vol.png", dir=str(output_dir)
    )
    os.close(volume_png_fd)
    volume_png_path = Path(volume_png_path_str)
    figure_volume.savefig(
        volume_png_path,
        dpi=PLOT_VOL_DPI,
        bbox_inches="tight",
        pad_inches=0.1,
        facecolor="black",
    )
    plt.close(figure_volume)
    return volume_png_path


def _try_save_tsnr_surface_panel(
    output_dir: Path,
    lh_surf_gii: Optional[Path],
    rh_surf_gii: Optional[Path],
    fs_subject_dir: Optional[Path],
    cmap_name: str,
    logger: logging.Logger,
) -> Optional[Path]:
    """Render optional surface tSNR panel and return temp PNG path when available."""
    if not (_is_valid_surface_map(lh_surf_gii) and _is_valid_surface_map(rh_surf_gii)):
        return None
    if fs_subject_dir is None or not fs_subject_dir.is_dir():
        return None

    left_inflated_mesh = fs_subject_dir / "surf" / "lh.inflated"
    right_inflated_mesh = fs_subject_dir / "surf" / "rh.inflated"
    if not (left_inflated_mesh.is_file() and right_inflated_mesh.is_file()):
        return None

    if not SURFPLOT_AVAILABLE:
        logger.warning("QC: tSNR surf panel skipped (surfplot unavailable)")
        return None

    surface_png_path: Optional[Path] = None
    try:
        left_vertex_tsnr = np.asarray(
            nib.load(str(lh_surf_gii)).darrays[0].data, dtype=np.float32
        )
        right_vertex_tsnr = np.asarray(
            nib.load(str(rh_surf_gii)).darrays[0].data, dtype=np.float32
        )
        combined_vertices = np.concatenate([left_vertex_tsnr, right_vertex_tsnr])
        surf_vmin = 0.0
        _, surf_vmax = _compute_display_range(combined_vertices, low_pct=0, high_pct=98)

        surf_plot = Plot(
            surf_lh=str(left_inflated_mesh),
            surf_rh=str(right_inflated_mesh),
            views=["lateral", "medial"],
            layout="row",
            size=(1600, 180),
            zoom=2,
        )
        surf_plot.add_layer(
            {
                "left": np.clip(left_vertex_tsnr, surf_vmin, surf_vmax),
                "right": np.clip(right_vertex_tsnr, surf_vmin, surf_vmax),
            },
            cmap=cmap_name,
            cbar=False,
        )

        figure_surface = surf_plot.build()
        figure_surface.patch.set_facecolor("black")
        for axis in figure_surface.axes:
            axis.set_facecolor("black")
        _tsnr_add_right_colorbar(
            figure_surface, surf_vmin, surf_vmax, cmap_name, text_color="white"
        )

        surface_png_fd, surface_png_path_str = tempfile.mkstemp(
            suffix="_tsnr_surf.png", dir=str(output_dir)
        )
        os.close(surface_png_fd)
        surface_png_path = Path(surface_png_path_str)
        figure_surface.savefig(
            surface_png_path,
            dpi=PLOT_VOL_DPI,
            bbox_inches="tight",
            pad_inches=0.1,
            facecolor="black",
        )
        plt.close(figure_surface)
        return surface_png_path
    except Exception as exc:
        logger.warning(f"QC: tSNR surface rendering failed: {exc}")
        if surface_png_path and surface_png_path.exists():
            surface_png_path.unlink(missing_ok=True)
        return None


def _stack_panels_to_output(
    panel_paths: List[Path],
    output_path: Path,
    hspace_ratio: float = 0.05,
) -> bool:
    """Vertically stack panels with configurable gap and black background."""
    if not panel_paths:
        return False

    panel_images = [Image.open(path).convert("RGB") for path in panel_paths]
    max_width = max(image.width for image in panel_images)
    resized_panels: List[Image.Image] = []
    for image in panel_images:
        if image.width != max_width:
            new_height = int(image.height * (max_width / image.width))
            image = image.resize(
                (max_width, new_height), resample=Image.Resampling.LANCZOS
            )
        resized_panels.append(image)

    if not resized_panels:
        return False

    avg_panel_height = sum(image.height for image in resized_panels) / len(
        resized_panels
    )
    gap_px = max(0, int(round(avg_panel_height * hspace_ratio)))
    total_height = sum(image.height for image in resized_panels) + gap_px * (
        len(resized_panels) - 1
    )
    stacked = Image.new("RGB", (max_width, total_height), color="black")
    paste_y = 0
    for image in resized_panels:
        paste_x = (max_width - image.width) // 2
        stacked.paste(image, (paste_x, paste_y))
        paste_y += image.height + gap_px

    stacked.save(output_path)
    return True


def create_tsnr_qc(
    session_tsnr_vol: Union[str, Path],
    save_f: Union[str, Path],
    lh_surf_gii: Optional[Union[str, Path]] = None,
    rh_surf_gii: Optional[Union[str, Path]] = None,
    fs_subject_dir: Optional[Union[str, Path]] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Dict[str, str]:
    """
    Generate combined tSNR QC snapshot with a volume panel and optional surface panel.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    output_path = Path(save_f)
    session_tsnr_vol = Path(session_tsnr_vol)
    lh_surf_gii = Path(lh_surf_gii) if lh_surf_gii is not None else None
    rh_surf_gii = Path(rh_surf_gii) if rh_surf_gii is not None else None
    fs_subject_dir = Path(fs_subject_dir) if fs_subject_dir is not None else None

    try:
        if not session_tsnr_vol.exists():
            logger.error(f"QC: session tSNR volume not found - {session_tsnr_vol}")
            return {}

        output_path.parent.mkdir(parents=True, exist_ok=True)
        colormap_name = "magma"
        volume_png_path = _save_tsnr_volume_panel(
            session_tsnr_vol=session_tsnr_vol,
            output_dir=output_path.parent,
            cmap_name=colormap_name,
        )
        surface_png_path = _try_save_tsnr_surface_panel(
            output_dir=output_path.parent,
            lh_surf_gii=lh_surf_gii,
            rh_surf_gii=rh_surf_gii,
            fs_subject_dir=fs_subject_dir,
            cmap_name=colormap_name,
            logger=logger,
        )

        panel_paths = [
            panel_path
            for panel_path in (volume_png_path, surface_png_path)
            if panel_path is not None and panel_path.exists()
        ]
        if not panel_paths:
            volume_png_path.unlink(missing_ok=True)
            return {}

        if not _stack_panels_to_output(
            panel_paths=panel_paths, output_path=output_path
        ):
            return {}

        for panel_path in panel_paths:
            panel_path.unlink(missing_ok=True)

        logger.info(f"QC: tSNR QC saved - {output_path.name}")
        return {"tsnr_qc": str(output_path)}
    except ImportError as exc:
        if "PIL" in str(exc):
            logger.warning(f"QC: Pillow missing, skipping tSNR QC - {exc}")
            return {}
        logger.error(f"QC: tSNR QC dependency error - {exc}")
        return {}
    except Exception as e:
        logger.error(f"QC: tSNR QC failed - {e}")
        return {}


def _create_before_after_comparison(
    before_data: Union[str, Path, np.ndarray],
    after_data: Union[str, Path, np.ndarray],
    num_cols: int = 6,
    col_margin: int = 1,
    save_f: Union[str, Path] = None,
    perspectives: Optional[List[str]] = ["axial"],
    before_after_labels: Optional[List[str]] = ["Before", "After"],
    logger: Optional[logging.Logger] = None,
) -> plt.Figure:
    """
    Create a before/after comparison of two images using the generic grid function.

    Args:
        before_data: Path to before image or numpy array
        after_data: Path to after image or numpy array
        num_cols: Number of columns (slices per orientation)
        perspectives: List of orientations to show

    Returns:
        Matplotlib figure with before/after comparison
    """
    # Common parameters for both figures
    grid_params = {
        "num_cols": num_cols,
        "col_margin": col_margin,
        "perspectives": perspectives,
        "figsize_per_col": (3, 3),
        "show_title": False,
        "overlay_data": None,
        "underlay_cmap": "gray",
    }

    # Create before and after figures
    fig_before = create_grid_mri_image(underlay_data=before_data, **grid_params)
    fig_after = create_grid_mri_image(underlay_data=after_data, **grid_params)

    # Add labels to figures and save them as temporary files
    temp_files = []
    saved_images = []

    for fig, label in zip([fig_before, fig_after], before_after_labels):
        # Add label to figure
        if fig.axes:
            ax = fig.axes[0]
            ax.text(
                -0.08,
                0.5,
                label,
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                color="white",
                rotation=90,
                ha="center",
                va="center",
            )

        # Save figure as temporary file
        temp_file = str(save_f).replace(".png", f"_{label.lower()}.png")
        fig.savefig(temp_file, dpi=PLOT_VOL_DPI, bbox_inches="tight", facecolor="black")
        temp_files.append(temp_file)

        if logger:
            logger.info(
                f"Output: bias correction QC {label.lower()} saved - {temp_file}"
            )

        # Load the saved image
        saved_images.append(Image.open(temp_file))

    # Create a new image with combined height
    img_before, img_after = saved_images
    total_width = max(img_before.width, img_after.width)
    total_height = img_before.height + img_after.height

    combined_img = Image.new("RGB", (total_width, total_height), color="black")
    combined_img.paste(img_before, (0, 0))
    combined_img.paste(img_after, (0, img_before.height))

    # Save the combined image
    combined_img.save(str(save_f), dpi=(PLOT_VOL_DPI, PLOT_VOL_DPI))

    if logger:
        logger.info(f"Output: bias correction QC comparison saved - {save_f}")

    # Clean up temporary files
    for temp_file in temp_files:
        os.remove(temp_file)

    # Close figures
    for fig in [fig_before, fig_after]:
        plt.close(fig)

    return None  # Function doesn't need to return a figure anymore


def _get_scene_file_path() -> Path:
    """Get path to Vol_Surface.scene resource file.

    Returns:
        Path to the scene file

    Raises:
        FileNotFoundError: If scene file is not found
    """
    # Get directory of this file (snapshots.py)
    module_dir = Path(__file__).parent
    scene_file = module_dir / "resources" / "Vol_Surface.scene"

    if not scene_file.exists():
        raise FileNotFoundError(
            f"Scene file not found: {scene_file}\n"
            "This file should be packaged with the installation. "
            "Please ensure nhp_mri_prep is properly installed and the scene file "
            "is present in quality_control/resources/Vol_Surface.scene"
        )
    return scene_file


def create_surf_recon_tissue_seg_qc(
    fs_subject_dir: Union[str, Path],
    save_f: Union[str, Path],
    modality: str = "anat",
    num_slices: int = 6,  # Not used in Workbench implementation, kept for compatibility
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Dict[str, str]:
    """
    Generate surface reconstruction tissue segmentation quality control visualization.

    Uses Connectome Workbench to create a 3D visualization of FreeSurfer surfaces
    (white and pial) overlaid on the brain volume. This replaces the previous
    rasterization-based 2D slice approach with a Workbench-based 3D rendering.

    Args:
        fs_subject_dir: Path to FreeSurfer subject directory (e.g., 'fastsurfer/sub-XXX')
        save_f: Full path for output PNG file (e.g., 'figures/sub-01_desc-surfReconTissueSeg_T1w.png')
        modality: Imaging modality ("anat" or "func")
        num_slices: Number of slices per orientation (not used, kept for compatibility)
        logger: Logger instance

    Returns:
        Dictionary with snapshot file paths: {f"{modality}_surf_recon_tissue_seg_overlay": str(output_path)}

    Raises:
        RuntimeError: If required tools (wb_command, mris_convert, mri_convert) are not available
        FileNotFoundError: If required input files or scene file are not found
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    fs_subject_dir = Path(fs_subject_dir)
    output_path = Path(save_f)
    logger.info(
        f"QC: creating {modality} surface reconstruction tissue segmentation overlay (Workbench-based)"
    )

    try:
        # Step 1: Validate external tools
        logger.info("QC: checking external tool dependencies...")
        if not check_dependency("wb_command", logger):
            raise RuntimeError(
                "wb_command not found. Connectome Workbench is required."
            )
        if not check_dependency("mri_convert", logger):
            raise RuntimeError("mri_convert not found. FreeSurfer is required.")

        # Step 2: Construct file paths from FreeSurfer directory structure
        surf_dir = fs_subject_dir / "surf"
        mri_dir = fs_subject_dir / "mri"

        # Surface files (using .white and .pial, not .smoothwm)
        lh_white_f = surf_dir / "lh.white"
        rh_white_f = surf_dir / "rh.white"
        lh_pial_f = surf_dir / "lh.pial"
        rh_pial_f = surf_dir / "rh.pial"

        # T1w brain file
        brain_f = mri_dir / "norm.mgz"

        # Step 3: Validate input files exist
        logger.info("QC: validating input files...")
        for file_path, name in [
            (brain_f, "norm.mgz"),
            (lh_white_f, "lh.white"),
            (rh_white_f, "rh.white"),
            (lh_pial_f, "lh.pial"),
            (rh_pial_f, "rh.pial"),
        ]:
            if not file_path.exists():
                logger.error(f"QC: {name} file not found - {file_path}")
                return {}

        # Step 4: Get scene file path
        scene_file = _get_scene_file_path()
        logger.info(f"QC: using scene file: {scene_file}")

        # Step 5: Create working directory (Nextflow work dir)
        # Use output_path.parent as the base (Nextflow manages this)
        work_dir = output_path.parent / "volsurf_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"QC: working directory: {work_dir}")

        # Step 6: Prepare GIFTI surfaces for scene rendering.
        # Prefer pre-generated surfaces from surf_dir; fallback to on-the-fly conversion.
        logger.info("QC: preparing GIFTI surfaces...")
        lh_white_gii = work_dir / "lh.white.surf.gii"
        rh_white_gii = work_dir / "rh.white.surf.gii"
        lh_pial_gii = work_dir / "lh.pial.surf.gii"
        rh_pial_gii = work_dir / "rh.pial.surf.gii"

        precomputed_giis = [
            (surf_dir / "lh.white.surf.gii", lh_white_f, lh_white_gii),
            (surf_dir / "rh.white.surf.gii", rh_white_f, rh_white_gii),
            (surf_dir / "lh.pial.surf.gii", lh_pial_f, lh_pial_gii),
            (surf_dir / "rh.pial.surf.gii", rh_pial_f, rh_pial_gii),
        ]

        if all(pre_gii.exists() for pre_gii, _, _ in precomputed_giis):
            logger.info("QC: using pre-generated GIFTI surfaces from surf directory")
            for pre_gii, _, out_gii in precomputed_giis:
                shutil.copy2(pre_gii, out_gii)
        else:
            logger.info(
                "QC: pre-generated GIFTI surfaces missing, using fallback conversion"
            )
            if not check_dependency("mris_convert", logger):
                raise RuntimeError(
                    "mris_convert not found. FreeSurfer is required for fallback GIFTI conversion."
                )

            for _, in_surf, out_gii in precomputed_giis:
                convert_fs_surface_to_gifti(in_surf, out_gii, apply_cras=True)

        # Step 7: Convert volume to NIfTI
        logger.info("QC: converting volume to NIfTI format...")
        brain_nii = work_dir / "norm.nii.gz"
        subprocess.run(
            ["mri_convert", str(brain_f), str(brain_nii)],
            check=True,
            capture_output=True,
            text=True,
        )

        # Step 8: Copy scene file to working directory
        logger.info("QC: copying scene file to working directory...")
        work_scene_file = work_dir / "Vol_Surface.scene"
        shutil.copy2(scene_file, work_scene_file)
        # Keep scene volume reference aligned with the converted file name.
        scene_text = work_scene_file.read_text()
        work_scene_file.write_text(
            scene_text.replace("brain.finalsurfs.nii.gz", "norm.nii.gz")
        )

        # Step 9: Render scene using Connectome Workbench
        logger.info("QC: rendering scene with Connectome Workbench...")
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            [
                "wb_command",
                "-show-scene",
                str(work_scene_file),
                "1",
                str(output_path),
                "2400",
                "1000",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        logger.info(
            f"QC: surface reconstruction tissue segmentation overlay saved - {os.path.basename(output_path)}"
        )
        return {f"{modality}_surf_recon_tissue_seg_overlay": str(output_path)}

    except FileNotFoundError as e:
        logger.error(f"QC: Required file not found - {e}")
        return {}
    except subprocess.CalledProcessError as e:
        logger.error(
            f"QC: Command failed - {e.cmd}: {e.stderr if e.stderr else 'no error output'}"
        )
        return {}
    except RuntimeError as e:
        logger.error(f"QC: {e}")
        return {}
    except Exception as e:
        logger.error(
            f"QC: surface reconstruction tissue segmentation overlay generation failed - {e}"
        )
        import traceback

        logger.debug(traceback.format_exc())
        return {}


def create_cortical_surf_and_measures_qc(
    fs_subject_dir: Union[str, Path],
    save_f: Union[str, Path],
    atlas_name: str = "ARM2",
    modality: str = "anat",
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Dict[str, str]:
    """
    Generate cortical surface and measures quality control plots.
    Creates surface plots showing different data on three surface types:
    - Row 1: smoothwm with curvature
    - Row 2: pial with segmentation (atlas labels)
    - Row 3: inflated with thickness

    Args:
        fs_subject_dir: Path to FreeSurfer subject directory (e.g., 'fastsurfer/sub-XXX')
        atlas_name: Name of the atlas without "atlas" suffix (default: "ARM2", will create "ARM2atlas.mapped.annot")
        save_f: Full path for output file (e.g., 'figures/sub-01_desc-corticalSurfAndMeasures_T1w.png')
        modality: Imaging modality ("anat" or "func")
        logger: Logger instance

    Returns:
        Dictionary with snapshot file paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not SURFPLOT_AVAILABLE:
        logger.warning(
            "QC: surfplot not available, skipping cortical surface and measures QC"
        )
        return {}

    fs_subject_dir = Path(fs_subject_dir)
    output_path = Path(save_f)
    logger.info(f"QC: creating {modality} cortical surface and measures plot")

    try:
        # Normalize atlas name: remove "atlas" suffix if present (for backward compatibility)
        # Then append "atlas" to match FreeSurferRecon naming convention
        normalized_atlas_name = (
            atlas_name.rstrip("atlas") if atlas_name.endswith("atlas") else atlas_name
        )

        # Construct file paths from FreeSurfer directory structure
        surf_dir = fs_subject_dir / "surf"
        label_dir = fs_subject_dir / "label"

        # Surface files
        smoothwm_surf_lh = surf_dir / "lh.smoothwm"
        smoothwm_surf_rh = surf_dir / "rh.smoothwm"
        pial_surf_lh = surf_dir / "lh.pial"
        pial_surf_rh = surf_dir / "rh.pial"
        inflated_surf_lh = surf_dir / "lh.inflated"
        inflated_surf_rh = surf_dir / "rh.inflated"

        # Data files
        curv_lh = surf_dir / "lh.curv"
        curv_rh = surf_dir / "rh.curv"
        thickness_lh = surf_dir / "lh.thickness"
        thickness_rh = surf_dir / "rh.thickness"

        # Annotation files (try mapped version first, then fallback)
        atlas_annot_lh = (
            label_dir / f"lh.aparc.{normalized_atlas_name}atlas.mapped.annot"
        )
        atlas_annot_rh = (
            label_dir / f"rh.aparc.{normalized_atlas_name}atlas.mapped.annot"
        )
        if not atlas_annot_lh.exists():
            atlas_annot_lh = label_dir / f"lh.aparc.{normalized_atlas_name}atlas.annot"
        if not atlas_annot_rh.exists():
            atlas_annot_rh = label_dir / f"rh.aparc.{normalized_atlas_name}atlas.annot"

        # Validate inputs
        required_files = [
            (smoothwm_surf_lh, "smoothwm_lh"),
            (smoothwm_surf_rh, "smoothwm_rh"),
            (pial_surf_lh, "pial_lh"),
            (pial_surf_rh, "pial_rh"),
            (inflated_surf_lh, "inflated_lh"),
            (inflated_surf_rh, "inflated_rh"),
            (atlas_annot_lh, "atlas_annot_lh"),
            (atlas_annot_rh, "atlas_annot_rh"),
            (curv_lh, "curv_lh"),
            (curv_rh, "curv_rh"),
            (thickness_lh, "thickness_lh"),
            (thickness_rh, "thickness_rh"),
        ]

        for file_path, name in required_files:
            if not file_path.exists():
                logger.error(f"QC: {name} file not found - {file_path}")
                return {}

        logger.info("QC: loading data files...")

        # Load annotations (for pial surface)
        try:
            lh_labels, _, lh_names = nib.freesurfer.read_annot(str(atlas_annot_lh))
            rh_labels, _, rh_names = nib.freesurfer.read_annot(str(atlas_annot_rh))
            logger.info(
                f"QC: loaded annotations: {len(lh_names)} labels for LH, {len(rh_names)} labels for RH"
            )
        except Exception as e:
            logger.error(f"QC: error loading annotations - {e}")
            return {}

        # Load curvature (for smoothwm surface)
        try:
            lh_curv = nib.freesurfer.read_morph_data(str(curv_lh))
            rh_curv = nib.freesurfer.read_morph_data(str(curv_rh))
            logger.info("QC: loaded curvature data")
        except Exception as e:
            logger.error(f"QC: error loading curvature - {e}")
            return {}

        # Load thickness (for inflated surface)
        try:
            lh_thickness = nib.freesurfer.read_morph_data(str(thickness_lh))
            rh_thickness = nib.freesurfer.read_morph_data(str(thickness_rh))
            logger.info("QC: loaded thickness data")
        except Exception as e:
            logger.error(f"QC: error loading thickness - {e}")
            return {}

        logger.info("QC: creating surface plots...")

        # Compute curvature distribution and symmetric range centered at 0
        all_curv = np.concatenate([lh_curv, rh_curv])
        curv_abs_max = max(
            abs(np.percentile(all_curv, 33)), abs(np.percentile(all_curv, 66))
        )
        curv_vmin = -curv_abs_max
        curv_vmax = curv_abs_max

        # Compute thickness percentiles for clipping
        all_thickness = np.concatenate([lh_thickness, rh_thickness])
        thickness_vmin = 1e-6  # minimum thickness to avoid 0 not showing in the plot
        thickness_vmax = np.percentile(all_thickness, 95)
        print(f"Thickness range: {thickness_vmin} to {thickness_vmax}")

        # Surface types with their surfaces and corresponding data
        surface_configs = [
            (
                "smoothwm",
                smoothwm_surf_lh,
                smoothwm_surf_rh,
                "curvature",
                {"left": lh_curv, "right": rh_curv},
            ),
            (
                "pial",
                pial_surf_lh,
                pial_surf_rh,
                "segmentation",
                {"left": lh_labels, "right": rh_labels},
            ),
            (
                "inflated",
                inflated_surf_lh,
                inflated_surf_rh,
                "thickness",
                {"left": lh_thickness, "right": rh_thickness},
            ),
        ]

        # Views: (hemisphere, view_name)
        views = [
            ("left", "lateral"),
            ("left", "medial"),
            ("right", "lateral"),
            ("right", "medial"),
        ]

        # Create individual plots for each surface type and view combination
        temp_dir = Path(tempfile.mkdtemp())
        temp_images = {}  # Dictionary: (surf_name, hemi, view) -> image_path

        try:
            for surf_name, surf_lh, surf_rh, data_type, data_dict in surface_configs:
                logger.info(
                    f"QC: creating {surf_name} surface plots with {data_type}..."
                )
                for hemi, view in views:
                    # Create plot for single hemisphere and view
                    plot_kwargs = {
                        "views": view,
                        "size": SURFACE_PLOT_SIZE,
                        "zoom": SURFACE_PLOT_ZOOM,
                    }
                    if hemi == "left":
                        plot_kwargs["surf_lh"] = str(surf_lh)
                        data_key = "left"
                    else:
                        plot_kwargs["surf_rh"] = str(surf_rh)
                        data_key = "right"

                    p = Plot(**plot_kwargs)

                    # Add data layer based on surface type
                    if data_type == "curvature":
                        data = np.clip(data_dict[data_key], curv_vmin, curv_vmax)
                        p.add_layer({data_key: data}, cmap="coolwarm_r", cbar=False)
                    elif data_type == "segmentation":
                        p.add_layer(
                            {data_key: data_dict[data_key]}, cmap="tab20", cbar=False
                        )
                    elif data_type == "thickness":  # thickness
                        data = np.clip(
                            data_dict[data_key], thickness_vmin, thickness_vmax
                        )
                        p.add_layer({data_key: data}, cmap="viridis", cbar=False)

                    fig = p.build()
                    temp_path = Path(temp_dir) / f"{surf_name}_{hemi}_{view}.png"
                    fig.savefig(
                        temp_path,
                        dpi=PLOT_SURFACE_DPI,
                        bbox_inches="tight",
                        pad_inches=0.05,
                        facecolor="white",
                    )
                    temp_images[(surf_name, hemi, view)] = temp_path
                    plt.close(fig)

            # Load and combine images
            logger.info("QC: combining images...")

            # Load all images and crop white space
            loaded_images = []
            for surf_name, _, _, _, _ in surface_configs:
                row_images = []
                for hemi, view in views:
                    img = Image.open(temp_images[(surf_name, hemi, view)])
                    img = _crop_white_space(img)
                    row_images.append(img)
                loaded_images.append(row_images)

            # Get dimensions - use the maximum width and height from all images
            max_width = max(img.width for row in loaded_images for img in row)
            max_height = max(img.height for row in loaded_images for img in row)

            # Create colorbars for curvature (row 0), parcellation label (row 1), and thickness (row 2)
            cbar_target_width = int(max_width * CBAR_TARGET_WIDTH_RATIO)
            cbar_height = max_height
            cbar_fig_width_inches = cbar_target_width / PLOT_CBAR_DPI
            cbar_fig_height_inches = cbar_height / PLOT_CBAR_DPI

            # Create colorbars using helper functions
            curv_cbar_img = _create_colorbar(
                cmap="coolwarm_r",
                vmin=curv_vmin,
                vmax=curv_vmax,
                label="Curvature",
                fig_width_inches=cbar_fig_width_inches,
                fig_height_inches=cbar_fig_height_inches,
                dpi=PLOT_CBAR_DPI,
                gradient_width_ratio=CBAR_GRADIENT_WIDTH_RATIO,
                temp_dir=temp_dir,
            )

            thickness_cbar_img = _create_colorbar(
                cmap="viridis",
                vmin=thickness_vmin,
                vmax=thickness_vmax,
                label="Thickness (mm)",
                fig_width_inches=cbar_fig_width_inches,
                fig_height_inches=cbar_fig_height_inches,
                dpi=PLOT_CBAR_DPI,
                gradient_width_ratio=CBAR_GRADIENT_WIDTH_RATIO,
                temp_dir=temp_dir,
            )

            parcellation_label_img = _create_label_image(
                label="Parcellation",
                fig_width_inches=cbar_fig_width_inches,
                fig_height_inches=cbar_fig_height_inches,
                dpi=PLOT_CBAR_DPI,
                gradient_width_ratio=CBAR_GRADIENT_WIDTH_RATIO,
                temp_dir=temp_dir,
            )

            base_height = max_height * 3 + SURFACE_SPACING * 2

            # Create surface area image (4 columns)
            surface_area_width = 4 * max_width + 3 * SURFACE_SPACING
            surface_img = Image.new(
                "RGB", (surface_area_width + 50, base_height), "white"
            )

            # Paste all surface images into the surface area image
            for row_idx, row_images in enumerate(loaded_images):
                for col_idx, img in enumerate(row_images):
                    img_to_paste = img.copy()

                    # Crop from center if too wide
                    if img_to_paste.width > max_width:
                        excess = img_to_paste.width - max_width
                        img_to_paste = img_to_paste.crop(
                            (
                                excess // 2,
                                0,
                                excess // 2 + max_width,
                                img_to_paste.height,
                            )
                        )

                    x_pos = (
                        col_idx * (max_width + SURFACE_SPACING)
                        + (max_width - img_to_paste.width) // 2
                    )
                    y_pos = (
                        row_idx * (max_height + SURFACE_SPACING)
                        + (max_height - img_to_paste.height) // 2
                    )
                    surface_img.paste(img_to_paste, (x_pos, y_pos))

            # Find actual right edge of surface content
            actual_surface_width = _find_content_width(surface_img)

            # Create colorbar area image (no white blank space on right)
            left_cbar_width = max(
                curv_cbar_img.width,
                parcellation_label_img.width,
                thickness_cbar_img.width,
            )
            cbar_area_width = left_cbar_width
            cbar_area_img = Image.new("RGB", (cbar_area_width, base_height), "white")

            # Paste colorbars: left side for all labels (aligned)
            max_cbar_height = max(
                curv_cbar_img.height,
                parcellation_label_img.height,
                thickness_cbar_img.height,
            )
            cbar_images = [curv_cbar_img, parcellation_label_img, thickness_cbar_img]
            for row_idx, cbar_img in enumerate(cbar_images):
                cbar_y_pos = (
                    row_idx * (max_height + SURFACE_SPACING)
                    + (max_height - max_cbar_height) // 2
                )
                cbar_area_img.paste(cbar_img, (0, cbar_y_pos))

            # Crop surface image and create final combined image
            if actual_surface_width < surface_img.width:
                surface_img = surface_img.crop(
                    (0, 0, actual_surface_width, surface_img.height)
                )

            actual_base_width = actual_surface_width + CBAR_SPACING + cbar_area_width
            actual_margin_x = int(actual_base_width * MARGIN_PERCENT)
            actual_margin_y = int(base_height * MARGIN_PERCENT)
            combined_img = Image.new(
                "RGB",
                (
                    actual_base_width + 2 * actual_margin_x,
                    base_height + 2 * actual_margin_y,
                ),
                "white",
            )

            combined_img.paste(surface_img, (actual_margin_x, actual_margin_y))
            combined_img.paste(
                cbar_area_img,
                (
                    actual_margin_x + actual_surface_width + CBAR_SPACING,
                    actual_margin_y,
                ),
            )

            # Save combined image
            output_path.parent.mkdir(parents=True, exist_ok=True)
            combined_img.save(output_path, dpi=(PLOT_VOL_DPI, PLOT_VOL_DPI))
            logger.info(
                f"QC: cortical surface and measures plot saved - {os.path.basename(output_path)}"
            )

            return {f"{modality}_cortical_surf_and_measures_overlay": str(output_path)}

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        logger.error(f"QC: cortical surface and measures plot generation failed - {e}")
        return {}
