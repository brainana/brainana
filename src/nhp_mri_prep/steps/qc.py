"""
Standalone QC step functions for Nextflow integration.

These functions generate quality control visualizations as separate steps.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .types import StepOutput
from ..quality_control import (
    create_bias_correction_qc,
    create_t1wt2w_combined_qc,
    create_skullstripping_qc,
    create_registration_qc,
    create_conform_qc,
    create_atlas_segmentation_qc,
    create_motion_correction_qc,
    create_surf_recon_tissue_seg_qc,
    create_cortical_surf_and_measures_qc,
    create_tsnr_qc,
)
from ..quality_control.snapshots import _create_before_after_comparison
from ..quality_control.reports import generate_qc_report
from ..utils.nextflow import ensure_stderr_logging_if_unconfigured

logger = logging.getLogger(__name__)
ensure_stderr_logging_if_unconfigured()


def qc_bias_correction(
    original_file: Path,
    corrected_file: Path,
    output_path: Path,
    modality: str = "anat",
    config: Optional[Dict[str, Any]] = None,
) -> StepOutput:
    """
    Generate bias correction QC snapshot.

    Args:
        original_file: Original (uncorrected) image
        corrected_file: Bias-corrected image
        output_path: Output path for QC snapshot
        modality: Modality ('anat' or 'func')
        config: Configuration dictionary (optional)

    Returns:
        StepOutput with QC file
    """
    if not config or not config.get("quality_control", {}).get("enabled", True):
        logger.info("QC: bias correction QC skipped (disabled in configuration)")
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_bias_correction", "skipped": True},
        )

    try:
        result = create_bias_correction_qc(
            image_original=str(original_file),
            image_corrected=str(corrected_file),
            save_f=str(output_path),
            modality=modality,
            logger=logger,
        )

        qc_file = Path(result.get("snapshot_file", output_path))

        return StepOutput(
            output_file=qc_file,
            metadata={"step": "qc_bias_correction", "modality": modality},
            qc_files=[qc_file],
        )
    except Exception as e:
        logger.warning(f"QC: bias correction QC failed - {e}")
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_bias_correction", "error": str(e)},
        )


def qc_skullstripping(
    underlay_file: Path,
    mask_file: Path,
    output_path: Path,
    modality: str = "anat",
    config: Optional[Dict[str, Any]] = None,
) -> StepOutput:
    """
    Generate skull stripping QC snapshot.

    Args:
        underlay_file: Original image (underlay)
        mask_file: Brain mask file
        output_path: Output path for QC snapshot
        modality: Modality ('anat' or 'func')
        config: Configuration dictionary (optional)

    Returns:
        StepOutput with QC file
    """
    if not config or not config.get("quality_control", {}).get("enabled", True):
        logger.info("QC: skull stripping QC skipped (disabled in configuration)")
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_skullstripping", "skipped": True},
        )

    try:
        result = create_skullstripping_qc(
            underlay_file=str(underlay_file),
            mask_file=str(mask_file),
            save_f=str(output_path),
            modality=modality,
            logger=logger,
        )

        qc_file = Path(result.get("snapshot_file", output_path))

        return StepOutput(
            output_file=qc_file,
            metadata={"step": "qc_skullstripping", "modality": modality},
            qc_files=[qc_file],
        )
    except Exception as e:
        logger.warning(f"QC: skull stripping QC failed - {e}")
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_skullstripping", "error": str(e)},
        )


def qc_registration(
    image_file: Path,
    template_file: Path,
    output_path: Path,
    modality: str = "anat2template",
    config: Optional[Dict[str, Any]] = None,
) -> StepOutput:
    """
    Generate registration QC snapshot.

    Args:
        image_file: Registered image
        template_file: Template/reference image
        output_path: Output path for QC snapshot
        modality: Modality string (e.g., 'anat2template', 'func2anat')
        config: Configuration dictionary (optional)

    Returns:
        StepOutput with QC file
    """
    if not config or not config.get("quality_control", {}).get("enabled", True):
        logger.info("QC: registration QC skipped (disabled in configuration)")
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_registration", "skipped": True},
        )

    try:
        result = create_registration_qc(
            image_file=str(image_file),
            template_file=str(template_file),
            save_f=str(output_path),
            modality=modality,
            logger=logger,
        )

        qc_file = Path(result.get("snapshot_file", output_path))

        return StepOutput(
            output_file=qc_file,
            metadata={"step": "qc_registration", "modality": modality},
            qc_files=[qc_file],
        )
    except Exception as e:
        logger.warning(f"QC: registration QC failed - {e}")
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_registration", "error": str(e)},
        )


def qc_conform(
    conformed_file: Path,
    template_file: Path,
    output_path: Path,
    modality: str = "anat",
    config: Optional[Dict[str, Any]] = None,
    full_fov_file: Optional[Path] = None,
    full_fov_template_file: Optional[Path] = None,
    fov_box_file: Optional[Path] = None,
    full_fov_output_path: Optional[Path] = None,
) -> StepOutput:
    """
    Generate conform QC snapshot.

    Args:
        conformed_file: Path to conformed image
        template_file: Path to template image used for conforming
        output_path: Output path for QC snapshot
        modality: Modality ('anat' or 'func')
        config: Configuration dictionary (optional)
        full_fov_file: Optional conformed image on the enlarged, uncropped grid. Given
            with the three below, an extra figure is rendered on that grid with the
            processing FOV drawn as a lavender box, so cropped-away hardware is visible.
        full_fov_template_file: Optional resampled template on that same enlarged grid
        fov_box_file: Optional binary mask of the target FOV within that grid
        full_fov_output_path: Output path for that extra figure

    Returns:
        StepOutput with QC file
    """
    if not config or not config.get("quality_control", {}).get("enabled", True):
        logger.info("QC: conform QC skipped (disabled in configuration)")
        return StepOutput(
            output_file=output_path, metadata={"step": "qc_conform", "skipped": True}
        )

    try:
        result = create_conform_qc(
            conformed_file=str(conformed_file),
            template_file=str(template_file),
            save_f=str(output_path),
            modality=modality,
            logger=logger,
            full_fov_file=full_fov_file,
            full_fov_template_file=full_fov_template_file,
            fov_box_file=fov_box_file,
            full_fov_save_f=full_fov_output_path,
        )

        qc_file = Path(result.get(f"{modality}_conform_overlay", output_path))
        qc_files = [qc_file]
        full_fov_qc = result.get(f"{modality}_conform_fullfov_overlay")
        if full_fov_qc:
            # Ordered full-FOV first: the standard figure is that box, enlarged.
            qc_files.insert(0, Path(full_fov_qc))

        return StepOutput(
            output_file=qc_file,
            metadata={"step": "qc_conform", "modality": modality},
            qc_files=qc_files,
        )
    except Exception as e:
        logger.warning(f"QC: conform QC failed - {e}")
        return StepOutput(
            output_file=output_path, metadata={"step": "qc_conform", "error": str(e)}
        )


def qc_atlas_segmentation(
    underlay_file: Path,
    atlas_file: Path,
    output_path: Path,
    modality: str = "anat",
    config: Optional[Dict[str, Any]] = None,
) -> StepOutput:
    """
    Generate atlas segmentation QC snapshot.

    Args:
        underlay_file: Path to underlay image (e.g., T1w brain image)
        atlas_file: Path to atlas segmentation file
        output_path: Output path for QC snapshot
        modality: Modality ('anat' or 'func')
        config: Configuration dictionary (optional)

    Returns:
        StepOutput with QC file
    """
    if not config or not config.get("quality_control", {}).get("enabled", True):
        logger.info("QC: atlas segmentation QC skipped (disabled in configuration)")
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_atlas_segmentation", "skipped": True},
        )

    try:
        result = create_atlas_segmentation_qc(
            underlay_file=str(underlay_file),
            atlas_file=str(atlas_file),
            save_f=str(output_path),
            modality=modality,
            logger=logger,
        )

        qc_file = Path(
            result.get(f"{modality}_atlas_segmentation_overlay", output_path)
        )

        return StepOutput(
            output_file=qc_file,
            metadata={"step": "qc_atlas_segmentation", "modality": modality},
            qc_files=[qc_file],
        )
    except Exception as e:
        logger.warning(f"QC: atlas segmentation QC failed - {e}")
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_atlas_segmentation", "error": str(e)},
        )


def qc_motion_correction(
    motion_params_file: Path,
    output_path: Path,
    input_file: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    confounds_file: Optional[Path] = None,
) -> StepOutput:
    """
    Generate motion correction QC plot.

    Args:
        motion_params_file: Motion parameters file (.tsv or .par)
        output_path: Output path for QC plot
        input_file: Optional input functional file (for metadata)
        config: Configuration dictionary (optional)
        confounds_file: Optional confounds TSV; its non-steady-state and motion-outlier columns
            are shaded as frame bands, matching the confounds QC figure

    Returns:
        StepOutput with QC file
    """
    if not config or not config.get("quality_control", {}).get("enabled", True):
        logger.info("QC: motion correction QC skipped (disabled in configuration)")
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_motion_correction", "skipped": True},
        )

    try:
        result = create_motion_correction_qc(
            motion_params=str(motion_params_file),
            save_f=str(output_path),
            input_file=str(input_file) if input_file else None,
            logger=logger,
            confounds_file=str(confounds_file) if confounds_file else None,
        )

        qc_file = Path(result.get("motion_plot", output_path))

        return StepOutput(
            output_file=qc_file,
            metadata={"step": "qc_motion_correction"},
            qc_files=[qc_file],
        )
    except Exception as e:
        logger.warning(f"QC: motion correction QC failed - {e}")
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_motion_correction", "error": str(e)},
        )


def qc_confounds(
    confounds_file: Path,
    output_path: Path,
    config: Optional[Dict[str, Any]] = None,
) -> StepOutput:
    """
    Generate confounds QC figure: compact panels with color-coded stats.

    Plots whichever of the following columns are present in the confounds TSV, one panel each
    (fMRIPrep order): ``global_signal`` (GS), ``csf``, ``white_matter`` (WM), ``std_dvars``
    (DVARS), ``framewise_displacement`` (FD). Tissue rows appear only when anatomical
    segmentation was available. Visualizes confound regressors only — BOLD is unchanged.

    Args:
        confounds_file: ``*_desc-confounds_timeseries.tsv``.
        output_path: Output PNG path.
        config: Configuration dictionary (optional).
    """
    if not config or not config.get("quality_control", {}).get("enabled", True):
        logger.info("QC: confounds QC skipped (disabled in configuration)")
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_confounds", "skipped": True},
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        from ..operations.confounds import (
            FD_OUTLIER_THRESHOLD_MM,
            STD_DVARS_OUTLIER_THRESHOLD,
        )
        from ..quality_control.mri_plotting import (
            CONFOUND_PANEL_SPECS,
            CONFOUNDS_QC_MARGINS,
            create_confounds_plot,
            save_timeseries_qc_figure,
        )

        df = pd.read_csv(str(confounds_file), sep="\t")
        panels = [c for c, _, _ in CONFOUND_PANEL_SPECS if c in df.columns]
        if not panels:
            raise ValueError("no recognized confound columns to plot")

        # Threshold reference lines come from the same effective config that produced this TSV in
        # this run. The JSON sidecar also records them, but QC_CONFOUNDS stages only the TSV.
        conf_cfg = (config or {}).get("func", {}).get("confounds", {}) or {}
        thresholds = {
            "fd_outlier_threshold_mm": conf_cfg.get(
                "fd_outlier_threshold_mm", FD_OUTLIER_THRESHOLD_MM
            ),
            "std_dvars_outlier_threshold": conf_cfg.get(
                "std_dvars_outlier_threshold", STD_DVARS_OUTLIER_THRESHOLD
            ),
        }

        fig = create_confounds_plot(df, thresholds=thresholds)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_timeseries_qc_figure(fig, output_path, margins=CONFOUNDS_QC_MARGINS)
        plt.close(fig)

        return StepOutput(
            output_file=output_path,
            metadata={
                "step": "qc_confounds",
                "panels": panels,
                "thresholds": thresholds,
            },
            qc_files=[output_path],
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"QC: confounds QC failed - {e}")
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_confounds", "error": str(e)},
        )


def qc_surf_recon_tissue_seg(
    fs_subject_dir: Path,
    output_path: Path,
    modality: str = "anat",
    config: Optional[Dict[str, Any]] = None,
) -> StepOutput:
    """
    Generate surface reconstruction tissue segmentation QC snapshot.

    Args:
        fs_subject_dir: Path to FreeSurfer subject directory
        output_path: Output path for QC snapshot
        modality: Modality ('anat' or 'func')
        config: Configuration dictionary (optional)

    Returns:
        StepOutput with QC file
    """
    if not config or not config.get("quality_control", {}).get("enabled", True):
        logger.info(
            "QC: surface reconstruction tissue segmentation QC skipped (disabled in configuration)"
        )
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_surf_recon_tissue_seg", "skipped": True},
        )

    try:
        result = create_surf_recon_tissue_seg_qc(
            fs_subject_dir=str(fs_subject_dir),
            save_f=str(output_path),
            modality=modality,
            logger=logger,
        )

        qc_file = Path(
            result.get(f"{modality}_surf_recon_tissue_seg_overlay", output_path)
        )

        return StepOutput(
            output_file=qc_file,
            metadata={"step": "qc_surf_recon_tissue_seg", "modality": modality},
            qc_files=[qc_file],
        )
    except Exception as e:
        logger.warning(
            f"QC: surface reconstruction tissue segmentation QC failed - {e}"
        )
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_surf_recon_tissue_seg", "error": str(e)},
        )


def qc_cortical_surf_and_measures(
    fs_subject_dir: Path,
    output_path: Path,
    atlas_name: str = "ARM2",
    modality: str = "anat",
    config: Optional[Dict[str, Any]] = None,
) -> StepOutput:
    """
    Generate cortical surface and measures QC snapshot.

    Args:
        fs_subject_dir: Path to FreeSurfer subject directory
        output_path: Output path for QC snapshot
        atlas_name: Atlas name (default: "ARM2")
        modality: Modality ('anat' or 'func')
        config: Configuration dictionary (optional)

    Returns:
        StepOutput with QC file
    """
    if not config or not config.get("quality_control", {}).get("enabled", True):
        logger.info(
            "QC: cortical surface and measures QC skipped (disabled in configuration)"
        )
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_cortical_surf_and_measures", "skipped": True},
        )

    try:
        result = create_cortical_surf_and_measures_qc(
            fs_subject_dir=str(fs_subject_dir),
            save_f=str(output_path),
            atlas_name=atlas_name,
            modality=modality,
            logger=logger,
        )

        qc_file = Path(result.get("snapshot_file", output_path))

        return StepOutput(
            output_file=qc_file,
            metadata={
                "step": "qc_cortical_surf_and_measures",
                "modality": modality,
                "atlas_name": atlas_name,
            },
            qc_files=[qc_file],
        )
    except Exception as e:
        logger.warning(f"QC: cortical surface and measures QC failed - {e}")
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_cortical_surf_and_measures", "error": str(e)},
        )


def qc_generate_report(
    snapshot_dir: Path,
    report_path: Path,
    config: Dict[str, Any],
    snapshot_paths: Optional[list] = None,
    run_status: Optional[Dict[str, Any]] = None,
) -> StepOutput:
    """
    Generate comprehensive QC report from snapshots.

    Args:
        snapshot_dir: Directory containing QC snapshots
        report_path: Output path for HTML report
        config: Configuration dictionary
        snapshot_paths: Optional list of specific snapshot paths (None = auto-discover)
        run_status: Optional pipeline run status (success/abort + error info) rendered
            into the report's "Run status" section.

    Returns:
        StepOutput with report file
    """
    if not config.get("quality_control", {}).get("enabled", True):
        logger.info("QC: report generation skipped (disabled in configuration)")
        return StepOutput(
            output_file=report_path, metadata={"step": "qc_report", "skipped": True}
        )

    try:
        result = generate_qc_report(
            snapshot_dir=str(snapshot_dir),
            report_path=str(report_path),
            config=config,
            logger=logger,
            snapshot_paths=snapshot_paths,
            run_status=run_status,
        )

        report_file = Path(result.get("report_file", report_path))

        return StepOutput(
            output_file=report_file,
            metadata={
                "step": "qc_report",
                "num_snapshots": result.get("num_snapshots", 0),
            },
            qc_files=[report_file],
        )
    except Exception as e:
        logger.warning(f"QC: report generation failed - {e}")
        return StepOutput(
            output_file=report_path, metadata={"step": "qc_report", "error": str(e)}
        )


def qc_within_ses_coreg(
    tmean_run1: Path,
    tmean_averaged: Path,
    output_path: Path,
    config: Optional[Dict[str, Any]] = None,
) -> StepOutput:
    """
    Generate within-session coregistration QC snapshot.

    Shows comparison: first row = single run (tmean of run 1),
    second row = coregistered average tmean.

    Args:
        tmean_run1: Tmean from first run (reference)
        tmean_averaged: Averaged tmean after coregistration
        output_path: Output path for QC snapshot
        config: Configuration dictionary (optional)

    Returns:
        StepOutput with QC file
    """
    if not config or not config.get("quality_control", {}).get("enabled", True):
        logger.info(
            "QC: within-session coregistration QC skipped (disabled in configuration)"
        )
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_within_ses_coreg", "skipped": True},
        )

    try:
        # Validate inputs
        if not tmean_run1.exists():
            logger.error(f"QC: tmean_run1 file not found - {tmean_run1}")
            return StepOutput(
                output_file=output_path,
                metadata={
                    "step": "qc_within_ses_coreg",
                    "error": "tmean_run1 not found",
                },
            )

        if not tmean_averaged.exists():
            logger.error(f"QC: tmean_averaged file not found - {tmean_averaged}")
            return StepOutput(
                output_file=output_path,
                metadata={
                    "step": "qc_within_ses_coreg",
                    "error": "tmean_averaged not found",
                },
            )

        # Ensure the parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create before/after comparison (run1 vs averaged)
        _create_before_after_comparison(
            str(tmean_run1),  # Original: single run
            str(tmean_averaged),  # Corrected: averaged after coreg
            num_cols=6,
            perspectives=["axial"],
            before_after_labels=["single run ref", "avg run after coreg"],
            save_f=str(output_path),
            logger=logger,
        )

        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_within_ses_coreg", "modality": "func"},
            qc_files=[output_path],
        )
    except Exception as e:
        logger.warning(f"QC: within-session coregistration QC failed - {e}")
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_within_ses_coreg", "error": str(e)},
        )


def qc_tsnr(
    session_tsnr_vol: Path,
    lh_surf_gii: Optional[Path],
    rh_surf_gii: Optional[Path],
    fs_subject_dir: Optional[Path],
    output_path: Path,
    config: Optional[Dict[str, Any]] = None,
) -> StepOutput:
    """
    Combined QC figure: session-average tSNR volume (axial grid) and optional surface maps.

    Surface panel is omitted if gifti paths are missing/dummy, ``fs_subject_dir`` is missing,
    or optional dependencies (surfplot, PIL) are unavailable.
    """
    if not config or not config.get("quality_control", {}).get("enabled", True):
        logger.info("QC: tSNR QC skipped (disabled in configuration)")
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_tsnr", "skipped": True},
        )

    try:
        if not session_tsnr_vol.exists():
            logger.error("QC: session tSNR volume not found — %s", session_tsnr_vol)
            return StepOutput(
                output_file=output_path,
                metadata={"step": "qc_tsnr", "error": "session_tsnr_missing"},
            )

        result = create_tsnr_qc(
            session_tsnr_vol=str(session_tsnr_vol),
            save_f=str(output_path),
            lh_surf_gii=str(lh_surf_gii) if lh_surf_gii else None,
            rh_surf_gii=str(rh_surf_gii) if rh_surf_gii else None,
            fs_subject_dir=str(fs_subject_dir) if fs_subject_dir else None,
            logger=logger,
        )

        if not result:
            return StepOutput(
                output_file=output_path,
                metadata={"step": "qc_tsnr", "error": "render_failed"},
            )

        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_tsnr", "modality": "func"},
            qc_files=[output_path],
        )
    except Exception as e:
        logger.warning("QC: tSNR QC failed — %s", e)
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_tsnr", "error": str(e)},
        )


def qc_t1wt2w_combined(
    t1w_before_file: Path,
    t1wt2w_combined_file: Path,
    output_path: Path,
    modality: str = "anat",
    mask_file: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> StepOutput:
    """
    Generate T1wT2wCombined QC snapshot.

    Shows before/after comparison: T1w after bias correction vs T1wT2wCombined image.
    Optionally applies a brain mask to both images before visualization.

    Args:
        t1w_before_file: T1w image after bias correction (before)
        t1wt2w_combined_file: T1wT2wCombined image (after)
        output_path: Output path for QC snapshot
        modality: Modality ('anat' or 'func')
        mask_file: Optional brain mask file (if provided, mask will be applied to both images)
        config: Configuration dictionary (optional)

    Returns:
        StepOutput with QC file
    """
    if not config or not config.get("quality_control", {}).get("enabled", True):
        logger.info("QC: T1wT2wCombined QC skipped (disabled in configuration)")
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_t1wt2w_combined", "skipped": True},
        )

    try:
        result = create_t1wt2w_combined_qc(
            image_before=str(t1w_before_file),
            image_combined=str(t1wt2w_combined_file),
            save_f=str(output_path),
            modality=modality,
            mask_file=str(mask_file) if mask_file else None,
            logger=logger,
        )

        qc_file = Path(
            result.get(f"{modality}_t1wt2w_combined_comparison", output_path)
        )

        return StepOutput(
            output_file=qc_file,
            metadata={
                "step": "qc_t1wt2w_combined",
                "modality": modality,
                "mask_applied": mask_file is not None,
            },
            qc_files=[qc_file],
        )
    except Exception as e:
        logger.warning(f"QC: T1wT2wCombined QC failed - {e}")
        return StepOutput(
            output_file=output_path,
            metadata={"step": "qc_t1wt2w_combined", "error": str(e)},
        )
