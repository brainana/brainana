"""
Standalone anatomical processing step functions.

These functions wrap the existing operations to provide standardized
inputs/outputs for Nextflow integration.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

from .types import StepInput, StepOutput
from ..operations.preprocessing import (
    conform_to_template,
    bias_correction,
    apply_segmentation,
    generate_t1wt2wcombined,
)
from ..operations.registration import (
    ants_register,
    ants_apply_transforms,
    flirt_apply_transforms,
)
from ..operations.sitk_rigid_registration import (
    apply_sitk_affine,
    conform_world_mat_path,
)
import SimpleITK as sitk
from ..utils.bids import get_bids_prefix
from ..utils.templates import (
    discover_atlases_in_space,
    get_template_manager,
    space_label_for,
)
from ..utils.mri import (
    normalize_anat_input,
    get_image_shape,
    shape_to_ants_input_type,
)
from fastsurfer_surfrecon.config import AtlasConfig, ReconSurfConfig
from fastsurfer_surfrecon.pipeline import ReconSurfPipeline

logger = logging.getLogger(__name__)


def anat_synthesis(
    anat_files: List[Path], working_dir: Path, config: Dict[str, Any]
) -> StepOutput:
    """
    Synthesize multiple anatomical runs (T1w, T2w, etc.) into a single image.

    This function coregisters and averages multiple anatomical runs from the same
    subject/session to create a single synthesized image for processing.

    Args:
        anat_files: List of anatomical file paths to synthesize (all same modality)
        working_dir: Working directory for intermediate files
        config: Configuration dictionary

    Returns:
        StepOutput with synthesized anatomical file
    """
    if not anat_files:
        return StepOutput(
            output_file=None,
            metadata={
                "step": "anat_synthesis",
                "synthesized": False,
                "num_runs": 0,
                "input_normalization": [],
            },
        )

    # ------------------------------------------------------------------
    # Normalize every input before anything else touches it. This is the single
    # process every anatomical passes through (T1w and T2w, subject- and
    # session-level, synthesis needed or not), so repairing here means no
    # downstream step has to know about malformed anatomical headers.
    # normalize_anat_input covers dimensionality, missing orientation, qform/sform
    # disagreement and container format in one read/write, and reports the header
    # defects that have no safe automatic repair.
    #
    # Basenames are preserved because the synthesis branch below parses BIDS
    # entities out of them. The destination MUST stay under working_dir: the
    # ANAT_SYNTHESIS process declares `path("*.nii.gz")` as its output and
    # publishes on the same pattern, both non-recursive over the task dir root,
    # so a stray file there would become a second channel element and a spurious
    # published derivative.
    norm_dir = Path(working_dir) / "input_normalized"
    normalized_files = []
    normalization_reports = []
    for anat_file in anat_files:
        normalized, report = normalize_anat_input(
            anat_file, norm_dir / Path(anat_file).name, logger=logger
        )
        normalized_files.append(normalized)
        normalization_reports.append({"source": str(anat_file), **report})
    anat_files = normalized_files

    if len(anat_files) <= 1:
        # No synthesis needed
        return StepOutput(
            output_file=anat_files[0],
            metadata={
                "step": "anat_synthesis",
                "synthesized": False,
                "num_runs": len(anat_files),
                "input_normalization": normalization_reports,
            },
        )

    # Determine modality from first file
    from ..utils.bids import parse_bids_entities, get_filename_stem

    first_stem = get_filename_stem(anat_files[0])
    modality = "T1w"  # default
    if "_T2w" in first_stem or first_stem.endswith("_T2w"):
        modality = "T2w"
    elif "_T1w" in first_stem or first_stem.endswith("_T1w"):
        modality = "T1w"

    logger.info(f"Step: synthesizing {len(anat_files)} {modality} runs")

    # Import synthesis function and BIDSFile
    from ..operations.synthesis_multiple_anat import synthesize_multiple_anatomical
    from ..utils.bids import BIDSFile

    bids_files = []
    for anat_file in anat_files:
        entities = parse_bids_entities(anat_file.name)
        bids_files.append(
            BIDSFile(
                path=str(anat_file),
                sub=entities.get("sub", "unknown"),
                ses=entities.get("ses"),
                modality="anat",
                suffix=modality,
                entities=entities,
                run=entities.get("run"),
            )
        )

    # Call synthesis function
    # The synthesis function now handles all file management internally using working_dir
    synthesized_file = synthesize_multiple_anatomical(
        anat_files=bids_files, working_dir=working_dir, logger=logger, config=config
    )

    if not synthesized_file:
        raise RuntimeError(f"Anatomical synthesis failed for {len(anat_files)} files")

    return StepOutput(
        output_file=Path(synthesized_file),
        metadata={
            "step": "anat_synthesis",
            "synthesized": True,
            "num_runs": len(anat_files),
            "input_normalization": normalization_reports,
        },
    )


def anat_conform(input: StepInput, template_file: Path) -> StepOutput:
    """
    Conform anatomical image to template space.

    Args:
        input: StepInput with input_file, working_dir, config, metadata
        template_file: Template file for conforming

    Returns:
        StepOutput with conformed file and transform files
    """
    if not input.config.get("anat.conform.enabled", True):
        logger.info("Step: conform skipped (disabled in configuration)")
        return StepOutput(
            output_file=input.input_file, metadata={"step": "conform", "skipped": True}
        )

    # Check if skullstripping is disabled - if so, skip internal skullstripping in conform
    skip_skullstripping = not input.config.get(
        "anat.skullstripping_segmentation.enabled", True
    )

    # Rigid backend: 'flirt' (FSL, default) or 'sitk' (FSL- and AFNI-free). The lite
    # pipeline sets this to 'sitk'. Resolve from either a dotted Config or a plain nested
    # dict (the lite notebook passes a plain dict, where dotted-string .get falls through).
    rigid_method = input.config.get("anat.conform.rigid_method") or input.config.get(
        "anat", {}
    ).get("conform", {}).get("rigid_method", "flirt")

    # Call operation
    result = conform_to_template(
        imagef=str(input.input_file),
        template_file=str(template_file),
        working_dir=str(input.working_dir),
        output_name=input.output_name or "anat_conformed.nii.gz",
        logger=logger,
        modal="anat",
        skip_skullstripping=skip_skullstripping,
        rigid_method=rigid_method,
        # Anatomical only: also emit the conform on a grid large enough to keep every
        # scanner-space voxel (recording chamber, head-post, neck). Leaf output.
        emit_full_fov=True,
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
        additional_files["template_resampled"] = Path(
            result["template_f"]
        )  # template_f is already the resampled template
    if result.get("imagef_conformed_full_fov"):
        additional_files["conformed_full_fov"] = Path(
            result["imagef_conformed_full_fov"]
        )
    if result.get("template_f_full_fov"):
        additional_files["template_resampled_full_fov"] = Path(
            result["template_f_full_fov"]
        )
    if result.get("fov_box"):
        additional_files["fov_box"] = Path(result["fov_box"])

    return StepOutput(
        output_file=output_file,
        metadata={
            "step": "conform",
            "modality": "anat",
            "template_file": str(template_file),
            "rigid_method": rigid_method,
            # Real backend that produced the transform (for the xfm sidecar GeneratedBy).
            "engine": result.get("engine", rigid_method),
            # Padding that turned the target FOV into the uncropped one (QC + sidecar).
            "full_fov_pad": result.get("full_fov_pad"),
        },
        additional_files=additional_files,
    )


def anat_bias_correction(
    input: StepInput, brain_mask: Optional[Path] = None
) -> StepOutput:
    """
    Perform bias field correction on anatomical image.

    Args:
        input: StepInput with input_file, working_dir, config, metadata
        brain_mask: Optional brain mask file (if provided and not dummy, will be used for bias correction)

    Returns:
        StepOutput with bias-corrected file and optionally brain-only file
    """
    if not input.config.get("anat.bias_correction.enabled", True):
        logger.info("Step: bias correction skipped (disabled in configuration)")
        return StepOutput(
            output_file=input.input_file,
            metadata={"step": "bias_correction", "skipped": True},
        )

    # Use mask if provided (mask is already validated as real in workflow)
    mask_path = brain_mask if brain_mask else None
    if mask_path:
        logger.info(f"Step: using brain mask for bias correction - {brain_mask.name}")

    # Call operation
    result = bias_correction(
        imagef=str(input.input_file),
        working_dir=str(input.working_dir),
        modal="anat",
        output_name=input.output_name or "anat_bias_corrected.nii.gz",
        config=input.config,
        logger=logger,
        maskf=str(mask_path) if mask_path else None,
    )

    output_file = Path(result["imagef_bias_corrected"])
    additional_files = {}

    # Add brain-only file if generated
    if result.get("imagef_brain"):
        additional_files["brain"] = Path(result["imagef_brain"])

    return StepOutput(
        output_file=output_file,
        metadata={
            "step": "bias_correction",
            "modality": "anat",
            "algorithm": input.config.get("anat", {})
            .get("bias_correction", {})
            .get("algorithm", "N4BiasFieldCorrection"),
            "mask_used": str(mask_path) if mask_path else None,
        },
        additional_files=additional_files,
    )


def anat_skullstripping(input: StepInput) -> StepOutput:
    """
    Perform skull stripping on anatomical image using fastsurfer_nn.

    Args:
        input: StepInput with input_file, working_dir, config, metadata

    Returns:
        StepOutput with skull-stripped file, brain mask, segmentation,
        atlas LUT (ColorLUT TSV, same base as segmentation), and optional hemimask
    """
    if not input.config.get("anat.skullstripping_segmentation.enabled", True):
        logger.info("Step: skull stripping skipped (disabled in configuration)")
        return StepOutput(
            output_file=input.input_file,
            metadata={"step": "skullstripping", "skipped": True},
        )

    # Call operation
    result = apply_segmentation(
        imagef=str(input.input_file),
        modal="anat",
        working_dir=str(input.working_dir),
        output_name=input.output_name or "anat_brain.nii.gz",
        config=input.config,
        logger=logger,
    )

    output_file = Path(result["imagef_skullstripped"])
    additional_files = {}

    if result.get("brain_mask"):
        additional_files["brain_mask"] = Path(result["brain_mask"])
    if result.get("segmentation"):
        additional_files["segmentation"] = Path(result["segmentation"])
    if result.get("hemimask"):
        additional_files["hemimask"] = Path(result["hemimask"])
    if result.get("input_cropped"):
        additional_files["input_cropped"] = Path(result["input_cropped"])
    if result.get("atlas_lut"):
        additional_files["atlas_lut"] = Path(result["atlas_lut"])

    return StepOutput(
        output_file=output_file,
        metadata={
            "step": "skullstripping",
            "modality": "anat",
            "method": "fastSurferCNN",
            "atlas_name": result.get("atlas_name"),
        },
        additional_files=additional_files,
    )


def anat_registration(
    input: StepInput, template_file: Path, template_name: str
) -> StepOutput:
    """
    Register anatomical image to template space.

    Args:
        input: StepInput with input_file, working_dir, config, metadata
        template_file: Template file for registration
        template_name: Template name (e.g., "NMT2Sym")

    Returns:
        StepOutput with registered file and transform files
    """
    if not input.config.get("registration.enabled", True):
        logger.info("Step: registration skipped (disabled in configuration)")
        return StepOutput(
            output_file=input.input_file,
            metadata={"step": "registration", "skipped": True},
        )

    # Get registration behavior from config
    reg_config = input.config.get("registration", {})
    xfm_type = reg_config.get("anat2template_xfm_type", "syn")
    enable_fireants = reg_config.get("enable_fireants", True)
    fireants_allow_cpu = reg_config.get("fireants_allow_cpu", True)

    logger.info(
        f"Step: registering anatomical image to template {template_name} "
        f"(xfm_type={xfm_type})"
    )

    # Call operation
    result = ants_register(
        movingf=str(input.input_file),
        fixedf=str(template_file),
        working_dir=str(input.working_dir),
        output_prefix="anat2template",
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
            "modality": "anat",
            "target": template_name,
            "xfm_type": xfm_type,
            "fireants_requested": enable_fireants,
            # Real backend used (fireants/ants/antspyx) after any runtime fallback.
            "engine": result.get("engine"),
        },
        additional_files=additional_files,
    )


def copy_atlas_sidecars(
    atlas_name: str,
    dest_dir: Path,
    source_dir: Optional[Path] = None,
) -> List[Path]:
    """Copy an atlas's associated sidecar files into dest_dir alongside its image.

    Copies the template-zoo files that share the ``atlas-{name}`` prefix — the label
    lookup table ``atlas-{name}.tsv`` and any docs ``atlas-{name}.md`` — verbatim, and
    the family bibliography ``references.bib`` renamed to ``atlas-{name}.bib``. These
    sidecars are space- and subject-independent, so the same set is copied into every
    output space (T1w / scanner / fsnative). Image and provenance files (``.nii.gz``,
    ``.json``) are skipped, and existing destination files are left untouched.

    Args:
        atlas_name: Atlas label, e.g. ``ARM2`` (the ``{name}`` in ``atlas-{name}``).
        dest_dir: Directory to copy sidecars into (the step's ``atlas/`` work dir).
        source_dir: Template-zoo family directory (parent of the atlas ``.nii.gz``).
            When None, it is located by globbing the atlas zoo for
            ``atlas-{name}_space-*.nii.gz`` — the same zoo atlas discovery uses.

    Returns:
        List of newly created destination paths (empty when nothing was copied).
    """
    if source_dir is None:
        atlas_root = get_template_manager().template_dir / "atlas"
        matches = sorted(atlas_root.glob(f"**/atlas-{atlas_name}_space-*.nii.gz"))
        if not matches:
            logger.debug(
                f"System: no template-zoo source for atlas {atlas_name}; no sidecars copied"
            )
            return []
        source_dir = matches[0].parent

    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []

    # Prefix-shared sidecars (atlas-{name}.tsv / .md), copied verbatim. The '.' after the
    # name excludes the label images, whose next character is '_' (atlas-{name}_space-...).
    for sidecar in sorted(source_dir.glob(f"atlas-{atlas_name}.*")):
        if sidecar.name.endswith(".nii.gz") or sidecar.suffix == ".json":
            continue
        dest = dest_dir / sidecar.name
        if dest.exists():
            continue
        shutil.copy2(sidecar, dest)
        copied.append(dest)

    # Family bibliography, renamed to carry the atlas- prefix so it travels with the atlas.
    bib = source_dir / "references.bib"
    if bib.is_file():
        dest = dest_dir / f"atlas-{atlas_name}.bib"
        if not dest.exists():
            shutil.copy2(bib, dest)
            copied.append(dest)

    return copied


def anat_backproject_atlases(
    inverse_xfm: Path,
    t1w_reference: Path,
    bids_name: Path,
    working_dir: Path,
    config: Dict[str, Any],
) -> StepOutput:
    """
    Backproject atlases from template space to T1w space using inverse transform.

    Discovers atlases in the template space (from config output_space), applies
    the inverse T1w->template transform to each, and writes outputs to
    working_dir/atlas/ with naming:
    atlas-{atlas_name}_space-T1w_{ses_prefix}.nii.gz.
    ses_prefix is sub (+ ses when present), derived from bids_name via
    get_bids_prefix (space/desc/modality entities are stripped).

    Args:
        inverse_xfm: Inverse transform (template -> T1w)
        t1w_reference: T1w image defining output grid (anat_after_bias)
        bids_name: BIDS filename of T1w output (e.g. sub-X_ses-Y_run-Z_desc-preproc_T1w.nii.gz)
        working_dir: Working directory; atlas outputs go to working_dir/atlas/
        config: Configuration dict (uses template.output_space)

    Returns:
        StepOutput with output_file=atlas_dir, additional_files={atlas_name: path}
    """
    effective_output_space = config.get("template", {}).get(
        "output_space", "NMT2Sym:res-05"
    )
    # A custom template file path resolves to the literal 'template' space, which has no
    # bundled atlases (discover_atlases_in_space returns [] below and the step no-ops).
    space_name = space_label_for(effective_output_space)
    parts = effective_output_space.split(":")
    template_res = parts[1] if len(parts) > 1 else None

    atlases = discover_atlases_in_space(
        space_name=space_name,
        template_res=template_res,
    )
    if not atlases:
        logger.info(
            f"Step: no atlases found in space {space_name}, skipping backproject"
        )
        atlas_dir = working_dir / "atlas"
        atlas_dir.mkdir(parents=True, exist_ok=True)
        return StepOutput(
            output_file=atlas_dir,
            metadata={
                "step": "backproject_atlases",
                "atlases_found": 0,
                "space": space_name,
            },
            additional_files={},
        )

    output_stem = get_bids_prefix(bids_name)

    atlas_dir = working_dir / "atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)

    additional_files: Dict[str, Path] = {}
    sidecars: List[Path] = []
    for atlas_name, atlas_path in atlases:
        output_name = f"atlas-{atlas_name}_space-T1w_{output_stem}.nii.gz"
        shape = get_image_shape(str(atlas_path), logger=logger)
        moving_type = shape_to_ants_input_type(shape)

        result = ants_apply_transforms(
            movingf=str(atlas_path),
            moving_type=moving_type,
            interpolation="NearestNeighbor",
            outputf_name=output_name,
            fixedf=str(t1w_reference),
            working_dir=str(atlas_dir),
            transformf=[str(inverse_xfm)],
            logger=logger,
            reff=str(t1w_reference),
            generate_tmean=False,
        )
        out_path = Path(result["imagef_registered"])
        additional_files[atlas_name] = out_path

        # Copy the atlas's associated sidecars (LUT/docs/refs) next to the image so they
        # travel with the backprojected label volume. atlas_path is the template-zoo
        # source, so its parent is the family dir holding the sidecars.
        sidecars.extend(
            copy_atlas_sidecars(atlas_name, atlas_dir, source_dir=atlas_path.parent)
        )

    return StepOutput(
        output_file=atlas_dir,
        metadata={
            "step": "backproject_atlases",
            "atlases_found": len(additional_files),
            "space": space_name,
            "sidecars": [str(p) for p in sidecars],
        },
        additional_files=additional_files,
    )


def anat_reproject_atlases_to_scanner(
    atlas_files: List[Path],
    conform_inverse_xfm: Path,
    scanner_reference: Path,
    working_dir: Path,
    rigid_method: str = "flirt",
) -> StepOutput:
    """
    Reproject T1w-space atlases to scanner space using conform inverse transform.

    Args:
        atlas_files: Paths to already backprojected T1w-space atlas images
        conform_inverse_xfm: Inverse conform transform (T1w -> scanner)
        scanner_reference: Scanner-space reference image defining output grid
        working_dir: Working directory; atlas outputs go to working_dir/atlas/
        rigid_method: Conform backend that produced the transform — 'flirt' (FSL, default)
            applies via ``flirt -applyxfm``; 'sitk' applies the world-space affine via
            SimpleITK (FSL-free), matching ``conform_to_template(rigid_method=...)``.

    Returns:
        StepOutput with output_file=atlas_dir, additional_files={atlas_name: path}
    """
    atlas_dir = working_dir / "atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)

    # For the sitk backend, the conform inverse is applied by inverting the stored
    # world-space affine (forward maps template->scanner, so scanner->template = inverse).
    sitk_world_mat = (
        conform_world_mat_path(conform_inverse_xfm) if rigid_method == "sitk" else None
    )

    additional_files: Dict[str, Path] = {}
    sidecars: List[Path] = []
    seen_atlas_names: set = set()
    for atlas_path in atlas_files:
        # Input: atlas-{name}_space-T1w_{stem}.nii.gz -> scanner: atlas-{name}_space-scanner_{stem}.nii.gz
        in_name = atlas_path.name
        if "_space-T1w_" in in_name:
            out_name = in_name.replace("_space-T1w_", "_space-scanner_", 1)
        else:
            stem = in_name.replace(".nii.gz", "")
            out_name = f"{stem}_space-scanner.nii.gz"
        if rigid_method == "sitk":
            result = apply_sitk_affine(
                movingf=str(atlas_path),
                outputf_name=out_name,
                reff=str(scanner_reference),
                working_dir=str(atlas_dir),
                world_mat=sitk_world_mat,
                interpolation=sitk.sitkNearestNeighbor,
                invert=True,
            )
        else:
            result = flirt_apply_transforms(
                movingf=str(atlas_path),
                outputf_name=out_name,
                reff=str(scanner_reference),
                working_dir=str(atlas_dir),
                transformf=str(conform_inverse_xfm),
                logger=logger,
                interpolation="nearestneighbour",
                generate_tmean=False,
            )
        out_path = Path(result["imagef_registered"])
        additional_files[out_name] = out_path

        # The sidecars (LUT/docs/refs) are space-independent, so carry the same set into
        # scanner space. Re-discover them from the template zoo by atlas name (this step
        # receives only projected .nii.gz files, not the zoo source). Copy once per atlas.
        atlas_name = _atlas_name_from_filename(in_name)
        if atlas_name not in seen_atlas_names:
            seen_atlas_names.add(atlas_name)
            sidecars.extend(copy_atlas_sidecars(atlas_name, atlas_dir))

    return StepOutput(
        output_file=atlas_dir,
        metadata={
            "step": "reproject_atlases_to_scanner",
            "atlases_found": len(additional_files),
            "sidecars": [str(p) for p in sidecars],
        },
        additional_files=additional_files,
    )


# lh/rh only appear as the FreeSurfer CLI --hemi value; filenames use _hemi-L/_hemi-R.
_SURF_HEMI_MAP = {"L": "lh", "R": "rh"}


def _atlas_name_from_filename(filename: str) -> str:
    """Extract the atlas label from 'atlas-{name}_space-...' -> '{name}'.

    Atlas backprojection writes ``atlas-{name}_space-T1w_{stem}.nii.gz``; we recover
    ``{name}`` to build the fsnative/surface output names.
    """
    stem = filename
    if stem.startswith("atlas-"):
        stem = stem[len("atlas-") :]
    idx = stem.find("_space-")
    if idx != -1:
        return stem[:idx]
    idx = stem.find("_")
    return stem[:idx] if idx != -1 else stem


def anat_project_atlases_to_surface(
    atlas_files: List[Path],
    fs_subject_dir: Path,
    bids_name: Path,
    working_dir: Path,
) -> StepOutput:
    """
    Project T1w-space atlases onto FastSurfer cortical surfaces (nearest-neighbor).

    Two sub-steps per atlas; labels must never be interpolated or smoothed:
      1. Resample the T1w-space label volume onto the FastSurfer conformed grid
         (``mri/T1.mgz``) with ``mri_vol2vol --regheader --interp nearest``. T1w and
         FastSurfer-conformed space share the same scanner-RAS frame, so this is a
         grid-only resample with no transform file.
      2. Sample the FastSurfer-space volume onto the lh/rh surfaces at mid-thickness
         with ``mri_vol2surf --projfrac 0.5 --interp nearest`` (no ``--surf-fwhm``),
         writing one ``.func.gii`` per hemisphere.

    ``fs_subject_dir`` is the FastSurfer subject folder (``SUBJECTS_DIR`` is its parent;
    FreeSurfer subject id is ``fs_subject_dir.name``). The directory existing is how
    "surface reconstruction succeeded" is detected.

    Outputs go to ``working_dir/atlas/`` with naming:
      ``atlas-{name}_space-fsnative_{stem}.nii.gz``
      ``atlas-{name}_space-fsnative_hemi-{L,R}_{stem}.func.gii``

    On missing tool / missing FastSurfer tree / a failed atlas: logs a warning and
    returns whatever completed, with ``skipped``/``reason`` in metadata (mirrors
    ``project_tsnr_to_surface``).
    """
    atlas_dir = working_dir / "atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)

    additional_files: Dict[str, Path] = {}
    metadata: Dict[str, Any] = {
        "step": "project_atlases_to_surface",
        "space": "fsnative",
    }

    fs_subject_directory = Path(fs_subject_dir)
    if not fs_subject_directory.is_dir():
        logger.warning(
            f"atlas surf: FastSurfer subject dir missing: {fs_subject_directory}"
        )
        metadata.update(
            {
                "skipped": True,
                "reason": "missing_fs_subject_dir",
                "atlases_projected": 0,
            }
        )
        return StepOutput(
            output_file=atlas_dir, metadata=metadata, additional_files=additional_files
        )

    # Target grid: prefer the conformed T1.mgz, fall back to orig.mgz.
    target_grid = fs_subject_directory / "mri" / "T1.mgz"
    if not target_grid.exists():
        target_grid = fs_subject_directory / "mri" / "orig.mgz"
    if not target_grid.exists():
        logger.warning(
            f"atlas surf: no target grid (T1.mgz/orig.mgz) under {fs_subject_directory}/mri"
        )
        metadata.update(
            {"skipped": True, "reason": "missing_target_grid", "atlases_projected": 0}
        )
        return StepOutput(
            output_file=atlas_dir, metadata=metadata, additional_files=additional_files
        )

    # Record the grid actually used so callers (e.g. sidecar provenance) need not
    # re-derive the T1.mgz/orig.mgz fallback and risk drifting from this choice.
    metadata["target_grid"] = str(target_grid)

    output_stem = get_bids_prefix(bids_name)
    freesurfer_subject_id = fs_subject_directory.name
    env = os.environ.copy()
    # Absolute SUBJECTS_DIR (resolving the staged symlink) matches project_tsnr_to_surface
    # and avoids FreeSurfer tools mis-resolving a relative '.' if they canonicalize the path.
    env["SUBJECTS_DIR"] = str(fs_subject_directory.resolve().parent)

    projected = 0
    sidecars: List[Path] = []
    seen_atlas_names: set = set()
    for atlas_path in atlas_files:
        atlas_path = Path(atlas_path)
        atlas_name = _atlas_name_from_filename(atlas_path.name)

        # Step 1: T1w -> FastSurfer grid (nearest neighbor, header-based resample).
        fsnative_vol = (
            atlas_dir / f"atlas-{atlas_name}_space-fsnative_{output_stem}.nii.gz"
        )
        vol2vol_cmd = [
            "mri_vol2vol",
            "--mov",
            str(atlas_path),
            "--targ",
            str(target_grid),
            "--regheader",
            "--interp",
            "nearest",
            "--o",
            str(fsnative_vol),
        ]
        try:
            subprocess.run(
                vol2vol_cmd, check=True, env=env, capture_output=True, text=True
            )
        except FileNotFoundError:
            logger.warning(
                "atlas surf: mri_vol2vol not found; skip surface projection."
            )
            metadata.update(
                {
                    "skipped": True,
                    "reason": "mri_vol2vol_not_found",
                    "atlases_projected": projected,
                }
            )
            return StepOutput(
                output_file=atlas_dir,
                metadata=metadata,
                additional_files=additional_files,
            )
        except subprocess.CalledProcessError as exc:
            logger.warning(
                f"atlas surf: mri_vol2vol failed for {atlas_path.name}: {exc.stderr}"
            )
            fsnative_vol.unlink(missing_ok=True)
            continue
        additional_files[fsnative_vol.name] = fsnative_vol

        # The sidecars (LUT/docs/refs) are space-independent; carry the same set into
        # fsnative space. Re-discover from the template zoo by atlas name (this step gets
        # only projected .nii.gz files, not the zoo source). Copy once per atlas.
        if atlas_name not in seen_atlas_names:
            seen_atlas_names.add(atlas_name)
            sidecars.extend(copy_atlas_sidecars(atlas_name, atlas_dir))

        # Step 2: FastSurfer volume -> lh/rh surfaces (nearest neighbor, mid-thickness).
        hemi_projected = False
        for hemi_code in ("L", "R"):
            surf_gii = (
                atlas_dir
                / f"atlas-{atlas_name}_space-fsnative_hemi-{hemi_code}_{output_stem}.func.gii"
            )
            vol2surf_cmd = [
                "mri_vol2surf",
                "--mov",
                str(fsnative_vol),
                "--regheader",
                freesurfer_subject_id,
                "--hemi",
                _SURF_HEMI_MAP[hemi_code],
                "--projfrac",
                "0.5",
                "--interp",
                "nearest",
                "--out_type",
                "gii",
                "--o",
                str(surf_gii),
            ]
            try:
                subprocess.run(
                    vol2surf_cmd, check=True, env=env, capture_output=True, text=True
                )
            except FileNotFoundError:
                logger.warning(
                    "atlas surf: mri_vol2surf not found; skip surface projection."
                )
                metadata.update(
                    {
                        "skipped": True,
                        "reason": "mri_vol2surf_not_found",
                        "atlases_projected": projected,
                    }
                )
                return StepOutput(
                    output_file=atlas_dir,
                    metadata=metadata,
                    additional_files=additional_files,
                )
            except subprocess.CalledProcessError as exc:
                logger.warning(
                    f"atlas surf: mri_vol2surf failed for {surf_gii.name}: {exc.stderr}"
                )
                surf_gii.unlink(missing_ok=True)
                continue
            additional_files[surf_gii.name] = surf_gii
            hemi_projected = True
        # Count the atlas only if at least one hemisphere surface was produced; both
        # vol2surf calls can fail (each hits `continue`) while vol2vol succeeded.
        if hemi_projected:
            projected += 1

    metadata["atlases_projected"] = projected
    metadata["sidecars"] = [str(p) for p in sidecars]
    return StepOutput(
        output_file=atlas_dir, metadata=metadata, additional_files=additional_files
    )


def anat_t2w_to_t1w_registration(input: StepInput, t1w_reference: Path) -> StepOutput:
    """
    Register T2w image to T1w space.

    Args:
        input: StepInput with input_file (reoriented T2w), working_dir, config, metadata
        t1w_reference: Path to T1w reference

    Returns:
        StepOutput with registered T2w file and transform files
    """
    # Step 1: ANTs rigid registration from T2w to T1w
    try:
        registration_result = ants_register(
            fixedf=str(t1w_reference),
            movingf=str(input.input_file),
            working_dir=str(input.working_dir),
            output_prefix="t2w_to_t1w_coreg",
            logger=logger,
            xfm_type="rigid",
        )
        forward_transform = registration_result.get("forward_transform")
        if not forward_transform:
            raise RuntimeError("ANTs registration did not produce a forward transform")
        xfm_forward_f = Path(forward_transform)
        xfm_inverse_f = (
            Path(registration_result.get("inverse_transform"))
            if registration_result.get("inverse_transform")
            else None
        )
    except Exception as e:
        logger.error(f"Error during ANTs registration: {e}")
        raise RuntimeError(
            f"T2w to T1w registration failed during ANTs registration: {e}"
        )

    # Step 2: Apply the transform (interpolation from config)
    interpolation = input.config.get("registration", {}).get("interpolation", "BSpline")
    try:
        apply_result = ants_apply_transforms(
            movingf=str(input.input_file),
            outputf_name="t2w_to_t1w_coreg_registered.nii.gz",
            fixedf=str(t1w_reference),
            working_dir=str(input.working_dir),
            transformf=[str(xfm_forward_f)],
            logger=logger,
            moving_type=0,
            interpolation=interpolation,
            reff=str(t1w_reference),
            generate_tmean=False,
        )
        output_file = Path(apply_result["imagef_registered"])
    except Exception as e:
        logger.error(f"Error during ANTs transformation application: {e}")
        raise RuntimeError(
            f"T2w to T1w registration failed when applying transformation: {e}"
        )

    additional_files = {}
    if xfm_forward_f.exists():
        additional_files["forward_transform"] = xfm_forward_f
    if xfm_inverse_f is not None and xfm_inverse_f.exists():
        additional_files["inverse_transform"] = xfm_inverse_f

    return StepOutput(
        output_file=output_file,
        metadata={
            "step": "t2w_to_t1w_registration",
            "modality": "T2w",
            "target": "T1w",
            "xfm_type": "rigid",
            # Real backend used (ants/antspyx; rigid never uses FireANTs).
            "engine": registration_result.get("engine"),
        },
        additional_files=additional_files,
    )


def anat_surface_reconstruction(
    input: StepInput,
    t1w_file: Path,
    segmentation_file: Path,
    brain_mask: Optional[Path] = None,
    arm6_atlas: Optional[Path] = None,
) -> StepOutput:
    """
    Perform surface reconstruction using fastsurfer_surfrecon.

    Args:
        input: StepInput with input_file, working_dir, config, metadata
        t1w_file: T1w image file (any T1w file, independent of preprocessing pipeline)
        segmentation_file: Segmentation file (from skullstripping step)
        brain_mask: Brain mask file (required for surface reconstruction)
        arm6_atlas: Optional ARM6 atlas for claustrum fix (stage s07b)

    Returns:
        StepOutput with surface reconstruction directory path
    """
    if not input.config.get("anat.surface_reconstruction.enabled", True):
        logger.info("Step: surface reconstruction skipped (disabled in configuration)")
        return StepOutput(
            output_file=input.input_file,
            metadata={"step": "surface_reconstruction", "skipped": True},
        )

    # Validate required inputs
    if not brain_mask:
        raise ValueError("brain_mask is required for surface reconstruction")

    # Get subject ID from metadata or derive from input
    base_subject_id = input.metadata.get("subject_id") or "unknown"
    if not base_subject_id.startswith("sub-"):
        base_subject_id = f"sub-{base_subject_id}"

    # Get session information from metadata
    session_id = input.metadata.get("session_id")
    session_count = input.metadata.get("session_count", 1)

    # Determine subject ID for FastSurfer based on session count and session_id
    # Logic:
    # - If session_id is None/empty: subject-level synthesis → use sub-XXX
    # - If session_id is not empty:
    #   - If session_count > 1: multiple sessions exist → use sub-XXX_ses-XXX
    #   - If session_count == 1: only one session → use sub-XXX
    if not session_id or session_id == "":
        # Subject-level synthesis (no session identifier)
        subject_id = base_subject_id
    elif session_count > 1:
        # Multiple sessions exist: use session identifier to avoid collisions
        # Extract session ID without 'ses-' prefix if present
        ses_id = session_id if not session_id.startswith("ses-") else session_id[4:]
        subject_id = f"{base_subject_id}_ses-{ses_id}"
    else:
        # Only one session: no need for session identifier
        subject_id = base_subject_id

    # Get atlas name from config (default to ARM2 for macaque)
    atlas_name = (
        input.config.get("anat", {})
        .get("skullstripping_segmentation", {})
        .get("atlas_name", "ARM2")
    )

    # Create subjects directory structure
    subjects_dir = Path(input.working_dir) / "fastsurfer"
    subject_dir = subjects_dir / subject_id

    logger.info(f"Step: Surface reconstruction for {subject_id}")
    logger.info(f"Step: Atlas = {atlas_name}")
    logger.info(f"Step: Subjects directory = {subjects_dir}")

    # Get LUT path using fastsurfer_surfrecon's AtlasConfig (has built-in fallbacks)
    atlas_config = AtlasConfig(name=atlas_name)
    lut_path = atlas_config.colorlut_path

    if lut_path is None or not lut_path.exists():
        raise FileNotFoundError(
            f"LUT file not found for atlas {atlas_name}. "
            f"AtlasConfig searched but could not locate ColorLUT file."
        )

    logger.info(f"Step: LUT path = {lut_path}")

    if arm6_atlas is not None:
        logger.info(f"Step: Optional ARM6 atlas detected: {arm6_atlas}")
        if not arm6_atlas.exists():
            raise FileNotFoundError(
                f"Step: Optional ARM6 atlas file not found at {arm6_atlas}"
            )
    else:
        logger.info(
            "Step: Optional ARM6 atlas not found; claustrum fix will be skipped"
        )

    # Validate required files exist before processing
    for path, name in [
        (t1w_file, "T1w image"),
        (segmentation_file, "segmentation"),
        (brain_mask, "mask"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Step: Required file not found: {name} at {path}")

    logger.info("Step: All required files validated")

    # Step 1: Prepare files for FreeSurfer using postprocess_for_freesurfer
    logger.info("Step: Preparing files for FreeSurfer format")
    try:
        from fastsurfer_nn.postprocessing.prepping_for_surfrecon import (
            postprocess_for_freesurfer,
        )

        prep_result = postprocess_for_freesurfer(
            t1w_image=str(t1w_file),
            segmentation=str(segmentation_file),
            mask=str(brain_mask),
            lut_path=str(lut_path),
            subject_dir=str(subject_dir),
            vox_size="min",
            orientation="lia",
            arm6_atlas=str(arm6_atlas) if arm6_atlas is not None else None,
        )

        if prep_result != 0:
            raise RuntimeError(f"File preparation failed: {prep_result}")

        logger.info("Step: File preparation completed successfully")
    except ImportError as e:
        logger.error(f"Step: Failed to import postprocess_for_freesurfer - {e}")
        raise
    except Exception as e:
        logger.error(f"Step: File preparation failed - {e}")
        raise

    # Step 2: Run surface reconstruction pipeline
    logger.info("Step: Running surface reconstruction pipeline")
    try:
        # Get thread count from config
        threads = input.config.get("processing", {}).get("threads", 1)

        # Create configuration using defaults from YAML, only override non-default values
        recon_config = ReconSurfConfig.with_defaults(
            subject_id=subject_id,
            subjects_dir=str(subjects_dir),
            atlas={"name": atlas_name},
            processing={
                "threads": threads,
                "skip_cc": True,
                "skip_talairach": True,
            },  # Default for macaque
            verbose=1,
        )

        # Run pipeline
        pipeline = ReconSurfPipeline(recon_config)
        pipeline.run()

        logger.info("Step: Surface reconstruction completed successfully")

    except Exception as e:
        logger.error(f"Step: Pipeline execution failed - {e}")
        raise

    return StepOutput(
        output_file=subject_dir,  # Return subject directory as output
        metadata={
            "step": "surface_reconstruction",
            "modality": "anat",
            "subject_id": subject_id,
            "atlas_name": atlas_name,
            "subjects_dir": str(subjects_dir),
        },
    )


def anat_t1wt2wcombined(
    t1w_file: Path,
    t2w_file: Path,
    segmentation_file: Path,
    segmentation_lut_file: Path,
    output_file: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> StepOutput:
    """
    Generate T1wT2wCombined image from T1w, T2w, and segmentation.

    This is a step-level wrapper around generate_t1wt2wcombined() that provides
    standardized input/output for Nextflow integration.

    Args:
        t1w_file: Path to T1w image file
        t2w_file: Path to T2w image file (must be in same space as T1w)
        segmentation_file: Path to segmentation file (e.g., aparc+aseg.orig.nii.gz)
        segmentation_lut_file: Path to segmentation LUT TSV file
        output_file: Path for output combined image
        config: Configuration dictionary
        metadata: Optional metadata dictionary (subject_id, session_id, etc.)

    Returns:
        StepOutput with combined image file and metadata
    """
    if metadata is None:
        metadata = {}

    logger.info("Step: generating T1wT2wCombined image")
    logger.info(f"  T1w: {t1w_file.name}")
    logger.info(f"  T2w: {t2w_file.name}")
    logger.info(f"  Segmentation: {segmentation_file.name}")

    # Call the preprocessing function
    result = generate_t1wt2wcombined(
        t1w_file=t1w_file,
        t2w_file=t2w_file,
        segmentation_file=segmentation_file,
        segmentation_lut_file=segmentation_lut_file,
        output_file=output_file,
        logger=logger,
    )

    output_path = Path(result["combined_image"])

    return StepOutput(
        output_file=output_path,
        metadata={
            "step": "t1wt2w_combined",
            "modality": "anat",
            "t1w_file": str(t1w_file),
            "t2w_file": str(t2w_file),
            "segmentation_file": str(segmentation_file),
            "t1w_gm_intensity": result.get("t1w_gm_intensity"),
            "t2w_gm_intensity": result.get("t2w_gm_intensity"),
            **metadata,
        },
    )
