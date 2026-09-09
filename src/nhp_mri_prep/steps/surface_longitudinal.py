"""
Longitudinal surface reconstruction steps.

Implements the two phases that ``anat.synthesis_level: "session_longitudinal"``
adds on top of the ordinary per-session reconstruction:

1. **Within-subject base template** -- the sessions' normalized volumes are
   registered into a common unbiased space with ``mri_robust_template`` (robust
   rigid registration, median averaging), the average is segmented, and a full
   reconstruction is run on it. The result is one mesh for the subject.
2. **Longitudinal per timepoint** -- each session's volume is resampled into
   base space and its reconstruction is seeded from the base's surfaces, so all
   timepoints share the base's vertex numbering while surface placement still
   uses that session's own intensities.

Why ``mri_robust_template`` and nothing else from FreeSurfer's longitudinal
stream: it is pure robust rigid registration plus averaging, with no atlas
priors and no species assumptions, so it transfers to macaque data unchanged.
``recon-all -base``/``-long`` would run a human GCA volume stream and clobber
the CNN segmentation the surfaces are built on.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import nibabel as nib

from fastsurfer_surfrecon.wrappers.longitudinal import (
    RCA_BASE_INIT_SAT,
    make_upright,
    mri_concatenate_lta,
    mri_convert_apply_lta,
    mri_robust_template,
)

from .types import StepInput, StepOutput

logger = logging.getLogger(__name__)

# Volumes each cross-sectional timepoint contributes to the base build.
# orig  -> the base's own orig.mgz
# norm  -> solves the rigid transforms (skull-stripped, so registration is driven
#          by brain tissue rather than by neck and head-post)
# mask  -> a binary mask template, kept for QC
#
# Deliberately not brainmask.mgz, which rca-base-init averages as though it were
# a binary mask: in brainana it is a *masked intensity volume*, and with the
# default get_t1=false it is literally a symlink to norm.mgz. mask.mgz is the
# real binary CNN mask.
BASE_INPUT_VOLUMES = ("orig.mgz", "norm.mgz", "mask.mgz")

BASE_INPUTS_DIRNAME = "base_inputs"


def collect_base_inputs(
    subject_dir: Path,
    out_dir: Path,
    subject_label: str,
    volumes: Sequence[str] = BASE_INPUT_VOLUMES,
) -> List[Path]:
    """Copy the volumes a base build needs out of a finished subject directory.

    Copied rather than linked, and with symlinks resolved, because these are
    staged into another task as individual files: a relative symlink to a
    sibling (which is what ``T1.mgz`` and ``brainmask.mgz`` are under the
    default ``get_t1: false``) would arrive dangling.

    Args:
        subject_dir: A finished FreeSurfer-style subject directory.
        out_dir: Directory to collect into. A per-subject subdirectory is
            created inside it so several timepoints can be staged together
            without colliding.
        subject_label: Name of that subdirectory, i.e. the timepoint's id.
        volumes: Filenames to copy from ``subject_dir/mri``.

    Returns:
        The copied file paths.

    Raises:
        FileNotFoundError: If any requested volume is missing. Failing here is
            better than emitting a partial set, because a base silently built
            from fewer volumes than intended is not detectable downstream.
    """
    subject_dir = Path(subject_dir)
    dest = Path(out_dir) / subject_label
    dest.mkdir(parents=True, exist_ok=True)

    copied = []
    for name in volumes:
        src = subject_dir / "mri" / name
        if not src.exists():
            raise FileNotFoundError(
                f"Cannot collect base inputs for {subject_label}: "
                f"{name} not found at {src}"
            )
        dst = dest / name
        shutil.copy2(src, dst, follow_symlinks=True)
        if not dst.is_file() or dst.stat().st_size == 0:
            raise RuntimeError(
                f"Collected {name} for {subject_label} is empty or not a "
                f"regular file: {dst}"
            )
        copied.append(dst)

    logger.info("Collected %d base input volume(s) for %s", len(copied), subject_label)
    return copied


def _geometry(path: Path) -> tuple:
    img = nib.load(str(path))
    zooms = tuple(round(float(z), 4) for z in img.header.get_zooms()[:3])
    return tuple(img.shape[:3]), zooms


def assert_consistent_geometry(volumes: Dict[str, Path]) -> tuple:
    """All timepoints must share one voxel grid before averaging.

    ``mri_robust_template`` produces a well-defined conformed output only when
    its inputs agree, and this is the one violation that would not announce
    itself: the base would be built on some arbitrary grid, the transforms would
    target it, and every timepoint's surfaces would be quietly misaligned.

    Under the default ``anat.conform.enabled: true`` this holds by construction,
    because every session is registered onto the same template-derived grid --
    which is also why that is the first thing to check when it fails.

    Args:
        volumes: Mapping of timepoint id to a volume from that timepoint.

    Returns:
        The single (shape, zooms) shared by all inputs.

    Raises:
        ValueError: If the geometries disagree.
    """
    geoms = {tp: _geometry(path) for tp, path in volumes.items()}
    distinct = set(geoms.values())
    if len(distinct) > 1:
        detail = "\n  ".join(
            f"{tp}: shape={g[0]} zooms={g[1]}" for tp, g in sorted(geoms.items())
        )
        raise ValueError(
            "Timepoints do not share a voxel grid, so an unbiased base template "
            "cannot be built from them:\n  "
            + detail
            + "\n\nFirst thing to check: anat.conform.enabled should be true, "
            "which puts every session on the same template-derived grid "
            "regardless of its acquired field of view. Otherwise the sessions "
            "were acquired at different resolutions and need resampling to a "
            "common grid first."
        )
    return next(iter(distinct))


def build_base_template(
    input: StepInput,
    timepoint_dirs: Dict[str, Path],
    base_subject_id: str,
    expected_timepoints: Optional[int] = None,
    iscale: bool = False,
    subsample: Optional[int] = None,
    arm6_atlas: Optional[Path] = None,
) -> StepOutput:
    """Build and reconstruct a subject's unbiased within-subject template.

    Args:
        input: StepInput carrying config, working_dir and metadata.
        timepoint_dirs: Mapping of cross-sectional timepoint id to the directory
            holding that timepoint's collected base-input volumes.
        base_subject_id: Directory name for the base, e.g. ``sub-01_base``.
        expected_timepoints: How many timepoints should have arrived. When this
            disagrees with what did, the base is built anyway but the shortfall
            is recorded and warned about -- surface reconstruction runs with
            ``errorStrategy 'ignore'``, so a failed session silently contributes
            nothing, and a longitudinal analysis must not quietly rest on a
            truncated set.
        iscale: Allow intensity scaling between timepoints. Off by default, to
            match ``rca-base-init``.
        subsample: Subsample threshold for large volumes.
        arm6_atlas: Optional ARM6 atlas *in base space*, for the claustrum fix
            and thin-WM enhancement. None today -- see the note at the
            reconstruction call for what that costs.

    Returns:
        StepOutput whose output_file is the base subject directory.
    """
    from ..operations.preprocessing import apply_segmentation
    from .anatomical import anat_surface_reconstruction

    if not timepoint_dirs:
        raise ValueError("build_base_template needs at least one timepoint")

    timepoints = sorted(timepoint_dirs)
    working_dir = Path(input.working_dir)
    subjects_dir = working_dir / "fastsurfer"
    base_dir = subjects_dir / base_subject_id
    base_mri = base_dir / "mri"
    base_transforms = base_mri / "transforms"
    base_scripts = base_dir / "scripts"
    for d in (base_mri, base_transforms, base_scripts):
        d.mkdir(parents=True, exist_ok=True)

    log_file = base_scripts / "long_base.log"

    shortfall = None
    if expected_timepoints is not None and len(timepoints) != expected_timepoints:
        shortfall = (
            f"expected {expected_timepoints} timepoint(s) but received "
            f"{len(timepoints)}: {', '.join(timepoints)}"
        )
        logger.warning(
            "Base template for %s is being built from an incomplete set (%s). "
            "Surface reconstruction ignores per-session failures, so check the "
            "cross-sectional logs for the missing session(s).",
            base_subject_id,
            shortfall,
        )

    def vol(tp: str, name: str) -> Path:
        path = Path(timepoint_dirs[tp]) / name
        if not path.exists():
            raise FileNotFoundError(
                f"Timepoint {tp} is missing {name} (looked in {path})"
            )
        return path

    # Grids must agree before anything is averaged.
    grid = assert_consistent_geometry({tp: vol(tp, "orig.mgz") for tp in timepoints})
    logger.info("Timepoints share grid shape=%s zooms=%s", grid[0], grid[1])

    base_orig = base_mri / "orig.mgz"
    # mri_robust_template writes a float average. orig.mgz must be uchar to
    # match a cross-sectional orig, so that surface reconstruction's conform
    # check passes and it does not re-save (and possibly re-grid) the volume.
    # longmc converts for the same reason.
    base_orig_float = base_mri / "orig_robusttemplate.mgz"
    norm_template = base_mri / "norm_template.mgz"
    ltas = [base_transforms / f"{tp}_to_{base_subject_id}.lta" for tp in timepoints]

    if len(timepoints) == 1:
        # Nothing to average. FreeSurfer's base stream uses make_upright here so
        # a single-session subject still traverses the same code path.
        tp = timepoints[0]
        logger.info("Single timepoint (%s): using make_upright for the base", tp)
        make_upright(
            vol(tp, "norm.mgz"), norm_template, ltas[0], log_file=log_file
        )
        shutil.copy2(vol(tp, "orig.mgz"), base_orig, follow_symlinks=True)
        shutil.copy2(vol(tp, "mask.mgz"), base_mri / "mask_template.mgz")
    else:
        # Pass 1: solve the rigid transforms on the skull-stripped volumes, so
        # registration is driven by brain tissue.
        mri_robust_template(
            movs=[vol(tp, "norm.mgz") for tp in timepoints],
            template=norm_template,
            ltas=ltas,
            average=1,  # median
            sat=RCA_BASE_INIT_SAT,
            iscale=iscale,
            subsample=subsample,
            # Spatial init is random by default, which would make the base
            # differ between runs and break -resume reproducibility.
            inittp=1,
            log_file=log_file,
        )
        # Pass 2: apply those transforms to the full-head volumes to get the
        # base's own orig.mgz. --noit means "resample and average, do not
        # re-solve", so the two passes stay consistent.
        mri_robust_template(
            movs=[vol(tp, "orig.mgz") for tp in timepoints],
            template=base_orig_float,
            ixforms=ltas,
            average=1,
            noit=True,
            sat=None,
            log_file=log_file,
        )
        mri_convert_apply_lta(
            base_orig_float, base_orig, lta=None, odt="uchar", log_file=log_file
        )
        # Pass 3: a binary mask template, for QC only. Nearest-neighbour and a
        # mean rather than a median, since this is a label volume.
        mri_robust_template(
            movs=[vol(tp, "mask.mgz") for tp in timepoints],
            template=base_mri / "mask_template.mgz",
            ixforms=ltas,
            average=0,
            noit=True,
            finalnearest=True,
            sat=None,
            log_file=log_file,
        )

    # Inverse transforms, for anyone needing to go base -> timepoint.
    for tp, lta in zip(timepoints, ltas):
        mri_concatenate_lta(
            lta,
            "identity.nofile",
            base_transforms / f"{base_subject_id}_to_{tp}.lta",
            invert1=True,
            log_file=log_file,
        )

    # FreeSurfer's own filename for the timepoint list.
    (base_scripts / "base-tps").write_text("\n".join(timepoints) + "\n")

    # The base grid must not move from here on: the transforms above target it,
    # and postprocess_for_freesurfer below re-saves orig.mgz. If a re-conform
    # changed the grid, every timepoint would be silently misaligned, so this is
    # a hard check rather than a warning.
    from fastsurfer_nn.data_loader.conform import is_conform

    vox_size = min(grid[1])
    if not is_conform(
        nib.load(str(base_orig)),
        vox_size=vox_size,
        orientation="lia",
        img_size="cube",
        # dtype is checked separately by the conversion above; what must hold
        # here is the *geometry*, since that is what the transforms target.
        dtype=None,
        verbose=False,
    ):
        raise RuntimeError(
            f"Base template {base_orig} is not conformed at {vox_size} mm "
            "(LIA, cube). Surface reconstruction would re-conform it and the "
            "timepoint-to-base transforms just written would no longer target "
            "the base's grid, misaligning every timepoint. This means the "
            "cross-sectional inputs were not conformed identically."
        )

    provenance: Dict[str, Any] = {
        "base_subject_id": base_subject_id,
        "timepoints": timepoints,
        "expected_timepoints": expected_timepoints,
        "incomplete": shortfall,
        "grid": {"shape": list(grid[0]), "zooms": list(grid[1])},
        "iscale": iscale,
        "sat": RCA_BASE_INIT_SAT,
        "average": "median",
        "ltas": {tp: str(lta) for tp, lta in zip(timepoints, ltas)},
    }
    (base_scripts / "long_base.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )

    # Segment the average itself, rather than mapping one timepoint's labels in.
    # The base is a median of N rigidly aligned scans of the same subject, so its
    # SNR beats any single session's. Mapping a timepoint's segmentation through
    # its transform would instead pick an arbitrary session and bake its errors
    # into every other timepoint -- the asymmetry the unbiased base exists to
    # remove -- and nearest-neighbour label resampling erodes exactly the thin
    # white matter that fix_V1_WM and the claustrum fix exist to repair.
    seg_work = working_dir / "base_segmentation"
    seg_work.mkdir(parents=True, exist_ok=True)
    base_nii = seg_work / f"{base_subject_id}_T1w.nii.gz"

    from fastsurfer_surfrecon.wrappers.mri import mri_convert

    mri_convert(base_orig, base_nii, log_file=log_file)

    logger.info("Segmenting base template %s", base_nii)
    seg_result = apply_segmentation(
        imagef=base_nii,
        modal="anat",
        working_dir=seg_work,
        output_name=f"{base_subject_id}_desc-brain_T1w.nii.gz",
        config=input.config,
    )

    # Reconstruct the base like any other subject. session_id="" with
    # session_count=1 makes the naming logic yield exactly base_subject_id.
    recon_input = StepInput(
        input_file=base_nii,
        working_dir=working_dir,
        config=input.config,
        output_name="surface_base_template",
        metadata={
            "subject_id": base_subject_id,
            "session_id": "",
            "session_count": 1,
        },
    )
    # ARM6 is not produced by segmentation -- it comes from a separate
    # template-atlas backprojection, in *session* space, so there is no
    # base-space ARM6 to hand over here. Consequence: the claustrum fix (s07b)
    # and the ARM6 thin-WM enhancement are skipped for the base, and therefore
    # for every timepoint seeded from it, while cross-sectional runs still get
    # them. That keeps base and timepoints mutually consistent, which is what
    # matters for comparing timepoints, but it does mean longitudinal surfaces
    # carry a systematic offset relative to the cross-sectional ones -- worth
    # remembering when comparing the two.
    if arm6_atlas is not None:
        logger.info("Step: base ARM6 atlas supplied: %s", arm6_atlas)
    else:
        logger.info(
            "Step: no base-space ARM6 atlas; claustrum fix and ARM6 WM "
            "enhancement will be skipped for the base and all its timepoints"
        )

    recon = anat_surface_reconstruction(
        recon_input,
        t1w_file=base_nii,
        segmentation_file=Path(seg_result["segmentation"]),
        brain_mask=Path(seg_result["brain_mask"]),
        arm6_atlas=arm6_atlas,
    )

    if recon.output_file.resolve() != base_dir.resolve():
        raise RuntimeError(
            f"Base reconstruction landed in {recon.output_file}, expected "
            f"{base_dir}. The subject-id naming logic and base_subject_id have "
            "diverged; downstream stages locate the base by name."
        )

    return StepOutput(
        output_file=base_dir,
        metadata={
            "step": "surface_base_template",
            "modality": "anat",
            "base_subject_id": base_subject_id,
            "timepoints": timepoints,
            "incomplete": shortfall,
            "subjects_dir": str(subjects_dir),
        },
        additional_files={
            f"lta_{tp}": lta for tp, lta in zip(timepoints, ltas)
        },
    )


def run_long_timepoint(
    input: StepInput,
    cross_subject_id: str,
    base_subject_id: str,
    tp_to_base_lta: Path,
    long_subject_id: Optional[str] = None,
) -> StepOutput:
    """Reconstruct one timepoint seeded from its subject's base template.

    The subject directory must already contain the base and the timepoint's
    cross-sectional reconstruction; stage 00 does the seeding and stages 08-12
    disable themselves, so this is an ordinary pipeline run with a longitudinal
    config.

    Args:
        input: StepInput carrying config, working_dir and metadata.
        cross_subject_id: The timepoint's cross-sectional directory name.
        base_subject_id: The base template's directory name.
        tp_to_base_lta: Transform from the timepoint's space into base space.
        long_subject_id: Output directory name. Defaults to
            ``<cross_subject_id>_long``.

    Returns:
        StepOutput whose output_file is the longitudinal subject directory.
    """
    from fastsurfer_surfrecon.config import ReconSurfConfig
    from fastsurfer_surfrecon.pipeline import ReconSurfPipeline

    subjects_dir = Path(input.working_dir) / "fastsurfer"
    long_subject_id = long_subject_id or f"{cross_subject_id}_long"

    atlas_name = (
        input.config.get("anat", {})
        .get("skullstripping_segmentation", {})
        .get("atlas_name", "ARM2")
    )
    threads = input.config.get("processing", {}).get("threads", 1)

    logger.info(
        "Longitudinal reconstruction of %s from base %s",
        long_subject_id,
        base_subject_id,
    )

    # Not routed through anat_surface_reconstruction: its
    # postprocess_for_freesurfer call would overwrite the base-space orig.mgz
    # that stage 00 resampled, and replace the inherited segmentation with a
    # freshly conformed native-space one.
    recon_config = ReconSurfConfig.with_defaults(
        subject_id=long_subject_id,
        subjects_dir=str(subjects_dir),
        atlas={"name": atlas_name},
        processing={
            "threads": threads,
            "skip_cc": True,
            "skip_talairach": True,
        },
        verbose=1,
        longitudinal=True,
        base_subject_id=base_subject_id,
        cross_subject_id=cross_subject_id,
        tp_to_base_lta=Path(tp_to_base_lta),
    )
    ReconSurfPipeline(recon_config).run()

    return StepOutput(
        output_file=subjects_dir / long_subject_id,
        metadata={
            "step": "surface_reconstruction_long",
            "modality": "anat",
            "subject_id": long_subject_id,
            "cross_subject_id": cross_subject_id,
            "base_subject_id": base_subject_id,
            "atlas_name": atlas_name,
            "subjects_dir": str(subjects_dir),
        },
    )
