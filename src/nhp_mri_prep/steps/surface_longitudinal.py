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
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import nibabel as nib

from fastsurfer_surfrecon.wrappers.longitudinal import (
    RCA_BASE_INIT_SAT,
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
BASE_INPUT_VOLUMES = (
    "orig.mgz",
    "norm.mgz",
    "mask.mgz",
    # Not used to build the base -- collected so the base's own segmentation can
    # be checked against a consensus of the sessions'. See fuse_timepoint_asegs.
    "aseg.auto_noCCseg.mgz",
)

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


def _affine(path: Path):
    """The full voxel-to-world mapping, for checks that a grid did not move.

    Shape and zooms alone would accept a translation or an axis flip, which is
    precisely what "the grid moved" means for a transform that targets it.
    """
    import numpy as np

    return np.asarray(nib.load(str(path)).affine, dtype=float)


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


def map_timepoint_asegs_to_base(
    aseg_by_timepoint: Dict[str, Path],
    ltas_by_timepoint: Dict[str, Path],
    out_dir: Path,
    log_file: Optional[Path] = None,
) -> Dict[str, Path]:
    """Resample each timepoint's segmentation into base space, nearest-neighbour.

    Purely diagnostic. The base's own segmentation comes from running the CNN on
    the robust average, which is the right choice -- the average has better SNR
    than any single session, and resampling one session's labels in would bake
    that session's errors and its arbitrary selection into every timepoint. But
    the CNN sees a different kind of volume there than it does cross-sectionally:
    a median of per-session, already bias-corrected, already robustly-rescaled
    uchar volumes. That is a domain shift, and whether it matters is an empirical
    question. These give something to compare the base against, so the question
    is answerable from the run's own outputs.

    Deliberately *not* fused into a single consensus volume. A majority vote needs
    a majority, and two timepoints is the accepted minimum -- every disagreeing
    voxel would then be a 1-1 tie broken by whatever the implementation happened
    to prefer, which is a bias dressed up as a consensus. Comparing against each
    timepoint separately needs no tie-break and additionally shows the spread.

    Args:
        aseg_by_timepoint: Each timepoint's conformed segmentation.
        ltas_by_timepoint: The matching timepoint-to-base transforms.
        out_dir: Directory for the resampled volumes.
        log_file: Log file path.

    Returns:
        Mapping of timepoint id to its segmentation in base space. Timepoints that
        could not be mapped are omitted rather than raising -- this is a
        diagnostic and must never cost the run its base.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapped: Dict[str, Path] = {}
    for tp in sorted(set(aseg_by_timepoint) & set(ltas_by_timepoint)):
        dst = out_dir / f"{tp}_aseg_in_base.mgz"
        try:
            mri_convert_apply_lta(
                aseg_by_timepoint[tp],
                dst,
                lta=ltas_by_timepoint[tp],
                # Labels: nearest neighbour, and no dtype coercion.
                odt=None,
                resample="nearest",
                log_file=log_file,
            )
        except Exception as exc:
            logger.warning(
                "Could not map %s's segmentation into base space: %s", tp, exc
            )
            continue
        mapped[tp] = dst
    logger.info(
        "Mapped %d of %d timepoint segmentations into base space",
        len(mapped),
        len(aseg_by_timepoint),
    )
    return mapped


def label_dice(a: Path, b: Path, min_voxels: int = 50) -> Dict[str, float]:
    """Per-label Dice between two label volumes on the same grid.

    Labels smaller than ``min_voxels`` in both volumes are skipped: Dice on a
    handful of voxels is dominated by single-voxel differences and would make the
    summary look alarming for no reason.
    """
    import numpy as np

    da = np.asanyarray(nib.load(str(a)).dataobj)
    db = np.asanyarray(nib.load(str(b)).dataobj)
    if da.shape != db.shape:
        raise ValueError(f"Label volumes differ in shape: {da.shape} vs {db.shape}")

    out: Dict[str, float] = {}
    for label in sorted(set(np.unique(da)) | set(np.unique(db))):
        if label == 0:
            continue
        ma, mb = da == label, db == label
        na, nb = int(ma.sum()), int(mb.sum())
        if na < min_voxels and nb < min_voxels:
            continue
        denom = na + nb
        out[str(int(label))] = (
            float(2.0 * int((ma & mb).sum()) / denom) if denom else 0.0
        )
    return out


def build_base_template(
    input: StepInput,
    timepoint_dirs: Dict[str, Path],
    base_subject_id: str,
    expected_timepoints: Optional[int] = None,
    iscale: bool = False,
    subsample: Optional[int] = None,
) -> StepOutput:
    """Build a subject's unbiased within-subject template volume.

    First of the three base stages. This one is CPU-only and does no
    segmentation and no reconstruction, so it can be sized and scheduled on its
    own: `mri_robust_template` needs memory proportional to timepoint count x
    cube size, while the reconstruction that follows needs a long wall clock and
    the segmentation between them needs a GPU. Keeping them separate also stops
    a multi-hour reconstruction from holding a GPU token.

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
        iscale: Allow intensity scaling between timepoints. Off by default to
            match ``rca-base-init``, but reachable because each session's
            ``orig.mgz`` was rescaled by its *own* robust factor during conform,
            so the volumes being averaged are not on a common intensity scale.
        subsample: Subsample threshold for large volumes.

    Returns:
        StepOutput with output_file=base subject directory and
        additional_files containing ``base_nii`` (the template as NIfTI, for the
        segmentation stage) and one ``lta_<timepoint>`` per timepoint.
    """
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

    if len(timepoints) < 2:
        # A single timepoint has nothing to average, and an "unbiased template"
        # of one scan is just that scan. FreeSurfer's base stream supports it
        # (via make_upright) only so that one-session subjects appear in
        # group-level tables; here it would add an interpolation for no gain,
        # and the emitted transform and the base volume have to stay mutually
        # consistent or every timepoint is resampled onto the wrong grid.
        raise ValueError(
            f"A within-subject base template needs at least 2 timepoints; got "
            f"{len(timepoints)} ({', '.join(timepoints)}). Single-session "
            "subjects are skipped by the workflow for this reason."
        )

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
    mask_template = base_mri / "mask_template.mgz"
    ltas = [base_transforms / f"{tp}_to_{base_subject_id}.lta" for tp in timepoints]
    # Intensity scales, when asked for. Pass 1 solves them and pass 2 must reuse
    # them: pass 2 runs --noit, so without --iscalein it averages on the original
    # unequal scales and the knob would have no effect on the base at all.
    iscale_files = (
        [base_transforms / f"{tp}_to_{base_subject_id}.iscale.txt" for tp in timepoints]
        if iscale
        else None
    )

    # Pass 1: solve the rigid transforms on the skull-stripped volumes, so
    # registration is driven by brain tissue.
    mri_robust_template(
        movs=[vol(tp, "norm.mgz") for tp in timepoints],
        template=norm_template,
        ltas=ltas,
        average=1,  # median
        sat=RCA_BASE_INIT_SAT,
        iscale=iscale,
        iscaleout=iscale_files,
        subsample=subsample,
        # Spatial init is random by default, which would make the base differ
        # between runs and break -resume reproducibility.
        inittp=1,
        log_file=log_file,
    )
    # Pass 2: apply those transforms to the full-head volumes to get the base's
    # own orig.mgz. --noit means "resample and average, do not re-solve", so the
    # two passes stay consistent.
    mri_robust_template(
        movs=[vol(tp, "orig.mgz") for tp in timepoints],
        template=base_orig_float,
        ixforms=ltas,
        iscalein=iscale_files,
        average=1,
        noit=True,
        sat=None,
        log_file=log_file,
    )
    mri_convert_apply_lta(
        base_orig_float, base_orig, lta=None, odt="uchar", log_file=log_file
    )
    # Pass 3: a binary mask template. Nearest-neighbour and a mean rather than a
    # median, since this is a label volume. Diagnostic only -- the reconstruction
    # takes its mask from the base's own segmentation. Kept because when a base
    # looks wrong, comparing this consensus of the sessions' masks against the
    # base's freshly computed one is the quickest way to tell a bad average from
    # a bad segmentation.
    mri_robust_template(
        movs=[vol(tp, "mask.mgz") for tp in timepoints],
        template=mask_template,
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

    # The sessions' own segmentations, in base space, for the reconstruction
    # stage to compare the base's freshly computed one against. See
    # map_timepoint_asegs_to_base for why this is worth recording.
    mapped_asegs: Dict[str, Path] = {}
    try:
        mapped_asegs = map_timepoint_asegs_to_base(
            aseg_by_timepoint={
                tp: vol(tp, "aseg.auto_noCCseg.mgz") for tp in timepoints
            },
            ltas_by_timepoint=dict(zip(timepoints, ltas)),
            # Inside the base tree so it travels to the reconstruction stage
            # without extra channel plumbing.
            out_dir=base_mri / "timepoint_asegs",
            log_file=log_file,
        )
    except Exception as exc:
        logger.warning("Timepoint segmentations could not be mapped: %s", exc)

    # The base grid must not move from here on: the transforms above target it,
    # and postprocess_for_freesurfer in the reconstruction stage re-saves
    # orig.mgz. If a re-conform changed the grid, every timepoint would be
    # silently misaligned, so this is a hard check rather than a warning.
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
            "cross-sectional inputs were not conformed identically -- check "
            "anat.conform.enabled, which is what normally guarantees it."
        )

    # A NIfTI copy for the segmentation and registration stage, which works in
    # BIDS-ish space rather than on .mgz.
    base_nii = base_dir / "mri" / f"{base_subject_id}_T1w.nii.gz"
    from fastsurfer_surfrecon.wrappers.mri import mri_convert

    mri_convert(base_orig, base_nii, log_file=log_file)

    # Record the grid *and the full affine* the transforms target, so the
    # reconstruction stage -- which runs in a different task with a different
    # working directory -- can prove nothing moved. Shape and zooms alone would
    # accept a translation or an axis flip.
    provenance: Dict[str, Any] = {
        "base_subject_id": base_subject_id,
        "timepoints": timepoints,
        "expected_timepoints": expected_timepoints,
        "incomplete": shortfall,
        "grid": {"shape": list(grid[0]), "zooms": list(grid[1])},
        "affine": [[float(x) for x in row] for row in _affine(base_orig)],
        "iscale": iscale,
        "subsample": subsample,
        "sat": RCA_BASE_INIT_SAT,
        "average": "median",
        "ltas": {tp: str(lta) for tp, lta in zip(timepoints, ltas)},
        "timepoint_asegs_in_base": {
            tp: str(path) for tp, path in sorted(mapped_asegs.items())
        },
    }
    (base_scripts / "long_base.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )

    additional: Dict[str, Path] = {
        f"lta_{tp}": lta for tp, lta in zip(timepoints, ltas)
    }
    additional["base_nii"] = base_nii
    additional["norm_template"] = norm_template
    additional["mask_template"] = mask_template

    return StepOutput(
        output_file=base_dir,
        metadata={
            "step": "surface_base_template",
            "modality": "anat",
            "subject_id": base_subject_id,
            "base_subject_id": base_subject_id,
            "timepoints": timepoints,
            "incomplete": shortfall,
            "subjects_dir": str(subjects_dir),
        },
        additional_files=additional,
    )


def segment_and_backproject_base(
    input: StepInput,
    base_nii: Path,
    base_subject_id: str,
    bids_name: Optional[str] = None,
) -> StepOutput:
    """Segment the base template and backproject the template atlases onto it.

    Second of the three base stages, and the only one that needs a GPU: the
    fastSurferCNN segmentation and the FireANTs template registration.

    Segmenting the average itself, rather than mapping one timepoint's labels in,
    is deliberate: the base is a median of N rigidly aligned scans of the same
    subject, so its SNR beats any single session's, whereas resampling one
    timepoint's segmentation would pick an arbitrary session and bake its errors
    into every other timepoint -- the asymmetry the unbiased base exists to
    remove -- and nearest-neighbour label resampling erodes exactly the thin
    white matter that fix_V1_WM and the claustrum fix exist to repair.

    The atlas backprojection is what closes the largest parity gap. ARM6 never
    comes from segmentation; it is a template atlas resampled through the inverse
    anat->template transform, exactly as ANAT_BACKPROJECT_ATLASES_TO_T1W does per
    session. Without it the base loses *two* things, and so does every timepoint
    seeded from it: the claustrum fix (s07b disables itself when
    aparc.ARM6atlas+aseg.orig.mgz is absent) and -- the larger effect -- the ARM2
    thin-WM enhancement, which propagates through aseg.presurf -> wm.mgz ->
    filled.mgz and thins the white-matter compartment in precisely the
    orbitofrontal and lateral-prefrontal regions where partial-volume thin WM
    makes white-surface placement fail.

    Args:
        input: StepInput carrying config, working_dir and metadata.
        base_nii: The base template as NIfTI, from ``build_base_template``.
        base_subject_id: The base's directory name.
        bids_name: BIDS stem the published derivative and atlas filenames are
            derived from. Defaults to ``<base_subject_id>_T1w.nii.gz``, but
            ``sub-X_base`` is not a valid entity chain, so callers should pass
            something canonical such as ``sub-X_acq-base_T1w.nii.gz``.

    Returns:
        StepOutput with output_file=the skull-stripped base and additional_files
        containing ``segmentation``, ``brain_mask`` and, when available,
        ``arm6_atlas``, ``hemimask``, ``atlas_lut`` and every backprojected
        atlas under its atlas name.
    """
    from ..operations.preprocessing import apply_segmentation
    from ..utils.templates import resolve_template
    from .anatomical import anat_backproject_atlases, anat_registration

    base_nii = Path(base_nii)
    working_dir = Path(input.working_dir)
    seg_work = working_dir / "base_segmentation"
    seg_work.mkdir(parents=True, exist_ok=True)

    anat_cfg = input.config.get("anat", {})
    if anat_cfg.get("surface_reconstruction", {}).get("use_t1wt2wcombined"):
        # Cross-sectionally the CNN is always run on the plain T1w
        # (ANAT_SKULLSTRIPPING takes anat_after_conform), never on the combined
        # image. The base's orig.mgz is an average of whatever fed surf recon, so
        # with this enabled the CNN sees a contrast it never sees otherwise.
        logger.warning(
            "anat.surface_reconstruction.use_t1wt2wcombined is enabled, so the "
            "base template is an average of T1w/T2w-combined volumes and its "
            "segmentation runs on that combined contrast. Cross-sectional "
            "segmentation always uses the plain T1w, so check the base's "
            "segmentation QC before trusting the surfaces."
        )

    bids_name = bids_name or f"{base_subject_id}_T1w.nii.gz"
    bids_stem = bids_name.replace(".nii.gz", "").replace("_T1w", "").replace("_T2w", "")

    logger.info("Step: segmenting base template %s", base_nii)
    seg_result = apply_segmentation(
        imagef=base_nii,
        modal="anat",
        working_dir=seg_work,
        output_name=f"{bids_stem}_desc-brain_T1w.nii.gz",
        config=input.config,
        logger=logger,
    )

    if seg_result.get("input_cropped"):
        # apply_segmentation can return a cropped input whose grid differs from
        # base_nii, in which case the mask belongs to the cropped image, not to
        # the base. Unreachable today (enable_crop_2round is hard-coded off) but
        # it would silently mis-place the mask, so refuse rather than guess.
        raise RuntimeError(
            "apply_segmentation returned input_cropped "
            f"({seg_result['input_cropped']}) for the base template. Its mask "
            "and segmentation are on the cropped grid, not the base's, so they "
            "cannot be used for the base reconstruction without re-anchoring."
        )

    missing = [
        k
        for k in ("segmentation", "brain_mask", "imagef_skullstripped")
        if not seg_result.get(k)
    ]
    if missing:
        # apply_segmentation returns segmentation only when the model produced
        # one, so a bare subscript here would surface as an opaque KeyError from
        # inside the GPU stage.
        raise RuntimeError(
            f"Segmentation of the base template did not produce {missing}. "
            "Surface reconstruction needs all three; check whether "
            "anat.skullstripping_segmentation is configured for a multi-class "
            "atlas model."
        )

    additional: Dict[str, Path] = {
        "segmentation": Path(seg_result["segmentation"]),
        "brain_mask": Path(seg_result["brain_mask"]),
        "imagef_skullstripped": Path(seg_result["imagef_skullstripped"]),
    }
    for optional in ("hemimask", "atlas_lut"):
        if seg_result.get(optional):
            additional[optional] = Path(seg_result[optional])

    atlas_name = seg_result.get("atlas_name", "ARM2")

    # --- Template atlases onto the base grid -------------------------------
    output_space = input.config.get("template", {}).get(
        "output_space", "NMT2Sym:res-05"
    )
    # space_label_for, not split(":"): a custom-template output_space is a file
    # path, and splitting it would leak an absolute path into filenames and
    # metadata. Every other caller in the repo uses the helper.
    from ..utils.templates import space_label_for

    template_name = space_label_for(output_space)
    arm6_atlas: Optional[Path] = None
    atlas_dir: Optional[Path] = None

    try:
        template_file = resolve_template(output_space)
    except Exception as exc:
        template_file = None
        logger.warning(
            "Could not resolve template %s (%s); the base gets no atlases, so "
            "the claustrum fix and ARM6 white-matter enhancement will be "
            "skipped for it and for every timepoint seeded from it.",
            output_space,
            exc,
        )

    if template_file is not None:
        reg_work = working_dir / "base_registration"
        reg_work.mkdir(parents=True, exist_ok=True)
        # Skull-stripped moving image, matching ANAT_REGISTRATION.
        reg = anat_registration(
            StepInput(
                input_file=additional["imagef_skullstripped"],
                working_dir=reg_work,
                config=input.config,
                output_name=f"{bids_stem}_space-{template_name}_T1w.nii.gz",
                metadata={"subject_id": base_subject_id, "session_id": ""},
            ),
            template_file=Path(template_file),
            template_name=template_name,
        )
        inverse_xfm = reg.additional_files.get("inverse_transform")
        if inverse_xfm is None:
            logger.warning(
                "Registration of the base to %s produced no inverse transform "
                "(registration disabled?); no atlases will be backprojected.",
                template_name,
            )
        else:
            atlas_work = working_dir / "base_atlases"
            atlas_work.mkdir(parents=True, exist_ok=True)
            projected = anat_backproject_atlases(
                inverse_xfm=Path(inverse_xfm),
                # The base grid, so the atlases land world-aligned to the volume
                # postprocess_for_freesurfer will conform.
                t1w_reference=base_nii,
                bids_name=Path(bids_name),
                working_dir=atlas_work,
                config=input.config,
            )
            atlas_dir = projected.output_file
            for name, path in projected.additional_files.items():
                additional[f"atlas_{name}"] = Path(path)
            arm6_atlas = projected.additional_files.get("ARM6")
            if arm6_atlas is None:
                # A custom template has no bundled atlases, so this is expected
                # there and matches the cross-sectional behaviour exactly.
                logger.warning(
                    "No ARM6 among the backprojected atlases for %s; the "
                    "claustrum fix and ARM6 white-matter enhancement will be "
                    "skipped for the base and every timepoint seeded from it.",
                    base_subject_id,
                )
            else:
                additional["arm6_atlas"] = Path(arm6_atlas)

    return StepOutput(
        output_file=additional["imagef_skullstripped"],
        metadata={
            "step": "surface_base_atlas",
            "modality": "anat",
            "subject_id": base_subject_id,
            "base_subject_id": base_subject_id,
            "atlas_name": atlas_name,
            "template": template_name,
            "arm6_atlas": str(arm6_atlas) if arm6_atlas else None,
            "atlas_dir": str(atlas_dir) if atlas_dir else None,
            "bids_name": bids_name,
        },
        additional_files=additional,
    )


def reconstruct_base(
    input: StepInput,
    base_dir: Path,
    base_nii: Path,
    base_subject_id: str,
    segmentation_file: Path,
    brain_mask: Path,
    arm6_atlas: Optional[Path] = None,
) -> StepOutput:
    """Reconstruct the base template's surfaces.

    Third of the three base stages: CPU-only and long-running, so it is sized
    like ANAT_SURFACE_RECONSTRUCTION and holds no GPU token.

    Args:
        input: StepInput carrying config, working_dir and metadata. Its
            working_dir must already contain ``fastsurfer/<base_subject_id>`` as
            produced by ``build_base_template``.
        base_dir: The staged base directory.
        base_nii: The base template as NIfTI.
        base_subject_id: The base's directory name.
        segmentation_file: Segmentation from ``segment_and_backproject_base``.
        brain_mask: Brain mask from the same.
        arm6_atlas: Base-space ARM6, when available. Passed *into*
            postprocess_for_freesurfer rather than written afterwards, because it
            also drives the ARM2 thin-WM enhancement that produces
            aparc+aseg.orig.mgz.

    Returns:
        StepOutput whose output_file is the reconstructed base directory.
    """
    from .anatomical import anat_surface_reconstruction

    base_dir = Path(base_dir)
    working_dir = Path(input.working_dir)
    subjects_dir = working_dir / "fastsurfer"

    if not input.config.get("anat", {}).get("surface_reconstruction", {}).get(
        "enabled", True
    ):
        logger.info("Step: base reconstruction skipped (disabled in configuration)")
        return StepOutput(
            output_file=base_dir,
            metadata={"step": "surface_base_recon", "skipped": True},
        )

    # Read back the grid the timepoint-to-base transforms target. Written by
    # build_base_template in a different task, so it travels in the tree rather
    # than in memory.
    provenance_path = base_dir / "scripts" / "long_base.json"
    if not provenance_path.exists():
        raise FileNotFoundError(
            f"{provenance_path} not found; the base template stage must run "
            "before its reconstruction."
        )
    provenance = json.loads(provenance_path.read_text())
    expected_grid = (
        tuple(provenance["grid"]["shape"]),
        tuple(round(float(z), 4) for z in provenance["grid"]["zooms"]),
    )
    expected_affine = provenance.get("affine")

    if arm6_atlas is not None:
        logger.info("Step: base ARM6 atlas supplied: %s", arm6_atlas)
    else:
        logger.info(
            "Step: no base-space ARM6 atlas; the claustrum fix and the ARM6 "
            "white-matter enhancement will be skipped for the base and for "
            "every timepoint seeded from it"
        )

    # session_id="" with session_count=1 makes the naming logic yield exactly
    # base_subject_id.
    recon = anat_surface_reconstruction(
        StepInput(
            input_file=Path(base_nii),
            working_dir=working_dir,
            config=input.config,
            output_name="surface_base_recon",
            metadata={
                "subject_id": base_subject_id,
                "session_id": "",
                "session_count": 1,
            },
        ),
        t1w_file=Path(base_nii),
        segmentation_file=Path(segmentation_file),
        brain_mask=Path(brain_mask),
        arm6_atlas=Path(arm6_atlas) if arm6_atlas else None,
    )

    if recon.output_file.resolve() != base_dir.resolve():
        raise RuntimeError(
            f"Base reconstruction landed in {recon.output_file}, expected "
            f"{base_dir}. The subject-id naming logic and base_subject_id have "
            "diverged; downstream stages locate the base by name."
        )

    # postprocess_for_freesurfer re-saves orig.mgz unconditionally, via a NIfTI
    # round trip. The transforms target the grid it had before that, so confirm
    # nothing moved -- otherwise every timepoint is resampled onto a grid the
    # base no longer has, and nothing downstream would report it.
    import numpy as np

    base_orig = base_dir / "mri" / "orig.mgz"
    final_geom = _geometry(base_orig)
    final_affine = _affine(base_orig)
    moved = final_geom != expected_grid
    if expected_affine is not None:
        moved = moved or not np.allclose(
            final_affine, np.asarray(expected_affine, dtype=float), atol=1e-4
        )
    if moved:
        raise RuntimeError(
            f"Base template geometry changed during reconstruction: was "
            f"shape={expected_grid[0]} zooms={expected_grid[1]}, now "
            f"shape={final_geom[0]} zooms={final_geom[1]}.\naffine before:\n"
            f"{expected_affine}\naffine after:\n{final_affine}\n"
            "The timepoint-to-base transforms target the original grid, so "
            "every timepoint would be misaligned. This is a bug in the base "
            "build, not a data problem."
        )

    # Segmentation agreement, once postprocess_for_freesurfer has written the
    # base's own conformed aseg. Diagnostic: a low Dice here is the signal that
    # segmenting a robust average shifted the CNN's input domain far enough to
    # matter. Never fatal.
    median_dice: Optional[float] = None
    base_aseg = base_dir / "mri" / "aseg.auto_noCCseg.mgz"
    mapped_dir = base_dir / "mri" / "timepoint_asegs"
    mapped = sorted(mapped_dir.glob("*_aseg_in_base.mgz")) if mapped_dir.is_dir() else []
    if base_aseg.exists() and mapped:
        try:
            per_timepoint: Dict[str, Dict[str, float]] = {}
            medians = []
            for path in mapped:
                tp = path.name.replace("_aseg_in_base.mgz", "")
                dice = label_dice(base_aseg, path)
                if not dice:
                    continue
                per_timepoint[tp] = dice
                values = sorted(dice.values())
                medians.append(values[len(values) // 2])
            if medians:
                median_dice = float(sum(medians) / len(medians))
                logger.info(
                    "Base segmentation vs %d timepoint segmentation(s): mean of "
                    "per-timepoint median Dice %.3f (range %.3f-%.3f)",
                    len(medians),
                    median_dice,
                    min(medians),
                    max(medians),
                )
                (base_dir / "scripts" / "base_segmentation_agreement.json").write_text(
                    json.dumps(
                        {
                            "description": (
                                "Per-label Dice between the base template's own "
                                "segmentation and each timepoint's segmentation "
                                "mapped into base space. Diagnostic for whether "
                                "segmenting a robust average shifted the CNN's "
                                "input domain; low values are the signal. Not "
                                "fused into a consensus on purpose -- with two "
                                "timepoints every disagreement is a tie, and "
                                "breaking it would be a bias dressed up as a "
                                "majority."
                            ),
                            "mean_of_per_timepoint_median_dice": median_dice,
                            "per_timepoint_median_dice": {
                                tp: sorted(d.values())[len(d) // 2]
                                for tp, d in per_timepoint.items()
                            },
                            "per_timepoint_per_label_dice": per_timepoint,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
        except Exception as exc:
            logger.warning("Could not compute segmentation agreement: %s", exc)
    else:
        logger.info(
            "No mapped timepoint segmentations available; skipping the base "
            "segmentation agreement check"
        )

    metadata = dict(recon.metadata)
    metadata.update(
        {
            "step": "surface_base_recon",
            "base_subject_id": base_subject_id,
            "timepoints": provenance.get("timepoints"),
            "incomplete": provenance.get("incomplete"),
            "arm6_atlas": str(arm6_atlas) if arm6_atlas else None,
            "subjects_dir": str(subjects_dir),
            "segmentation_agreement_median_dice": median_dice,
        }
    )
    return StepOutput(output_file=base_dir, metadata=metadata)


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

    anat_cfg = input.config.get("anat", {})
    if not anat_cfg.get("surface_reconstruction", {}).get("enabled", True):
        # anat_surface_reconstruction() makes this check for the cross-sectional
        # path; this function calls ReconSurfPipeline directly, so it has to make
        # it too or the knob would apply to some paths and not others.
        logger.info(
            "Step: longitudinal reconstruction skipped (disabled in configuration)"
        )
        return StepOutput(
            output_file=subjects_dir / long_subject_id,
            metadata={"step": "surface_reconstruction_long", "skipped": True},
        )

    atlas_name = anat_cfg.get("skullstripping_segmentation", {}).get(
        "atlas_name", "ARM2"
    )
    threads = input.config.get("processing", {}).get("threads", 1)
    long_cfg = anat_cfg.get("surface_reconstruction", {}).get("longitudinal", {})

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
        long_max_cbv_dist=float(long_cfg.get("max_cbv_dist", 3.5)),
        long_pial_blend_weight=float(long_cfg.get("pial_blend_weight", 0.25)),
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

# ---------------------------------------------------------------------------
# Within-subject change statistics
# ---------------------------------------------------------------------------

# Morphometry maps worth fitting a rate to. thickness is the usual endpoint;
# area and curv come along because they cost nothing extra once the surfaces
# are read.
DEFAULT_MEASURES = ("thickness", "area", "curv")


def parse_session_time(session_label: str) -> Optional[float]:
    """Pull a numeric time out of a session label, or None if there isn't one.

    Handles the two shapes that actually occur in BIDS session labels here:
    ``ses-12months`` -> 12.0 and ``ses-004`` -> 4.0.

    Note what this is *not*: a real elapsed time. For labels that are just scan
    indices it yields the index, which orders the timepoints correctly but makes
    a fitted "rate" per-scan rather than per-unit-time. Supply explicit times
    when the spacing matters.
    """
    if session_label is None:
        return None
    label = str(session_label)
    if label.startswith("ses-"):
        label = label[4:]
    match = re.match(r"^(\d+(?:\.\d+)?)", label)
    return float(match.group(1)) if match else None


def write_qdec_table(
    path: Path,
    base_subject_id: str,
    long_ids: Sequence[str],
    times: Sequence[float],
) -> Path:
    """Write a FreeSurfer qdec table for the longitudinal tools.

    Written by hand rather than via ``long_qdec_table``, whose ``--cross``
    conversion assumes FreeSurfer's ``<tp>.long.<base>`` directory naming, which
    this pipeline deliberately does not use. The FreeSurfer tools themselves only
    read the ``fsid`` / ``fsid-base`` columns, so arbitrary names are fine.
    """
    if len(long_ids) != len(times):
        raise ValueError(
            f"{len(long_ids)} timepoints but {len(times)} times; each timepoint "
            "needs exactly one time value"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["fsid fsid-base time"]
    lines += [
        f"{long_id} {base_subject_id} {time:g}"
        for long_id, time in zip(long_ids, times)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def _read_morph(path: Path):
    import nibabel.freesurfer.io as fsio

    return fsio.read_morph_data(str(path))


def parse_aparc_stats(path: Path) -> Dict[str, Dict[str, float]]:
    """Parse a FreeSurfer ``?h.aparc.*.stats`` table into {roi: {column: value}}.

    Column names come from the file's own ``# ColHeaders`` line rather than being
    assumed, because the set varies with what ``mris_anatomical_stats`` was asked
    for -- notably eTIV is absent here, since talairach registration is skipped
    for macaque data.
    """
    columns: List[str] = []
    rows: Dict[str, Dict[str, float]] = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            if "ColHeaders" in line:
                columns = line.split("ColHeaders", 1)[1].split()
            continue
        if not line.strip():
            continue
        fields = line.split()
        if not columns or len(fields) < 2:
            continue
        roi = fields[0]
        values = {}
        for name, raw in zip(columns[1:], fields[1:]):
            try:
                values[name] = float(raw)
            except ValueError:
                continue
        rows[roi] = values
    return rows


def collect_change_stats(
    base_dir: Path,
    long_dirs: Dict[str, Path],
    times: Optional[Dict[str, float]] = None,
    measures: Sequence[str] = DEFAULT_MEASURES,
    hemis: Sequence[str] = ("lh", "rh"),
    atlas_name: str = "ARM2",
) -> Dict[str, Any]:
    """Fit within-subject rates of change across a subject's longitudinal recons.

    Per-vertex differencing is valid here without any surface registration: every
    timepoint inherited the base's mesh, so vertex *i* is the same anatomical
    point in all of them. That is the payoff of the longitudinal stream, and it
    is also why the vertex-count check below is a hard error -- a mismatch means
    the seeding did not take and the numbers would be meaningless.

    Implemented in numpy rather than by shelling out to ``long_mris_slopes``,
    so it is testable without FreeSurfer and needs no ``?h.sphere.reg`` (which
    brainana does not produce). ``long_mris_slopes`` remains usable on these
    outputs via the qdec table also written here.

    Args:
        base_dir: The base template's subject directory. Outputs are written
            into its ``surf/`` and ``stats/``.
        long_dirs: Mapping of longitudinal subject id to its directory.
        times: Optional explicit time per longitudinal id. Falls back to parsing
            the session label out of the id.
        measures: Morphometry maps to fit.
        hemis: Hemispheres to process.
        atlas_name: Atlas whose ROI stats table to summarise.

    Returns:
        A summary dict: the timepoints and times used, the per-measure outputs
        written, and any measures skipped with the reason.
    """
    import numpy as np

    base_dir = Path(base_dir)
    if len(long_dirs) < 2:
        raise ValueError(
            f"Rates of change need at least 2 timepoints; got {len(long_dirs)}"
        )

    long_ids = sorted(long_dirs)

    resolved_times: Dict[str, float] = {}
    ordinal_fallback: List[str] = []
    for long_id in long_ids:
        if times and long_id in times:
            resolved_times[long_id] = float(times[long_id])
            continue
        match = re.search(r"_ses-([^_]+)", long_id)
        parsed = parse_session_time(match.group(1)) if match else None
        if parsed is None:
            # Labels like ses-preop or ses-M06 carry no number. Fall back to the
            # position in the sorted order, which still gives a correctly
            # *ordered* fit -- rather than refusing to produce any statistics.
            # The rate is then per-scan, not per-unit-time; the summary records
            # which timepoints were treated this way.
            parsed = float(long_ids.index(long_id) + 1)
            ordinal_fallback.append(long_id)
        resolved_times[long_id] = parsed

    if ordinal_fallback:
        logger.warning(
            "No numeric session label for %s; using scan order as the time "
            "variable, so the fitted rate is per scan rather than per unit "
            "time. Pass times explicitly if the spacing matters.",
            ", ".join(ordinal_fallback),
        )

    if len(set(resolved_times.values())) < 2:
        raise ValueError(
            f"All timepoints resolved to the same time ({resolved_times}); a "
            "rate cannot be fitted. Pass times explicitly."
        )

    t = np.array([resolved_times[i] for i in long_ids], dtype=float)

    surf_out = base_dir / "surf"
    stats_out = base_dir / "stats"
    surf_out.mkdir(parents=True, exist_ok=True)
    stats_out.mkdir(parents=True, exist_ok=True)

    write_qdec_table(
        base_dir / "scripts" / "long.qdec.table.dat",
        base_dir.name,
        long_ids,
        [resolved_times[i] for i in long_ids],
    )

    summary: Dict[str, Any] = {
        "base_subject_id": base_dir.name,
        "timepoints": long_ids,
        "times": resolved_times,
        "ordinal_time_fallback": ordinal_fallback,
        "vertex_outputs": {},
        "roi_tables": {},
        "skipped": {},
    }

    import nibabel as nib

    for hemi in hemis:
        for measure in measures:
            paths = {
                long_id: Path(long_dirs[long_id]) / "surf" / f"{hemi}.{measure}"
                for long_id in long_ids
            }
            missing = [i for i, p in paths.items() if not p.exists()]
            if missing:
                summary["skipped"][f"{hemi}.{measure}"] = (
                    f"missing in {', '.join(missing)}"
                )
                continue

            stack = [_read_morph(paths[i]) for i in long_ids]
            counts = {i: arr.shape[0] for i, arr in zip(long_ids, stack)}
            if len(set(counts.values())) > 1:
                raise RuntimeError(
                    f"{hemi}.{measure} vertex counts differ across timepoints: "
                    f"{counts}. All timepoints must inherit the base's mesh; a "
                    "mismatch means the longitudinal seeding did not take, so "
                    "per-vertex comparison is invalid."
                )

            data = np.vstack(stack)  # (n_timepoints, n_vertices)

            # Ordinary least squares against time, per vertex.
            t_centered = t - t.mean()
            denom = float((t_centered**2).sum())
            rate = (t_centered[:, None] * (data - data.mean(axis=0))).sum(
                axis=0
            ) / denom
            mean_map = data.mean(axis=0)
            # Symmetrised percent change per unit time, the quantity
            # long_mris_slopes calls spc. Guarded against a zero mean.
            with np.errstate(divide="ignore", invalid="ignore"):
                spc = np.where(mean_map != 0, 100.0 * rate / mean_map, 0.0)

            for suffix, values in (
                ("rate", rate),
                ("avg", mean_map),
                ("spc", spc),
            ):
                out = surf_out / f"{hemi}.long.{measure}-{suffix}.mgh"
                img = nib.MGHImage(
                    values.astype(np.float32).reshape(-1, 1, 1), np.eye(4)
                )
                nib.save(img, str(out))
                summary["vertex_outputs"][f"{hemi}.{measure}-{suffix}"] = str(out)

        # ROI-level table, from the stats each timepoint already wrote.
        stats_name = f"{hemi}.aparc.{atlas_name}atlas.mapped.stats"
        per_tp = {}
        for long_id in long_ids:
            stats_path = Path(long_dirs[long_id]) / "stats" / stats_name
            if stats_path.exists():
                per_tp[long_id] = parse_aparc_stats(stats_path)
        if len(per_tp) < 2:
            summary["skipped"][f"{hemi}.roi"] = (
                f"{stats_name} present in {len(per_tp)} timepoint(s); need 2"
            )
            continue

        # Only the timepoints that actually had a stats file. Iterating long_ids
        # here would KeyError on any timepoint whose stats were missing, which
        # under errorStrategy 'ignore' would lose all of the subject's stats
        # silently.
        stats_ids = [i for i in long_ids if i in per_tp]
        rois = sorted(set.intersection(*(set(per_tp[i]) for i in stats_ids)))

        csv_lines = ["roi,measure,slope,mean,spc,n_timepoints"]
        for roi in rois:
            # Columns intersected across the timepoints for *this* ROI, rather
            # than sampled from one arbitrary ROI: mris_anatomical_stats can emit
            # different column sets, and a column present only elsewhere would
            # KeyError here.
            roi_columns = sorted(
                set.intersection(*(set(per_tp[i][roi]) for i in stats_ids))
            )
            for column in roi_columns:
                usable = [i for i in stats_ids if roi in per_tp[i]]
                y = np.array(
                    [per_tp[i][roi][column] for i in usable], dtype=float
                )
                tt = np.array([resolved_times[i] for i in usable], dtype=float)
                if y.size < 2 or len(set(tt)) < 2:
                    continue
                ttc = tt - tt.mean()
                slope = float((ttc * (y - y.mean())).sum() / (ttc**2).sum())
                mean = float(y.mean())
                pct = 100.0 * slope / mean if mean != 0 else 0.0
                csv_lines.append(
                    f"{roi},{column},{slope:.6g},{mean:.6g},{pct:.6g},{y.size}"
                )
        csv_path = stats_out / f"{hemi}.long.roi-rates.csv"
        csv_path.write_text("\n".join(csv_lines) + "\n")
        summary["roi_tables"][hemi] = str(csv_path)

    (stats_out / "long.change-stats.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    logger.info(
        "Wrote change statistics for %s across %d timepoint(s)",
        base_dir.name,
        len(long_ids),
    )
    return summary
