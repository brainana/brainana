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

    # Snapshot the grid the transforms above target, so the post-reconstruction
    # check below can prove it did not move.
    base_affine_before = _affine(base_orig)

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

    # The reconstruction re-saved orig.mgz (postprocess_for_freesurfer does so
    # unconditionally, via a nifti round trip). The transforms written above
    # target the grid it had *before* that, so confirm it did not move. If it
    # did, every timepoint would be resampled onto a grid the base no longer
    # has, and nothing downstream would report it -- the surfaces would simply
    # be wrong.
    import numpy as np

    final_geom = _geometry(base_orig)
    final_affine = _affine(base_orig)
    if final_geom != grid or not np.allclose(
        final_affine, base_affine_before, atol=1e-4
    ):
        raise RuntimeError(
            f"Base template geometry changed during reconstruction: was "
            f"shape={grid[0]} zooms={grid[1]}, now shape={final_geom[0]} "
            f"zooms={final_geom[1]}.\naffine before:\n{base_affine_before}"
            f"\naffine after:\n{final_affine}\n"
            "The timepoint-to-base transforms target the original grid, so "
            "every timepoint would be misaligned. This is a bug in the base "
            "build, not a data problem."
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
