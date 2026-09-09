"""
FreeSurfer longitudinal-stream command wrappers.

Provides Python interfaces to the tools used to build a within-subject base
template and to map timepoints into its space: ``mri_robust_template``,
``make_upright`` and ``mri_concatenate_lta``.

These are the only pieces of FreeSurfer's longitudinal stream that are reused
verbatim. ``mri_robust_template`` is pure robust rigid registration plus
median averaging -- no atlas priors and no species assumptions -- which is why
it transfers to macaque data unchanged, while ``recon-all -base``/``-long``
would apply human GCA priors and clobber the CNN segmentation the surfaces are
built on.
"""

from pathlib import Path
from typing import Optional, Sequence
import logging

from .base import run_fs_command

logger = logging.getLogger(__name__)

# rca-base-init's own saturation value for the norm pass. Kept verbatim so the
# numerical behaviour matches FreeSurfer's base construction.
RCA_BASE_INIT_SAT = 4.685


def mri_robust_template(
    movs: Sequence[Path],
    template: Path,
    ltas: Optional[Sequence[Path]] = None,
    ixforms: Optional[Sequence[Path]] = None,
    average: int = 1,
    sat: Optional[float] = RCA_BASE_INIT_SAT,
    satit: bool = False,
    noit: bool = False,
    finalnearest: bool = False,
    iscale: bool = False,
    iscaleout: Optional[Sequence[Path]] = None,
    iscalein: Optional[Sequence[Path]] = None,
    subsample: Optional[int] = None,
    inittp: Optional[int] = None,
    fixtp: bool = False,
    log_file: Optional[Path] = None,
    cmd_log_file: Optional[Path] = None,
) -> Path:
    """
    Construct an unbiased within-subject template from several volumes.

    Wraps ``mri_robust_template``, which iteratively builds a mean/median volume
    and robustly registers every input to it (6 DOF rigid; the documented
    "6-7 DOF" seventh parameter is intensity scaling, not geometric scale).

    Parameters
    ----------
    movs : sequence of Path
        Input volumes, one per timepoint. All should share a voxel grid.
    template : Path
        Output template volume.
    ltas : sequence of Path, optional
        Output transforms, one per input, mapping each input into template space.
    ixforms : sequence of Path, optional
        Initial transforms to apply, one per input. Used with ``noit`` to
        resample a second volume set through transforms solved on another.
    average : int, default=1
        0 = mean, 1 = median.
    sat : float, optional
        Outlier saturation/sensitivity. Ignored when ``satit`` is set.
    satit : bool, default=False
        Auto-detect saturation. The documented alternative to ``sat`` for head
        or full-brain scans.
    noit : bool, default=False
        Do not iterate; just apply ``ixforms`` and average.
    finalnearest : bool, default=False
        Use nearest-neighbour for the final resample. For label/mask volumes.
    iscale : bool, default=False
        Allow intensity scaling. Off in ``rca-base-init``; relevant when tissue
        contrast changes between timepoints (e.g. ongoing myelination).
    iscaleout : sequence of Path, optional
        Write the solved intensity scales, one per input. Implies ``iscale``.
        Needed when a second ``--noit`` pass must reuse them -- solving scales and
        then averaging a different volume set without them leaves the knob inert.
    iscalein : sequence of Path, optional
        Reuse intensity scales solved by an earlier pass.
    subsample : int, optional
        Subsample if any axis exceeds this size. Escape hatch for large
        high-resolution volumes.
    inittp : int, optional
        Timepoint to use for spatial initialisation. Default in the binary is
        random, which makes results vary between runs; pass a value for
        reproducibility.
    fixtp : bool, default=False
        Map everything to the init timepoint, leaving it unresampled.
    log_file : Path, optional
        Log file path.
    cmd_log_file : Path, optional
        Command log (fastsurfer_recon.cmd format).

    Returns
    -------
    Path
        The output template path.

    Raises
    ------
    ValueError
        If ``ltas`` or ``ixforms`` is given with a length that does not match
        ``movs``. Mismatched lists would silently pair the wrong transform with
        the wrong timepoint, which misaligns every downstream surface.
    """
    movs = [Path(m) for m in movs]
    if not movs:
        raise ValueError("mri_robust_template needs at least one --mov volume")
    if ltas is not None and len(ltas) != len(movs):
        raise ValueError(
            f"ltas has {len(ltas)} entries but movs has {len(movs)}; "
            "each input needs exactly one output transform"
        )
    if ixforms is not None and len(ixforms) != len(movs):
        raise ValueError(
            f"ixforms has {len(ixforms)} entries but movs has {len(movs)}; "
            "each input needs exactly one initial transform"
        )

    template = Path(template)
    template.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str | Path] = ["mri_robust_template", "--mov", *movs]
    cmd += ["--template", template]
    cmd += ["--average", str(average)]

    if satit:
        cmd += ["--satit"]
    elif sat is not None:
        cmd += ["--sat", str(sat)]

    if ltas is not None:
        for lta in ltas:
            Path(lta).parent.mkdir(parents=True, exist_ok=True)
        cmd += ["--lta", *[Path(x) for x in ltas]]
    if ixforms is not None:
        cmd += ["--ixforms", *[Path(x) for x in ixforms]]
    if noit:
        cmd += ["--noit"]
    if finalnearest:
        cmd += ["--finalnearest"]
    if iscale or iscaleout is not None:
        cmd += ["--iscale"]
    if iscaleout is not None:
        if len(iscaleout) != len(movs):
            raise ValueError(
                f"iscaleout has {len(iscaleout)} entries but movs has "
                f"{len(movs)}; each input needs exactly one scale file"
            )
        for f in iscaleout:
            Path(f).parent.mkdir(parents=True, exist_ok=True)
        cmd += ["--iscaleout", *[Path(x) for x in iscaleout]]
    if iscalein is not None:
        if len(iscalein) != len(movs):
            raise ValueError(
                f"iscalein has {len(iscalein)} entries but movs has "
                f"{len(movs)}; each input needs exactly one scale file"
            )
        cmd += ["--iscalein", *[Path(x) for x in iscalein]]
    if subsample is not None:
        cmd += ["--subsample", str(subsample)]
    if inittp is not None:
        cmd += ["--inittp", str(inittp)]
    if fixtp:
        cmd += ["--fixtp"]

    expected = [template] + ([Path(x) for x in ltas] if ltas else [])
    logger.info(
        "Building robust template from %d volume(s) -> %s", len(movs), template
    )
    run_fs_command(
        cmd,
        log_file=log_file,
        cmd_log_file=cmd_log_file,
        expect_outputs=expected,
    )
    return template


def mri_convert_apply_lta(
    input_vol: Path,
    output_vol: Path,
    lta: Optional[Path] = None,
    odt: Optional[str] = "uchar",
    resample: str = "cubic",
    log_file: Optional[Path] = None,
    cmd_log_file: Optional[Path] = None,
) -> Path:
    """
    Resample a volume through an LTA onto the transform's target geometry.

    This is FreeSurfer's ``longmc`` command verbatim::

        mri_convert -at <lta> -odt uchar -rt cubic <in> <out>

    It is what puts a longitudinal timepoint into *base* space, and it is the
    reason base surfaces can afterwards be copied rather than transformed:
    volumes and surfaces then share one geometry.

    Note the single-dash flags. The generic ``mri_convert`` wrapper renders
    keyword arguments as ``--long-form`` options, which these are not, so this
    builds the command directly.

    Parameters
    ----------
    input_vol : Path
        Input volume, in the source (cross-sectional) space.
    output_vol : Path
        Output volume, on the LTA target's geometry.
    lta : Path, optional
        Transform whose target geometry the output adopts. When None, no
        transform is applied and this is a plain convert -- used to write the
        base template as uchar, which is what ``longmc`` also does so that the
        volume matches a cross-sectional orig.mgz and is not re-conformed later.
    odt : str, optional
        Output data type. ``longmc`` uses ``uchar`` for orig.mgz; pass None to
        leave the type unchanged, which is what label volumes need.
    resample : str, default="cubic"
        Interpolation. Use ``nearest`` for label volumes.
    log_file : Path, optional
        Log file path.
    cmd_log_file : Path, optional
        Command log (fastsurfer_recon.cmd format).

    Returns
    -------
    Path
        The output volume path.
    """
    output_vol = Path(output_vol)
    output_vol.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str | Path] = ["mri_convert"]
    if lta is not None:
        cmd += ["-at", Path(lta)]
    if odt is not None:
        cmd += ["-odt", odt]
    if lta is not None:
        cmd += ["-rt", resample]
    cmd += [Path(input_vol), output_vol]

    if lta is None:
        logger.info("Converting %s -> %s (odt=%s)", input_vol, output_vol, odt)
    else:
        logger.info("Resampling %s through %s -> %s", input_vol, lta, output_vol)
    run_fs_command(
        cmd,
        log_file=log_file,
        cmd_log_file=cmd_log_file,
        expect_outputs=[output_vol],
    )
    return output_vol


def make_upright(
    input_vol: Path,
    output_vol: Path,
    output_lta: Path,
    log_file: Optional[Path] = None,
    cmd_log_file: Optional[Path] = None,
) -> Path:
    """
    Reorient a single volume upright, emitting the transform that did it.

    This is the single-timepoint degenerate case of building a base template:
    with one input there is nothing to average, so FreeSurfer's base stream
    uses this instead. Keeping it lets a one-session subject go through the
    same code path as a multi-session one.

    Parameters
    ----------
    input_vol : Path
        Input volume.
    output_vol : Path
        Output upright volume.
    output_lta : Path
        Output transform mapping input to output.
    log_file : Path, optional
        Log file path.
    cmd_log_file : Path, optional
        Command log (fastsurfer_recon.cmd format).

    Returns
    -------
    Path
        The output volume path.
    """
    output_vol = Path(output_vol)
    output_lta = Path(output_lta)
    output_vol.parent.mkdir(parents=True, exist_ok=True)
    output_lta.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str | Path] = [
        "make_upright",
        Path(input_vol),
        output_vol,
        output_lta,
    ]
    logger.info("Making %s upright -> %s", input_vol, output_vol)
    run_fs_command(
        cmd,
        log_file=log_file,
        cmd_log_file=cmd_log_file,
        expect_outputs=[output_vol, output_lta],
    )
    return output_vol


def mri_concatenate_lta(
    lta1: Path,
    lta2: Path,
    output_lta: Path,
    invert1: bool = False,
    invert_out: bool = False,
    log_file: Optional[Path] = None,
    cmd_log_file: Optional[Path] = None,
) -> Path:
    """
    Concatenate (and optionally invert) LTA transforms.

    The common use here is inverting a timepoint-to-base transform, for which
    FreeSurfer's own idiom is to concatenate against the literal string
    ``identity.nofile``::

        mri_concatenate_lta -invert1 tp_to_base.lta identity.nofile base_to_tp.lta

    Parameters
    ----------
    lta1 : Path
        First transform. May be the literal ``identity.nofile``.
    lta2 : Path
        Second transform. May be the literal ``identity.nofile``.
    output_lta : Path
        Output transform.
    invert1 : bool, default=False
        Invert the first transform before concatenating.
    invert_out : bool, default=False
        Invert the concatenated result.
    log_file : Path, optional
        Log file path.
    cmd_log_file : Path, optional
        Command log (fastsurfer_recon.cmd format).

    Returns
    -------
    Path
        The output transform path.
    """
    output_lta = Path(output_lta)
    output_lta.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str | Path] = ["mri_concatenate_lta"]
    if invert1:
        cmd += ["-invert1"]
    if invert_out:
        cmd += ["-invert_out"]
    # identity.nofile is a sentinel the binary understands, not a real path, so
    # it must not be turned into a Path and resolved.
    cmd += [str(lta1) if str(lta1) == "identity.nofile" else Path(lta1)]
    cmd += [str(lta2) if str(lta2) == "identity.nofile" else Path(lta2)]
    cmd += [output_lta]

    logger.info("Concatenating LTAs -> %s", output_lta)
    run_fs_command(
        cmd,
        log_file=log_file,
        cmd_log_file=cmd_log_file,
        expect_outputs=[output_lta],
    )
    return output_lta


__all__ = [
    "RCA_BASE_INIT_SAT",
    "mri_robust_template",
    "mri_convert_apply_lta",
    "make_upright",
    "mri_concatenate_lta",
]
