"""
FreeSurfer mris_* command wrappers.

Provides Python interfaces to FreeSurfer mris_* surface tools.
"""

from pathlib import Path
from typing import Optional
import logging

from .base import run_fs_command, to_relative_path

logger = logging.getLogger(__name__)


def mris_info(
    surface: Path,
    log_file: Optional[Path] = None,
    subject_dir: Optional[Path] = None,
) -> str:
    """
    Get information about a surface.

    Parameters
    ----------
    surface : Path
        Surface file
    log_file : Path, optional
        Log file path
    subject_dir : Path, optional
        Subject directory. If provided, converts paths to relative from subject_dir.

    Returns
    -------
    str
        Surface information text
    """
    # Convert paths to relative if subject_dir provided
    if subject_dir:
        subject_dir = Path(subject_dir).resolve()
        surface = to_relative_path(surface, subject_dir)

    cmd = ["mris_info", str(surface)]
    result = run_fs_command(
        cmd, log_file=log_file, subject_dir=subject_dir, capture_output=True
    )
    return result.stdout


def mris_extract_main_component(
    input_surf: Path,
    output_surf: Path,
    log_file: Optional[Path] = None,
    subject_dir: Optional[Path] = None,
) -> Path:
    """
    Extract the largest connected component from a surface.

    Parameters
    ----------
    input_surf : Path
        Input surface
    output_surf : Path
        Output surface (can be same as input)
    log_file : Path, optional
        Log file path
    subject_dir : Path, optional
        Subject directory. If provided, converts paths to relative from subject_dir.

    Returns
    -------
    Path
        Output surface path
    """
    # Convert paths to relative if subject_dir provided
    if subject_dir:
        subject_dir = Path(subject_dir).resolve()
        input_surf = to_relative_path(input_surf, subject_dir)
        output_surf = to_relative_path(output_surf, subject_dir)

    cmd = [
        "mris_extract_main_component",
        str(input_surf),
        str(output_surf),
    ]
    run_fs_command(
        cmd, log_file=log_file, subject_dir=subject_dir, expect_outputs=[output_surf]
    )
    return output_surf


def mris_remesh(
    input_surf: Path,
    output_surf: Path,
    desired_face_area: Optional[float] = None,
    remesh: bool = False,
    iters: Optional[int] = None,
    log_file: Optional[Path] = None,
    subject_dir: Optional[Path] = None,
) -> Path:
    """
    Remesh surface to target face area or using remesh mode.

    Parameters
    ----------
    input_surf : Path
        Input surface
    output_surf : Path
        Output remeshed surface
    desired_face_area : float, optional
        Target average face area (used if remesh=False)
    remesh : bool, default=False
        Use --remesh --iters mode
    iters : int, optional
        Number of remesh iterations (required if remesh=True)
    log_file : Path, optional
        Log file path
    subject_dir : Path, optional
        Subject directory. If provided, converts paths to relative from subject_dir.

    Returns
    -------
    Path
        Output surface path
    """
    # Convert paths to relative if subject_dir provided
    if subject_dir:
        subject_dir = Path(subject_dir).resolve()
        input_surf = to_relative_path(input_surf, subject_dir)
        output_surf = to_relative_path(output_surf, subject_dir)

    cmd = ["mris_remesh"]

    if remesh:
        if iters is None:
            raise ValueError("iters must be specified when remesh=True")
        cmd.extend(["--remesh", "--iters", str(iters)])
    else:
        if desired_face_area is None:
            raise ValueError("desired_face_area must be specified when remesh=False")
        cmd.extend(["--desired-face-area", str(desired_face_area)])
        # Add --iters if provided (optional flagged argument)
        if iters is not None:
            cmd.extend(["--iters", str(iters)])

    cmd.extend(["--input", str(input_surf), "--output", str(output_surf)])
    run_fs_command(
        cmd, log_file=log_file, subject_dir=subject_dir, expect_outputs=[output_surf]
    )
    return output_surf


def mris_smooth(
    input_surf: Path,
    output_surf: Path,
    n_iterations: int = 10,
    nw: bool = True,
    seed: Optional[int] = None,
    log_file: Optional[Path] = None,
    subject_dir: Optional[Path] = None,
) -> Path:
    """
    Smooth a surface.

    Parameters
    ----------
    input_surf : Path
        Input surface
    output_surf : Path
        Output smoothed surface
    n_iterations : int, default=10
        Number of smoothing iterations
    nw : bool, default=True
        Normalize weights
    seed : int, optional
        Random seed
    log_file : Path, optional
        Log file path
    subject_dir : Path, optional
        Subject directory. If provided, converts paths to relative from subject_dir.

    Returns
    -------
    Path
        Output surface path
    """
    # Convert paths to relative if subject_dir provided
    if subject_dir:
        subject_dir = Path(subject_dir).resolve()
        input_surf = to_relative_path(input_surf, subject_dir)
        output_surf = to_relative_path(output_surf, subject_dir)

    cmd = [
        "mris_smooth",
        "-n",
        str(n_iterations),
    ]

    if nw:
        cmd.append("-nw")
    if seed is not None:
        cmd.extend(["-seed", str(seed)])

    cmd.extend([str(input_surf), str(output_surf)])
    run_fs_command(
        cmd, log_file=log_file, subject_dir=subject_dir, expect_outputs=[output_surf]
    )
    return output_surf


def mris_inflate(
    input_surf: Path,
    output_surf: Path,
    n_iterations: Optional[int] = None,
    no_save_sulc: bool = True,
    log_file: Optional[Path] = None,
    subject_dir: Optional[Path] = None,
) -> Path:
    """
    Inflate surface to sphere.

    Parameters
    ----------
    input_surf : Path
        Input surface (e.g., smoothwm.nofix or smoothwm)
    output_surf : Path
        Output inflated surface (e.g., inflated.nofix or inflated)
    n_iterations : int, optional
        Number of inflation iterations (None = use FreeSurfer default).
        For monkey data, typically use 11 (less inflation than default ~15-20).
    no_save_sulc : bool, default=True
        Skip saving sulc file during inflation
    log_file : Path, optional
        Log file path
    subject_dir : Path, optional
        Subject directory. If provided, converts paths to relative from subject_dir.

    Returns
    -------
    Path
        Output surface path
    """
    # Convert paths to relative if subject_dir provided
    if subject_dir:
        subject_dir = Path(subject_dir).resolve()
        input_surf = to_relative_path(input_surf, subject_dir)
        output_surf = to_relative_path(output_surf, subject_dir)

    cmd = ["mris_inflate"]

    if no_save_sulc:
        cmd.append("-no-save-sulc")

    if n_iterations is not None:
        cmd.extend(["-n", str(n_iterations)])

    cmd.extend([str(input_surf), str(output_surf)])
    run_fs_command(
        cmd, log_file=log_file, subject_dir=subject_dir, expect_outputs=[output_surf]
    )
    return output_surf


def mris_place_surface(
    input_surf: Path,
    output_surf: Path,
    hemi: str,
    wm: Path,
    invol: Path,
    aseg: Path,
    adgws_in: Optional[Path] = None,
    white: bool = False,
    pial: bool = False,
    threads: int = 1,
    log_file: Optional[Path] = None,
    subject_dir: Optional[Path] = None,
    subjects_dir: Optional[Path] = None,
    **kwargs,
) -> Path:
    """
    Place white or pial surface.

    This is a complex command with many options. See FreeSurfer documentation
    for full details.

    Parameters
    ----------
    input_surf : Path
        Input surface (e.g., white.preaparc or white)
    output_surf : Path
        Output surface (e.g., white or pial)
    hemi : str
        Hemisphere ('lh' or 'rh')
    wm : Path
        White matter segmentation
    invol : Path
        Input volume (e.g., brain.finalsurfs.mgz)
    aseg : Path
        Aseg segmentation (e.g., aseg.presurf.mgz)
    adgws_in : Path, optional
        Auto-detected gray/white stats file
    white : bool
        Place white surface
    pial : bool
        Place pial surface
    threads : int
        Number of threads
    log_file : Path, optional
        Log file path
    subject_dir : Path, optional
        Subject directory. If provided, converts paths to relative from subject_dir.
    subjects_dir : Path, optional
        SUBJECTS_DIR path. If None, uses environment variable or derives from subject_dir.
    **kwargs
        Additional arguments:
        - rip_label: cortex label file
        - rip_bg: flag
        - rip_surf: surface to rip from
        - pin_medial_wall: cortex label
        - repulse_surf: surface to repulse from
        - white_surf: white surface (for pial)
        - aparc: parcellation annotation
        - nsmooth: number of smoothing iterations
        - max_cbv_dist: maximum distance
        - blend_surf: blend surface
        - i: input surface (alternative to positional)

    Returns
    -------
    Path
        Output surface path
    """
    # Convert paths to relative if subject_dir provided
    if subject_dir:
        subject_dir = Path(subject_dir).resolve()
        input_surf = to_relative_path(input_surf, subject_dir)
        output_surf = to_relative_path(output_surf, subject_dir)
        wm = to_relative_path(wm, subject_dir)
        invol = to_relative_path(invol, subject_dir)
        aseg = to_relative_path(aseg, subject_dir)
        if adgws_in:
            adgws_in = to_relative_path(adgws_in, subject_dir)

    # Derive subjects_dir from subject_dir if not provided
    if subjects_dir is None and subject_dir:
        # subject_dir is typically subjects_dir / subject_id, so parent is subjects_dir
        subjects_dir = subject_dir.parent

    cmd = ["mris_place_surface"]

    # Required arguments
    if adgws_in:
        cmd.extend(["--adgws-in", str(adgws_in)])
    cmd.extend(["--seg", str(aseg)])
    cmd.extend(["--threads", str(threads)])
    cmd.extend(["--wm", str(wm)])
    cmd.extend(["--invol", str(invol)])
    cmd.append(f"--{hemi}")
    cmd.extend(["--o", str(output_surf)])

    # Surface type
    if white:
        cmd.append("--white")
    if pial:
        cmd.append("--pial")

    # Optional arguments from kwargs
    kwarg_map = {
        "rip_label": "--rip-label",
        "rip_bg": "--rip-bg",
        "rip_surf": "--rip-surf",
        "pin_medial_wall": "--pin-medial-wall",
        "repulse_surf": "--repulse-surf",
        "white_surf": "--white-surf",
        "aparc": "--aparc",
        "nsmooth": "--nsmooth",
        "max_cbv_dist": "--max-cbv-dist",
        "blend_surf": "--blend-surf",
        "i": "--i",
    }

    def _render(value):
        """Path values become subject-dir-relative, everything else is str()."""
        if isinstance(value, Path) and subject_dir:
            value = to_relative_path(value, subject_dir)
        return str(value)

    for key, value in kwargs.items():
        if key in kwarg_map:
            flag = kwarg_map[key]
            if value is True:
                cmd.append(flag)
            elif isinstance(value, (tuple, list)):
                # Multi-argument options, e.g. --blend-surf <weight> <surf>.
                # These must stay separate argv items; joining them into one
                # string would hand FreeSurfer a single token containing a
                # space, which it does not parse back apart.
                cmd.append(flag)
                cmd.extend(_render(v) for v in value)
            elif value is not False and value is not None:
                cmd.append(flag)
                cmd.append(_render(value))

    # Input surface (if not in kwargs)
    if "--i" not in cmd:
        cmd.extend(["--i", str(input_surf)])

    # Set SUBJECTS_DIR in environment if provided
    env = None
    if subjects_dir:
        env = {"SUBJECTS_DIR": str(subjects_dir)}

    run_fs_command(
        cmd,
        log_file=log_file,
        subject_dir=subject_dir,
        env=env,
        expect_outputs=[output_surf],
    )
    return output_surf


def mris_place_surface_curv_map(
    surface: Path,
    output_curv: Path,
    n_smooth: int = 2,
    n_iterations: int = 10,
    log_file: Optional[Path] = None,
    subject_dir: Optional[Path] = None,
) -> Path:
    """
    Compute curvature map from surface.

    Parameters
    ----------
    surface : Path
        Input surface
    output_curv : Path
        Output curvature file
    n_smooth : int, default=2
        Smoothing iterations
    n_iterations : int, default=10
        Number of iterations
    log_file : Path, optional
        Log file path
    subject_dir : Path, optional
        Subject directory. If provided, converts paths to relative from subject_dir.

    Returns
    -------
    Path
        Output curvature file path
    """
    # Convert paths to relative if subject_dir provided
    if subject_dir:
        subject_dir = Path(subject_dir).resolve()
        surface = to_relative_path(surface, subject_dir)
        output_curv = to_relative_path(output_curv, subject_dir)

    cmd = [
        "mris_place_surface",
        "--curv-map",
        str(surface),
        str(n_smooth),
        str(n_iterations),
        str(output_curv),
    ]
    run_fs_command(cmd, log_file=log_file, subject_dir=subject_dir)
    return output_curv


def mris_place_surface_area_map(
    surface: Path,
    output_area: Path,
    log_file: Optional[Path] = None,
    subject_dir: Optional[Path] = None,
) -> Path:
    """
    Compute area map from surface.

    Parameters
    ----------
    surface : Path
        Input surface
    output_area : Path
        Output area file
    log_file : Path, optional
        Log file path
    subject_dir : Path, optional
        Subject directory. If provided, converts paths to relative from subject_dir.

    Returns
    -------
    Path
        Output area file path
    """
    # Convert paths to relative if subject_dir provided
    if subject_dir:
        subject_dir = Path(subject_dir).resolve()
        surface = to_relative_path(surface, subject_dir)
        output_area = to_relative_path(output_area, subject_dir)

    cmd = [
        "mris_place_surface",
        "--area-map",
        str(surface),
        str(output_area),
    ]
    run_fs_command(cmd, log_file=log_file, subject_dir=subject_dir)
    return output_area


def mris_place_surface_thickness(
    white_surf: Path,
    pial_surf: Path,
    output_thickness: Path,
    n_smooth: int = 20,
    n_iterations: int = 5,
    log_file: Optional[Path] = None,
    subject_dir: Optional[Path] = None,
) -> Path:
    """
    Compute thickness map between white and pial surfaces.

    Parameters
    ----------
    white_surf : Path
        White matter surface
    pial_surf : Path
        Pial surface
    output_thickness : Path
        Output thickness file
    n_smooth : int, default=20
        Smoothing iterations
    n_iterations : int, default=5
        Number of iterations
    log_file : Path, optional
        Log file path
    subject_dir : Path, optional
        Subject directory. If provided, converts paths to relative from subject_dir.

    Returns
    -------
    Path
        Output thickness file path
    """
    # Convert paths to relative if subject_dir provided
    if subject_dir:
        subject_dir = Path(subject_dir).resolve()
        white_surf = to_relative_path(white_surf, subject_dir)
        pial_surf = to_relative_path(pial_surf, subject_dir)
        output_thickness = to_relative_path(output_thickness, subject_dir)

    cmd = [
        "mris_place_surface",
        "--thickness",
        str(white_surf),
        str(pial_surf),
        str(n_smooth),
        str(n_iterations),
        str(output_thickness),
    ]
    run_fs_command(cmd, log_file=log_file, subject_dir=subject_dir)
    return output_thickness


def mris_fix_topology(
    subject: str,
    hemi: str,
    sphere: Path,
    inflated: Path,
    orig: Path,
    output_premesh: Path,
    mgz: bool = True,
    ga: bool = True,
    seed: int = 1234,
    log_file: Optional[Path] = None,
    subjects_dir: Optional[Path] = None,
) -> Path:
    """
    Fix topological defects in surface.

    Parameters
    ----------
    subject : str
        Subject ID
    hemi : str
        Hemisphere ('lh' or 'rh')
    sphere : Path
        Sphere surface (e.g., qsphere.nofix) - can be absolute or relative
    inflated : Path
        Inflated surface (e.g., inflated.nofix) - can be absolute or relative
    orig : Path
        Original surface (e.g., orig.nofix) - can be absolute or relative
    output_premesh : Path
        Output premesh surface (e.g., orig.premesh) - can be absolute or relative
    mgz : bool, default=True
        Use mgz format
    ga : bool, default=True
        Use -ga flag
    seed : int, default=1234
        Random seed
    log_file : Path, optional
        Log file path
    subjects_dir : Path, optional
        Subjects directory. If provided, command runs from subject's scripts directory
        and uses relative filenames.

    Returns
    -------
    Path
        Output premesh surface path
    """
    cmd = ["mris_fix_topology"]

    if mgz:
        cmd.append("-mgz")

    # mris_fix_topology expects relative filenames (without hemisphere prefix)
    # when run from the subject's scripts directory
    if subjects_dir:
        subjects_dir = Path(subjects_dir).resolve()

        # Extract just the filename (remove hemisphere prefix if present)
        def get_rel_filename(path: Path) -> str:
            name = path.name
            # Remove hemisphere prefix if present (e.g., "lh.qsphere.nofix" -> "qsphere.nofix")
            if name.startswith(f"{hemi}."):
                return name[len(hemi) + 1 :]
            return name

        sphere_name = get_rel_filename(sphere)
        inflated_name = get_rel_filename(inflated)
        orig_name = get_rel_filename(orig)
        output_name = get_rel_filename(output_premesh)

        cmd.extend(["-sphere", sphere_name])
        cmd.extend(["-inflated", inflated_name])
        cmd.extend(["-orig", orig_name])
        cmd.extend(["-out", output_name])

        # Run from subject's scripts directory
        subject_scripts_dir = subjects_dir / subject / "scripts"
        subject_scripts_dir.mkdir(parents=True, exist_ok=True)
        # Use subject_scripts_dir for both logging and execution
        subject_dir = subject_scripts_dir
    else:
        # Use absolute paths (legacy behavior)
        cmd.extend(["-sphere", str(sphere)])
        cmd.extend(["-inflated", str(inflated)])
        cmd.extend(["-orig", str(orig)])
        cmd.extend(["-out", str(output_premesh)])
        subject_dir = None

    if ga:
        cmd.append("-ga")
    cmd.extend(["-seed", str(seed)])
    cmd.extend([subject, hemi])

    # Set SUBJECTS_DIR in environment if provided
    env = None
    if subjects_dir:
        env = {"SUBJECTS_DIR": str(subjects_dir)}

    # Pass subject_dir (scripts directory) for both logging and execution
    run_fs_command(cmd, log_file=log_file, subject_dir=subject_dir, env=env)
    return output_premesh


def mris_remove_intersection(
    input_surf: Path,
    output_surf: Path,
    log_file: Optional[Path] = None,
    subject_dir: Optional[Path] = None,
) -> Path:
    """
    Remove surface intersections.

    Parameters
    ----------
    input_surf : Path
        Input surface (can be same as output for in-place)
    output_surf : Path
        Output surface (can be same as input)
    log_file : Path, optional
        Log file path
    subject_dir : Path, optional
        Subject directory. If provided, converts paths to relative from subject_dir.

    Returns
    -------
    Path
        Output surface path
    """
    # Convert paths to relative if subject_dir provided
    if subject_dir:
        subject_dir = Path(subject_dir).resolve()
        input_surf = to_relative_path(input_surf, subject_dir)
        output_surf = to_relative_path(output_surf, subject_dir)

    cmd = [
        "mris_remove_intersection",
        str(input_surf),
        str(output_surf),
    ]
    run_fs_command(
        cmd, log_file=log_file, subject_dir=subject_dir, expect_outputs=[output_surf]
    )
    return output_surf


def mris_autodet_gwstats(
    output_stats: Path,
    input_vol: Path,
    wm_vol: Path,
    surface: Path,
    log_file: Optional[Path] = None,
) -> Path:
    """
    Auto-detect gray/white statistics.

    Parameters
    ----------
    output_stats : Path
        Output statistics file (e.g., autodet.gw.stats.{hemi}.dat)
    input_vol : Path
        Input volume (e.g., brain.finalsurfs.mgz)
    wm_vol : Path
        White matter volume (e.g., wm.mgz)
    surface : Path
        Surface file (e.g., orig.premesh)
    log_file : Path, optional
        Log file path

    Returns
    -------
    Path
        Output statistics file path
    """
    cmd = [
        "mris_autodet_gwstats",
        "--o",
        str(output_stats),
        "--i",
        str(input_vol),
        "--wm",
        str(wm_vol),
        "--surf",
        str(surface),
    ]
    run_fs_command(cmd, log_file=log_file)
    return output_stats


def mris_curvature_stats(
    subject: str,
    hemi: str,
    output_stats: Path,
    surface_name: str = "smoothwm",
    curvatures: list[str] = None,
    write_curvature_files: bool = True,
    mgz: bool = True,
    log_file: Optional[Path] = None,
    subjects_dir: Optional[Path] = None,
) -> Path:
    """
    Compute curvature statistics.

    Parameters
    ----------
    subject : str
        Subject ID
    hemi : str
        Hemisphere ('lh' or 'rh')
    output_stats : Path
        Output statistics file (e.g., {hemi}.curv.stats)
    surface_name : str, default="smoothwm"
        Surface name (e.g., "smoothwm")
    curvatures : list[str], optional
        Curvature types (e.g., ["curv", "sulc"]). Defaults to ["curv", "sulc"]
    write_curvature_files : bool, default=True
        Write curvature files (-m flag)
    mgz : bool, default=True
        Use mgz format (-G flag)
    log_file : Path, optional
        Log file path
    subjects_dir : Path, optional
        SUBJECTS_DIR path. If None, uses environment variable.

    Returns
    -------
    Path
        Output statistics file path
    """
    if curvatures is None:
        curvatures = ["curv", "sulc"]

    cmd = ["mris_curvature_stats"]

    if write_curvature_files:
        cmd.append("-m")
    if mgz:
        cmd.append("--writeCurvatureFiles")
    cmd.append("-G")
    cmd.extend(["-o", str(output_stats)])
    cmd.extend(["-F", surface_name])
    cmd.extend([subject, hemi] + curvatures)

    # Set SUBJECTS_DIR if provided
    env = None
    if subjects_dir:
        env = {"SUBJECTS_DIR": str(subjects_dir)}

    run_fs_command(cmd, log_file=log_file, env=env)
    return output_stats


def mris_volmask(
    subject: str,
    aseg_name: str = "aseg.presurf",
    label_left_white: int = 2,
    label_left_ribbon: int = 3,
    label_right_white: int = 41,
    label_right_ribbon: int = 42,
    save_ribbon: bool = True,
    log_file: Optional[Path] = None,
    subjects_dir: Optional[Path] = None,
) -> None:
    """
    Create cortical ribbon volume mask.

    Parameters
    ----------
    subject : str
        Subject ID
    aseg_name : str, default="aseg.presurf"
        Aseg name (without .mgz extension)
    label_left_white : int, default=2
        Left white matter label
    label_left_ribbon : int, default=3
        Left ribbon label
    label_right_white : int, default=41
        Right white matter label
    label_right_ribbon : int, default=42
        Right ribbon label
    save_ribbon : bool, default=True
        Save ribbon volume (--save-ribbon flag)
    log_file : Path, optional
        Log file path
    subjects_dir : Path, optional
        SUBJECTS_DIR path. If None, uses environment variable.
    """
    cmd = [
        "mris_volmask",
        "--aseg_name",
        aseg_name,
        "--label_left_white",
        str(label_left_white),
        "--label_left_ribbon",
        str(label_left_ribbon),
        "--label_right_white",
        str(label_right_white),
        "--label_right_ribbon",
        str(label_right_ribbon),
    ]

    if save_ribbon:
        cmd.append("--save_ribbon")

    cmd.append(subject)

    # Set SUBJECTS_DIR if provided
    env = None
    if subjects_dir:
        env = {"SUBJECTS_DIR": str(subjects_dir)}

    run_fs_command(cmd, log_file=log_file, env=env)


def mris_register(
    input_sphere: Path,
    target_atlas: Path,
    output_sphere: Path,
    curv: bool = True,
    norot: bool = False,
    nosulc: bool = False,
    rotate: Optional[str] = None,
    threads: int = 1,
    log_file: Optional[Path] = None,
) -> Path:
    """
    Register sphere to atlas.

    Parameters
    ----------
    input_sphere : Path
        Input sphere surface
    target_atlas : Path
        Target atlas (e.g., folding atlas)
    output_sphere : Path
        Output registered sphere
    curv : bool, default=True
        Use curvature
    norot : bool, default=False
        No rotation
    nosulc : bool, default=False
        No sulcus
    rotate : str, optional
        Rotation angles (e.g., "alpha beta gamma")
    threads : int
        Number of threads
    log_file : Path, optional
        Log file path

    Returns
    -------
    Path
        Output sphere path
    """
    cmd = ["mris_register"]

    if curv:
        cmd.append("-curv")
    if norot:
        cmd.append("-norot")
    if nosulc:
        cmd.append("-nosulc")
    if rotate:
        cmd.extend(["-rotate", rotate])
    if threads > 1:
        cmd.extend(["-threads", str(threads)])

    cmd.extend(
        [
            str(input_sphere),
            str(target_atlas),
            str(output_sphere),
        ]
    )

    run_fs_command(cmd, log_file=log_file)
    return output_sphere


def mris_ca_label(
    subject: str,
    hemi: str,
    sphere_reg: Path,
    atlas: Path,
    output_annot: Path,
    cortex_label: Path,
    aseg: Path,
    seed: int = 1234,
    long_flag: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> Path:
    """
    Create cortical parcellation using classifier atlas.

    Parameters
    ----------
    subject : str
        Subject ID
    hemi : str
        Hemisphere ('lh' or 'rh')
    sphere_reg : Path
        Registered sphere surface
    atlas : Path
        Classifier atlas (.gcs file)
    output_annot : Path
        Output annotation file
    cortex_label : Path
        Cortex label file
    aseg : Path
        Aseg segmentation
    seed : int, default=1234
        Random seed
    long_flag : str, optional
        Longitudinal flag (e.g., "-long -R base_annot")
    log_file : Path, optional
        Log file path

    Returns
    -------
    Path
        Output annotation path
    """
    cmd = [
        "mris_ca_label",
        "-l",
        str(cortex_label),
        "-aseg",
        str(aseg),
        "-seed",
        str(seed),
    ]

    if long_flag:
        cmd.extend(long_flag.split())

    cmd.extend(
        [
            subject,
            hemi,
            str(sphere_reg),
            str(atlas),
            str(output_annot),
        ]
    )

    run_fs_command(cmd, log_file=log_file)
    return output_annot


def mris_curvature(
    surface: Path,
    hemi: str,
    seed: Optional[int] = None,
    thresh: Optional[float] = None,
    normalize: bool = False,
    area: Optional[int] = None,
    weights: bool = False,
    distances: Optional[tuple[int, int]] = None,
    log_file: Optional[Path] = None,
) -> None:
    """
    Compute surface curvature.

    Parameters
    ----------
    surface : Path
        Input surface (e.g., white.preaparc or inflated)
    hemi : str
        Hemisphere ('lh' or 'rh')
    seed : int, optional
        Random seed
    thresh : float, optional
        Threshold value (e.g., 0.999)
    normalize : bool, default=False
        Normalize curvature (-n flag)
    area : int, optional
        Area smoothing iterations (-a flag)
    weights : bool, default=False
        Use weights (-w flag)
    distances : tuple[int, int], optional
        Distance parameters (e.g., (10, 10))
    log_file : Path, optional
        Log file path

    Note
    ----
    mris_curvature writes curvature files directly to the surface directory.
    The output files are named {hemi}.{surface_name}.H and {hemi}.{surface_name}.K
    """
    cmd = ["mris_curvature"]

    if weights:
        cmd.append("-w")
    if seed is not None:
        cmd.extend(["-seed", str(seed)])
    if thresh is not None:
        cmd.extend(["-thresh", str(thresh)])
    if normalize:
        cmd.append("-n")
    if area is not None:
        cmd.extend(["-a", str(area)])
    if distances:
        cmd.extend(["-distances", str(distances[0]), str(distances[1])])

    # Surface name (without path, FreeSurfer expects just the name)
    surface_name = surface.name
    if surface_name.startswith(f"{hemi}."):
        surface_name = surface_name[len(f"{hemi}.") :]

    cmd.append(surface_name)

    # Change to surface directory for execution (FreeSurfer expects to be in surf dir)
    surf_dir = surface.parent
    run_fs_command(cmd, log_file=log_file, cwd=str(surf_dir))


def mris_anatomical_stats(
    subject: str,
    hemi: str,
    surface: Path,
    annotation: Path,
    output_stats: Path,
    cortex_label: Path,
    ctab: Optional[Path] = None,
    noxfm: bool = False,
    log_file: Optional[Path] = None,
    subjects_dir: Optional[Path] = None,
    **kwargs,
) -> Path:
    """
    Compute anatomical statistics from surface parcellation.

    Parameters
    ----------
    subject : str
        Subject ID
    hemi : str
        Hemisphere ('lh' or 'rh')
    surface : Path
        Surface file (e.g., white)
    annotation : Path
        Annotation file
    output_stats : Path
        Output statistics file
    cortex_label : Path
        Cortex label file
    ctab : Path, optional
        Color table file
    noxfm : bool, default=False
        Skip transform (for non-human)
    log_file : Path, optional
        Log file path
    subjects_dir : Path, optional
        SUBJECTS_DIR path. If None, uses environment variable.
    **kwargs
        Additional arguments (e.g., -th3, -mgz, -b, -f)

    Returns
    -------
    Path
        Output stats file path
    """
    cmd = ["mris_anatomical_stats"]

    # Common flags
    if "-th3" not in kwargs and "th3" not in kwargs:
        cmd.append("-th3")
    if "-mgz" not in kwargs and "mgz" not in kwargs:
        cmd.append("-mgz")
    if "-b" not in kwargs and "b" not in kwargs:
        cmd.append("-b")

    cmd.extend(["-cortex", str(cortex_label)])
    cmd.extend(["-f", str(output_stats)])
    cmd.extend(["-a", str(annotation)])

    if ctab:
        cmd.extend(["-c", str(ctab)])
    # Note: -noxfm flag support varies by FreeSurfer version
    # FastSurfer uses it when no_talairach is set, but some FreeSurfer versions
    # don't support this flag. If the flag is not supported, the command will fail.
    # For compatibility, we skip the flag if it's not available in this version.
    # The statistics should still compute correctly without it.
    #
    # Future enhancement: Add FreeSurfer version detection to conditionally use -noxfm.
    # This would require parsing FreeSurfer's build-stamp.txt or running a version
    # check command. For now, we skip the flag to maintain compatibility across versions.
    # if noxfm:
    #     cmd.append("-noxfm")

    # Add kwargs
    for key, value in kwargs.items():
        if key.startswith("-"):
            cmd.append(key)
            if value is not True:
                cmd.append(str(value))
        else:
            cmd.append(f"-{key.replace('_', '-')}")
            if value is not True:
                cmd.append(str(value))

    # Extract surface name (e.g., "white" from "/path/to/surf/lh.white")
    # When SUBJECTS_DIR is set, mris_anatomical_stats constructs the path automatically
    surface_name = surface.name
    # Remove hemisphere prefix if present (e.g., "lh.white" -> "white")
    if surface_name.startswith(f"{hemi}."):
        surface_name = surface_name[len(f"{hemi}.") :]

    cmd.extend([subject, hemi, surface_name])

    # Set SUBJECTS_DIR if provided
    env = None
    if subjects_dir:
        env = {"SUBJECTS_DIR": str(subjects_dir)}

    run_fs_command(cmd, log_file=log_file, env=env)
    return output_stats


__all__ = [
    "mris_info",
    "mris_extract_main_component",
    "mris_remesh",
    "mris_smooth",
    "mris_inflate",
    "mris_curvature",
    "mris_fix_topology",
    "mris_remove_intersection",
    "mris_autodet_gwstats",
    "mris_curvature_stats",
    "mris_volmask",
    "mris_place_surface",
    "mris_place_surface_curv_map",
    "mris_place_surface_area_map",
    "mris_place_surface_thickness",
    "mris_register",
    "mris_ca_label",
    "mris_anatomical_stats",
]
