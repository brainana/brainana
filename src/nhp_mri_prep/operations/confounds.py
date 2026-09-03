"""fMRIPrep-compatible functional confound regressors.

This module computes a BIDS/fMRIPrep-style ``*_desc-confounds_timeseries.tsv`` (plus a JSON
sidecar) from brainana's functional preprocessing outputs so that downstream tools that expect
fMRIPrep derivatives — notably nilearn's ``load_confounds`` / ``load_confounds_strategy`` — work
directly on brainana output.

IMPORTANT — confounds are regressors only:
    Nothing here modifies, filters, scrubs, or denoises the BOLD image. This module only *computes
    and writes regressor columns*. ``motion_outlier##`` / ``non_steady_state_outlier##`` are merely
    indicator columns the downstream user may optionally include in their own model; no volumes are
    removed and no temporal filtering is applied to the data.

Implementation is a lean, dependency-light reimplementation of the public fMRIPrep / Power et al.
formulas (no nipype/niworkflows): framewise displacement (Power, sum-of-abs, macaque radius), DVARS
and standardized DVARS (IQR/AR1 standardization matching nipype's ``compute_dvars``), non-steady-state
detection (MAD outlier on the global mean), tissue mean signals from an anatomical segmentation, and
the 24-parameter motion expansion. CompCor and cosine regressors are intentionally out of scope.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import nibabel as nib

# Reuse the macaque head radius already defined for motion correction.
from .preprocessing import MACAQUE_HEAD_RADIUS_MM

# --- Defaults (each overridable under func.confounds, except where noted) ------------------------
# Radius (mm) converting rotational deltas to mm-equivalent displacement, defaulting to the macaque
# head. Overridable via config (func.confounds.fd_radius_mm) for other NHP species; FD and rmsd
# values are only comparable across runs computed at the same radius, which is why the radius used
# is recorded per column in the JSON sidecar.
FD_RADIUS_MM: float = MACAQUE_HEAD_RADIUS_MM  # 27.0
# Threshold (mm) above which a volume is FLAGGED in motion_outlier## (no data removed).
# Macaque-scaled from the human Power-2012 default (0.5 mm): FD's rotation->mm conversion uses a
# 50 mm sphere for humans vs FD_RADIUS_MM (27 mm) here, so the threshold scales by the radius ratio
# 0.5 * (27 / 50) ~= 0.27 -> 0.25 mm. Overridable via config (func.confounds.fd_outlier_threshold_mm).
# Independently confirmed against PRIME-DE: pooled over the anesthetized sites (515 runs / 247,746
# frames), which carry no real head motion and so measure this estimator's noise floor, FD has
# median 0.034 mm and p99 0.258 mm. 0.25 mm therefore sits at the 98.9th percentile of motionless
# data (~1.1% false positives) and separates the regimes by ~50x: 1.1% of anesthetized frames
# flagged vs 55.9% of awake ones. Do not retune without redoing that comparison.
FD_OUTLIER_THRESHOLD_MM: float = 0.25
# Standardized-DVARS threshold above which a volume is FLAGGED in motion_outlier## (no data removed).
STD_DVARS_OUTLIER_THRESHOLD: float = 1.5
# Intensity normalization used by fMRIPrep/nipype before computing DVARS.
_DVARS_INTENSITY_NORMALIZATION: float = 1000.0

# FSL mcflirt .par column order (rotations in radians, translations in mm).
_PAR_COLUMNS: List[str] = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
# fMRIPrep emits translations first, then rotations.
_MOTION_ORDER: List[str] = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]

# Tissue classification primarily uses the LUT's explicit ``region`` column (brainana/ARM2 ColorLUT
# has one: WM/CSF/cortex/subcortex). These NAME substring patterns are only a fallback for generic
# LUTs that lack a region column (matched against the most descriptive name column available).
_CSF_NAME_PATTERNS: Tuple[str, ...] = (
    "ventricle",
    "csf",
    "cerebrospinal",
    "cerebralspinalfluid",
    "choroid",
)
_WM_NAME_PATTERNS: Tuple[str, ...] = (
    "whitematter",
    "white-matter",
    "white_matter",
    "white matter",
)


# =================================================================================================
# Generic helpers
# =================================================================================================
def _expand_regressor(values: np.ndarray) -> Dict[str, np.ndarray]:
    """Return the fMRIPrep expansion of a single regressor.

    Produces ``{raw, _derivative1, _power2, _derivative1_power2}``. The first derivative sample is
    NaN (written as ``n/a``), matching fMRIPrep.
    """
    values = np.asarray(values, dtype=float)
    deriv = np.empty_like(values)
    deriv[0] = np.nan
    deriv[1:] = np.diff(values)
    return {
        "": values,
        "_derivative1": deriv,
        "_power2": values**2,
        "_derivative1_power2": deriv**2,
    }


def _add_expanded(
    columns: "Dict[str, np.ndarray]", name: str, values: np.ndarray
) -> None:
    """Insert ``name`` and its three expansion terms into ``columns`` (fMRIPrep naming)."""
    for suffix, col in _expand_regressor(values).items():
        columns[f"{name}{suffix}"] = col


def _as_3d_bool_mask(mask: np.ndarray) -> np.ndarray:
    """Coerce a mask to a 3D boolean array, dropping a singleton 4th (time) axis if present."""
    arr = np.asarray(mask)
    if arr.ndim == 4:
        arr = arr[..., 0]
    return arr > 0


# =================================================================================================
# Motion parameters
# =================================================================================================
def load_motion_params(par_file: Union[str, Path]) -> pd.DataFrame:
    """Load motion parameters into a DataFrame in fMRIPrep column order.

    Accepts either a raw FSL mcflirt ``.par`` (6 whitespace columns, no header, ordered
    rot_x,rot_y,rot_z,trans_x,trans_y,trans_z) or brainana's named motion TSV (header with
    ``rot_x..trans_z`` columns). Returns columns ``trans_x, trans_y, trans_z, rot_x, rot_y, rot_z``.
    """
    par_file = str(par_file)
    with open(par_file) as fh:
        first = fh.readline()

    def _is_float(tok: str) -> bool:
        try:
            float(tok)
            return True
        except ValueError:
            return False

    tokens = first.split()
    # A header row has at least one non-numeric token (robust to scientific notation like 1e-5).
    has_header = bool(tokens) and not all(_is_float(t) for t in tokens)

    if has_header:
        df = pd.read_csv(par_file, sep=None, engine="python")
        missing = [c for c in _MOTION_ORDER if c not in df.columns]
        if missing:
            raise ValueError(
                f"Motion TSV {par_file} missing required columns: {missing}"
            )
        return df[_MOTION_ORDER].astype(float).reset_index(drop=True)

    arr = np.loadtxt(par_file)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 6:
        raise ValueError(
            f"Motion parameter file {par_file} has {arr.shape[1]} columns, expected >= 6"
        )
    df = pd.DataFrame(arr[:, :6], columns=_PAR_COLUMNS)
    return df[_MOTION_ORDER]


def expand_motion_params(motion_df: pd.DataFrame) -> "Dict[str, np.ndarray]":
    """24-parameter expansion (6 params x {raw, derivative1, power2, derivative1_power2})."""
    columns: Dict[str, np.ndarray] = {}
    for name in _MOTION_ORDER:
        _add_expanded(columns, name, motion_df[name].to_numpy())
    return columns


def compute_framewise_displacement(
    motion_df: pd.DataFrame, radius_mm: float = FD_RADIUS_MM
) -> np.ndarray:
    """Power et al. (2012) framewise displacement (sum of absolute backward differences).

    Rotations (radians) are converted to mm-equivalent surface displacement using ``radius_mm``.
    The first sample is NaN (``n/a``), matching fMRIPrep.
    """
    rot = motion_df[["rot_x", "rot_y", "rot_z"]].to_numpy() * radius_mm
    trans = motion_df[["trans_x", "trans_y", "trans_z"]].to_numpy()
    combined = np.hstack([trans, rot])
    diff = np.abs(np.diff(combined, axis=0)).sum(axis=1)
    fd = np.empty(combined.shape[0], dtype=float)
    fd[0] = np.nan
    fd[1:] = diff
    return fd


def _euler_rotation_matrix(angles: np.ndarray) -> np.ndarray:
    """Rotation matrix from FSL-style Euler angles (radians), composed as ``Rx @ Ry @ Rz``."""
    rx, ry, rz = float(angles[0]), float(angles[1]), float(angles[2])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_x @ rot_y @ rot_z


def compute_rmsd(
    motion_df: pd.DataFrame, radius_mm: float = FD_RADIUS_MM
) -> np.ndarray:
    """Relative RMS head displacement (the fMRIPrep ``rmsd`` column).

    Reimplements FSL's frame-to-frame RMS deviation (Jenkinson, 1999) directly from the motion
    parameters, so no external ``mcflirt -stats`` file is needed. For each consecutive pair the
    relative rigid transform ``(M, t)`` is formed from the absolute rotations/translations, and the
    RMS deviation over a sphere of radius ``radius_mm`` (macaque head radius) is

        rmsd = sqrt( (radius^2 / 5) * trace(A^T A) + t^T t ),   A = M - I

    The first sample is NaN (``n/a``), matching fMRIPrep.
    """
    rots = motion_df[["rot_x", "rot_y", "rot_z"]].to_numpy(dtype=float)
    trans = motion_df[["trans_x", "trans_y", "trans_z"]].to_numpy(dtype=float)
    n = len(motion_df)
    rmsd = np.full(n, np.nan, dtype=float)
    eye = np.eye(3)
    r2_over_5 = (radius_mm * radius_mm) / 5.0
    prev_rot = _euler_rotation_matrix(rots[0]) if n else eye
    for i in range(1, n):
        cur_rot = _euler_rotation_matrix(rots[i])
        # Relative transform T_i ∘ T_{i-1}^{-1}: rotation M, translation t.
        m = cur_rot @ prev_rot.T
        t = trans[i] - m @ trans[i - 1]
        a = m - eye
        rmsd[i] = np.sqrt(r2_over_5 * np.trace(a.T @ a) + t @ t)
        prev_rot = cur_rot
    return rmsd


# =================================================================================================
# DVARS  (reimplements nipype.algorithms.confounds.compute_dvars)
# =================================================================================================
def _ar1(series: np.ndarray) -> np.ndarray:
    """Lag-1 autocorrelation (Yule-Walker order 1) per row of a (voxels x time) array."""
    demeaned = series - series.mean(axis=1, keepdims=True)
    num = (demeaned[:, 1:] * demeaned[:, :-1]).sum(axis=1)
    den = (demeaned * demeaned).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ar1 = np.where(den > 0, num / den, 0.0)
    return ar1


def compute_dvars(
    bold_data: np.ndarray,
    mask: np.ndarray,
    intensity_normalization: float = _DVARS_INTENSITY_NORMALIZATION,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (dvars, std_dvars) following nipype/fMRIPrep.

    Args:
        bold_data: 4D array (x, y, z, t).
        mask: 3D boolean/int array; voxels > 0 are used.
        intensity_normalization: median-scaling target (fMRIPrep default 1000; 0 disables).

    Returns:
        ``(dvars, std_dvars)`` 1D arrays of length T, first sample NaN (``n/a``).
    """
    idx = _as_3d_bool_mask(mask)
    mfunc = bold_data[idx].astype(np.float64)  # (Nvox, T)
    if mfunc.ndim != 2 or mfunc.shape[1] < 2:
        raise ValueError("BOLD must be 4D with >= 2 timepoints for DVARS")

    if intensity_normalization != 0:
        positive = mfunc[mfunc > 0]
        median = np.median(positive) if positive.size else 0.0
        if median > 0:
            mfunc = (mfunc / median) * intensity_normalization

    # Robust per-voxel temporal SD (IQR / 1.349).
    func_sd = (
        np.percentile(mfunc, 75, axis=1) - np.percentile(mfunc, 25, axis=1)
    ) / 1.349
    # Drop zero-variance voxels (fMRIPrep remove_zerovariance default).
    keep = func_sd > 0
    mfunc = mfunc[keep]
    func_sd = func_sd[keep]
    if mfunc.shape[0] == 0:
        n = bold_data.shape[3]
        return np.full(n, np.nan), np.full(n, np.nan)

    ar1 = _ar1(mfunc)
    diff_sdhat = np.sqrt(2.0 * (1.0 - ar1)) * func_sd
    diff_sd_mean = diff_sdhat.mean()

    func_diff = np.diff(mfunc, axis=1)  # (Nvox, T-1)
    dvars_nstd = np.sqrt(np.square(func_diff).mean(axis=0))
    if diff_sd_mean > 0:
        dvars_stdz = dvars_nstd / diff_sd_mean
    else:
        dvars_stdz = np.full_like(dvars_nstd, np.nan)

    n = bold_data.shape[3]
    dvars = np.full(n, np.nan)
    std_dvars = np.full(n, np.nan)
    dvars[1:] = dvars_nstd
    std_dvars[1:] = dvars_stdz
    return dvars, std_dvars


# =================================================================================================
# Non-steady-state (dummy scan) detection  (reimplements nipype is_outlier / NonSteadyStateDetector)
# =================================================================================================
def detect_nonsteady_volumes(bold_data: np.ndarray, thresh: float = 3.5) -> int:
    """Number of initial non-steady-state volumes via MAD outlier on the global mean signal."""
    n = bold_data.shape[3]
    global_signal = bold_data.reshape(-1, n).mean(axis=0)
    median = np.median(global_signal)
    diff = np.abs(global_signal - median)
    mad = np.median(diff)
    if mad == 0:
        return 0
    modified_z = 0.6745 * diff / mad
    count = 0
    for z in modified_z:
        if z <= thresh:
            break
        count += 1
    return count


# =================================================================================================
# Outliers
# =================================================================================================
def _one_hot_outliers(flags: np.ndarray, prefix: str) -> "Dict[str, np.ndarray]":
    """One indicator column per flagged index: ``{prefix}{NN}`` with a single 1.0."""
    columns: Dict[str, np.ndarray] = {}
    n = flags.shape[0]
    indices = np.flatnonzero(flags)
    for i, vol in enumerate(indices):
        col = np.zeros(n, dtype=float)
        col[vol] = 1.0
        columns[f"{prefix}{i:02d}"] = col
    return columns


def compute_motion_outliers(
    fd: np.ndarray,
    std_dvars: Optional[np.ndarray],
    fd_thresh: float = FD_OUTLIER_THRESHOLD_MM,
    dvars_thresh: float = STD_DVARS_OUTLIER_THRESHOLD,
    n_nonsteady: int = 0,
) -> "Dict[str, np.ndarray]":
    """``motion_outlier##`` indicator columns (union of FD and std_dvars threshold crossings).

    The first ``n_nonsteady`` frames are excluded from flagging: those volumes are already captured
    by ``non_steady_state_outlier##`` columns, and re-flagging them here would emit a duplicate,
    perfectly collinear one-hot column (a rank-deficient design matrix downstream).
    """
    flags = np.nan_to_num(fd, nan=0.0) > fd_thresh
    if std_dvars is not None:
        flags = flags | (np.nan_to_num(std_dvars, nan=0.0) > dvars_thresh)
    if n_nonsteady > 0:
        flags[: max(0, n_nonsteady)] = False
    return _one_hot_outliers(flags, "motion_outlier")


def nonsteady_outlier_columns(n_volumes: int, n_dummy: int) -> "Dict[str, np.ndarray]":
    """``non_steady_state_outlier##`` indicator columns for the first ``n_dummy`` volumes."""
    flags = np.zeros(n_volumes, dtype=bool)
    flags[: max(0, n_dummy)] = True
    return _one_hot_outliers(flags, "non_steady_state_outlier")


# =================================================================================================
# Tissue masks + mean signals
# =================================================================================================
def _classify_labels(lut: pd.DataFrame) -> Tuple[List[int], List[int]]:
    """Return (csf_labels, wm_labels) from an atlas LUT.

    Prefers an explicit ``region`` column (the brainana/ARM2 ColorLUT carries one with values
    ``WM``/``CSF``/``cortex``/``subcortex``) — this is authoritative and species-agnostic. Falls back
    to substring matching on the most descriptive name column for generic LUTs without a region column.
    The label *value* column is the segmentation's voxel id (``ID``/``index``/first column); ids may be
    negative (ARM2 uses negative ids for WM/CSF), which is fine for ``np.isin``.
    """
    cols = {c.lower(): c for c in lut.columns}
    id_col = cols.get("id") or cols.get("index") or cols.get("label") or lut.columns[0]

    def _safe_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    csf, wm = [], []
    region_col = cols.get("region")
    if region_col is not None:
        for _, row in lut.iterrows():
            label = _safe_int(row[id_col])
            if label is None:
                continue
            region = str(row[region_col]).strip().lower()
            if region == "csf":
                csf.append(label)
            elif region == "wm":
                wm.append(label)
        return csf, wm

    # Fallback: substring match on the most descriptive name column available.
    name_col = (
        cols.get("name_full")
        or cols.get("labelname")
        or cols.get("name")
        or cols.get("structure")
        or lut.columns[min(1, len(lut.columns) - 1)]
    )
    for _, row in lut.iterrows():
        label = _safe_int(row[id_col])
        if label is None or label == 0:
            continue
        name = str(row[name_col]).lower()
        if any(p in name for p in _CSF_NAME_PATTERNS):
            csf.append(label)
        elif any(p in name for p in _WM_NAME_PATTERNS):
            wm.append(label)
    return csf, wm


def _resample_label_to_reference(
    label_file: Union[str, Path], reference_file: Union[str, Path]
) -> np.ndarray:
    """Nearest-neighbor resample a label image onto a reference grid using SimpleITK.

    Both are read directly from disk (geometry from the NIfTI headers). A 4D reference (e.g. a BOLD
    timeseries) is collapsed to its first volume for geometry. Returns the resampled labels as an
    int32 array in **nibabel (x, y, z) order** (SimpleITK's array axes are reversed, so we transpose).
    """
    import SimpleITK as sitk

    seg = sitk.ReadImage(str(label_file))
    ref = sitk.ReadImage(str(reference_file))
    if ref.GetDimension() == 4:
        size = list(ref.GetSize())
        size[3] = 0  # collapse the time axis -> 3D reference geometry
        ref = sitk.Extract(ref, size, [0, 0, 0, 0])
    resampled = sitk.Resample(
        seg, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, seg.GetPixelIDValue()
    )
    arr = sitk.GetArrayFromImage(resampled)  # (z, y, x)
    return np.transpose(arr, (2, 1, 0)).astype(
        np.int32
    )  # -> (x, y, z) to match nibabel


def build_tissue_masks(
    seg_file: Union[str, Path],
    lut_file: Union[str, Path],
    reference_file: Union[str, Path],
    erode_iterations: int = 1,
    logger: Optional[logging.Logger] = None,
) -> "Dict[str, np.ndarray]":
    """Build eroded CSF / WM / combined tissue masks on the BOLD (reference) grid.

    The segmentation is resampled (nearest-neighbor, via SimpleITK) onto ``reference_file``'s grid.
    Tissue classes are identified by the LUT *region* column (or names as fallback), not hard-coded
    integer codes. Masks are lightly eroded to limit partial-volume contamination; if erosion empties a
    mask, the un-eroded mask is used and a warning is logged (no silent all-zero columns).

    Returns a dict possibly containing ``csf``, ``white_matter``, ``csf_wm`` boolean 3D masks.
    """
    from scipy.ndimage import binary_erosion

    if logger is None:
        logger = logging.getLogger(__name__)

    seg = _resample_label_to_reference(seg_file, reference_file)

    lut = pd.read_csv(str(lut_file), sep=None, engine="python")
    csf_labels, wm_labels = _classify_labels(lut)
    if not csf_labels and not wm_labels:
        raise ValueError(
            f"Atlas LUT {lut_file} yielded no CSF/WM labels; cannot build tissue masks. "
            "Check the LUT naming scheme."
        )
    logger.info(f"Confounds: CSF labels {csf_labels}; WM labels {wm_labels}")

    def _mask_for(labels: List[int]) -> Optional[np.ndarray]:
        if not labels:
            return None
        raw = np.isin(seg, labels)
        if not raw.any():
            return None
        eroded = (
            binary_erosion(raw, iterations=erode_iterations)
            if erode_iterations
            else raw
        )
        if not eroded.any():
            logger.warning(
                "Confounds: tissue mask empty after erosion; using un-eroded mask "
                f"(labels {labels})"
            )
            return raw
        return eroded

    masks: Dict[str, np.ndarray] = {}
    csf_mask = _mask_for(csf_labels)
    wm_mask = _mask_for(wm_labels)
    if csf_mask is not None:
        masks["csf"] = csf_mask
    if wm_mask is not None:
        masks["white_matter"] = wm_mask
    if csf_mask is not None or wm_mask is not None:
        combined = np.zeros(seg.shape, dtype=bool)
        if csf_mask is not None:
            combined |= csf_mask
        if wm_mask is not None:
            combined |= wm_mask
        masks["csf_wm"] = combined
    return masks


def mean_signal(bold_data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean BOLD signal across in-mask voxels for each volume."""
    idx = _as_3d_bool_mask(mask)
    if not idx.any():
        return np.full(bold_data.shape[3], np.nan)
    return bold_data[idx].mean(axis=0)


# =================================================================================================
# Assembly
# =================================================================================================
def _ordered_columns(columns: "Dict[str, np.ndarray]") -> List[str]:
    """Order columns to roughly mirror fMRIPrep (cosmetic only; nilearn loads by name)."""

    def base_blocks(prefixes: List[str]) -> List[str]:
        out = []
        for p in prefixes:
            for s in ("", "_derivative1", "_power2", "_derivative1_power2"):
                if f"{p}{s}" in columns:
                    out.append(f"{p}{s}")
        return out

    order: List[str] = []
    order += base_blocks(["global_signal", "csf", "white_matter", "csf_wm"])
    order += [
        c
        for c in ("std_dvars", "dvars", "framewise_displacement", "rmsd")
        if c in columns
    ]
    order += base_blocks(_MOTION_ORDER)
    order += sorted(c for c in columns if c.startswith("non_steady_state_outlier"))
    order += sorted(c for c in columns if c.startswith("motion_outlier"))
    # Anything not explicitly placed goes last, stable.
    order += [c for c in columns if c not in order]
    return order


def _build_json_sidecar(
    columns: "Dict[str, np.ndarray]",
    has_tissue: bool,
    tissue_note: Optional[str],
    mask_note: Optional[str] = None,
    fd_outlier_threshold: float = FD_OUTLIER_THRESHOLD_MM,
    std_dvars_outlier_threshold: float = STD_DVARS_OUTLIER_THRESHOLD,
    radius_mm: float = FD_RADIUS_MM,
) -> Dict[str, Any]:
    """Per-column metadata sidecar (fMRIPrep-style, minimal)."""
    meta: Dict[str, Any] = {}
    for name in columns:
        if name == "framewise_displacement":
            meta[name] = {
                "Method": "Power et al., 2012 (sum of absolute differences)",
                "RotationRadiusMM": radius_mm,
                "Units": "mm",
            }
        elif name == "rmsd":
            meta[name] = {
                "Method": "Relative RMS head displacement (Jenkinson, 1999 sphere formula)",
                "RotationRadiusMM": radius_mm,
                "Units": "mm",
            }
        elif name == "dvars":
            meta[name] = {"Method": "Power et al., 2012 (non-standardized DVARS)"}
        elif name == "std_dvars":
            meta[name] = {"Method": "Power et al., 2012 (standardized DVARS, IQR/AR1)"}
        elif name in ("global_signal", "csf", "white_matter", "csf_wm"):
            meta[name] = {"Method": "Mean signal", "Mask": name}
        elif name.startswith("motion_outlier"):
            meta[name] = {
                "Method": "Outlier flag",
                "FDThresholdMM": fd_outlier_threshold,
                "StdDVARSThreshold": std_dvars_outlier_threshold,
            }
        elif name.startswith("non_steady_state_outlier"):
            meta[name] = {"Method": "Non-steady-state (dummy) volume indicator"}
    meta["__global__"] = {
        "Note": (
            "Confounds are regressors only: the BOLD image is not modified, filtered, scrubbed, or "
            "denoised. Outlier columns are indicators for optional downstream use."
        ),
        "TissueRegressorsProduced": has_tissue,
    }
    if tissue_note:
        meta["__global__"]["TissueRegressorNote"] = tissue_note
    if mask_note:
        meta["__global__"]["BrainMaskNote"] = mask_note
    return meta


def compute_confounds(
    bold_file: Union[str, Path],
    motion_par_file: Optional[Union[str, Path]],
    working_dir: Union[str, Path],
    output_prefix: str,
    brain_mask_file: Optional[Union[str, Path]] = None,
    seg_file: Optional[Union[str, Path]] = None,
    seg_lut_file: Optional[Union[str, Path]] = None,
    n_dummy_min: int = 0,
    radius_mm: float = FD_RADIUS_MM,
    fd_outlier_threshold: float = FD_OUTLIER_THRESHOLD_MM,
    std_dvars_outlier_threshold: float = STD_DVARS_OUTLIER_THRESHOLD,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    """Compute an fMRIPrep-compatible confounds TSV + JSON sidecar.

    Tissue mean signals are only produced when both ``seg_file`` and ``seg_lut_file`` are given
    (i.e. a T1w-space anatomical segmentation is available). Without them, motion-only confounds are
    written and the JSON records that tissue regressors were skipped.

    Returns ``{"confounds_tsv": path, "confounds_json": path}``.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    work_dir = Path(working_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    bold_img = nib.load(str(bold_file))
    bold_data = np.asanyarray(bold_img.dataobj, dtype=np.float32)
    if bold_data.ndim != 4:
        raise ValueError(f"Expected 4D BOLD, got shape {bold_data.shape}")
    n_volumes = bold_data.shape[3]
    # A single volume has no timeseries: every derivative/DVARS/FD column would be n/a. Refuse to
    # write a degenerate file (callers run this under errorStrategy 'ignore', so the run is skipped).
    if n_volumes < 2:
        raise ValueError(
            f"BOLD has {n_volumes} volume(s); confounds require a timeseries (>= 2 volumes)"
        )

    columns: Dict[str, np.ndarray] = {}

    # --- Motion (24-param) + FD + rmsd -----------------------------------------------------------
    if motion_par_file is not None and Path(motion_par_file).exists():
        motion_df = load_motion_params(motion_par_file)
        if len(motion_df) != n_volumes:
            logger.warning(
                f"Confounds: motion params ({len(motion_df)}) != BOLD volumes ({n_volumes})"
            )
        columns.update(expand_motion_params(motion_df))
        fd = compute_framewise_displacement(motion_df, radius_mm=radius_mm)
        columns["framewise_displacement"] = fd
        # rmsd is derived directly from the motion parameters (no external mcflirt _rel.rms file).
        columns["rmsd"] = compute_rmsd(motion_df, radius_mm=radius_mm)
    else:
        logger.warning(
            "Confounds: no motion parameters available; motion columns omitted"
        )
        fd = np.full(n_volumes, np.nan)

    # --- Brain mask (required for DVARS + global signal) -----------------------------------------
    # A real brain mask is required for these columns. We deliberately do NOT fall back to a
    # variance/whole-FOV mask: that would yield mask-based confounds spanning non-brain voxels,
    # i.e. columns that carry fMRIPrep names but a different meaning. Missing is better than wrong —
    # so these columns are skipped (and the reason is recorded in the JSON) when no valid mask exists.
    brain_mask = None
    mask_note: Optional[str] = None
    if brain_mask_file is not None and Path(brain_mask_file).exists():
        mask_img = nib.load(str(brain_mask_file))
        if mask_img.shape[:3] == bold_img.shape[:3]:
            # Masks are sometimes stored 4D with a singleton time axis (x, y, z, 1);
            # _as_3d_bool_mask drops it and binarizes so the index matches the BOLD spatial axes.
            brain_mask = _as_3d_bool_mask(np.asanyarray(mask_img.dataobj))
        else:
            mask_note = (
                "brain mask grid does not match BOLD; global_signal/DVARS skipped"
            )
            logger.warning(f"Confounds: {mask_note}")
    else:
        mask_note = "no brain mask available; global_signal/DVARS skipped"
        logger.info(f"Confounds: {mask_note}")

    # --- DVARS + global signal (only with a valid brain mask) ------------------------------------
    std_dvars: Optional[np.ndarray] = None
    if brain_mask is not None:
        try:
            dvars, std_dvars = compute_dvars(bold_data, brain_mask)
            columns["dvars"] = dvars
            columns["std_dvars"] = std_dvars
        except ValueError as exc:
            logger.warning(f"Confounds: DVARS skipped ({exc})")
            std_dvars = None
        _add_expanded(columns, "global_signal", mean_signal(bold_data, brain_mask))

    # --- Tissue signals (conditional on segmentation) --------------------------------------------
    has_tissue = False
    tissue_note: Optional[str] = None
    if seg_file is not None and seg_lut_file is not None:
        try:
            masks = build_tissue_masks(seg_file, seg_lut_file, bold_file, logger=logger)
            for tissue in ("csf", "white_matter", "csf_wm"):
                if tissue in masks:
                    _add_expanded(
                        columns, tissue, mean_signal(bold_data, masks[tissue])
                    )
                    has_tissue = True
            if not has_tissue:
                tissue_note = "Segmentation provided but no CSF/WM labels resolved."
        except Exception as exc:  # noqa: BLE001 - tissue regressors are best-effort
            logger.warning(f"Confounds: tissue regressors skipped ({exc})")
            tissue_note = f"Tissue regressors skipped: {exc}"
    else:
        tissue_note = (
            "No anatomical segmentation available; tissue regressors not produced."
        )
        logger.info(f"Confounds: {tissue_note}")

    # --- Outliers --------------------------------------------------------------------------------
    n_dummy = max(detect_nonsteady_volumes(bold_data), int(n_dummy_min))
    columns.update(nonsteady_outlier_columns(n_volumes, n_dummy))
    columns.update(
        compute_motion_outliers(
            fd,
            std_dvars,
            fd_thresh=fd_outlier_threshold,
            dvars_thresh=std_dvars_outlier_threshold,
            n_nonsteady=n_dummy,
        )
    )

    # --- Write TSV + JSON ------------------------------------------------------------------------
    order = _ordered_columns(columns)
    df = pd.DataFrame({name: columns[name] for name in order})
    tsv_path = f"{output_prefix}.tsv"
    json_path = f"{output_prefix}.json"
    df.to_csv(tsv_path, sep="\t", index=False, na_rep="n/a")
    with open(json_path, "w") as fh:
        json.dump(
            _build_json_sidecar(
                columns,
                has_tissue,
                tissue_note,
                mask_note,
                fd_outlier_threshold=fd_outlier_threshold,
                std_dvars_outlier_threshold=std_dvars_outlier_threshold,
                radius_mm=radius_mm,
            ),
            fh,
            indent=2,
        )

    logger.info(
        f"Confounds: wrote {len(order)} columns x {n_volumes} volumes "
        f"(tissue={'yes' if has_tissue else 'no'}, dummies={n_dummy})"
    )
    return {"confounds_tsv": tsv_path, "confounds_json": json_path}
