"""SimpleITK rigid registration: a FSL-free, drop-in replacement for FLIRT in the
conform step.

Two layers live here:
- The FLIRT-faithful rigid search + multi-scale schedule (ports flirt.cc search_cost),
  exposed via ``sitk_flirt_register``.
- The conform-facing wrappers ``sitk_register`` / ``sitk_apply_transforms`` /
  ``apply_sitk_affine`` plus the modality profiles (``_SITK_PROFILE_T1W``,
  ``_SITK_PROFILE_FUNC``) and ``sitk_config_for_modality`` that select a single,
  deterministic parameter set.

This module was consolidated from the former ``scripts/sitk_flirt_search.py`` and the
reusable ``_sitk_*`` helpers in ``scripts/test_rigid_reg.py`` so both the benchmark and
``conform_to_template(rigid_method="sitk")`` share one source of truth.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from nhp_mri_prep.operations.validation import (
    ensure_working_directory,
    validate_input_file,
    validate_output_file,
)

logger = logging.getLogger("nhp_mri_prep.sitk_rigid_registration")

# Bump when search/schedule semantics change (benchmark resume invalidation).
SITK_PIPELINE_REV = "corratio_powell_finesearch_v9"

# GEOMETRY seed is preferred only when its fixed/moving overlap beats the COG seed
# by this relative margin, so near-ties keep the FLIRT-faithful COG seed (no
# regression on cases where COG overlap already matches GEOMETRY overlap).
_SITK_SEED_GEOMETRY_MARGIN = 0.1

_SITK_SEARCH_RANGE_DEG = (-180.0, 180.0)
_SITK_STAGE1_TARGET_MM = 8.0
_SITK_STAGE4_TARGET_MM = 4.0
_SITK_SCHEDULE_SCALES_MM = (4.0, 2.0, 1.0)
_SITK_FLIRT_TOP_K = 10
_SITK_TRANSLATION_REFINE_CONVERGENCE = 1e-4
# Corratio + Powell refinement (FLIRT-faithful path): multi-scale pyramid, max Powell
# iterations per scale, and the cap on how many coarse local minima we refine.
_SITK_POWELL_SCALES_MM = (8.0, 4.0, 2.0, 1.0)
_SITK_POWELL_MAXITER = 120
_SITK_CORRATIO_MAX_MINIMA = 8
# FLIRT finesearch: local rotation grid step (deg) and the coarse-cost-range fraction
# used to gate which coarse minima survive (fallback ≈ the old 20th-percentile cut).
_SITK_FINE_STEP_DEG_DEFAULT = 15
_SITK_COST_THRESH_FRACTION_DEFAULT = 0.2


def _unwrap_euler3d(tx: sitk.Transform) -> sitk.Euler3DTransform:
    if isinstance(tx, sitk.Euler3DTransform):
        return tx
    inner: sitk.Transform = (
        tx.GetNthTransform(0) if isinstance(tx, sitk.CompositeTransform) else tx
    )
    result = sitk.Euler3DTransform()
    result.SetParameters(inner.GetParameters())
    result.SetFixedParameters(inner.GetFixedParameters())
    return result


def _set_sitk_metric(
    reg: sitk.ImageRegistrationMethod,
    sitk_config: dict[str, Any],
    *,
    search_stage: bool = False,
    schedule_stage: bool = False,
) -> None:
    if schedule_stage:
        metric = sitk_config.get(
            "schedule_metric", sitk_config.get("search_metric", "Correlation")
        )
    elif search_stage:
        metric = sitk_config.get("search_metric", sitk_config["metric"])
    else:
        metric = sitk_config["metric"]
    if metric == "Correlation":
        reg.SetMetricAsCorrelation()
    elif metric == "MattesMI":
        bins = int(sitk_config.get("number_of_histogram_bins", 32))
        reg.SetMetricAsMattesMutualInformation(bins)
    else:
        raise ValueError(f"Unknown SimpleITK metric: {metric}")


def _sitk_min_spacing(img: sitk.Image) -> float:
    return float(min(img.GetSpacing()))


def _sitk_shrink_for_target_mm(img: sitk.Image, target_mm: float) -> int:
    return max(1, int(round(target_mm / _sitk_min_spacing(img))))


def _sitk_apply_sampling(
    reg: sitk.ImageRegistrationMethod,
    sitk_config: dict[str, Any],
    modality: str,
    *,
    heavy: bool = False,
    full: bool = False,
) -> None:
    # Full (all-voxel) sampling for the coarse search: the 8/4 mm images are tiny so
    # it is nearly free, and it removes the random-sampling noise that otherwise lets a
    # 180 deg flip out-rank the true pose (their cost gap is only ~0.12-0.17).
    if full:
        reg.SetMetricSamplingStrategy(reg.NONE)
        return
    pct = float(sitk_config.get("search_sampling_pct", 0.2))
    if heavy and modality == "func":
        pct = max(pct, 0.25)
    reg.SetMetricSamplingPercentage(pct)
    reg.SetMetricSamplingStrategy(reg.RANDOM)


def _sitk_rotation_samples_deg(
    deg_min: float, deg_max: float, step_deg: float
) -> np.ndarray:
    """Rotation grid CENTERED on 0 (identity): samples 0, +/-step, +/-2*step, ... in range.

    A rotation search must always try identity. The old linspace(deg_min, deg_max, n)
    only landed on 0 when the interval count was even, so the default 40 deg step over
    (-180, 180) skipped 0 while sampling the 180 deg flip exactly — biasing the search to
    flips and never generating the (correct) near-identity pose for cases whose true
    rotation is small (e.g. 032142 ~4.6 deg, 032123 ~12.8 deg). Centering on 0, like
    FLIRT's search, samples identity and symmetric +/- rotations and drops the exact
    180 deg endpoint.
    """
    step = abs(float(step_deg))
    if step <= 0.0:
        return np.array([0.0])
    k_lo = int(np.ceil(deg_min / step))
    k_hi = int(np.floor(deg_max / step))
    if k_hi < k_lo:
        return np.array([0.0])
    return np.arange(k_lo, k_hi + 1, dtype=np.float64) * step


def _sitk_image_cog_mm(img: sitk.Image) -> np.ndarray:
    arr = sitk.GetArrayFromImage(img).astype(np.float64)
    weights = np.maximum(arr, 0.0)
    total = float(weights.sum())
    if total <= 0:
        idx = [(float(s) - 1.0) / 2.0 for s in img.GetSize()]
        return np.array(
            img.TransformContinuousIndexToPhysicalPoint(idx), dtype=np.float64
        )
    z_idx, y_idx, x_idx = np.indices(weights.shape)
    cz = float((weights * z_idx).sum() / total)
    cy = float((weights * y_idx).sum() / total)
    cx = float((weights * x_idx).sum() / total)
    return np.array(
        img.TransformContinuousIndexToPhysicalPoint((cx, cy, cz)), dtype=np.float64
    )


def _sitk_fixed_geometric_center(fixed: sitk.Image) -> tuple[float, float, float]:
    return fixed.TransformContinuousIndexToPhysicalPoint(
        [(float(sz) - 1.0) / 2.0 for sz in fixed.GetSize()]
    )


def _sitk_euler_from_rot_trans(
    center: tuple[float, float, float],
    rx_rad: float,
    ry_rad: float,
    rz_rad: float,
    tx_param: float,
    ty_param: float,
    tz_param: float,
) -> sitk.Euler3DTransform:
    tx_obj = sitk.Euler3DTransform()
    tx_obj.SetCenter(center)
    tx_obj.SetParameters([rx_rad, ry_rad, rz_rad, tx_param, ty_param, tz_param])
    return tx_obj


def _sitk_flirt_cog_seed(
    fixed: sitk.Image,
    moving: sitk.Image,
    center: tuple[float, float, float],
    rx_rad: float,
    ry_rad: float,
    rz_rad: float,
) -> sitk.Euler3DTransform:
    """COG init for the fixed->moving resample transform: place the fixed COG onto the
    moving COG, i.e. tx(ref_cog) = mov_cog, so trans = mov_cog - R @ (ref_cog - c) - c.

    The transform is consumed by ``sitk.Resample(moving, fixed, tx)``, which maps points
    from the FIXED domain into the MOVING domain (each fixed voxel samples moving at
    ``tx(p)``). Overlap therefore requires the fixed COG to sample the moving COG. The
    previous form (``ref_cog - R @ (mov_cog - c) - c``) instead enforced tx(mov_cog) =
    ref_cog — the inverse direction — leaving tx(ref_cog) displaced by ~2x the COG offset
    and, crucially, rotation-invariant (rotating pivots around the wrong correspondence).
    That was harmless when ref_cog ~ mov_cog but produced zero overlap for off-centre FOVs
    (e.g. sub-03), trapping the corratio search in a flat (no-overlap) cost region.
    """
    ref_cog = _sitk_image_cog_mm(fixed)
    mov_cog = _sitk_image_cog_mm(moving)
    rot_tx = _sitk_euler_from_rot_trans(center, rx_rad, ry_rad, rz_rad, 0.0, 0.0, 0.0)
    rotation = np.array(rot_tx.GetMatrix(), dtype=np.float64).reshape(3, 3)
    c = np.array(center, dtype=np.float64)
    t_param = mov_cog - rotation @ (ref_cog - c) - c
    return _sitk_euler_from_rot_trans(
        center,
        rx_rad,
        ry_rad,
        rz_rad,
        float(t_param[0]),
        float(t_param[1]),
        float(t_param[2]),
    )


def _sitk_overlap_voxels(
    fixed: sitk.Image, moving: sitk.Image, tx: sitk.Euler3DTransform
) -> int:
    """Count voxels where fixed and resampled moving are both positive."""
    resampled = sitk.Resample(
        moving, fixed, tx, sitk.sitkLinear, 0.0, moving.GetPixelID()
    )
    fixed_arr = sitk.GetArrayFromImage(fixed)
    moving_arr = sitk.GetArrayFromImage(resampled)
    return int(((fixed_arr > 0) & (moving_arr > 0)).sum())


def _sitk_geometry_seed(
    fixed: sitk.Image,
    moving: sitk.Image,
    rx_rad: float,
    ry_rad: float,
    rz_rad: float,
) -> sitk.Euler3DTransform:
    """GEOMETRY-centered init with explicit rotation (fallback when COG seed has no overlap)."""
    geom = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    params = list(geom.GetParameters())
    params[0:3] = [rx_rad, ry_rad, rz_rad]
    out = sitk.Euler3DTransform()
    out.SetFixedParameters(geom.GetFixedParameters())
    out.SetParameters(params)
    return out


def _sitk_seed_transform(
    fixed: sitk.Image,
    moving: sitk.Image,
    center: tuple[float, float, float],
    rx_rad: float,
    ry_rad: float,
    rz_rad: float,
) -> sitk.Euler3DTransform:
    """Pick the seed (COG vs GEOMETRY) with the larger fixed/moving overlap.

    The COG seed is FLIRT-faithful, but a whole-head / off-center FOV (neck,
    shoulders) drags the intensity centre off the brain, leaving little or no
    overlap with the template. Choosing by overlap — rather than "COG unless
    overlap is exactly zero" — also rescues partial-overlap traps (e.g. 032142),
    where COG overlap is nonzero yet far smaller than the GEOMETRY-centred seed.
    """
    cog_seed = _sitk_flirt_cog_seed(fixed, moving, center, rx_rad, ry_rad, rz_rad)
    cog_overlap = _sitk_overlap_voxels(fixed, moving, cog_seed)
    geom_seed = _sitk_geometry_seed(fixed, moving, rx_rad, ry_rad, rz_rad)
    geom_overlap = _sitk_overlap_voxels(fixed, moving, geom_seed)
    if geom_overlap > cog_overlap * (1.0 + _SITK_SEED_GEOMETRY_MARGIN):
        logger.debug(
            "GEOMETRY seed wins overlap (geom=%d > cog=%d) rot=%.0f,%.0f,%.0f deg",
            geom_overlap,
            cog_overlap,
            np.degrees(rx_rad),
            np.degrees(ry_rad),
            np.degrees(rz_rad),
        )
        return geom_seed
    return cog_seed


def _sitk_refine_translation_only(
    fixed: sitk.Image,
    moving: sitk.Image,
    init_tx: sitk.Euler3DTransform,
    sitk_config: dict[str, Any],
    modality: str,
    *,
    learning_rate_mm: float,
    iters: int,
    full_sampling: bool = False,
) -> sitk.Euler3DTransform:
    reg = sitk.ImageRegistrationMethod()
    _set_sitk_metric(reg, sitk_config, search_stage=True)
    reg.SetInterpolator(sitk.sitkLinear)
    _sitk_apply_sampling(reg, sitk_config, modality, heavy=True, full=full_sampling)
    reg.SetOptimizerAsGradientDescent(
        learningRate=learning_rate_mm,
        numberOfIterations=iters,
        convergenceMinimumValue=_SITK_TRANSLATION_REFINE_CONVERGENCE,
        convergenceWindowSize=5,
    )
    reg.SetOptimizerScales([1e6, 1e6, 1e6, 1.0, 1.0, 1.0])
    reg.SetInitialTransform(init_tx, inPlace=False)
    return _unwrap_euler3d(reg.Execute(fixed, moving))


def _sitk_eval_metric(
    fixed: sitk.Image,
    moving: sitk.Image,
    tx: sitk.Euler3DTransform,
    sitk_config: dict[str, Any],
    modality: str,
    *,
    full_sampling: bool = False,
) -> float:
    """Evaluate search metric at tx without moving parameters (ITK needs >=1 iter)."""
    reg = sitk.ImageRegistrationMethod()
    _set_sitk_metric(reg, sitk_config, search_stage=True)
    reg.SetInterpolator(sitk.sitkLinear)
    _sitk_apply_sampling(reg, sitk_config, modality, full=full_sampling)
    reg.SetOptimizerAsGradientDescent(
        learningRate=0.0,
        numberOfIterations=1,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=1,
    )
    # Freeze all DOF so the single iteration only samples the metric at init.
    reg.SetOptimizerScales([1e6] * 6)
    reg.SetInitialTransform(tx, inPlace=False)
    reg.Execute(fixed, moving)
    return float(reg.GetMetricValue())


def _sitk_corratio_cost(
    fixed: sitk.Image, moving: sitk.Image, tx: sitk.Euler3DTransform, nbins: int = 32
) -> float:
    """FLIRT-style correlation ratio CR(fixed | moving), negated so lower = better.

    Bins the resampled moving intensities and measures the residual within-bin variance
    of the fixed image. Unlike normalized cross-correlation it is robust to the 180 deg
    flip ambiguity at coarse scale (verified: it ranks the true pose above a flip by a
    ~2x larger margin), which is what NCC's thin margin cannot do once the coarse pose
    is only crudely positioned.
    """
    resampled = sitk.Resample(
        moving, fixed, tx, sitk.sitkLinear, 0.0, moving.GetPixelID()
    )
    x = sitk.GetArrayViewFromImage(fixed).astype(np.float64).ravel()
    y = sitk.GetArrayViewFromImage(resampled).astype(np.float64).ravel()
    ymin, ymax = float(y.min()), float(y.max())
    tot_var = float(x.var())
    if ymax <= ymin or tot_var <= 0.0:
        return 0.0
    bins = np.clip(((y - ymin) / (ymax - ymin) * nbins).astype(np.intp), 0, nbins - 1)
    n = np.bincount(bins, minlength=nbins).astype(np.float64)
    sx = np.bincount(bins, weights=x, minlength=nbins)
    sx2 = np.bincount(bins, weights=x * x, minlength=nbins)
    nz = n > 0
    within = np.zeros(nbins, dtype=np.float64)
    within[nz] = sx2[nz] - (sx[nz] ** 2) / n[nz]  # = n_k * Var_k(fixed)
    cr = 1.0 - within.sum() / (x.size * tot_var)
    return -float(cr)


def _sitk_rank_cost(
    fixed: sitk.Image,
    moving: sitk.Image,
    tx: sitk.Euler3DTransform,
    sitk_config: dict[str, Any],
    modality: str,
    *,
    full_sampling: bool = False,
) -> float:
    """Scalar cost used to RANK coarse-search poses (lower = better).

    Defaults to the SimpleITK search metric; set ``search_rank_metric='CorrelationRatio'``
    to rank by correlation ratio instead (translation refinement still uses the SimpleITK
    metric, which has no corratio). Opt-in: the func/MattesMI path is unaffected.
    """
    if sitk_config.get("search_rank_metric") == "CorrelationRatio":
        return _sitk_corratio_cost(
            fixed, moving, tx, int(sitk_config.get("corratio_bins", 32))
        )
    return _sitk_eval_metric(
        fixed, moving, tx, sitk_config, modality, full_sampling=full_sampling
    )


def _sitk_copy_euler(tx: sitk.Euler3DTransform) -> sitk.Euler3DTransform:
    out = sitk.Euler3DTransform()
    out.SetFixedParameters(tx.GetFixedParameters())
    out.SetParameters(tx.GetParameters())
    return out


def _sitk_fine_offsets_deg(coarse_step_deg: float, fine_step_deg: float) -> np.ndarray:
    """FLIRT finesearch offsets (incl. 0) within +/-coarse_step/2, stepped by fine_step.

    These tile the angular gap *owned* by each retained coarse minimum at the finer
    rotation resolution. Degenerates to [0.0] when fine_step >= coarse_step/2 (or is
    non-positive), so the fine search then just re-evaluates the coarse minimum — the
    pre-finesearch behaviour. coarse=40/fine=15 -> [-15, 0, 15]; coarse=30/fine=10 ->
    [-10, 0, 10].
    """
    half = abs(float(coarse_step_deg)) / 2.0
    fine = abs(float(fine_step_deg))
    if fine <= 0.0 or half <= 0.0:
        return np.array([0.0])
    k = int(np.floor(half / fine))
    if k <= 0:
        return np.array([0.0])
    return np.arange(-k, k + 1, dtype=np.float64) * fine


def _sitk_cost_range_threshold(costs: np.ndarray, fraction: float) -> float:
    """Keep-threshold at ``min + fraction*(max-min)`` over the finite coarse costs.

    Range-relative (not "% of best cost") so it is monotone and sign-safe for corratio's
    negative costs. Returns +inf for an empty/all-non-finite input so nothing is dropped.
    """
    arr = np.asarray(list(costs), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("inf")
    cmin = float(arr.min())
    cmax = float(arr.max())
    return cmin + float(fraction) * (cmax - cmin)


def _sitk_fine_rotation_search(
    fixed_8: sitk.Image,
    moving_8: sitk.Image,
    center: tuple[float, float, float],
    base_deg: tuple[float, float, float],
    offsets_deg: np.ndarray,
    sitk_config: dict[str, Any],
    modality: str,
    lr_tx: float,
    tx_iters: int,
    score_fn: Any,
) -> tuple[float, sitk.Euler3DTransform]:
    """Translation-optimise a fine rotation grid around ``base_deg`` and return the best.

    For every (rx,ry,rz) on ``base_deg + offsets`` the rotation is held fixed while the
    translation is optimised (FLIRT-faithful), then ``score_fn(tx)`` ranks the pose
    (lower = better). Returns ``(best_score, best_tx)``.
    """
    best_score = float("inf")
    best_tx: sitk.Euler3DTransform | None = None
    for dx in offsets_deg:
        for dy in offsets_deg:
            for dz in offsets_deg:
                seed = _sitk_seed_transform(
                    fixed_8,
                    moving_8,
                    center,
                    np.deg2rad(base_deg[0] + dx),
                    np.deg2rad(base_deg[1] + dy),
                    np.deg2rad(base_deg[2] + dz),
                )
                refined = _sitk_refine_translation_only(
                    fixed_8,
                    moving_8,
                    seed,
                    sitk_config,
                    modality,
                    learning_rate_mm=lr_tx,
                    iters=tx_iters,
                    full_sampling=True,
                )
                score = float(score_fn(refined))
                if score < best_score:
                    best_score = score
                    best_tx = refined
    return best_score, best_tx


def _sitk_flirt_search_cost(
    fixed_8: sitk.Image,
    moving_8: sitk.Image,
    fixed_4: sitk.Image,
    moving_4: sitk.Image,
    sitk_config: dict[str, Any],
    modality: str,
) -> list[tuple[float, sitk.Euler3DTransform]]:
    """FLIRT-faithful coarse rotation search returning the lowest-cost candidates.

    For every rotation on the coarse grid we optimise the translation, then score the
    cost *at that rotation's own optimised translation*, and rank rotations by cost.

    The previous implementation discarded each rotation's cost and instead trilinearly
    interpolated the optimised translations across the (coarse, ~40 deg-spaced) grid
    before scoring a finer rotation grid. That interpolation handed good rotations a
    wrong translation, so their cost looked bad and they were filtered out before the
    schedule — verified on 032142/032123: the correctly-aligned coarse cell was found
    (full-res masked NCC ~0.17/0.26) yet never survived into the candidate list (~0.02/
    0.11). Scoring each rotation at its own translation, with no interpolation, keeps it.
    """
    t0 = time.perf_counter()
    deg_min, deg_max = sitk_config.get("search_range_deg", _SITK_SEARCH_RANGE_DEG)
    coarse_step = float(sitk_config["coarse_step_deg"])
    fine_step = float(sitk_config.get("fine_step_deg", _SITK_FINE_STEP_DEG_DEFAULT))
    cost_frac = float(
        sitk_config.get("cost_thresh_fraction", _SITK_COST_THRESH_FRACTION_DEFAULT)
    )
    tx_iters = int(sitk_config.get("search_tx_iters", 50))
    lr_tx = _sitk_min_spacing(fixed_8) * 0.5

    center = _sitk_fixed_geometric_center(fixed_8)
    rx_c = _sitk_rotation_samples_deg(deg_min, deg_max, coarse_step)
    ry_c = _sitk_rotation_samples_deg(deg_min, deg_max, coarse_step)
    rz_c = _sitk_rotation_samples_deg(deg_min, deg_max, coarse_step)
    logger.info(
        "FLIRT search coarse: %dx%dx%d rotations @ ~%.1f mm, tx_iters=%d",
        len(rx_c),
        len(ry_c),
        len(rz_c),
        _sitk_min_spacing(fixed_8),
        tx_iters,
    )

    def _rank4(tx: sitk.Euler3DTransform) -> float:
        # Score at the finer 4 mm scale with full (all-voxel) sampling.
        return _sitk_rank_cost(
            fixed_4, moving_4, tx, sitk_config, modality, full_sampling=True
        )

    # Coarse pass: optimise TRANSLATION only, holding each grid rotation fixed (FLIRT's
    # search_cost does the same). Refining rotation here lets poses drift into
    # high-correlation 180 deg flips; FLIRT instead stays robust via its corratio cost +
    # per-local-minimum refinement.
    coarse: list[tuple[float, tuple[float, float, float]]] = []
    for rx in rx_c:
        for ry in ry_c:
            for rz in rz_c:
                seed = _sitk_seed_transform(
                    fixed_8,
                    moving_8,
                    center,
                    np.deg2rad(rx),
                    np.deg2rad(ry),
                    np.deg2rad(rz),
                )
                refined = _sitk_refine_translation_only(
                    fixed_8,
                    moving_8,
                    seed,
                    sitk_config,
                    modality,
                    learning_rate_mm=lr_tx,
                    iters=tx_iters,
                    full_sampling=True,
                )
                coarse.append((_rank4(refined), (float(rx), float(ry), float(rz))))
    coarse.sort(key=lambda r: r[0])
    thresh = _sitk_cost_range_threshold([c for c, _ in coarse], cost_frac)
    retained = [(c, rot) for c, rot in coarse if c <= thresh][:_SITK_FLIRT_TOP_K]
    logger.info(
        "FLIRT search coarse done: %d cells in %.1f s; best cost=%.6f; "
        "%d retained (cost<=%.6f)",
        len(coarse),
        time.perf_counter() - t0,
        coarse[0][0],
        len(retained),
        thresh,
    )

    # FLIRT finesearch: refine each retained coarse minimum on a local rotation grid
    # (fine_step within +/-coarse_step/2) before the schedule.
    fine_offsets = _sitk_fine_offsets_deg(coarse_step, fine_step)
    scored: list[tuple[float, sitk.Euler3DTransform]] = []
    for _, rot in retained:
        score, tx = _sitk_fine_rotation_search(
            fixed_8,
            moving_8,
            center,
            rot,
            fine_offsets,
            sitk_config,
            modality,
            lr_tx,
            tx_iters,
            _rank4,
        )
        if tx is not None:
            scored.append((score, tx))
    scored.sort(key=lambda r: r[0])

    # Polish the top candidates' translation at 4 mm and re-rank before the schedule.
    t1 = time.perf_counter()
    lr_tx4 = _sitk_min_spacing(fixed_4) * 0.5
    results: list[tuple[float, sitk.Euler3DTransform]] = []
    for _, tx in scored[:_SITK_FLIRT_TOP_K]:
        refined = _sitk_refine_translation_only(
            fixed_4,
            moving_4,
            tx,
            sitk_config,
            modality,
            learning_rate_mm=lr_tx4,
            iters=tx_iters,
            full_sampling=True,
        )
        results.append((_rank4(refined), refined))
    results.sort(key=lambda r: r[0])
    logger.info(
        "FLIRT search refine top-%d @ 4 mm: best cost=%.6f (%.1f s, total %.1f s)",
        len(results),
        results[0][0],
        time.perf_counter() - t1,
        time.perf_counter() - t0,
    )
    return results[:_SITK_FLIRT_TOP_K]


def _sitk_rigid_refine_at_scale(
    fixed: sitk.Image,
    moving: sitk.Image,
    init_tx: sitk.Euler3DTransform,
    sitk_config: dict[str, Any],
    modality: str,
    target_mm: float,
    iters: int,
) -> sitk.Euler3DTransform:
    shrink = _sitk_shrink_for_target_mm(fixed, target_mm)
    shrink_factors = [shrink]
    sigmas = [max(0.0, target_mm / 2.0)]
    lr = _sitk_min_spacing(fixed) * 0.5 * float(sitk_config.get("learning_rate", 1.0))

    reg = sitk.ImageRegistrationMethod()
    _set_sitk_metric(reg, sitk_config, schedule_stage=True)
    reg.SetInterpolator(sitk.sitkLinear)
    _sitk_apply_sampling(reg, sitk_config, modality, heavy=True)
    reg.SetOptimizerAsGradientDescent(
        learningRate=lr,
        numberOfIterations=iters,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(shrink_factors)
    reg.SetSmoothingSigmasPerLevel(sigmas)
    reg.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(True)
    reg.SetInitialTransform(init_tx, inPlace=False)
    init_metric = _sitk_eval_metric(fixed, moving, init_tx, sitk_config, modality)
    out_tx = _unwrap_euler3d(reg.Execute(fixed, moving))
    out_metric = float(reg.GetMetricValue())
    if out_metric > init_metric:
        logger.info(
            "FLIRT schedule @ %.1f mm: metric worsened (%.6f > %.6f); keeping init",
            target_mm,
            out_metric,
            init_metric,
        )
        return _sitk_copy_euler(init_tx)
    return out_tx


def _sitk_flirt_schedule(
    fixed: sitk.Image,
    moving: sitk.Image,
    candidates: list[tuple[float, sitk.Euler3DTransform]],
    sitk_config: dict[str, Any],
    modality: str,
) -> sitk.Euler3DTransform:
    """Simplified defaultschedule: 4 mm top-3, 2 mm best, 1 mm polish."""
    if not candidates:
        raise RuntimeError("FLIRT schedule: no search candidates")
    schedule_iters = int(sitk_config.get("schedule_iters", 40))
    scales = sitk_config.get("schedule_scales_mm", _SITK_SCHEDULE_SCALES_MM)

    pool = list(candidates[:3])
    logger.info("FLIRT schedule: re-score %d candidates", len(pool))
    rescored: list[tuple[float, sitk.Euler3DTransform]] = []
    for _, tx in pool:
        m = _sitk_rank_cost(fixed, moving, tx, sitk_config, modality)
        rescored.append((m, tx))
    rescored.sort(key=lambda x: x[0])

    best_tx = _sitk_copy_euler(rescored[0][1])
    for scale_mm in scales:
        logger.info(
            "FLIRT schedule: rigid optimise @ %.1f mm (%d iters)",
            scale_mm,
            schedule_iters,
        )
        best_tx = _sitk_rigid_refine_at_scale(
            fixed,
            moving,
            best_tx,
            sitk_config,
            modality,
            float(scale_mm),
            schedule_iters,
        )
    return best_tx


def _sitk_euler_params(
    center: tuple[float, float, float], params: np.ndarray
) -> sitk.Euler3DTransform:
    tx = sitk.Euler3DTransform()
    tx.SetCenter(center)
    tx.SetParameters([float(v) for v in params])
    return tx


def _sitk_find_cost_minima(
    grid: np.ndarray, thresh: float
) -> list[tuple[int, int, int]]:
    """Local minima (26-neighbour) of the cost-vs-rotation grid, below `thresh`.

    Port of FLIRT find_cost_minima: refining every basin (not just the global best)
    keeps a correctly-oriented but globally-suboptimal pose in contention.
    """
    nx, ny, nz = grid.shape
    minima: list[tuple[int, int, int]] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if grid[i, j, k] > thresh:
                    continue
                sub = grid[
                    max(0, i - 1) : i + 2, max(0, j - 1) : j + 2, max(0, k - 1) : k + 2
                ]
                if grid[i, j, k] <= float(sub.min()) + 1e-9:
                    minima.append((i, j, k))
    return minima


def _sitk_corratio_powell_refine(
    fixed: sitk.Image,
    moving: sitk.Image,
    center: tuple[float, float, float],
    params0: np.ndarray,
    scales_mm: tuple[float, ...],
    nbins: int,
    maxiter: int,
) -> np.ndarray:
    """Multi-scale Powell refinement of the 6 rigid params, minimising corratio.

    FLIRT optimises corratio with Powell (derivative-free). Gradient descent on
    correlation drifts into the higher-correlation 180 deg flip at coarse scale; Powell
    on corratio holds the true basin (verified: converges to FLIRT's exact rotation).
    """
    from scipy.optimize import minimize  # lazy: only the corratio path needs scipy

    p = np.array(params0, dtype=np.float64)
    for mm in scales_mm:
        shrink = _sitk_shrink_for_target_mm(fixed, mm)
        fs = sitk.Shrink(fixed, [shrink] * 3) if shrink > 1 else fixed
        ms = sitk.Shrink(moving, [shrink] * 3) if shrink > 1 else moving

        def _cost(q: np.ndarray) -> float:
            return _sitk_corratio_cost(fs, ms, _sitk_euler_params(center, q), nbins)

        p = minimize(
            _cost,
            p,
            method="Powell",
            options={"maxiter": maxiter, "xtol": 1e-3, "ftol": 1e-4},
        ).x
    return p


def _sitk_corratio_register(
    fixed: sitk.Image,
    moving: sitk.Image,
    fixed_8: sitk.Image,
    moving_8: sitk.Image,
    fixed_4: sitk.Image,
    moving_4: sitk.Image,
    sitk_config: dict[str, Any],
    modality: str,
) -> sitk.Euler3DTransform:
    """FLIRT-faithful corratio path (search_rank_metric='CorrelationRatio').

    Coarse corratio cost-vs-rotation grid (translation-only per cell) -> find_cost_minima
    below a ``cost_thresh_fraction`` cost-range gate -> FLIRT finesearch (local rotation
    grid at ``fine_step_deg`` around each minimum) PLUS the identity/COG seed -> refine
    each with corratio+Powell -> pick the best by final corratio. The identity seed is
    FLIRT's own initial estimate; it rescues co-oriented truths whose near-identity region
    is not itself a coarse local minimum (e.g. 032123), which local-minima selection alone
    would miss.
    """
    t0 = time.perf_counter()
    deg_min, deg_max = sitk_config.get("search_range_deg", _SITK_SEARCH_RANGE_DEG)
    coarse_step = float(sitk_config["coarse_step_deg"])
    fine_step = float(sitk_config.get("fine_step_deg", _SITK_FINE_STEP_DEG_DEFAULT))
    cost_frac = float(
        sitk_config.get("cost_thresh_fraction", _SITK_COST_THRESH_FRACTION_DEFAULT)
    )
    tx_iters = int(sitk_config.get("search_tx_iters", 50))
    lr_tx = _sitk_min_spacing(fixed_8) * 0.5
    nbins = int(sitk_config.get("corratio_bins", 32))
    center = _sitk_fixed_geometric_center(fixed_8)
    rs = _sitk_rotation_samples_deg(deg_min, deg_max, coarse_step)
    n = len(rs)

    def _corratio4(tx: sitk.Euler3DTransform) -> float:
        return _sitk_corratio_cost(fixed_4, moving_4, tx, nbins)

    grid = np.full((n, n, n), np.inf, dtype=np.float64)
    for i, rx in enumerate(rs):
        for j, ry in enumerate(rs):
            for k, rz in enumerate(rs):
                seed = _sitk_seed_transform(
                    fixed_8,
                    moving_8,
                    center,
                    np.deg2rad(rx),
                    np.deg2rad(ry),
                    np.deg2rad(rz),
                )
                refined = _sitk_refine_translation_only(
                    fixed_8,
                    moving_8,
                    seed,
                    sitk_config,
                    modality,
                    learning_rate_mm=lr_tx,
                    iters=tx_iters,
                    full_sampling=True,
                )
                grid[i, j, k] = _corratio4(refined)

    thresh = _sitk_cost_range_threshold(grid.ravel(), cost_frac)
    minima = _sitk_find_cost_minima(grid, thresh)
    minima.sort(key=lambda c: grid[c])
    minima = minima[:_SITK_CORRATIO_MAX_MINIMA]

    # FLIRT finesearch: refine each retained coarse minimum on a local rotation grid
    # (fine_step within +/-coarse_step/2) before corratio+Powell.
    fine_offsets = _sitk_fine_offsets_deg(coarse_step, fine_step)
    seeds: list[np.ndarray] = []
    for i, j, k in minima:
        base = (float(rs[i]), float(rs[j]), float(rs[k]))
        _, tx = _sitk_fine_rotation_search(
            fixed_8,
            moving_8,
            center,
            base,
            fine_offsets,
            sitk_config,
            modality,
            lr_tx,
            tx_iters,
            _corratio4,
        )
        if tx is not None:
            seeds.append(np.array(tx.GetParameters(), dtype=np.float64))
    # FLIRT's initial estimate: identity rotation at the COG/geometry seed.
    id_seed = _sitk_seed_transform(fixed_8, moving_8, center, 0.0, 0.0, 0.0)
    seeds.append(np.array(id_seed.GetParameters(), dtype=np.float64))
    logger.info(
        "FLIRT corratio search: %d coarse cells, %d minima (thresh=%.6f) + identity "
        "seed (%.1f s)",
        grid.size,
        len(minima),
        thresh,
        time.perf_counter() - t0,
    )

    scales = sitk_config.get("powell_scales_mm", _SITK_POWELL_SCALES_MM)
    maxiter = int(sitk_config.get("powell_maxiter", _SITK_POWELL_MAXITER))
    best_cost = float("inf")
    best_params = seeds[-1]
    for p0 in seeds:
        p = _sitk_corratio_powell_refine(
            fixed, moving, center, p0, scales, nbins, maxiter
        )
        cost = _sitk_corratio_cost(fixed, moving, _sitk_euler_params(center, p), nbins)
        if cost < best_cost:
            best_cost = cost
            best_params = p
    logger.info(
        "FLIRT corratio refine: %d seeds, best corratio cost=%.6f (total %.1f s)",
        len(seeds),
        best_cost,
        time.perf_counter() - t0,
    )
    return _sitk_euler_params(center, best_params)


def sitk_flirt_register(
    fixed: sitk.Image,
    moving: sitk.Image,
    sitk_config: dict[str, Any],
    modality: str,
) -> sitk.Euler3DTransform:
    """Full FLIRT-like pipeline on in-memory images."""
    shrink_8 = _sitk_shrink_for_target_mm(fixed, _SITK_STAGE1_TARGET_MM)
    shrink_4 = _sitk_shrink_for_target_mm(fixed, _SITK_STAGE4_TARGET_MM)
    fixed_8 = sitk.Shrink(fixed, [shrink_8] * 3)
    moving_8 = sitk.Shrink(moving, [shrink_8] * 3)
    fixed_4 = sitk.Shrink(fixed, [shrink_4] * 3)
    moving_4 = sitk.Shrink(moving, [shrink_4] * 3)

    # FLIRT-faithful corratio + Powell + find_cost_minima path (anat default); the
    # correlation/MattesMI gradient-descent path is unchanged for func and other configs.
    if sitk_config.get("search_rank_metric") == "CorrelationRatio":
        return _sitk_corratio_register(
            fixed, moving, fixed_8, moving_8, fixed_4, moving_4, sitk_config, modality
        )

    candidates = _sitk_flirt_search_cost(
        fixed_8, moving_8, fixed_4, moving_4, sitk_config, modality
    )
    return _sitk_flirt_schedule(fixed, moving, candidates, sitk_config, modality)


# ===========================================================================
# Conform-facing layer: modality profiles, transform I/O, register + apply
# (consolidated from scripts/test_rigid_reg.py).
# ===========================================================================


@dataclass(frozen=True)
class SitkModalityProfile:
    """FLIRT-like search + schedule knobs and optional grid sweeps."""

    pyramid_target_mm: tuple[float, ...]
    metric: str
    search_metric: str = "Correlation"
    schedule_metric: str | None = None
    histogram_bins: tuple[int, ...] = ()
    learning_rates: tuple[float, ...] = (1.0,)
    coarse_step_deg_options: tuple[int, ...] = (40,)
    fine_step_deg_options: tuple[int, ...] = (15,)
    cost_thresh_fraction_options: tuple[float, ...] = (0.2,)
    search_tx_iters_options: tuple[int, ...] = (50,)
    schedule_iters_options: tuple[int, ...] = (40,)
    search_sampling_pct: float = 0.2
    # "CorrelationRatio" selects the FLIRT-faithful corratio + Powell + find_cost_minima
    # search/refine path (anat); None keeps the correlation/MattesMI gradient-descent path.
    search_rank_metric: str | None = None


# T1w / anatomical conform profile (single deterministic config: each *_options is len 1).
# Mirrors FSL FLIRT anat defaults: corratio, coarsesearch 40 deg, finesearch 15 deg.
_SITK_PROFILE_T1W = SitkModalityProfile(
    pyramid_target_mm=(8.0, 4.0, 2.0),
    metric="Correlation",
    search_metric="Correlation",
    search_rank_metric="CorrelationRatio",
    histogram_bins=(32,),
    coarse_step_deg_options=(40,),
    fine_step_deg_options=(15,),
    cost_thresh_fraction_options=(0.1,),
    search_tx_iters_options=(30,),
    schedule_iters_options=(50,),
)
# Reserved for a future T2w-specific profile; until then T2w uses the T1w profile.
_SITK_PROFILE_T2W = _SITK_PROFILE_T1W
_SITK_PROFILE_FUNC = SitkModalityProfile(
    pyramid_target_mm=(8.0, 4.0, 2.0),
    metric="MattesMI",
    search_metric="MattesMI",
    schedule_metric="MattesMI",
    histogram_bins=(32,),
    learning_rates=(1.0,),
    coarse_step_deg_options=(30,),
    fine_step_deg_options=(10,),
    cost_thresh_fraction_options=(0.1,),
    search_tx_iters_options=(30,),
    schedule_iters_options=(50,),
)


def sitk_profile_for_modality(modality: str) -> SitkModalityProfile:
    """Return the SimpleITK profile for a modality (``func`` vs anatomical T1w)."""
    return _SITK_PROFILE_FUNC if modality == "func" else _SITK_PROFILE_T1W


def _sitk_lr_tag(learning_rate: float) -> str:
    return f"lr{learning_rate:.1f}".replace(".", "p")


def _sitk_cost_thresh_tag(fraction: float) -> str:
    return f"ct{int(round(fraction * 100))}"


def _sitk_param_label(
    *,
    coarse_step_deg: int,
    fine_step_deg: int,
    cost_thresh_fraction: float,
    search_tx_iters: int,
    schedule_iters: int,
    histogram_bins: int | None = None,
    learning_rate: float = 1.0,
    include_coarse: bool = False,
    include_bins: bool = False,
    include_learning_rate: bool = False,
) -> str:
    coarse_part = f"cs{coarse_step_deg:02d}_" if include_coarse else ""
    bins_part = (
        f"_b{histogram_bins}" if include_bins and histogram_bins is not None else ""
    )
    lr_part = f"_{_sitk_lr_tag(learning_rate)}" if include_learning_rate else ""
    ct = _sitk_cost_thresh_tag(cost_thresh_fraction)
    return (
        f"sitk_flirt_{coarse_part}fs{fine_step_deg:02d}_{ct}_ti{search_tx_iters}_si{schedule_iters}"
        f"{bins_part}{lr_part}"
    )


def _build_sitk_param_grid(profile: SitkModalityProfile) -> list[dict[str, Any]]:
    """FLIRT-faithful search + schedule; sweep coarse/fine step, threshold, tx/schedule iters."""
    include_lr = len(profile.learning_rates) > 1
    include_bins = profile.metric == "MattesMI" and len(profile.histogram_bins) > 0
    include_coarse = len(profile.coarse_step_deg_options) > 1
    grid: list[dict[str, Any]] = []
    bin_values: tuple[int | None, ...] = (
        profile.histogram_bins if include_bins else (None,)
    )
    lr_values = profile.learning_rates if include_lr else (1.0,)
    for histogram_bin in bin_values:
        for learning_rate in lr_values:
            for coarse_step_deg in profile.coarse_step_deg_options:
                for fine_step_deg in profile.fine_step_deg_options:
                    for cost_thresh in profile.cost_thresh_fraction_options:
                        for search_tx_iters in profile.search_tx_iters_options:
                            for schedule_iters in profile.schedule_iters_options:
                                label = _sitk_param_label(
                                    coarse_step_deg=coarse_step_deg,
                                    fine_step_deg=fine_step_deg,
                                    cost_thresh_fraction=cost_thresh,
                                    search_tx_iters=search_tx_iters,
                                    schedule_iters=schedule_iters,
                                    histogram_bins=histogram_bin,
                                    learning_rate=learning_rate,
                                    include_coarse=include_coarse,
                                    include_bins=include_bins,
                                    include_learning_rate=include_lr,
                                )
                                schedule_metric = (
                                    profile.schedule_metric
                                    if profile.schedule_metric is not None
                                    else profile.metric
                                )
                                entry: dict[str, Any] = {
                                    "metric": profile.metric,
                                    "search_metric": profile.search_metric,
                                    "schedule_metric": schedule_metric,
                                    "search_range_deg": _SITK_SEARCH_RANGE_DEG,
                                    "coarse_step_deg": coarse_step_deg,
                                    "fine_step_deg": fine_step_deg,
                                    "cost_thresh_fraction": cost_thresh,
                                    "search_tx_iters": search_tx_iters,
                                    "schedule_iters": schedule_iters,
                                    "search_sampling_pct": profile.search_sampling_pct,
                                    "learning_rate": learning_rate,
                                    "label": label,
                                }
                                if histogram_bin is not None:
                                    entry["number_of_histogram_bins"] = histogram_bin
                                if profile.search_rank_metric is not None:
                                    entry[
                                        "search_rank_metric"
                                    ] = profile.search_rank_metric
                                grid.append(entry)
    return grid


def sitk_config_for_modality(modality: str) -> dict[str, Any]:
    """Single deterministic ``sitk_config`` for the conform step (no grid sweep).

    Each profile's ``*_options`` tuples are length-1, so the built grid has exactly one
    entry; we return it with a stable ``label`` keyed on modality.
    """
    profile = sitk_profile_for_modality(modality)
    config = _build_sitk_param_grid(profile)[0]
    config["label"] = "func" if modality == "func" else "anat"
    return config


def _sitk_pipeline_rev_path(variant_dir: Path) -> Path:
    return variant_dir / ".sitk_pipeline_rev"


def _sitk_ensure_3d(img: sitk.Image) -> sitk.Image:
    """Return a 3-D version of img, computing the temporal mean if img is 4-D.

    If the image is already 3-D it is returned unchanged.
    """
    if img.GetDimension() != 4:
        return img
    arr = sitk.GetArrayFromImage(img)  # shape (t, z, y, x) in SimpleITK array order
    mean_arr = arr.mean(axis=0).astype(np.float32)
    out = sitk.GetImageFromArray(mean_arr)
    out.SetSpacing(img.GetSpacing()[:3])
    out.SetOrigin(img.GetOrigin()[:3])
    out.SetDirection([img.GetDirection()[i] for i in (0, 1, 2, 4, 5, 6, 8, 9, 10)])
    return out


def _sitk_tx_to_matrix(tx: sitk.Euler3DTransform) -> np.ndarray:
    """Convert centered Euler3DTransform to 4x4 affine (world-space, SimpleITK LPS)."""
    matrix = np.eye(4, dtype=np.float64)
    rotation = np.array(tx.GetMatrix(), dtype=np.float64).reshape(3, 3)
    center = np.array(tx.GetCenter(), dtype=np.float64)
    translation = np.array(tx.GetTranslation(), dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation + center - rotation @ center
    return matrix


def _sitk_affine_lps(img: sitk.Image) -> np.ndarray:
    """Voxel-index -> physical (SimpleITK LPS world) 4x4 affine for an image."""
    direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.array(img.GetSpacing(), dtype=np.float64)
    origin = np.array(img.GetOrigin(), dtype=np.float64)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin
    return affine


def _sitk_fsl_scale(img: sitk.Image) -> np.ndarray:
    """Voxel-index -> FSL-mm scaling for an image (diag(pixdim), x-flipped for a
    positive affine determinant — what FSL itself calls NEUROLOGICAL storage order —
    matching how FSL/FLIRT build their internal mm coordinates).

    Note the flip offset ``(size[0] - 1) * spacing[0]``: it depends on the grid size, so
    the FSL frame of a padded reference differs from the unpadded one along x by
    ``pad_right``, not ``pad_left``. See :func:`fsl_mat_for_new_reference`."""
    spacing = np.array(img.GetSpacing(), dtype=np.float64)
    size = img.GetSize()
    scale = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    if np.linalg.det(_sitk_affine_lps(img)[:3, :3]) > 0:
        scale[0, 0] = -spacing[0]
        scale[0, 3] = (size[0] - 1) * spacing[0]
    return scale


def _sitk_tx_to_fsl_matrix(
    tx: sitk.Euler3DTransform, fixed: sitk.Image, moving: sitk.Image
) -> np.ndarray:
    """Convert the sitk world transform to an FSL FLIRT matrix.

    Produces a drop-in for FLIRT's ``conform_scanner2native.mat`` — usable with
    ``flirt -in <brain> -ref <template> -applyxfm -init``. The sitk transform maps
    fixed(template)->moving(brain) in LPS world; FLIRT's matrix maps in->ref in FSL-mm,
    so we go world -> voxel-to-voxel -> FSL-mm and invert the direction. Verified by
    applying via FSL applyxfm: reproduces the registration (NCC matches FLIRT).
    """
    world = _sitk_tx_to_matrix(tx)  # template-world -> brain-world (LPS)
    a_fixed = _sitk_affine_lps(fixed)
    a_moving = _sitk_affine_lps(moving)
    vox2vox = np.linalg.inv(a_moving) @ world @ a_fixed  # template-vox -> brain-vox
    s_fixed = _sitk_fsl_scale(fixed)
    s_moving = _sitk_fsl_scale(moving)
    return np.linalg.inv(s_moving @ vox2vox @ np.linalg.inv(s_fixed))


def _sitk_world_mat_path(mat_path: Path) -> Path:
    """Sidecar path holding the world-space affine (for sitk resume reconstruction)."""
    return mat_path.with_suffix(".world.mat")


def conform_world_mat_path(conform_xfm: "Path | str") -> Path:
    """Locate the ``.world.mat`` sidecar for a conform transform (forward or inverse).

    ``sitk_register`` writes ``<prefix>.mat``, ``<prefix>_inverse.mat`` and the
    ``<prefix>.world.mat`` sidecar. Given either the forward or the ``_inverse`` ``.mat``
    path, return the shared world-space sidecar so :func:`apply_sitk_affine` can reload the
    affine (use ``invert=True`` to apply the inverse direction).
    """
    p = Path(conform_xfm)
    stem = p.stem
    if stem.endswith("_inverse"):
        stem = stem[: -len("_inverse")]
    return _sitk_world_mat_path(p.with_name(stem + p.suffix))


def _sitk_load_world_for_resume(mat_path: Path) -> np.ndarray:
    """Load the world-space affine for resume: the `.world.mat` sidecar (new FSL-primary
    layout) if present, else the primary `.mat` (legacy runs that stored world there).
    """
    world = _sitk_world_mat_path(mat_path)
    return np.loadtxt(world if world.is_file() else mat_path)


def _sitk_save_transform_artifacts(
    tx: sitk.Euler3DTransform,
    mat_path: Path,
    fixed: sitk.Image,
    moving: sitk.Image,
    *,
    param_set: str | None = None,
) -> None:
    """Save the transform as a FLIRT-compatible FSL matrix (primary .mat) plus its inverse
    (`_inverse.mat`), mirroring FLIRT's conform_scanner2native[/_inverse].mat outputs, and
    the world-space affine as a `.world.mat` sidecar used to rebuild the tx on resume.
    """
    forward_fsl = _sitk_tx_to_fsl_matrix(tx, fixed, moving)
    np.savetxt(mat_path, forward_fsl)
    inverse_path = mat_path.with_name(f"{mat_path.stem}_inverse{mat_path.suffix}")
    np.savetxt(inverse_path, np.linalg.inv(forward_fsl))
    np.savetxt(_sitk_world_mat_path(mat_path), _sitk_tx_to_matrix(tx))
    if param_set is not None:
        _sitk_pipeline_rev_path(mat_path.parent).write_text(
            SITK_PIPELINE_REV + "\n", encoding="utf-8"
        )


def _sitk_transform_from_mat(
    mat: np.ndarray, registration_fixedf: Path
) -> sitk.Euler3DTransform:
    """Rebuild centered Euler3DTransform from saved 4x4 affine + registration fixed image."""
    fixed = _sitk_ensure_3d(
        sitk.ReadImage(
            str(validate_input_file(registration_fixedf, logger)), sitk.sitkFloat32
        )
    )
    center = _sitk_fixed_geometric_center(fixed)
    rotation = mat[:3, :3].astype(np.float64)
    t_full = mat[:3, 3].astype(np.float64)
    c = np.array(center, dtype=np.float64)
    t_param = t_full - c + rotation @ c
    tx = sitk.Euler3DTransform()
    tx.SetCenter(center)
    tx.SetMatrix(rotation.reshape(-1).tolist())
    tx.SetTranslation(t_param.tolist())
    return tx


def sitk_register(
    fixedf: Path,
    movingf: Path,
    work_dir: Path,
    output_prefix: str,
    sitk_config: dict[str, Any],
    modality: str = "anat",
) -> dict[str, Any]:
    """Rigid registration: FLIRT search_cost + default schedule (sitk_flirt_search).

    Returns ``{"forward_transform": <.mat path>, "transform_obj": <Euler3DTransform>}``.
    Writes ``<output_prefix>.mat`` (FSL forward), ``<output_prefix>_inverse.mat`` (FSL
    inverse) and ``<output_prefix>.world.mat`` (LPS world affine).
    """
    fixed_path = validate_input_file(fixedf, logger)
    moving_path = validate_input_file(movingf, logger)
    work_dir = ensure_working_directory(work_dir, logger)

    fixed = _sitk_ensure_3d(sitk.ReadImage(str(fixed_path), sitk.sitkFloat32))
    moving = _sitk_ensure_3d(sitk.ReadImage(str(moving_path), sitk.sitkFloat32))

    label = sitk_config.get("label", "default")
    prof = sitk_profile_for_modality(modality)
    logger.info(
        "SimpleITK FLIRT register (%s): coarse=%s° fine=%s° ct=%.2f "
        "ti=%s si=%s search=%s schedule=%s (rev=%s)",
        label,
        sitk_config.get("coarse_step_deg", prof.coarse_step_deg_options[0]),
        sitk_config.get("fine_step_deg", _SITK_FINE_STEP_DEG_DEFAULT),
        sitk_config.get("cost_thresh_fraction", 0.2),
        sitk_config.get("search_tx_iters", 50),
        sitk_config.get("schedule_iters", 40),
        sitk_config.get("search_metric", prof.search_metric),
        sitk_config.get(
            "schedule_metric",
            prof.schedule_metric if prof.schedule_metric else prof.metric,
        ),
        SITK_PIPELINE_REV,
    )
    t0 = time.perf_counter()
    final_tx = sitk_flirt_register(fixed, moving, sitk_config, modality)
    logger.info(
        "SimpleITK FLIRT pipeline finished in %.1f s (label=%s)",
        time.perf_counter() - t0,
        label,
    )

    forward_transform = work_dir / f"{output_prefix}.mat"
    _sitk_save_transform_artifacts(
        final_tx, forward_transform, fixed, moving, param_set=label
    )
    validate_output_file(forward_transform, logger)
    logger.info("SimpleITK transform saved to %s", forward_transform)

    return {
        "forward_transform": str(forward_transform),
        "transform_obj": final_tx,
    }


def sitk_apply_transforms(
    movingf: Path,
    outputf_name: str,
    reff: Path,
    work_dir: Path,
    transform_obj: sitk.Euler3DTransform,
    interpolation: int = sitk.sitkLinear,
) -> dict[str, str]:
    """Apply an in-memory SimpleITK rigid transform to an image via resampling."""
    moving_path = validate_input_file(movingf, logger)
    ref_path = validate_input_file(reff, logger)
    work_dir = ensure_working_directory(work_dir, logger)

    fixed = _sitk_ensure_3d(sitk.ReadImage(str(ref_path), sitk.sitkFloat32))
    moving = _sitk_ensure_3d(sitk.ReadImage(str(moving_path), sitk.sitkFloat32))
    resampled = sitk.Resample(
        moving,
        fixed,
        transform_obj,
        interpolation,
        0.0,
        moving.GetPixelID(),
    )

    output_path = work_dir / outputf_name
    sitk.WriteImage(resampled, str(output_path))
    validate_output_file(output_path, logger)
    return {"imagef_registered": str(output_path)}


def sitk_resample_to_spacing(
    in_path: "Path | str",
    out_path: "Path | str",
    target_spacing: "tuple[float, float, float]",
    interpolation: int = sitk.sitkBSpline,
    logger: logging.Logger = logger,
) -> str:
    """Resample a 3-D image to a target voxel spacing (FSL/AFNI-free).

    Equivalent to ``3dresample -dxyz <sx> <sy> <sz> -rmode Cu``: keeps the image's origin,
    direction and physical field-of-view, recomputing the voxel grid size for the new
    spacing. BSpline interpolation matches AFNI's cubic resampling for template prep. Used
    by the ``rigid_method='sitk'`` conform path so it needs no AFNI binary.
    """
    in_p = validate_input_file(in_path, logger)
    img = sitk.ReadImage(str(in_p))
    in_spacing = np.array(img.GetSpacing(), dtype=np.float64)
    in_size = np.array(img.GetSize(), dtype=np.int64)
    target = np.asarray(target_spacing, dtype=np.float64)
    out_size = [
        max(1, int(round(float(sz) * float(sp) / float(tp))))
        for sz, sp, tp in zip(in_size, in_spacing, target)
    ]
    resampled = sitk.Resample(
        img,
        out_size,
        sitk.Transform(),  # identity: pure grid change in the same physical space
        interpolation,
        img.GetOrigin(),
        tuple(float(t) for t in target),
        img.GetDirection(),
        0.0,
        img.GetPixelID(),
    )
    sitk.WriteImage(resampled, str(out_path))
    validate_output_file(out_path, logger)
    return str(out_path)


def apply_sitk_affine(
    movingf: Path,
    outputf_name: str,
    reff: Path,
    working_dir: Path,
    world_mat: "Path | str | np.ndarray",
    *,
    interpolation: int = sitk.sitkLinear,
    invert: bool = False,
) -> dict[str, str]:
    """Apply a stored world-space (LPS) affine to an image via SimpleITK resampling.

    FSL-free counterpart to ``flirt_apply_transforms`` for the conform forward/inverse.
    ``world_mat`` is a 4x4 numpy array or a path to a ``.world.mat`` sidecar written by
    :func:`_sitk_save_transform_artifacts`. The world matrix maps fixed(template)->moving
    in LPS, which is exactly what SimpleITK's Resample expects (output grid = ``reff``);
    pass ``invert=True`` for the inverse direction (e.g. template->scanner when
    reprojecting atlases). No registration fixed image is needed — the matrix carries the
    full affine, so there is no transform-center to reconstruct.
    """
    moving_path = validate_input_file(movingf, logger)
    ref_path = validate_input_file(reff, logger)
    working_dir = ensure_working_directory(working_dir, logger)

    if isinstance(world_mat, np.ndarray):
        mat = np.asarray(world_mat, dtype=np.float64)
    else:
        mat = np.loadtxt(str(validate_input_file(world_mat, logger)))
    mat = np.asarray(mat, dtype=np.float64)
    if invert:
        mat = np.linalg.inv(mat)

    transform = sitk.AffineTransform(3)
    transform.SetMatrix(mat[:3, :3].reshape(-1).tolist())
    transform.SetTranslation(mat[:3, 3].tolist())

    # Read in native pixel type so label maps keep their integer dtype.
    moving = _sitk_ensure_3d(sitk.ReadImage(str(moving_path)))
    ref = _sitk_ensure_3d(sitk.ReadImage(str(ref_path)))
    resampled = sitk.Resample(
        moving, ref, transform, interpolation, 0.0, moving.GetPixelID()
    )

    output_path = working_dir / outputf_name
    sitk.WriteImage(resampled, str(output_path))
    validate_output_file(output_path, logger)
    return {"imagef_registered": str(output_path)}


# ---------------------------------------------------------------------------
# Full-FOV conform: grid maths for an enlarged reference that crops nothing
# ---------------------------------------------------------------------------
# The conform reference grid is derived entirely from the template (padded 10%,
# resampled to the input resolution), so it is sized for a brain, not a head:
# a recording chamber, head-post or the neck falls outside it and is cropped.
# The helpers below size a strictly larger, voxel-aligned reference that contains
# every voxel of the moving image, so the extra output is a super-volume of the
# standard conformed image rather than a differently-gridded sibling.

# Guard rails for the enlarged grid. The conform resolution is min(input spacing)
# applied to *all three* axes, so an anisotropic scan over a whole-head FOV can
# explode: 0.2x0.2x2.0 mm at 1024x1024x40 becomes ~4.2e8 voxels (1.7 GB float32).
DEFAULT_FULL_FOV_MAX_VOXELS = 512**3  # ~1.34e8 voxels, ~537 MB as float32
DEFAULT_FULL_FOV_MAX_GROWTH = 3.0  # per-axis cap on (pad_left + pad_right) / size


def fsl_mat_to_world_affine(
    mat: "np.ndarray | Path | str",
    fixedf: "Path | str",
    movingf: "Path | str",
) -> np.ndarray:
    """FSL FLIRT matrix -> LPS world affine (fixed-world -> moving-world).

    Exact algebraic inverse of :func:`_sitk_tx_to_fsl_matrix`: FLIRT's matrix maps
    moving-FSL-mm -> fixed-FSL-mm, and FSL-mm is ``index * pixdim`` (x-flipped when
    the stored orientation has a positive affine determinant), so

        world = a_moving @ inv(s_moving) @ inv(mat) @ s_fixed @ inv(a_fixed)

    ``fixedf`` must be the grid the matrix is *applied* with (``template_for_xfm``),
    not the grid it was estimated on (``template_f_for_reg``) — the goal is to
    reproduce where FLIRT actually puts the voxels.
    """
    m = (
        np.asarray(mat, dtype=np.float64)
        if isinstance(mat, np.ndarray)
        else np.loadtxt(str(validate_input_file(mat, logger)))
    )
    fixed = _sitk_ensure_3d(sitk.ReadImage(str(validate_input_file(fixedf, logger))))
    moving = _sitk_ensure_3d(sitk.ReadImage(str(validate_input_file(movingf, logger))))

    a_fixed, a_moving = _sitk_affine_lps(fixed), _sitk_affine_lps(moving)
    s_fixed, s_moving = _sitk_fsl_scale(fixed), _sitk_fsl_scale(moving)
    return (
        a_moving
        @ np.linalg.inv(s_moving)
        @ np.linalg.inv(np.asarray(m, dtype=np.float64))
        @ s_fixed
        @ np.linalg.inv(a_fixed)
    )


def world_mat_to_vox2vox(
    world_mat: np.ndarray, reff: "Path | str", movingf: "Path | str"
) -> np.ndarray:
    """Moving voxel index -> reference voxel index, from an LPS world affine.

    ``world_mat`` maps reference-world -> moving-world (the direction
    :func:`sitk.Resample` consumes), so the index-space map is
    ``inv(A_ref) @ inv(world) @ A_moving``.
    """
    ref = _sitk_ensure_3d(sitk.ReadImage(str(validate_input_file(reff, logger))))
    moving = _sitk_ensure_3d(sitk.ReadImage(str(validate_input_file(movingf, logger))))
    return (
        np.linalg.inv(_sitk_affine_lps(ref))
        @ np.linalg.inv(np.asarray(world_mat, dtype=np.float64))
        @ _sitk_affine_lps(moving)
    )


def fsl_mat_for_new_reference(
    mat: "np.ndarray | Path | str", old_reff: "Path | str", new_reff: "Path | str"
) -> np.ndarray:
    """Re-express an FSL matrix for a reference grid with a different FOV.

    FSL-mm coordinates are tied to the reference's own voxel grid, so a matrix
    estimated against ``old_reff`` is invalid once the grid is padded. Changing
    frames through world space gives

        mat_new = S_new @ inv(A_new) @ A_old @ inv(S_old) @ mat_old

    For a pure zero-pad this collapses to a translation, but *not* a symmetric one:
    on the x axis of a positive-determinant reference FSL measures from the far edge
    (``(size-1) * pixdim``), so that component moves with ``pad_right`` while the
    others move with ``pad_left``. Hand-rolling ``t += spacing * pad_left`` therefore
    introduces an x error of ``spacing * (pad_left_x - pad_right_x)`` on exactly the
    templates this pipeline ships. Compute it through S/A instead.
    """
    m = (
        np.asarray(mat, dtype=np.float64)
        if isinstance(mat, np.ndarray)
        else np.loadtxt(str(validate_input_file(mat, logger)))
    )
    old = _sitk_ensure_3d(sitk.ReadImage(str(validate_input_file(old_reff, logger))))
    new = _sitk_ensure_3d(sitk.ReadImage(str(validate_input_file(new_reff, logger))))

    change_of_frame = (
        _sitk_fsl_scale(new)
        @ np.linalg.inv(_sitk_affine_lps(new))
        @ _sitk_affine_lps(old)
        @ np.linalg.inv(_sitk_fsl_scale(old))
    )
    if not np.allclose(change_of_frame[:3, :3], np.eye(3), atol=1e-9):
        raise ValueError(
            "Reference grids differ by more than a translation (spacing or direction "
            "changed); the FSL matrix cannot be re-expressed by padding alone."
        )
    return change_of_frame @ np.asarray(m, dtype=np.float64)


def reference_padding_to_cover(
    vox2vox: np.ndarray,
    moving_shape: "tuple[int, int, int]",
    ref_shape: "tuple[int, int, int]",
    *,
    margin: int = 1,
    max_growth: float = DEFAULT_FULL_FOV_MAX_GROWTH,
) -> "tuple[np.ndarray, np.ndarray]":
    """Voxel padding of the reference grid needed to contain the whole moving image.

    Maps the eight corners of the moving image's *outer-edge* box
    (``-0.5 .. size-0.5``) through ``vox2vox`` and takes the axis-aligned hull. Edges,
    not centres: SimpleITK's linear interpolator returns data over ``[-0.5, size-0.5)``,
    so centre corners would discard a nonzero half-voxel slab on every face — exactly
    the outermost data (the chamber rim) this output exists to keep. An affine map
    sends the convex hull of the corners to the hull of their images, so eight corners
    are exact even for oblique geometry.

    Both pads are clamped at zero, which makes the padded grid a strict superset of the
    reference, so the two differ by a whole number of voxels and the standard conformed
    image sits at ``pad_left`` inside the enlarged one.

    Args:
        vox2vox: 4x4 moving-index -> reference-index affine.
        moving_shape: Spatial shape of the moving image, in ``(x, y, z)`` index order.
        ref_shape: Spatial shape of the reference grid, in ``(x, y, z)`` index order.
        margin: Extra voxels per face. One by default: it costs almost nothing, covers
            FLIRT's zero-pad ramp (nonzero support out to a full voxel, wider than
            SimpleITK's edge clamp) and absorbs the floor/ceil rounding.
        max_growth: Per-axis sanity cap on total padding as a multiple of the reference
            size. A NaN or garbage transform yields an absurd hull; this catches it
            before anything is allocated.

    Returns:
        ``(pad_left, pad_right)``, non-negative int arrays of shape ``(3,)``.

    Raises:
        ValueError: If ``vox2vox`` is not finite, or the padding exceeds ``max_growth``.
    """
    vox2vox = np.asarray(vox2vox, dtype=np.float64)
    if not np.isfinite(vox2vox).all():
        raise ValueError(f"Non-finite moving->reference transform:\n{vox2vox}")

    moving_shape = np.asarray(moving_shape, dtype=np.float64)[:3]
    ref_shape = np.asarray(ref_shape, dtype=np.int64)[:3]

    # Eight outer-edge corners of the moving image, as homogeneous columns.
    lows, highs = np.full(3, -0.5), moving_shape - 0.5
    corners = np.array(
        [
            [
                lows[0] if not i & 1 else highs[0],
                lows[1] if not i & 2 else highs[1],
                lows[2] if not i & 4 else highs[2],
                1.0,
            ]
            for i in range(8)
        ]
    ).T
    mapped = (vox2vox @ corners)[:3]

    lo = np.floor(mapped.min(axis=1)).astype(np.int64) - margin
    hi = np.ceil(mapped.max(axis=1)).astype(np.int64) + margin

    pad_left = np.maximum(0, -lo).astype(int)
    pad_right = np.maximum(0, hi - (ref_shape - 1)).astype(int)

    growth = (pad_left + pad_right) / np.maximum(ref_shape, 1)
    if np.any(growth > max_growth):
        raise ValueError(
            f"Implausible full-FOV padding (left={list(pad_left)}, right={list(pad_right)}) "
            f"for reference shape {list(ref_shape)}: grows by up to {growth.max():.1f}x, "
            f"cap is {max_growth}x. The rigid transform is probably wrong."
        )
    return pad_left, pad_right
