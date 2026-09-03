"""Unit tests for fMRIPrep-compatible confound computation (operations/confounds.py).

These cover the dependency-light numpy paths (motion expansion, FD, DVARS, outliers, non-steady
detection, assembly + JSON). Nilearn integration (load_confounds, clean_img) is in
test_nilearn_confounds.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import pytest

from nhp_mri_prep.operations import confounds as C


def _write_par(path: Path, n: int = 20) -> np.ndarray:
    """Write a synthetic FSL .par (rot_x,rot_y,rot_z,trans_x,trans_y,trans_z) and return it."""
    rng = np.random.RandomState(0)
    arr = np.cumsum(rng.normal(0, 0.01, size=(n, 6)), axis=0)
    np.savetxt(str(path), arr)
    return arr


def test_motion_expansion_naming_and_order():
    df = pd.DataFrame({c: np.arange(5, dtype=float) for c in C._MOTION_ORDER})
    cols = C.expand_motion_params(df)
    # 6 params x 4 terms = 24 columns
    assert len(cols) == 24
    for base in C._MOTION_ORDER:
        assert base in cols
        assert f"{base}_derivative1" in cols
        assert f"{base}_power2" in cols
        assert f"{base}_derivative1_power2" in cols
    # derivative first sample is NaN
    assert np.isnan(cols["trans_x_derivative1"][0])
    # power2 == raw**2
    np.testing.assert_allclose(cols["trans_x_power2"], df["trans_x"].to_numpy() ** 2)


def test_load_motion_params_reorders_to_fmriprep(tmp_path):
    par = tmp_path / "mc.par"
    arr = _write_par(par, n=10)
    df = C.load_motion_params(par)
    assert list(df.columns) == C._MOTION_ORDER
    # trans_x in fMRIPrep order == column 3 of the FSL .par
    np.testing.assert_allclose(df["trans_x"].to_numpy(), arr[:, 3])
    np.testing.assert_allclose(df["rot_x"].to_numpy(), arr[:, 0])


def test_framewise_displacement_power_formula():
    # Construct a known step in trans_x at frame 2.
    n = 5
    df = pd.DataFrame({c: np.zeros(n) for c in C._MOTION_ORDER})
    df.loc[2:, "trans_x"] = 1.0  # 1mm jump at frame 2
    fd = C.compute_framewise_displacement(df, radius_mm=27.0)
    assert np.isnan(fd[0])
    assert fd[1] == pytest.approx(0.0)
    assert fd[2] == pytest.approx(1.0)  # |Δtrans_x| = 1mm, all else 0
    assert fd[3] == pytest.approx(0.0)
    # Rotation contribution scales by radius.
    df2 = pd.DataFrame({c: np.zeros(n) for c in C._MOTION_ORDER})
    df2.loc[1:, "rot_z"] = 0.01
    fd2 = C.compute_framewise_displacement(df2, radius_mm=27.0)
    assert fd2[1] == pytest.approx(0.01 * 27.0)


def test_compute_rmsd_zero_motion_and_known_translation():
    # No motion -> rmsd all zero after the n/a first sample.
    df0 = pd.DataFrame({c: np.zeros(4) for c in C._MOTION_ORDER})
    r0 = C.compute_rmsd(df0, radius_mm=27.0)
    assert np.isnan(r0[0])
    assert np.allclose(r0[1:], 0.0)

    # Pure 1mm translation in x at frame 1 -> rmsd == 1mm (rotation term zero).
    df = pd.DataFrame({c: np.zeros(3) for c in C._MOTION_ORDER})
    df.loc[1:, "trans_x"] = 1.0
    r = C.compute_rmsd(df, radius_mm=27.0)
    assert r[1] == pytest.approx(1.0)
    assert r[2] == pytest.approx(0.0)


def test_dvars_constant_image_is_zero_or_nan():
    # Constant-in-time image -> zero temporal difference -> zero-variance voxels dropped -> NaN.
    data = np.ones((4, 4, 4, 10), dtype=np.float32)
    mask = np.ones((4, 4, 4), dtype=np.uint8)
    dvars, std_dvars = C.compute_dvars(data, mask)
    assert np.isnan(dvars[0]) and np.isnan(std_dvars[0])


def test_dvars_shapes_and_first_nan():
    rng = np.random.RandomState(1)
    data = rng.normal(100, 5, size=(5, 5, 5, 12)).astype(np.float32)
    mask = np.ones((5, 5, 5), dtype=np.uint8)
    dvars, std_dvars = C.compute_dvars(data, mask)
    assert dvars.shape == (12,) and std_dvars.shape == (12,)
    assert np.isnan(dvars[0]) and np.isnan(std_dvars[0])
    assert np.all(np.isfinite(dvars[1:]))


def test_detect_nonsteady_volumes():
    rng = np.random.RandomState(2)
    data = rng.normal(100, 1, size=(4, 4, 4, 30)).astype(np.float32)
    # Make first 2 volumes bright outliers (typical T1 saturation).
    data[..., 0] += 80
    data[..., 1] += 60
    n = C.detect_nonsteady_volumes(data)
    assert n == 2


def test_motion_outliers_union():
    fd = np.array([np.nan, 0.1, 0.9, 0.2])
    std_dvars = np.array([np.nan, 0.1, 0.1, 2.0])
    cols = C.compute_motion_outliers(fd, std_dvars, fd_thresh=0.5, dvars_thresh=1.5)
    # Frame 2 (FD) and frame 3 (DVARS) flagged -> two one-hot columns.
    assert len(cols) == 2
    assert cols["motion_outlier00"][2] == 1.0
    assert cols["motion_outlier01"][3] == 1.0


def test_motion_outliers_exclude_nonsteady_frames():
    # A high-motion frame inside the non-steady-state window must NOT get a motion_outlier column
    # (it is already captured by non_steady_state_outlier## -> avoids a duplicate collinear column).
    fd = np.array([np.nan, 0.9, 0.1, 0.9])
    std_dvars = np.array([np.nan, 2.0, 0.1, 0.1])
    cols = C.compute_motion_outliers(
        fd, std_dvars, fd_thresh=0.5, dvars_thresh=1.5, n_nonsteady=2
    )
    # Frames 0 and 1 are excluded; only frame 3 survives.
    assert len(cols) == 1
    assert cols["motion_outlier00"][3] == 1.0
    assert cols["motion_outlier00"][1] == 0.0


def test_load_motion_params_named_tsv(tmp_path):
    # brainana's intermediate motion table (named header incl. enorm) must load + reorder.
    n = 6
    cols = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
    d = pd.DataFrame({c: np.arange(n, dtype=float) for c in cols})
    d["enorm"] = 0.0
    p = tmp_path / "desc-motion_timeseries.tsv"
    d.to_csv(p, sep="\t", index=False)
    df = C.load_motion_params(p)
    assert list(df.columns) == C._MOTION_ORDER
    assert len(df) == n


def test_load_motion_params_scientific_notation_no_header(tmp_path):
    # Raw .par with scientific notation must NOT be mistaken for a header (the 'e').
    p = tmp_path / "mc.par"
    p.write_text(
        "1.0e-05 2.0e-03 -1.0e-04 0.5 -0.5 0.25\n2.0e-05 3.0e-03 -2.0e-04 0.6 -0.4 0.20\n"
    )
    df = C.load_motion_params(p)
    assert list(df.columns) == C._MOTION_ORDER
    assert df["trans_x"].iloc[0] == 0.5


def test_classify_labels_uses_region_column_arm2_style():
    # Mirror the real ARM2 ColorLUT: 'region' column is authoritative; ids are negative for WM/CSF;
    # 'name' holds short codes (ctxWM/latVent) that name-substring matching would miss.
    lut = pd.DataFrame(
        {
            "ID": [0, 2, -1, -1001, -2, -1002],
            "LabelName": [
                "Background",
                "cortex-rh-ACgG",
                "WM-rh-ctxWM",
                "WM-lh-ctxWM",
                "CSF-rh-latVent",
                "CSF-lh-latVent",
            ],
            "region": ["", "cortex", "WM", "WM", "CSF", "CSF"],
            "name": ["", "ACgG", "ctxWM", "ctxWM", "latVent", "latVent"],
            "name_full": [
                "",
                "anterior_cingulate_gyrus",
                "cerebral_white_matter",
                "cerebral_white_matter",
                "lateral_ventricle",
                "lateral_ventricle",
            ],
        }
    )
    csf, wm = C._classify_labels(lut)
    assert set(wm) == {-1, -1001}
    assert set(csf) == {-2, -1002}


def test_build_tissue_masks_with_segmentation(tmp_path):
    # Tissue path uses SimpleITK to resample the seg onto the BOLD grid (no nilearn).
    # Build a tiny segmentation: label 2 = white matter block, label 4 = CSF block.
    seg = np.zeros((10, 10, 10), dtype=np.int16)
    seg[2:8, 2:8, 2:8] = 2  # WM region (large so erosion leaves voxels)
    seg[0:3, 0:3, 0:3] = 4  # CSF region
    affine = np.eye(4)
    seg_file = tmp_path / "aseg.nii.gz"
    nib.save(nib.Nifti1Image(seg, affine), str(seg_file))
    lut = tmp_path / "lut.tsv"
    # name-only LUT (no 'region' col) exercises the substring fallback classifier.
    pd.DataFrame(
        {
            "index": [0, 2, 4],
            "name": [
                "Background",
                "Left-Cerebral-White-Matter",
                "Left-Lateral-Ventricle",
            ],
        }
    ).to_csv(lut, sep="\t", index=False)

    # Reference BOLD on the same grid (4D), passed as a file path.
    ref_file = tmp_path / "bold.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((10, 10, 10, 5), dtype=np.float32), affine),
        str(ref_file),
    )
    masks = C.build_tissue_masks(seg_file, lut, ref_file, erode_iterations=1)
    assert "white_matter" in masks and masks["white_matter"].any()
    assert "csf" in masks
    assert "csf_wm" in masks


def test_compute_confounds_motion_only_end_to_end(tmp_path):
    n = 20
    par = tmp_path / "mc.par"
    _write_par(par, n=n)
    rng = np.random.RandomState(3)
    bold = nib.Nifti1Image(
        rng.normal(100, 5, size=(6, 6, 6, n)).astype(np.float32), affine=np.eye(4)
    )
    bold_file = tmp_path / "bold.nii.gz"
    nib.save(bold, str(bold_file))
    mask_file = tmp_path / "mask.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones((6, 6, 6), dtype=np.uint8), np.eye(4)), str(mask_file)
    )

    out = C.compute_confounds(
        bold_file=bold_file,
        motion_par_file=par,
        working_dir=tmp_path / "work",
        output_prefix=str(tmp_path / "sub-X_desc-confounds_timeseries"),
        brain_mask_file=mask_file,
    )
    df = pd.read_csv(out["confounds_tsv"], sep="\t")
    assert len(df) == n
    # Core fMRIPrep columns present (mask supplied -> global/DVARS computed).
    for col in [
        "trans_x",
        "trans_x_derivative1",
        "trans_x_power2",
        "trans_x_derivative1_power2",
        "rot_z",
        "global_signal",
        "global_signal_derivative1",
        "framewise_displacement",
        "dvars",
        "std_dvars",
    ]:
        assert col in df.columns, f"missing {col}"
    # Tissue columns absent without segmentation.
    assert "csf" not in df.columns
    assert "white_matter" not in df.columns
    # enorm is intentionally NOT in the confounds file (framewise_displacement replaces it).
    assert "enorm" not in df.columns
    # First derivative / FD sample is n/a.
    assert df["framewise_displacement"].isna().iloc[0]
    assert df["trans_x_derivative1"].isna().iloc[0]

    # JSON sidecar present and records no tissue regressors.
    meta = json.loads(Path(out["confounds_json"]).read_text())
    assert meta["__global__"]["TissueRegressorsProduced"] is False
    assert "framewise_displacement" in meta
    assert meta["framewise_displacement"]["RotationRadiusMM"] == C.FD_RADIUS_MM


def test_compute_confounds_accepts_4d_singleton_mask(tmp_path):
    # Brain masks are sometimes stored 4D as (x, y, z, 1); the boolean index must still be 3D.
    n = 18
    par = tmp_path / "mc.par"
    _write_par(par, n=n)
    rng = np.random.RandomState(7)
    nib.save(
        nib.Nifti1Image(
            rng.normal(100, 5, size=(5, 5, 5, n)).astype(np.float32), np.eye(4)
        ),
        str(tmp_path / "bold.nii.gz"),
    )
    # 4D mask with a singleton time axis (the shape that crashed FUNC_COMPUTE_CONFOUNDS).
    nib.save(
        nib.Nifti1Image(np.ones((5, 5, 5, 1), dtype=np.uint8), np.eye(4)),
        str(tmp_path / "mask.nii.gz"),
    )
    out = C.compute_confounds(
        bold_file=tmp_path / "bold.nii.gz",
        motion_par_file=par,
        working_dir=tmp_path / "work",
        output_prefix=str(tmp_path / "sub-Z_desc-confounds_timeseries"),
        brain_mask_file=tmp_path / "mask.nii.gz",
    )
    df = pd.read_csv(out["confounds_tsv"], sep="\t")
    # Mask-based columns are produced (mask was valid, just 4D).
    for col in ("global_signal", "dvars", "std_dvars"):
        assert col in df.columns
    assert np.isfinite(df["dvars"].to_numpy()[1:]).all()


def test_compute_confounds_without_mask_skips_global_and_dvars(tmp_path):
    n = 16
    par = tmp_path / "mc.par"
    _write_par(par, n=n)
    rng = np.random.RandomState(4)
    bold = nib.Nifti1Image(
        rng.normal(100, 5, size=(5, 5, 5, n)).astype(np.float32), affine=np.eye(4)
    )
    bold_file = tmp_path / "bold.nii.gz"
    nib.save(bold, str(bold_file))

    out = C.compute_confounds(
        bold_file=bold_file,
        motion_par_file=par,
        working_dir=tmp_path / "work",
        output_prefix=str(tmp_path / "sub-Y_desc-confounds_timeseries"),
        brain_mask_file=None,  # no mask -> mask-based columns must be skipped, not faked
    )
    df = pd.read_csv(out["confounds_tsv"], sep="\t")
    # Mask-based columns omitted rather than computed over a bogus whole-FOV mask.
    for col in ("global_signal", "dvars", "std_dvars"):
        assert col not in df.columns, f"{col} should be absent without a brain mask"
    # Motion-only confounds still present (rmsd is mask-independent, derived from motion params).
    assert "trans_x" in df.columns
    assert "framewise_displacement" in df.columns
    assert "rmsd" in df.columns
    assert np.isfinite(df["rmsd"].to_numpy()[1:]).all()
    # JSON records why mask-based columns were skipped.
    meta = json.loads(Path(out["confounds_json"]).read_text())
    assert "BrainMaskNote" in meta["__global__"]


def test_compute_confounds_single_volume_raises_and_writes_nothing(tmp_path):
    # A 1-volume BOLD has no timeseries -> refuse to write a degenerate file.
    bold_file = tmp_path / "bold.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones((4, 4, 4, 1), dtype=np.float32), np.eye(4)),
        str(bold_file),
    )
    prefix = str(tmp_path / "sub-S_desc-confounds_timeseries")
    with pytest.raises(ValueError, match="volume"):
        C.compute_confounds(
            bold_file=bold_file,
            motion_par_file=None,
            working_dir=tmp_path / "work",
            output_prefix=prefix,
        )
    # No TSV/JSON written.
    assert not Path(f"{prefix}.tsv").exists()
    assert not Path(f"{prefix}.json").exists()


def test_compute_confounds_dummy_frame_not_double_flagged(tmp_path):
    # A bright initial volume (non-steady-state) that is ALSO a DVARS outlier must appear only in
    # non_steady_state_outlier##, never as a duplicate motion_outlier## column.
    n = 20
    par = tmp_path / "mc.par"
    _write_par(par, n=n)
    rng = np.random.RandomState(11)
    data = rng.normal(100, 5, size=(6, 6, 6, n)).astype(np.float32)
    data[
        ..., 0
    ] += 400  # strong T1-saturation spike -> dummy + huge DVARS at frame 1 boundary
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(tmp_path / "bold.nii.gz"))
    nib.save(
        nib.Nifti1Image(np.ones((6, 6, 6), dtype=np.uint8), np.eye(4)),
        str(tmp_path / "mask.nii.gz"),
    )
    out = C.compute_confounds(
        bold_file=tmp_path / "bold.nii.gz",
        motion_par_file=par,
        working_dir=tmp_path / "work",
        output_prefix=str(tmp_path / "sub-D_desc-confounds_timeseries"),
        brain_mask_file=tmp_path / "mask.nii.gz",
    )
    df = pd.read_csv(out["confounds_tsv"], sep="\t")
    assert any(c.startswith("non_steady_state_outlier") for c in df.columns)
    # No two one-hot outlier columns are identical (would be perfectly collinear).
    onehot = [c for c in df.columns if "outlier" in c]
    seen = set()
    for c in onehot:
        key = tuple(df[c].fillna(0.0).to_numpy())
        assert key not in seen, f"duplicate collinear outlier column: {c}"
        seen.add(key)


def test_compute_confounds_configurable_fd_threshold(tmp_path):
    # Lowering the FD threshold flags more frames, and the JSON reports the value actually used.
    n = 20
    par = tmp_path / "mc.par"
    _write_par(par, n=n)
    rng = np.random.RandomState(12)
    nib.save(
        nib.Nifti1Image(
            rng.normal(100, 5, size=(5, 5, 5, n)).astype(np.float32), np.eye(4)
        ),
        str(tmp_path / "bold.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.uint8), np.eye(4)),
        str(tmp_path / "mask.nii.gz"),
    )

    def _run(fd_thresh, tag):
        out = C.compute_confounds(
            bold_file=tmp_path / "bold.nii.gz",
            motion_par_file=par,
            working_dir=tmp_path / f"work_{tag}",
            output_prefix=str(tmp_path / f"sub-{tag}_desc-confounds_timeseries"),
            brain_mask_file=tmp_path / "mask.nii.gz",
            fd_outlier_threshold=fd_thresh,
            std_dvars_outlier_threshold=1e9,  # disable DVARS flagging to isolate FD
        )
        df = pd.read_csv(out["confounds_tsv"], sep="\t")
        meta = json.loads(Path(out["confounds_json"]).read_text())
        n_flagged = sum(c.startswith("motion_outlier") for c in df.columns)
        return n_flagged, meta

    loose, _ = _run(10.0, "loose")  # no frame exceeds 10mm FD
    tight, meta_tight = _run(0.001, "tight")  # essentially every moving frame exceeds
    assert tight > loose
    # JSON reports the configured threshold, not the module default.
    mo_meta = next(v for k, v in meta_tight.items() if k.startswith("motion_outlier"))
    assert mo_meta["FDThresholdMM"] == 0.001


# -------------------------------------------------------------------------------------------------
# FD rotation radius: configurable, and reported truthfully
# -------------------------------------------------------------------------------------------------


def _synthetic_run(tmp_path: Path, n: int = 20):
    """Write a synthetic .par + BOLD + mask and return their paths."""
    par = tmp_path / "mc.par"
    _write_par(par, n=n)
    rng = np.random.RandomState(21)
    nib.save(
        nib.Nifti1Image(
            rng.normal(100, 5, size=(5, 5, 5, n)).astype(np.float32), np.eye(4)
        ),
        str(tmp_path / "bold.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.uint8), np.eye(4)),
        str(tmp_path / "mask.nii.gz"),
    )
    return par, tmp_path / "bold.nii.gz", tmp_path / "mask.nii.gz"


def test_sidecar_reports_the_radius_actually_used(tmp_path):
    """The sidecar must echo the radius passed in, not the module constant.

    It previously hardcoded ``FD_RADIUS_MM``, so an override produced a sidecar that disagreed with
    the FD values in the very TSV it described.
    """
    par, bold, mask = _synthetic_run(tmp_path)
    override = 50.0
    assert override != C.FD_RADIUS_MM

    out = C.compute_confounds(
        bold_file=bold,
        motion_par_file=par,
        working_dir=tmp_path / "work",
        output_prefix=str(tmp_path / "sub-x_desc-confounds_timeseries"),
        brain_mask_file=mask,
        radius_mm=override,
    )
    meta = json.loads(Path(out["confounds_json"]).read_text())
    assert meta["framewise_displacement"]["RotationRadiusMM"] == override
    assert meta["rmsd"]["RotationRadiusMM"] == override

    # And the radius genuinely scaled the values it describes.
    df = pd.read_csv(out["confounds_tsv"], sep="\t")
    motion = pd.read_csv(par, sep=r"\s+", header=None)
    motion.columns = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
    expected = C.compute_framewise_displacement(motion, radius_mm=override)
    assert np.allclose(
        df["framewise_displacement"].to_numpy()[1:], expected[1:], atol=1e-9
    )


def test_func_compute_confounds_forwards_config_radius(tmp_path):
    """func.confounds.fd_radius_mm must reach the FD values.

    Guards the config -> step -> operation path: nothing else covers it, so a dropped ``.get()``
    would silently pin every run back to the 27 mm macaque default.
    """
    from nhp_mri_prep.steps.functional import func_compute_confounds
    from nhp_mri_prep.steps.types import StepInput

    par, bold, mask = _synthetic_run(tmp_path)

    def _run(radius, tag):
        result = func_compute_confounds(
            StepInput(
                input_file=bold,
                working_dir=tmp_path / f"work_{tag}",
                config={"func": {"confounds": {"fd_radius_mm": radius}}},
                output_name="confounds",
                metadata={},
            ),
            motion_par_file=par,
            brain_mask_file=mask,
        )
        df = pd.read_csv(result.output_file, sep="\t")
        return df["framewise_displacement"].to_numpy()[1:]

    fd_27 = _run(27.0, "r27")
    fd_54 = _run(54.0, "r54")
    # Doubling the radius doubles only the rotational term, so FD must grow but not exactly 2x.
    assert np.all(fd_54 > fd_27)
    assert not np.allclose(fd_54, 2 * fd_27)
