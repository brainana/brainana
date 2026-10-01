"""Placing a subject in NMT2Sym world before the NMT2Sym registrations (nmt2sym_frame).

The template prior, the V1 white-matter fix and the template surface register NMT2Sym
material with FireANTs, which needs the pair roughly aligned. Conform aligns to the
output template, so for MEBRAINS/D99/Yerkes19, custom templates and conform off the
subject first has to be placed in NMT2Sym world.
"""

import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nibabel.processing import resample_from_to

from nhp_mri_prep.operations import nmt2sym_frame as nf

ZOO = Path(__file__).resolve().parents[1] / "template_zoo" / "template"
NMT_MASK = ZOO / "NMT2Sym" / "tpl-NMT2Sym_res-05_brainmask.nii.gz"


def _dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def _rotation_x(deg):
    t = np.radians(deg)
    m = np.eye(4)
    m[1:3, 1:3] = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
    return m


# --- which frame -------------------------------------------------------------------


@pytest.mark.parametrize(
    "config, expected",
    [
        ({}, ("identity", "NMT2Sym")),
        ({"template": {"output_space": "NMT2Sym:res-1"}}, ("identity", "NMT2Sym")),
        ({"template": {"output_space": "NMT2Asym:res-05"}}, ("identity", "NMT2Asym")),
        ({"template": {"output_space": "MEBRAINS"}}, ("table", "MEBRAINS")),
        ({"template": {"output_space": "D99:res-05"}}, ("table", "D99")),
        ({"template": {"output_space": "Yerkes19"}}, ("table", "Yerkes19")),
        ({"template": {"output_space": "/data/my_template.nii.gz"}}, ("rigid", None)),
        ({"anat": {"conform": {"enabled": False}}}, ("rigid", None)),
    ],
)
def test_frame_source(config, expected):
    assert nf.frame_source(config) == expected


def test_identity_never_touches_the_image(tmp_path):
    frame = nf.nmt2sym_frame({}, tmp_path)
    assert frame.is_identity and np.array_equal(frame.to_nmt2sym, np.eye(4))
    src = tmp_path / "img.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), np.float32), np.eye(4)), str(src))
    assert nf.reframe(src, frame, tmp_path / "out.nii.gz") == src


# --- ITK text and the bundled table -------------------------------------------------


def test_itk_affine_round_trip(tmp_path):
    m = _rotation_x(12.0)
    m[:3, 3] = [1.5, -20.0, 14.0]
    path = nf.write_itk_affine(tmp_path / "x.txt", m)
    assert np.allclose(nf.read_itk_affine(path), m)


def test_itk_affine_reads_a_centred_transform(tmp_path):
    """ITK: p' = R (p - c) + c + t."""
    r = _rotation_x(30.0)[:3, :3]
    c, t = np.array([5.0, -3.0, 2.0]), np.array([1.0, 2.0, 3.0])
    path = tmp_path / "c.txt"
    path.write_text(
        "#Insight Transform File V1.0\n#Transform 0\nTransform: AffineTransform_double_3_3\n"
        f"Parameters: {' '.join(map(str, [*r.ravel(), *t]))}\n"
        f"FixedParameters: {' '.join(map(str, c))}\n"
    )
    p = np.array([7.0, 1.0, -4.0])
    got = nf.read_itk_affine(path)
    assert np.allclose(got[:3, :3] @ p + got[:3, 3], r @ (p - c) + c + t)


@pytest.mark.parametrize(
    "family, mask",
    [
        ("MEBRAINS", "MEBRAINS/tpl-MEBRAINS_res-05_T1w_brain"),
        ("D99", "D99/tpl-D99_res-05_brainmask"),
        ("Yerkes19", "Yerkes19/tpl-Yerkes19_res-05_brainmask"),
    ],
)
def test_table_places_the_template_brain_on_nmt2sym(tmp_path, family, mask):
    """The bundled rigid, applied by header, puts the template's brain on NMT2Sym's.

    The wrong direction must not: that is the mistake this guards against.
    """
    frame = nf.nmt2sym_frame({"template": {"output_space": family}}, tmp_path)
    assert frame.source == "table"
    src = tmp_path / "mask.nii.gz"
    img = nib.load(str(ZOO / f"{mask}.nii.gz"))
    nib.save(nib.Nifti1Image((np.asanyarray(img.dataobj) > 0).astype(np.uint8), img.affine), str(src))
    ref = nib.load(str(NMT_MASK))
    nmt = np.asanyarray(ref.dataobj) > 0

    def placed_dice(f):
        moved = nib.load(str(nf.reframe(src, f, tmp_path / "placed.nii.gz")))
        return _dice(np.asanyarray(resample_from_to(moved, ref, order=0).dataobj) > 0, nmt)

    wrong = nf.Nmt2SymFrame(np.linalg.inv(frame.to_nmt2sym), "table")
    assert placed_dice(frame) > 0.88
    assert placed_dice(wrong) < 0.6
    assert frame.provenance()["TranslationMm"] > 20


def test_reframe_keeps_the_voxels(tmp_path):
    data = np.random.default_rng(0).random((5, 6, 7)).astype(np.float32)
    affine = np.diag([0.5, 0.5, 0.5, 1.0])
    src = tmp_path / "img.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(src))
    w = _rotation_x(10.0)
    w[:3, 3] = [0.0, 20.0, 15.0]
    out = nib.load(str(nf.reframe(src, nf.Nmt2SymFrame(w, "table"), tmp_path / "out.nii.gz")))
    assert np.array_equal(np.asanyarray(out.dataobj), data)
    assert np.allclose(out.affine, w @ affine, atol=1e-5)


# --- runtime rigid ------------------------------------------------------------------


@pytest.mark.parametrize("method", ["flirt", "sitk"])
def test_rigid_recovers_a_displaced_brain(tmp_path, method):
    """A copy of the NMT2Sym brain moved 25 mm and pitched 10 degrees, as D99 is."""
    if method == "flirt" and shutil.which("flirt") is None:
        pytest.skip("FSL flirt not installed")
    brain = nib.load(str(ZOO / "NMT2Sym" / "tpl-NMT2Sym_res-1_T1w_brain.nii.gz"))
    true = _rotation_x(10.0)
    true[:3, 3] = [0.0, 19.8, 15.0]  # subject world -> NMT2Sym world
    moved = tmp_path / "subject_brain.nii.gz"
    nib.save(
        nib.Nifti1Image(np.asarray(brain.dataobj, np.float32), np.linalg.inv(true) @ brain.affine),
        str(moved),
    )
    config = {"anat": {"conform": {"enabled": False, "rigid_method": method}}}
    frame = nf.nmt2sym_frame(config, tmp_path / "work", moved)
    assert frame.source == "rigid" and frame.detail == method
    centroid = nib.affines.apply_affine(
        brain.affine, np.argwhere(np.asanyarray(brain.dataobj) > 0).mean(0)
    )
    subject_point = np.linalg.solve(true, np.append(centroid, 1.0))
    assert np.linalg.norm((frame.to_nmt2sym @ subject_point)[:3] - centroid) < 1.0
    rot_err = np.degrees(np.arccos(np.clip((np.trace(frame.to_nmt2sym[:3, :3].T @ true[:3, :3]) - 1) / 2, -1, 1)))
    assert rot_err < 1.0


def test_rigid_needs_a_brain(tmp_path):
    with pytest.raises(ValueError):
        nf.nmt2sym_frame({"anat": {"conform": {"enabled": False}}}, tmp_path)


# --- ANTs point order (template surface) --------------------------------------------


@pytest.mark.skipif(shutil.which("antsApplyTransformsToPoints") is None, reason="ANTs not installed")
def test_post_transform_is_applied_after_xfm(tmp_path, monkeypatch):
    """antsApplyTransformsToPoints applies its -t list to points in the order given."""
    monkeypatch.setenv("FREESURFER_HOME", str(tmp_path))  # run_fs_command requires it
    from fastsurfer_surfrecon.processing.template_init import transform_points_ants

    rot = _rotation_x(90.0)  # LPS point maps
    shift = np.eye(4)
    shift[:3, 3] = [10.0, 0.0, 0.0]
    xfm = nf.write_itk_affine(tmp_path / "rot.txt", rot)
    post = nf.write_itk_affine(tmp_path / "shift.txt", shift)
    ras = np.array([[1.0, 2.0, 3.0]])
    got = transform_points_ants(ras, xfm, post=post)
    lps = np.append(ras[0] * [-1, -1, 1], 1.0)
    expected = (shift @ rot @ lps)[:3] * [-1, -1, 1]
    assert np.allclose(got[0], expected, atol=1e-4)
