"""Anatomical conform registers the brain, not the scan.

A brain in one corner of a large field of view (neck and shoulders below it) used to
send FLIRT's +/-180 degree search off course, even with the skull strip right.
Conform now crops the stripped brain to its bounding box for registration and
applies the transform to the uncropped input. This pins the end result: a pitched
NMT2Sym brain embedded high in a 190 x 240 x 240 mm grid conforms to within 1 degree.
"""

import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from nhp_mri_prep.operations.preprocessing import conform_to_template
from nhp_mri_prep.operations.sitk_rigid_registration import fsl_mat_to_world_affine

ZOO = Path(__file__).resolve().parents[1] / "template_zoo" / "template"
_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


def _pitch(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    r = np.eye(4)
    r[1:3, 1:3] = [[c, -s], [s, c]]
    return r


@pytest.mark.parametrize("method", ["flirt", "sitk"])
def test_brain_in_a_corner_of_a_large_fov_conforms(tmp_path, method):
    if method == "flirt" and not (shutil.which("flirt") and shutil.which("3dresample")):
        pytest.skip("FSL flirt / AFNI 3dresample not installed")
    brain = nib.load(str(ZOO / "NMT2Sym" / "tpl-NMT2Sym_res-1_T1w_brain.nii.gz"))
    data = np.asarray(brain.dataobj, np.float32)

    # Embed the brain high in a body-sized grid, as a head sits above the shoulders.
    big = np.zeros((190, 240, 240), np.float32)
    offset = np.array([60, 120, 150])
    end = np.minimum(offset + data.shape, big.shape)
    big[tuple(slice(o, e) for o, e in zip(offset, end))] = data[
        tuple(slice(0, e - o) for o, e in zip(offset, end))
    ]
    affine = brain.affine.copy()
    affine[:3, 3] -= affine[:3, :3] @ offset
    true = _pitch(30.0)  # NMT2Sym world -> subject world
    subject = tmp_path / "subject.nii.gz"
    nib.save(nib.Nifti1Image(big, true @ affine), str(subject))

    result = conform_to_template(
        imagef=str(subject),
        template_file=str(ZOO / "NMT2Sym" / "tpl-NMT2Sym_res-05_T1w_brain.nii.gz"),
        working_dir=str(tmp_path / "work"),
        output_name="conformed.nii.gz",
        modal="anat",
        skip_skullstripping=True,
        rigid_method=method,
    )

    assert (tmp_path / "work" / "brain_for_reg.nii.gz").exists()
    # The published matrix refers to the uncropped input.
    world = _LPS @ fsl_mat_to_world_affine(
        result["forward_xfm"], result["template_f"], subject
    ) @ _LPS
    rot_err = np.degrees(
        np.arccos(np.clip((np.trace(world[:3, :3] @ true[:3, :3].T) - 1) / 2, -1, 1))
    )
    assert rot_err < 1.0
    # And the conformed image is on the template grid with the brain inside it.
    conformed = nib.load(result["imagef_conformed"])
    assert conformed.shape == nib.load(result["template_f"]).shape
    assert np.asanyarray(conformed.dataobj).sum() > 0.9 * big.sum() * np.prod(
        brain.header.get_zooms()[:3]
    ) / np.prod(conformed.header.get_zooms()[:3])
