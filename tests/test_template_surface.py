"""anat.surface_reconstruction.template_surface: config, V1-fill resolution, transform choice.

The template surface prior needs a subject-to-NMT2Sym transform. It must reuse the
anatomical registration when that targeted NMT2Sym, and register on its own
otherwise, so that the prior does not depend on template.output_space.
"""

from pathlib import Path

import pytest

from nhp_mri_prep.config.config_validation import validate_config
from nhp_mri_prep.operations.preprocessing import resolve_fix_v1_wm
from nhp_mri_prep.steps import anatomical
from nhp_mri_prep.steps.types import StepInput


def _cfg(surface=True, template=True):
    return {"anat": {"surface_reconstruction": {"enabled": surface, "template_surface": {"enabled": template}}}}


# --- fix_V1_WM -------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, config, modal, expected",
    [
        ("auto", _cfg(), "anat", False),  # template surfaces: V1 white comes from the template
        ("auto", _cfg(template=False), "anat", True),  # tessellated surfaces: the fill helped
        ("auto", _cfg(surface=False), "anat", True),  # no surfaces: keep v3.0.0 behaviour
        ("auto", {}, "anat", False),  # defaults: template surfaces on
        ("auto", _cfg(template=False), "func", False),
        (True, _cfg(), "anat", True),  # explicit values are honoured as given
        (False, _cfg(template=False), "anat", False),
    ],
)
def test_fix_v1_wm_resolution(value, config, modal, expected):
    assert resolve_fix_v1_wm(value, config, modal) is expected


def _fscnn(value):
    return {"anat": {"skullstripping_segmentation": {"fastSurferCNN": {"fix_V1_WM": value}}}}


@pytest.mark.parametrize("value", [True, False, "auto"])
def test_fix_v1_wm_accepts_bool_or_auto(value):
    validate_config(_fscnn(value))


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_fix_v1_wm_rejects_other_values(value):
    with pytest.raises(ValueError, match="fix_V1_WM"):
        validate_config(_fscnn(value))


@pytest.mark.parametrize("bad", [{"enabled": "yes"}, "on", ["enabled"]])
def test_template_surface_is_validated(bad):
    with pytest.raises(ValueError, match="template_surface"):
        validate_config({"anat": {"surface_reconstruction": {"template_surface": bad}}})


# --- transform choice ------------------------------------------------------------


@pytest.fixture
def step_input(tmp_path):
    return StepInput(input_file=tmp_path / "t1w.nii.gz", working_dir=tmp_path, config={}, metadata={})


@pytest.fixture
def fake_register(monkeypatch, tmp_path):
    """Record ants_register calls and return a transform, without registering anything."""
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        out = Path(kwargs["working_dir"]) / "surface_prior_to_NMT2Sym_fwd.nii.gz"
        out.write_text("xfm")
        return {"forward_transform": str(out)}

    monkeypatch.setattr(anatomical, "ants_register", fake)
    # The step skull-strips the T1w before registering; tiny real images keep that honest.
    import nibabel as nib
    import numpy as np

    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.float32), np.eye(4)), str(tmp_path / "t1w.nii.gz"))
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.uint8), np.eye(4)), str(tmp_path / "mask.nii.gz"))
    return calls


def _xfm(tmp_path, name):
    p = tmp_path / name
    p.write_text("xfm")
    return p


def test_reuses_the_run_registration_to_nmt2sym(step_input, tmp_path, fake_register):
    xfm = _xfm(tmp_path, "sub-01_from-T1w_to-NMT2Sym_mode-image_xfm.nii.gz")
    got, source = anatomical.surface_prior_transform(
        step_input, tmp_path / "t1w.nii.gz", tmp_path / "mask.nii.gz", xfm, tmp_path / "reg"
    )
    assert (got, source) == (xfm, "reused")
    assert fake_register == []


@pytest.mark.parametrize(
    "name",
    [
        "sub-01_from-T1w_to-MEBRAINS_mode-image_xfm.nii.gz",  # another template
        "sub-01_from-T1w_to-T1w_mode-image_xfm.mat",  # registration off (passthrough)
        None,  # no transform at all (e.g. the longitudinal base)
    ],
)
def test_registers_to_nmt2sym_otherwise(step_input, tmp_path, fake_register, name):
    xfm = _xfm(tmp_path, name) if name else None
    got, source = anatomical.surface_prior_transform(
        step_input, tmp_path / "t1w.nii.gz", tmp_path / "mask.nii.gz", xfm, tmp_path / "reg"
    )
    assert source == "registered" and got.exists()
    (call,) = fake_register
    assert call["fixedf"].endswith("tpl-NMT2Sym_res-05_T1w_brain.nii.gz")
    assert call["xfm_type"] == "syn"  # registration.anat2template_xfm_type default


def test_registration_off_passthrough_is_not_reused(tmp_path, fake_register):
    """With registration off, the identity passthrough is still named _to-NMT2Sym_."""
    step_input = StepInput(
        input_file=tmp_path / "t1w.nii.gz",
        working_dir=tmp_path,
        config={"registration": {"enabled": False}},
        metadata={},
    )
    xfm = _xfm(tmp_path, "sub-01_from-T1w_to-NMT2Sym_mode-image_xfm.h5")
    _, source = anatomical.surface_prior_transform(
        step_input, tmp_path / "t1w.nii.gz", tmp_path / "mask.nii.gz", xfm, tmp_path / "reg"
    )
    assert source == "registered"
    assert len(fake_register) == 1


def test_empty_placeholder_is_not_a_transform(step_input, tmp_path, fake_register):
    """Nextflow passes an empty placeholder when no registration output joined."""
    placeholder = tmp_path / "dummy_template_xfm.dummy"
    placeholder.write_text("")
    _, source = anatomical.surface_prior_transform(
        step_input, tmp_path / "t1w.nii.gz", tmp_path / "mask.nii.gz", placeholder, tmp_path / "reg"
    )
    assert source == "registered"
