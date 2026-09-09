"""`ReconSurfConfig`'s longitudinal fields must fail loudly, not degrade quietly.

A longitudinal run whose base template is missing does not raise later -- it
silently becomes a cross-sectional run on a half-seeded tree, which looks like a
success and yields surfaces with no cross-timepoint vertex correspondence. That
is the failure mode these tests exist to make impossible.
"""

import pytest
from pydantic import ValidationError

from fastsurfer_surfrecon.config import ReconSurfConfig


@pytest.fixture
def subjects_dir(tmp_path, monkeypatch):
    """A subjects dir with a base, a cross-sectional timepoint, and an LTA."""
    monkeypatch.setenv("FREESURFER_HOME", str(tmp_path / "fs"))
    (tmp_path / "fs").mkdir()
    sd = tmp_path / "subjects"
    (sd / "sub-01_base" / "mri").mkdir(parents=True)
    (sd / "sub-01_ses-a" / "mri").mkdir(parents=True)
    (sd / "sub-01_ses-a_long").mkdir(parents=True)
    lta = sd / "sub-01_base" / "mri" / "transforms"
    lta.mkdir(parents=True)
    (lta / "sub-01_ses-a_to_sub-01_base.lta").write_text("dummy\n")
    return sd


def _cfg(subjects_dir, **kwargs):
    base = dict(
        subject_id="sub-01_ses-a_long",
        subjects_dir=subjects_dir,
        atlas={"name": "ARM2"},
        processing={},
        verbose=0,
    )
    base.update(kwargs)
    return ReconSurfConfig.with_defaults(**base)


def _lta(subjects_dir):
    return (
        subjects_dir
        / "sub-01_base"
        / "mri"
        / "transforms"
        / "sub-01_ses-a_to_sub-01_base.lta"
    )


def test_cross_sectional_config_needs_no_longitudinal_fields(subjects_dir):
    """The default must stay off, so every existing caller is unaffected."""
    cfg = _cfg(subjects_dir)
    assert cfg.longitudinal is False
    assert cfg.base_subject_id is None
    assert cfg.base_subject_dir is None
    assert cfg.cross_subject_dir is None


def test_longitudinal_accepts_a_complete_specification(subjects_dir):
    cfg = _cfg(
        subjects_dir,
        longitudinal=True,
        base_subject_id="sub-01_base",
        cross_subject_id="sub-01_ses-a",
        tp_to_base_lta=_lta(subjects_dir),
    )
    assert cfg.longitudinal is True
    assert cfg.base_subject_dir == subjects_dir / "sub-01_base"
    assert cfg.cross_subject_dir == subjects_dir / "sub-01_ses-a"
    # Paths are resolved absolute, like the other optional path fields.
    assert cfg.tp_to_base_lta.is_absolute()


@pytest.mark.parametrize(
    "omit, expected",
    [
        ("base_subject_id", "requires base_subject_id"),
        ("tp_to_base_lta", "requires tp_to_base_lta"),
        ("cross_subject_id", "requires cross_subject_id"),
    ],
)
def test_longitudinal_rejects_a_partial_specification(subjects_dir, omit, expected):
    kwargs = dict(
        longitudinal=True,
        base_subject_id="sub-01_base",
        cross_subject_id="sub-01_ses-a",
        tp_to_base_lta=_lta(subjects_dir),
    )
    del kwargs[omit]
    with pytest.raises(ValidationError, match=expected):
        _cfg(subjects_dir, **kwargs)


def test_longitudinal_fields_without_the_flag_are_an_error(subjects_dir):
    """Silently ignoring them would hide a typo'd flag name."""
    with pytest.raises(ValidationError, match="longitudinal is False"):
        _cfg(subjects_dir, base_subject_id="sub-01_base")


def test_missing_base_directory_is_rejected(subjects_dir):
    with pytest.raises(ValidationError, match="Base subject directory not found"):
        _cfg(
            subjects_dir,
            longitudinal=True,
            base_subject_id="sub-01_nonexistent_base",
            cross_subject_id="sub-01_ses-a",
            tp_to_base_lta=_lta(subjects_dir),
        )


def test_missing_cross_sectional_directory_is_rejected(subjects_dir):
    with pytest.raises(
        ValidationError, match="Cross-sectional subject directory not found"
    ):
        _cfg(
            subjects_dir,
            longitudinal=True,
            base_subject_id="sub-01_base",
            cross_subject_id="sub-01_ses-nope",
            tp_to_base_lta=_lta(subjects_dir),
        )


def test_missing_lta_is_rejected(subjects_dir):
    with pytest.raises(ValidationError, match="tp_to_base_lta not found"):
        _cfg(
            subjects_dir,
            longitudinal=True,
            base_subject_id="sub-01_base",
            cross_subject_id="sub-01_ses-a",
            tp_to_base_lta=subjects_dir / "sub-01_base" / "mri" / "nope.lta",
        )


def test_a_timepoint_cannot_seed_itself(subjects_dir):
    """base == subject would copy a surface onto itself and skip s08-s12."""
    with pytest.raises(ValidationError, match="cannot seed itself"):
        _cfg(
            subjects_dir,
            subject_id="sub-01_base",
            longitudinal=True,
            base_subject_id="sub-01_base",
            cross_subject_id="sub-01_ses-a",
            tp_to_base_lta=_lta(subjects_dir),
        )
