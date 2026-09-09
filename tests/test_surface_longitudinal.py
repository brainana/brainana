"""Guards for the longitudinal base-template step.

Two things here are worth pinning down because their failure is silent rather
than loud:

- Staging base inputs must resolve symlinks. Under the default ``get_t1: false``
  some volumes in a subject tree are *relative* symlinks to siblings, and a
  relative link staged into another task's work directory arrives dangling.
- Timepoints must share a voxel grid before averaging. If they do not, the base
  is built on an arbitrary grid, the transforms target that grid, and every
  timepoint's surfaces are misaligned without anything reporting an error.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from nhp_mri_prep.steps.surface_longitudinal import (
    BASE_INPUT_VOLUMES,
    assert_consistent_geometry,
    collect_base_inputs,
)


def _write_vol(path, shape=(4, 4, 4), zooms=(0.8, 0.8, 0.8)):
    path.parent.mkdir(parents=True, exist_ok=True)
    affine = np.diag(list(zooms) + [1.0])
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine), str(path))
    return path


def _subject_tree(root, name, shape=(4, 4, 4), zooms=(0.8, 0.8, 0.8)):
    mri = root / name / "mri"
    for vol in BASE_INPUT_VOLUMES:
        _write_vol(mri / vol, shape=shape, zooms=zooms)
    return root / name


class TestCollectBaseInputs:
    def test_collects_every_required_volume(self, tmp_path):
        subject = _subject_tree(tmp_path / "subjects", "sub-01_ses-a")
        out = tmp_path / "staged"
        copied = collect_base_inputs(subject, out, "sub-01_ses-a")
        assert {p.name for p in copied} == set(BASE_INPUT_VOLUMES)
        assert all(p.is_file() and p.stat().st_size > 0 for p in copied)

    def test_uses_a_per_timepoint_subdirectory(self, tmp_path):
        """Several timepoints get staged together and must not collide."""
        subjects = tmp_path / "subjects"
        out = tmp_path / "staged"
        for tp in ("sub-01_ses-a", "sub-01_ses-b"):
            collect_base_inputs(_subject_tree(subjects, tp), out, tp)
        assert (out / "sub-01_ses-a" / "orig.mgz").exists()
        assert (out / "sub-01_ses-b" / "orig.mgz").exists()

    def test_resolves_symlinks_into_real_files(self, tmp_path):
        """A relative symlink staged elsewhere would arrive dangling."""
        subject = _subject_tree(tmp_path / "subjects", "sub-01_ses-a")
        real = subject / "mri" / "norm.mgz"
        link = subject / "mri" / "mask.mgz"
        link.unlink()
        link.symlink_to("norm.mgz")  # relative, like s05 writes
        assert link.is_symlink()

        out = tmp_path / "staged"
        collect_base_inputs(subject, out, "sub-01_ses-a")
        staged = out / "sub-01_ses-a" / "mask.mgz"
        assert not staged.is_symlink()
        assert staged.read_bytes() == real.read_bytes()

    def test_missing_volume_fails_loudly(self, tmp_path):
        """Better than emitting a partial set no one downstream can detect."""
        subject = _subject_tree(tmp_path / "subjects", "sub-01_ses-a")
        (subject / "mri" / "norm.mgz").unlink()
        with pytest.raises(FileNotFoundError, match="norm.mgz"):
            collect_base_inputs(subject, tmp_path / "staged", "sub-01_ses-a")


class TestGeometryAssertion:
    def test_matching_grids_pass_and_report_the_grid(self, tmp_path):
        vols = {
            tp: _write_vol(tmp_path / tp / "orig.mgz", (8, 8, 8), (0.8, 0.8, 0.8))
            for tp in ("ses-a", "ses-b", "ses-c")
        }
        shape, zooms = assert_consistent_geometry(vols)
        assert shape == (8, 8, 8)
        assert zooms == (0.8, 0.8, 0.8)

    def test_differing_voxel_sizes_are_rejected(self, tmp_path):
        vols = {
            "ses-a": _write_vol(tmp_path / "a.nii.gz", (8, 8, 8), (0.8, 0.8, 0.8)),
            "ses-b": _write_vol(tmp_path / "b.nii.gz", (8, 8, 8), (0.5, 0.5, 0.5)),
        }
        with pytest.raises(ValueError, match="do not share a voxel grid"):
            assert_consistent_geometry(vols)

    def test_differing_shapes_are_rejected(self, tmp_path):
        vols = {
            "ses-a": _write_vol(tmp_path / "a.nii.gz", (8, 8, 8), (0.8, 0.8, 0.8)),
            "ses-b": _write_vol(tmp_path / "b.nii.gz", (8, 9, 8), (0.8, 0.8, 0.8)),
        }
        with pytest.raises(ValueError, match="do not share a voxel grid"):
            assert_consistent_geometry(vols)

    def test_error_names_conform_as_the_first_thing_to_check(self, tmp_path):
        """anat.conform.enabled is what normally makes this hold."""
        vols = {
            "ses-a": _write_vol(tmp_path / "a.nii.gz", (8, 8, 8), (0.8, 0.8, 0.8)),
            "ses-b": _write_vol(tmp_path / "b.nii.gz", (8, 8, 8), (0.5, 0.5, 0.5)),
        }
        with pytest.raises(ValueError) as exc:
            assert_consistent_geometry(vols)
        assert "anat.conform.enabled" in str(exc.value)
        # And it should say which timepoints disagreed.
        assert "ses-a" in str(exc.value) and "ses-b" in str(exc.value)
