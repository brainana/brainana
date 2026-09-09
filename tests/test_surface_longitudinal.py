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

import shutil
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


class TestSegmentationConsensus:
    """The diagnostic that makes the CNN domain-shift question answerable.

    The base's segmentation comes from running the CNN on a robust average, which
    is the right choice but feeds the network a different kind of volume than it
    sees cross-sectionally. Rather than argue about whether that matters, the run
    records a per-label Dice against a majority vote of the sessions' own
    segmentations.
    """

    def _write_labels(self, path, array, zooms=(0.8, 0.8, 0.8)):
        path.parent.mkdir(parents=True, exist_ok=True)
        affine = np.diag(list(zooms) + [1.0])
        nib.save(nib.MGHImage(array.astype(np.uint8), affine), str(path))
        return path

    def _fake_apply(self, monkeypatch):
        """Stand in for `mri_convert -at`; transforms here are identities."""
        from nhp_mri_prep.steps import surface_longitudinal as m

        def fake_apply(input_vol, output_vol, lta=None, **kwargs):
            Path(output_vol).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_vol, output_vol)
            return output_vol

        monkeypatch.setattr(m, "mri_convert_apply_lta", fake_apply)

    def test_maps_every_timepoint_into_base_space(self, tmp_path, monkeypatch):
        from nhp_mri_prep.steps import surface_longitudinal as m

        self._fake_apply(monkeypatch)
        arrays = {"ses-a": np.array([[[2, 5]]]), "ses-b": np.array([[[3, 5]]])}
        asegs = {
            tp: self._write_labels(tmp_path / tp / "aseg.mgz", a)
            for tp, a in arrays.items()
        }
        ltas = {tp: tmp_path / f"{tp}.lta" for tp in arrays}
        for lta in ltas.values():
            lta.write_text("dummy\n")

        mapped = m.map_timepoint_asegs_to_base(asegs, ltas, tmp_path / "out")
        assert set(mapped) == {"ses-a", "ses-b"}
        assert all(p.exists() for p in mapped.values())
        assert all(p.name.endswith("_aseg_in_base.mgz") for p in mapped.values())

    def test_no_consensus_is_formed(self, tmp_path, monkeypatch):
        """Two timepoints cannot have a majority, so none is invented.

        A vote over two volumes resolves every disagreement by tie-break, which
        would be an implementation preference presented as agreement.
        """
        from nhp_mri_prep.steps import surface_longitudinal as m

        assert not hasattr(m, "fuse_timepoint_asegs")

    def test_an_unmappable_timepoint_is_dropped_not_fatal(self, tmp_path, monkeypatch):
        """This is a diagnostic; it must never cost the run its base template."""
        from nhp_mri_prep.steps import surface_longitudinal as m

        def flaky(input_vol, output_vol, lta=None, **kwargs):
            if "ses-b" in str(input_vol):
                raise RuntimeError("simulated mri_convert failure")
            Path(output_vol).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_vol, output_vol)
            return output_vol

        monkeypatch.setattr(m, "mri_convert_apply_lta", flaky)
        asegs = {
            tp: self._write_labels(tmp_path / tp / "aseg.mgz", np.array([[[1]]]))
            for tp in ("ses-a", "ses-b")
        }
        ltas = {tp: tmp_path / f"{tp}.lta" for tp in asegs}
        for lta in ltas.values():
            lta.write_text("dummy\n")

        mapped = m.map_timepoint_asegs_to_base(asegs, ltas, tmp_path / "out")
        assert set(mapped) == {"ses-a"}

    def test_dice_is_one_for_identical_volumes(self, tmp_path):
        from nhp_mri_prep.steps.surface_longitudinal import label_dice

        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[2:8, 2:8, 2:8] = 3
        a = self._write_labels(tmp_path / "a.mgz", arr)
        b = self._write_labels(tmp_path / "b.mgz", arr)
        dice = label_dice(a, b)
        assert dice == {"3": pytest.approx(1.0)}

    def test_dice_falls_with_disagreement(self, tmp_path):
        from nhp_mri_prep.steps.surface_longitudinal import label_dice

        a_arr = np.zeros((10, 10, 10), dtype=np.uint8)
        a_arr[2:8, 2:8, 2:8] = 3
        b_arr = np.zeros((10, 10, 10), dtype=np.uint8)
        b_arr[5:8, 2:8, 2:8] = 3  # half the extent
        a = self._write_labels(tmp_path / "a.mgz", a_arr)
        b = self._write_labels(tmp_path / "b.mgz", b_arr)
        dice = label_dice(a, b)
        assert 0.0 < dice["3"] < 1.0

    def test_tiny_labels_are_skipped(self, tmp_path):
        """Dice over a handful of voxels is noise, and would look alarming."""
        from nhp_mri_prep.steps.surface_longitudinal import label_dice

        a_arr = np.zeros((10, 10, 10), dtype=np.uint8)
        a_arr[0, 0, 0] = 7  # one voxel
        a_arr[2:8, 2:8, 2:8] = 3
        b_arr = a_arr.copy()
        b_arr[0, 0, 0] = 0
        a = self._write_labels(tmp_path / "a.mgz", a_arr)
        b = self._write_labels(tmp_path / "b.mgz", b_arr)
        dice = label_dice(a, b)
        assert "7" not in dice
        assert "3" in dice

    def test_mismatched_grids_are_rejected(self, tmp_path):
        from nhp_mri_prep.steps.surface_longitudinal import label_dice

        a = self._write_labels(tmp_path / "a.mgz", np.zeros((4, 4, 4), dtype=np.uint8))
        b = self._write_labels(tmp_path / "b.mgz", np.zeros((4, 4, 5), dtype=np.uint8))
        with pytest.raises(ValueError, match="differ in shape"):
            label_dice(a, b)
