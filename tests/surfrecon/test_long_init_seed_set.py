"""Seeding a longitudinal timepoint must actually stop s08-s12 from running.

The seed list in `stages/s00_long_init.py` is only correct insofar as it
satisfies the completion rules of the geometry stages, and those rules live in
those stages. So rather than asserting against a copy of the list, these tests
run the real stage and then ask the real s08-s12 classes whether they would
re-run. If anyone changes a completion rule, this fails instead of silently
re-tessellating a mesh that was supposed to be inherited.
"""

import json
import warnings

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import pytest

from fastsurfer_surfrecon.config import ReconSurfConfig
from fastsurfer_surfrecon.io.subjects_dir import SubjectsDir
from fastsurfer_surfrecon.stages.base import StageOutputError
from fastsurfer_surfrecon.stages.s00_long_init import (
    _BASE_VOLUMES,
    _SURFACE_SEEDS,
    LongTimepointInit,
)
from fastsurfer_surfrecon.stages.s08_tessellation import Tessellation
from fastsurfer_surfrecon.stages.s09_smoothing import Smoothing
from fastsurfer_surfrecon.stages.s10_inflation import Inflation
from fastsurfer_surfrecon.stages.s11_spherical_projection import SphericalProjection
from fastsurfer_surfrecon.stages.s12_topology_fix import TopologyFix

GEOMETRY_STAGES = (
    Tessellation,
    Smoothing,
    Inflation,
    SphericalProjection,
    TopologyFix,
)

ATLAS = "ARM2"

# The grid the base template defines and every timepoint must land on. Real
# volumes rather than placeholder bytes because the stage now compares the
# resampled timepoint against the base's orig.mgz, and a geometry check cannot
# be exercised by files that have no geometry.
BASE_SHAPE = (4, 4, 4)
BASE_ZOOMS = (0.8, 0.8, 0.8)


def _affine(zooms=BASE_ZOOMS, translation=(0.0, 0.0, 0.0)):
    affine = np.diag([*zooms, 1.0]).astype(float)
    affine[:3, 3] = translation
    return affine


# Distinct fill values, so "which tree did this volume come from" is answerable
# from the data rather than from a sentinel byte string.
BASE_FILL = 1
CROSS_FILL = 7


def _write_vol(
    path, shape=BASE_SHAPE, zooms=BASE_ZOOMS, translation=(0.0, 0.0, 0.0), fill=0
):
    """A minimal real MGH volume on a named grid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.MGHImage(
        np.full(shape, fill, dtype=np.uint8), _affine(zooms, translation)
    )
    nib.save(img, str(path))


# Every base surface the seed table reads, plus the two placement anchors.
BASE_SURFACES = sorted(set(_SURFACE_SEEDS.values()))


def _write_surf(path):
    """A minimal valid FreeSurfer surface: one triangle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fsio.write_geometry(
            path,
            np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            np.array([[0, 1, 2]], dtype=np.int32),
        )


@pytest.fixture
def long_setup(tmp_path, monkeypatch):
    """A base template and a cross-sectional timepoint, ready to seed from."""
    monkeypatch.setenv("FREESURFER_HOME", str(tmp_path / "fs"))
    (tmp_path / "fs").mkdir()

    sd_root = tmp_path / "subjects"
    base_id, cross_id = "sub-01_base", "sub-01_ses-a"
    long_id = f"{cross_id}_long"

    base_mri = sd_root / base_id / "mri"
    base_mri.mkdir(parents=True)
    for name in list(_BASE_VOLUMES) + [
        f"aparc.{ATLAS}atlas+aseg.orig.mgz",
        "aseg.mgz",
        # Not in _BASE_VOLUMES -- the timepoint rebuilds its own -- but the
        # geometry guard reads it as the grid every seed is defined on.
        "orig.mgz",
    ]:
        _write_vol(base_mri / name, fill=BASE_FILL)
    for hemi in ("lh", "rh"):
        for name in BASE_SURFACES:
            _write_surf(sd_root / base_id / "surf" / f"{hemi}.{name}")
    (sd_root / base_id / "scripts").mkdir(parents=True)
    (sd_root / base_id / "scripts" / "base-tps").write_text(f"{cross_id}\n")

    cross_mri = sd_root / cross_id / "mri"
    cross_mri.mkdir(parents=True)
    _write_vol(cross_mri / "orig.mgz", fill=CROSS_FILL)

    lta_dir = base_mri / "transforms"
    lta_dir.mkdir(parents=True)
    lta = lta_dir / f"{cross_id}_to_{base_id}.lta"
    lta.write_text("dummy\n")

    config = ReconSurfConfig.with_defaults(
        subject_id=long_id,
        subjects_dir=sd_root,
        atlas={"name": ATLAS},
        processing={},
        verbose=0,
        longitudinal=True,
        base_subject_id=base_id,
        cross_subject_id=cross_id,
        tp_to_base_lta=lta,
    )
    sd = SubjectsDir(sd_root, long_id)
    sd.setup()
    return config, sd


def _stub_apply_lta(shape=BASE_SHAPE, zooms=BASE_ZOOMS, translation=(0.0, 0.0, 0.0)):
    """Stand in for `mri_convert -at`, on a geometry the caller chooses.

    The real command needs FreeSurfer. What the stage cares about is the grid the
    output lands on -- which is the LTA's *destination* geometry, so making that
    a parameter is what lets a mis-targeted transform be tested at all.
    """

    def fake_apply_lta(input_vol, output_vol, lta, **kwargs):
        from pathlib import Path

        # Carry the input's voxel values across, as a real resample would, so a
        # caller can still tell which tree the volume came from.
        fill = int(np.asanyarray(nib.load(str(input_vol)).dataobj).flat[0])
        _write_vol(Path(output_vol), shape, zooms, translation, fill=fill)
        return output_vol

    return fake_apply_lta


@pytest.fixture
def seeded(long_setup, monkeypatch):
    """Run the real stage, with only the FreeSurfer resample stubbed out."""
    config, sd = long_setup

    monkeypatch.setattr(
        "fastsurfer_surfrecon.stages.s00_long_init.mri_convert_apply_lta",
        _stub_apply_lta(),
    )
    stage = LongTimepointInit(config, sd)
    stage._run()
    return config, sd, stage


def test_geometry_stages_will_not_rerun_after_seeding(seeded):
    """The whole point: s08-s12 must be disabled or already satisfied."""
    config, sd, _ = seeded
    for stage_cls in GEOMETRY_STAGES:
        for hemi in ("lh", "rh"):
            stage = stage_cls(config, sd, hemi)
            assert (
                stage.is_disabled() or stage.should_skip()
            ), f"{stage_cls.__name__} ({hemi}) would re-run after seeding"


def test_seeding_also_satisfies_skip_rules_without_the_disable_flag(seeded):
    """Belt and braces: the seeds alone should satisfy the completion rules.

    `is_disabled()` is what makes the skip explicit in the log, but the seeds
    must independently satisfy `should_skip()`, because that is what a resumed
    run and the surface QC step rely on. Checking this with the flag forced off
    catches a seed list that only "works" because of the flag.
    """
    config, sd, _ = seeded
    cross_sectional = config.model_copy(
        update={
            "longitudinal": False,
            "base_subject_id": None,
            "cross_subject_id": None,
            "tp_to_base_lta": None,
        }
    )
    for stage_cls in GEOMETRY_STAGES:
        for hemi in ("lh", "rh"):
            stage = stage_cls(cross_sectional, sd, hemi)
            assert not stage.is_disabled(), "flag should be off for this check"
            missing = [p for p in stage.expected_outputs() if not p.exists()]
            assert stage.should_skip(), (
                f"{stage_cls.__name__} ({hemi}) not satisfied by seeds; "
                f"missing: {[str(m) for m in missing]}"
            )


def test_verify_outputs_passes_on_a_correctly_seeded_tree(seeded):
    _, _, stage = seeded
    stage.verify_outputs()


def test_verify_outputs_catches_an_incomplete_seeding(seeded):
    """Deleting one seed must fail loudly rather than re-running a stage."""
    from fastsurfer_surfrecon.stages.base import StageOutputError

    config, sd, stage = seeded
    sd.hemi_surf("lh", "qsphere").unlink()
    with pytest.raises((StageOutputError, Exception)) as exc:
        stage.verify_outputs()
    assert "qsphere" in str(exc.value)


def test_verify_outputs_detects_seed_table_drift(seeded, monkeypatch):
    """The scenario the live-query safety net exists for.

    Deleting a seed is already caught by the base class, because every seed is
    also one of this stage's declared outputs. The case only the live query can
    catch is a geometry stage gaining an output the seed table does not cover --
    i.e. someone edits s08-s12 and forgets s00_long_init. Simulated here by
    adding a requirement to s12.
    """
    from fastsurfer_surfrecon.stages.base import StageOutputError
    from fastsurfer_surfrecon.stages import s12_topology_fix

    config, sd, stage = seeded
    original = s12_topology_fix.TopologyFix.expected_outputs

    def with_extra_output(self):
        return original(self) + [self.hemi_path("newly.required.surface")]

    monkeypatch.setattr(
        s12_topology_fix.TopologyFix, "expected_outputs", with_extra_output
    )
    with pytest.raises(StageOutputError) as exc:
        stage.verify_outputs()
    assert "TopologyFix" in str(exc.value)
    assert "newly.required.surface" in str(exc.value)


def test_orig_comes_from_the_cross_sectional_run_not_the_base(seeded):
    """orig.mgz must be the timepoint's own scan, resampled -- not base's."""
    config, sd, _ = seeded
    provenance = json.loads((sd.scripts_dir / "long_init.json").read_text())
    assert provenance["orig_from"].endswith(f"{config.cross_subject_id}/mri/orig.mgz")
    # Seeding orig.mgz from the base would freeze the base's intensities into
    # every timepoint and erase the signal being measured, so check the data,
    # not just the provenance note.
    seeded_fill = np.asanyarray(nib.load(str(sd.mri("orig.mgz"))).dataobj).flat[0]
    assert seeded_fill == CROSS_FILL != BASE_FILL


def test_orig_is_seeded_from_base_white_surface(seeded):
    """The pivot of the design: ?h.orig is the base's *white*, not its orig."""
    config, sd, _ = seeded
    provenance = json.loads((sd.scripts_dir / "long_init.json").read_text())
    for hemi in ("lh", "rh"):
        assert provenance["seeded"][f"surf/{hemi}.orig"].endswith(f"{hemi}.white")


def test_intensity_volumes_are_not_seeded(seeded):
    """norm/T1/brainmask must be rebuilt from this timepoint's own intensities.

    Seeding them would freeze the base's intensities into every timepoint and
    erase the signal being measured.
    """
    _, sd, _ = seeded
    for name in ("norm.mgz", "T1.mgz", "brainmask.mgz", "nu.mgz", "orig_nu.mgz"):
        assert not sd.mri(name).exists(), f"{name} must not be seeded"


def test_filled_is_not_seeded(seeded):
    """s07 gates brain.finalsurfs.mgz behind `not filled.exists()`."""
    _, sd, _ = seeded
    assert not sd.mri("filled.mgz").exists()
    assert not sd.mri("brain.finalsurfs.mgz").exists()


def test_curv_is_not_seeded(seeded):
    """Seeding curv would suppress `recon-all -curvHK`, which s19 needs."""
    _, sd, _ = seeded
    for hemi in ("lh", "rh"):
        assert not sd.hemi_surf(hemi, "curv").exists()


def test_provenance_records_every_seed(seeded):
    _, sd, _ = seeded
    provenance = json.loads((sd.scripts_dir / "long_init.json").read_text())
    expected = {f"mri/{v}" for v in _BASE_VOLUMES}
    expected |= {
        f"surf/{hemi}.{dst}" for hemi in ("lh", "rh") for dst in _SURFACE_SEEDS
    }
    assert expected <= set(provenance["seeded"])


def test_bookkeeping_files_identify_the_timepoint(seeded):
    """long.base/long.cross, not the directory name, are the identity marker."""
    config, sd, _ = seeded
    assert (sd.scripts_dir / "long.base").read_text().strip() == (
        config.base_subject_id
    )
    assert (sd.scripts_dir / "long.cross").read_text().strip() == (
        config.cross_subject_id
    )
    assert (sd.scripts_dir / "long.base-tps").exists()


# -------------------------------------------------------------------------------------------------
# The LTA must actually target the base's grid
# -------------------------------------------------------------------------------------------------


def _seed_with(config, sd, monkeypatch, **stub_kwargs):
    monkeypatch.setattr(
        "fastsurfer_surfrecon.stages.s00_long_init.mri_convert_apply_lta",
        _stub_apply_lta(**stub_kwargs),
    )
    LongTimepointInit(config, sd)._run()


def test_matching_geometry_records_what_it_checked_against(seeded):
    """The happy path leaves a trace, so an unchecked tree is recognisable later."""
    config, sd, _ = seeded
    provenance = json.loads((sd.scripts_dir / "long_init.json").read_text())
    assert provenance["geometry_checked_against"].endswith(
        f"{config.base_subject_id}/mri/orig.mgz"
    )
    assert provenance["grid"] == {
        "shape": list(BASE_SHAPE),
        "zooms": list(BASE_ZOOMS),
    }


@pytest.mark.parametrize(
    "stub_kwargs, expected",
    [
        ({"shape": (4, 4, 5)}, "shape"),
        ({"zooms": (0.8, 0.8, 1.0)}, "zooms"),
        # The case shape and zooms alone would wave through, which is exactly
        # why the affine is compared: a pure shift of the destination grid.
        ({"translation": (0.5, 0.0, 0.0)}, "affine"),
    ],
)
def test_an_lta_targeting_another_grid_is_fatal(
    long_setup, monkeypatch, stub_kwargs, expected
):
    """`mri_convert -at` adopts the LTA's *destination* geometry.

    Everything seeded afterwards is on the *base's* geometry. If they disagree,
    the timepoint is misaligned against the very mesh it inherited -- and nothing
    downstream reports it, because per-vertex differencing still succeeds. It
    just measures the wrong vertices.
    """
    config, sd = long_setup
    with pytest.raises(StageOutputError) as exc:
        _seed_with(config, sd, monkeypatch, **stub_kwargs)
    message = str(exc.value)
    assert expected in message
    assert str(config.tp_to_base_lta) in message
    assert config.base_subject_id in message


def test_nothing_is_seeded_when_the_geometry_check_fails(long_setup, monkeypatch):
    """The check must run before step 3, not after it.

    A half-seeded tree is worse than none: the surfaces would satisfy s08-s12's
    completion rules, so a resumed run would skip straight past the geometry it
    never actually inherited.
    """
    config, sd = long_setup
    with pytest.raises(StageOutputError):
        _seed_with(config, sd, monkeypatch, shape=(4, 4, 5))
    assert not sd.hemi_surf("lh", "orig").exists()
    assert not sd.mri("aseg.presurf.mgz").exists()
    assert not (sd.scripts_dir / "long_init.json").exists()


def test_a_base_without_orig_is_reported_clearly(long_setup, monkeypatch):
    """orig.mgz is not in _BASE_VOLUMES, so its absence is a plausible mistake."""
    config, sd = long_setup
    (config.subjects_dir / config.base_subject_id / "mri" / "orig.mgz").unlink()
    with pytest.raises(FileNotFoundError, match="orig.mgz"):
        _seed_with(config, sd, monkeypatch)


def test_stage_is_disabled_for_cross_sectional_runs(long_setup):
    """A cross-sectional run must be entirely unaffected by this stage."""
    config, sd = long_setup
    cross_sectional = config.model_copy(
        update={
            "longitudinal": False,
            "base_subject_id": None,
            "cross_subject_id": None,
            "tp_to_base_lta": None,
        }
    )
    assert LongTimepointInit(cross_sectional, sd).is_disabled()
