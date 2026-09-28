"""Stage 11 builds the spherical map mris_fix_topology segments defects on.

The spectral projection folded more than FreeSurfer's quasi-homeomorphic sphere,
and the folds enlarged and merged defects that the topology fix then cut away
(on the test cohort, most visibly the occipital pole). FreeSurfer's map is now the
default; the spectral projection stays available. FreeSurfer is mocked.
"""

import shutil

import nibabel.freesurfer.io as fsio
import pytest

from fastsurfer_surfrecon.config import ReconSurfConfig
from fastsurfer_surfrecon.io.subjects_dir import SubjectsDir
from fastsurfer_surfrecon.stages import s11_spherical_projection as s11
from fastsurfer_surfrecon.wrappers import mris

from test_s12_topology_fix import _icosphere


@pytest.fixture
def stage(tmp_path):
    config = ReconSurfConfig.with_defaults(
        subject_id="sub-01",
        subjects_dir=tmp_path,
        atlas={"name": "ARM2"},
        processing={"parallel_hemis": False, "threads": 1},
        verbose=0,
    )
    sd = SubjectsDir(tmp_path, "sub-01")
    sd.setup()
    v, f = _icosphere()
    for name in ("inflated.nofix", "smoothwm.nofix"):
        fsio.write_geometry(str(sd.surf_dir / f"lh.{name}"), v, f)
    return s11.SphericalProjection(config, sd, "lh")


def test_freesurfer_qsphere_is_the_default(stage, monkeypatch):
    assert stage.config.processing.use_fs_qsphere is True
    calls = {}

    def fake_sphere(input_surf, output_surf, **kw):
        calls["input"] = input_surf
        shutil.copy(input_surf, output_surf)
        return output_surf

    monkeypatch.setattr(s11, "mris_sphere_quick", fake_sphere)
    monkeypatch.setattr(
        s11, "spherically_project_surface", lambda **_: pytest.fail("spectral ran")
    )
    stage._run()

    # recon-all -qsphere maps the inflated surface, not smoothwm
    assert calls["input"] == stage.hemi_path("inflated.nofix")
    assert stage.hemi_path("qsphere.nofix").exists()
    assert stage.hemi_path("sphere").exists()


def test_spectral_projection_is_selectable(stage, monkeypatch):
    stage.config.processing.use_fs_qsphere = False
    calls = {}

    def fake_spectral(input_path, output_path, **kw):
        calls["input"] = input_path
        shutil.copy(input_path, output_path)

    monkeypatch.setattr(s11, "spherically_project_surface", fake_spectral)
    monkeypatch.setattr(
        s11, "mris_sphere_quick", lambda **_: pytest.fail("FreeSurfer qsphere ran")
    )
    stage._run()

    assert calls["input"] == stage.hemi_path("smoothwm.nofix")
    assert stage.hemi_path("qsphere.nofix").exists()


def test_missing_input_names_the_stage_that_makes_it(stage):
    stage.hemi_path("inflated.nofix").unlink()
    with pytest.raises(FileNotFoundError, match="s10"):
        stage._run()


def test_wrapper_runs_the_recon_all_command(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        mris, "run_fs_command", lambda cmd, **kw: seen.update(cmd=cmd, **kw)
    )
    subject = tmp_path / "sub-01"
    mris.mris_sphere_quick(
        subject / "surf/rh.inflated.nofix",
        subject / "surf/rh.qsphere.nofix",
        subject_dir=subject,
    )
    assert seen["cmd"] == [
        "mris_sphere", "-q", "-p", "6", "-a", "128", "-seed", "1234",
        "surf/rh.inflated.nofix", "surf/rh.qsphere.nofix",
    ]  # fmt: skip


def test_config_files_without_the_key_still_load():
    from fastsurfer_surfrecon.config import ProcessingConfig

    assert "use_fs_qsphere" not in {
        name for name, f in ProcessingConfig.model_fields.items() if f.is_required()
    }


def test_inflation_reruns_when_s11_needs_its_output_again(stage):
    """s12 deleted inflated.nofix; qsphere.nofix is then removed to redo s11.

    s10 used to skip because sphere/inflated exist, leaving s11 without input.
    """
    from fastsurfer_surfrecon.stages.s10_inflation import Inflation

    s10 = Inflation(stage.config, stage.sd, "lh")
    stage.hemi_path("inflated.nofix").unlink()
    for name in ("sphere", "inflated"):
        shutil.copy(stage.hemi_path("smoothwm.nofix"), stage.hemi_path(name))
    assert not s10.should_skip()

    # Nothing downstream needs it: the later outputs still prove s10 ran.
    shutil.copy(stage.hemi_path("smoothwm.nofix"), stage.hemi_path("qsphere.nofix"))
    assert s10.should_skip()
    stage.hemi_path("qsphere.nofix").unlink()
    stage.config.processing.use_fs_qsphere = False  # spectral reads smoothwm.nofix
    assert s10.should_skip()
