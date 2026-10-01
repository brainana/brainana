"""Every stage must declare how it decides it is complete.

A stage that neither declares expected_outputs() nor overrides should_skip()
would silently fall back to "never skip, never verify" -- which is how the s12
resume hazard was able to exist unnoticed. This test makes that choice
explicit and reviewable.
"""

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastsurfer_surfrecon import stages as stages_pkg
from fastsurfer_surfrecon.stages.base import HemisphereStage, PipelineStage

# Stages that intentionally run every time, with the reason.
ALWAYS_RUN = {
    "ComputeMorphometry": "cheap; always recomputed from current surfaces",
    "CorticalRibbon": "needs both hemispheres; guarded upstream",
    "Statistics": "needs ribbon.mgz; cheap to recompute",
    "AsegRefinement": "recon-all sub-steps manage their own outputs",
    "AparcMapping": "recon-all sub-steps manage their own outputs",
    "WMParcMapping": "recon-all sub-steps manage their own outputs",
}


def _stage_classes():
    for name in dir(stages_pkg):
        obj = getattr(stages_pkg, name)
        if (
            inspect.isclass(obj)
            and issubclass(obj, PipelineStage)
            and obj not in (PipelineStage, HemisphereStage)
            and not inspect.isabstract(obj)
        ):
            yield name, obj


def test_stage_classes_are_discoverable():
    found = dict(_stage_classes())
    assert len(found) >= 20, f"expected the full stage set, found {sorted(found)}"


@pytest.mark.parametrize("name,cls", sorted(_stage_classes()))
def test_stage_declares_a_completion_rule(name, cls):
    """Either declare outputs, override should_skip, or be listed as always-run."""
    declares_outputs = "expected_outputs" in cls.__dict__
    overrides_skip = "should_skip" in cls.__dict__

    if name in ALWAYS_RUN:
        assert overrides_skip, (
            f"{name} is listed in ALWAYS_RUN but does not override should_skip; "
            "remove it from the list or add the override"
        )
        return

    assert declares_outputs or overrides_skip, (
        f"{name} declares neither expected_outputs() nor should_skip(). It will "
        "never be skipped and never have its outputs verified. Declare its "
        "outputs, or add it to ALWAYS_RUN with a reason."
    )


def test_s12_skip_set_excludes_sphere():
    """Encodes the reasoning behind s12's output set as an assertion.

    `sphere` is written by BOTH s11 and s12. Including it in s12's declaration
    would let s11's output satisfy s12's skip check, re-creating the class of
    bug this work fixed. `qsphere` is s12's alone and is written last.
    """
    from fastsurfer_surfrecon.stages.s12_topology_fix import TopologyFix

    stage = TopologyFix.__new__(TopologyFix)
    stage.config = SimpleNamespace()
    stage.sd = SimpleNamespace(surf_dir=Path("/nonexistent/surf"))
    stage.hemi = "lh"

    names = [p.name for p in stage.expected_outputs()]

    assert "lh.sphere" not in names, "s11 also writes sphere; it cannot gate s12"
    assert "lh.qsphere" in names, "s12's last write must be part of its skip check"
    assert {"lh.orig", "lh.smoothwm", "lh.inflated"} <= set(names)


def test_s02_normalises_on_the_aseg_mapped_volume(tmp_path, monkeypatch, freesurfer_home):
    """WM normalisation reads labels 2/41 as WM, which only holds in the aseg
    mapping; the raw atlas volume has cortex at ID 2 (ARM2: right ACgG)."""
    from fastsurfer_surfrecon.config import ReconSurfConfig
    from fastsurfer_surfrecon.io.subjects_dir import SubjectsDir
    from fastsurfer_surfrecon.stages import s02_bias_correction as s02

    config = ReconSurfConfig.with_defaults(
        subject_id="sub-01",
        subjects_dir=tmp_path,
        atlas={"name": "ARM2"},
        processing={"parallel_hemis": False, "threads": 1},
        verbose=0,
    )
    sd = SubjectsDir(tmp_path, "sub-01")
    sd.setup()
    for name in ("aseg.auto_noCCseg.mgz", "aparc.ARM2atlas+aseg.orig.mgz"):
        sd.mri(name).write_bytes(b"")
    seen = {}
    monkeypatch.setattr(s02, "bias_correct_and_normalize", lambda **kw: seen.update(kw))
    s02.BiasCorrection(config, sd)._run()
    assert seen["aseg_path"] == sd.mri("aseg.auto_noCCseg.mgz")
