"""Stage 12 paths that the pre-orig gate used to turn into aborted hemispheres.

Two field-reported failures, neither of which needs a real defect to test:

* ``mris_fix_topology -ga`` crashing outright (FreeSurfer 7.4.1, "stack smashing
  detected") must fall back to the default search rather than kill the
  hemisphere.
* A premesh that is closed, consistently wound and genus 0 but *inside-out*
  skipped pymeshfix (whose predicate does not look at the sign) and then failed
  the gate, which requires outward normals. It must be flipped before the gate.

FreeSurfer is mocked throughout; meshes are synthesised in numpy.
"""

import json
import shutil

import nibabel.freesurfer.io as fsio
import numpy as np
import pytest

from fastsurfer_surfrecon.config import ReconSurfConfig
from fastsurfer_surfrecon.io.subjects_dir import SubjectsDir
from fastsurfer_surfrecon.processing.surface_fix import validate_surface
from fastsurfer_surfrecon.stages import s12_topology_fix as s12
from fastsurfer_surfrecon.wrappers.base import FreeSurferError


def _icosphere(levels: int = 4):
    """Unit icosphere (2562 vertices at level 4), outward-wound."""
    t = (1.0 + 5**0.5) / 2.0
    v = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]  # fmt: skip
    f = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]  # fmt: skip
    verts = [np.array(p, float) / np.linalg.norm(p) for p in v]
    faces = f
    for _ in range(levels):
        cache, new_faces = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces
    return np.array(verts) * 30.0, np.array(faces, dtype=np.int32)


@pytest.fixture
def stage(tmp_path, monkeypatch, freesurfer_home):
    config = ReconSurfConfig.with_defaults(
        subject_id="sub-01",
        subjects_dir=tmp_path,
        atlas={"name": "ARM2"},
        processing={"parallel_hemis": False, "threads": 1},
        verbose=0,
    )
    sd = SubjectsDir(tmp_path, "sub-01")
    sd.setup()

    # Every FreeSurfer step after the gate: copy input to output.
    def _copy(input_surf=None, output_surf=None, **_):
        if input_surf != output_surf:
            shutil.copy(input_surf, output_surf)
        return output_surf

    monkeypatch.setattr(s12, "mris_remove_intersection", _copy)
    monkeypatch.setattr(s12, "mris_smooth", _copy)
    monkeypatch.setattr(s12, "mris_inflate", _copy)
    monkeypatch.setattr(
        s12,
        "spherically_project_surface",
        lambda input_path, output_path, **_: shutil.copy(input_path, output_path),
    )
    return s12.TopologyFix(config, sd, "lh")


def _write_nofix_inputs(stage):
    v, f = _icosphere()
    for name in ("orig.nofix", "inflated.nofix", "qsphere.nofix"):
        fsio.write_geometry(str(stage.hemi_path(name)), v, f)


def test_inside_out_premesh_is_flipped_before_gate(stage):
    v, f = _icosphere()
    fsio.write_geometry(str(stage.hemi_path("orig.premesh")), v, f[:, ::-1].copy())
    assert not validate_surface(stage.hemi_path("orig.premesh"))["is_outward"]

    stage._run()  # used to raise SurfaceInvariantError at the pre-orig gate

    orig = validate_surface(stage.hemi_path("orig"))
    assert orig["is_closed"] and orig["is_oriented"] and orig["is_outward"]
    assert orig["euler"] == 2
    record = json.loads((stage.sd.scripts_dir / "lh.topology_fix.json").read_text())
    assert record["flipped_inside_out"] is True
    assert record["pymeshfix"] is False


def test_outward_premesh_is_left_alone(stage):
    v, f = _icosphere()
    fsio.write_geometry(str(stage.hemi_path("orig.premesh")), v, f)
    stage._run()
    record = json.loads((stage.sd.scripts_dir / "lh.topology_fix.json").read_text())
    assert record["flipped_inside_out"] is False
    assert not stage.hemi_path("orig.premesh.insideout").exists()


def test_ga_failure_falls_back_to_default_search(stage, monkeypatch):
    _write_nofix_inputs(stage)
    calls = []

    def fake_fix(ga, output_premesh, orig, **_):
        calls.append(ga)
        if ga:
            output_premesh.write_bytes(b"partial")  # a crash can leave debris
            raise FreeSurferError("*** stack smashing detected ***", returncode=-6)
        shutil.copy(orig, output_premesh)

    monkeypatch.setattr(s12, "mris_fix_topology", fake_fix)
    stage._run()

    assert calls == [True, False]
    record = json.loads((stage.sd.scripts_dir / "lh.topology_fix.json").read_text())
    assert record["mode"] == "no_ga_fallback"
    assert validate_surface(stage.hemi_path("orig"))["euler"] == 2


def test_ga_disabled_skips_the_ga_attempt(stage, monkeypatch):
    _write_nofix_inputs(stage)
    stage.config.processing.topology_fix_ga = False
    calls = []

    def fake_fix(ga, output_premesh, orig, **_):
        calls.append(ga)
        shutil.copy(orig, output_premesh)

    monkeypatch.setattr(s12, "mris_fix_topology", fake_fix)
    stage._run()
    assert calls == [False]
    record = json.loads((stage.sd.scripts_dir / "lh.topology_fix.json").read_text())
    assert record["mode"] == "no_ga"


def test_resume_keeps_recorded_mode(stage, monkeypatch):
    """A resume that reuses the premesh must not erase how it was made."""
    _write_nofix_inputs(stage)

    def fake_fix(ga, output_premesh, orig, **_):
        if ga:
            raise FreeSurferError("crash")
        shutil.copy(orig, output_premesh)

    monkeypatch.setattr(s12, "mris_fix_topology", fake_fix)
    stage._run()
    # Second run: premesh exists, mris_fix_topology is not called again.
    monkeypatch.setattr(
        s12, "mris_fix_topology", lambda **_: pytest.fail("should not rerun")
    )
    stage._run()
    record = json.loads((stage.sd.scripts_dir / "lh.topology_fix.json").read_text())
    assert record["mode"] == "no_ga_fallback"


# --------------------------------------------------------------------------
# A -ga run that "succeeds" with a broken mesh (field case: sub-032215 rh came
# back open, euler 0; pymeshfix closed it by cutting away the occipital pole)
# --------------------------------------------------------------------------


def _with_hole(n_faces_removed: int):
    """Icosphere with the faces around vertex 0 grown out to a hole."""
    v, f = _icosphere()
    centre = v[0]
    order = np.argsort(np.linalg.norm(v[f].mean(axis=1) - centre, axis=1))
    return v, np.delete(f, order[:n_faces_removed], axis=0)


def _fake_fix(ga_mesh, noga_mesh, calls):
    def fake_fix(ga, output_premesh, **_):
        calls.append(ga)
        v, f = ga_mesh if ga else noga_mesh
        fsio.write_geometry(str(output_premesh), v, f)
        # Overlays land beside the premesh; tag them with the search that wrote them.
        labels = (
            output_premesh.parent / f"{output_premesh.name.split('.')[0]}.defect_labels"
        )
        labels.write_text("ga" if ga else "no_ga")

    return fake_fix


def _record(stage):
    return json.loads((stage.sd.scripts_dir / "lh.topology_fix.json").read_text())


def test_open_ga_mesh_is_rescued_by_the_default_search(stage, monkeypatch):
    _write_nofix_inputs(stage)
    calls = []
    monkeypatch.setattr(
        s12, "mris_fix_topology", _fake_fix(_with_hole(400), _icosphere(), calls)
    )
    stage._run()

    assert calls == [True, False]
    record = _record(stage)
    assert record["mode"] == "no_ga_rescue"
    assert record["ga"]["premesh_closed"] is False
    assert record["no_ga"]["vertices"] == 2562
    assert validate_surface(stage.hemi_path("orig"))["n_vertices"] == 2562
    assert stage.hemi_path("defect_labels").read_text() == "no_ga"


def test_ga_mesh_is_kept_when_the_default_search_is_worse(stage, monkeypatch):
    _write_nofix_inputs(stage)
    calls = []
    monkeypatch.setattr(
        s12,
        "mris_fix_topology",
        _fake_fix(_with_hole(60), _with_hole(600), calls),
    )
    stage._run()

    assert calls == [True, False]
    record = _record(stage)
    assert record["mode"] == "ga"
    assert record["ga"]["vertices"] > record["no_ga"]["vertices"]
    # The overlays describe the mesh that became orig.
    assert stage.hemi_path("defect_labels").read_text() == "ga"
    assert not stage.hemi_path("defect_labels.ga").exists()


def test_orientation_slip_does_not_trigger_a_second_search(stage, monkeypatch):
    _write_nofix_inputs(stage)
    v, f = _icosphere()
    f = f.copy()
    f[0] = f[0][::-1]  # one face wound backwards: closed, euler 2, not oriented
    calls = []
    monkeypatch.setattr(s12, "mris_fix_topology", _fake_fix((v, f), None, calls))
    stage._run()

    assert calls == [True]
    record = _record(stage)
    assert record["mode"] == "ga"
    assert "no_ga" not in record


def test_resume_after_rescue_keeps_the_non_ga_mesh(stage, monkeypatch):
    """A step after the gate fails, then the stage is resumed.

    orig.premesh (the GA mesh the rescue rejected) still exists, so a resume
    that re-derived orig from it would promote the cut-away surface.
    """
    _write_nofix_inputs(stage)
    calls = []
    monkeypatch.setattr(
        s12, "mris_fix_topology", _fake_fix(_with_hole(400), _icosphere(), calls)
    )

    def smooth_crash(**_):
        raise FreeSurferError("mris_smooth died", returncode=1)

    real_smooth = s12.mris_smooth
    monkeypatch.setattr(s12, "mris_smooth", smooth_crash)
    with pytest.raises(FreeSurferError):
        stage._run()
    assert _record(stage)["mode"] == "no_ga_rescue"

    monkeypatch.setattr(s12, "mris_smooth", real_smooth)
    monkeypatch.setattr(
        s12, "mris_fix_topology", lambda **_: pytest.fail("should not rerun")
    )
    stage._run()

    record = _record(stage)
    assert record["mode"] == "no_ga_rescue"
    assert record["no_ga"]["vertices"] == 2562
    assert validate_surface(stage.hemi_path("orig"))["n_vertices"] == 2562


def test_resume_without_record_still_rescues_a_failed_ga_mesh(stage, monkeypatch):
    """The run died between mris_fix_topology -ga and writing the record."""
    _write_nofix_inputs(stage)
    fsio.write_geometry(str(stage.hemi_path("orig.premesh")), *_with_hole(400))
    calls = []
    monkeypatch.setattr(
        s12, "mris_fix_topology", _fake_fix(_with_hole(400), _icosphere(), calls)
    )
    stage._run()

    assert calls == [False]
    assert _record(stage)["mode"] == "no_ga_rescue"
    assert validate_surface(stage.hemi_path("orig"))["n_vertices"] == 2562
