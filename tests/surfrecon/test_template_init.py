"""Template-initialised surfaces (s12b) and the V1 freeze in s13/s15.

The geometry is where a silent error would hide: a sign slip between RAS and
ANTs' LPS, or the wrong centre when moving between scanner and surface RAS,
would still produce a perfectly valid mesh -- just in the wrong place. So these
tests pin the conventions numerically.
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from nibabel.freesurfer import read_geometry, read_label, write_geometry
from pydantic import ValidationError

from fastsurfer_surfrecon.config import ReconSurfConfig
from fastsurfer_surfrecon.processing.template_init import (
    scanner_to_surface,
    surface_to_scanner,
    transform_points_ants,
    fraction_in_mask,
    volume_info_from_mgz,
    warp_template_surface,
    write_rip_label,
    write_vertex_label,
)
from fastsurfer_surfrecon.stages.s08_tessellation import Tessellation
from fastsurfer_surfrecon.stages.s12_topology_fix import TopologyFix
from fastsurfer_surfrecon.stages.s12b_template_init import TemplateInit, frozen_label

REPO = Path(__file__).resolve().parent.parent.parent
TEMPLATE = REPO / "template_zoo" / "fastsurfer" / "sub-NMT2Sym"
needs_ants = pytest.mark.skipif(
    shutil.which("antsApplyTransformsToPoints") is None,
    reason="antsApplyTransformsToPoints not on PATH",
)


def _mgz(path, affine, shape=(20, 24, 22)):
    nib.save(nib.MGHImage(np.zeros(shape, dtype=np.uint8), affine), str(path))
    return path


def _oblique_affine():
    """A 0.5 mm grid, rotated and off-centre, so no convention holds by accident."""
    rot = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    affine = np.eye(4)
    affine[:3, :3] = rot * 0.5
    affine[:3, 3] = [12.3, -7.1, 4.4]
    return affine


def _itk_translation(path, lps_shift):
    """An ITK text affine that translates points by ``lps_shift`` (LPS mm)."""
    t = " ".join(str(v) for v in lps_shift)
    path.write_text(
        "#Insight Transform File V1.0\n#Transform 0\n"
        "Transform: AffineTransform_double_3_3\n"
        f"Parameters: 1 0 0 0 1 0 0 0 1 {t}\nFixedParameters: 0 0 0\n"
    )
    return path


# --- geometry conventions --------------------------------------------------------


@pytest.mark.parametrize("affine", [_oblique_affine(), np.diag([-0.5, 0.5, 0.5, 1.0])], ids=["oblique", "lia-like"])
def test_surface_scanner_conversion_matches_freesurfer(tmp_path, affine):
    """Surface RAS <-> scanner RAS agrees with FreeSurfer's own two vox2ras, for any orientation.

    Surface RAS is always LIA-framed, so ``scanner = surface + c_ras`` would be
    wrong for an oblique volume; this pins the general conversion s12b uses.
    """
    mgz = _mgz(tmp_path / "orig.mgz", affine)
    hdr = nib.load(str(mgz)).header
    info = volume_info_from_mgz(mgz)
    vox = np.c_[np.random.default_rng(0).uniform(0, 20, (50, 3)), np.ones(50)].T
    scanner = (hdr.get_vox2ras() @ vox)[:3].T
    surface = (hdr.get_vox2ras_tkr() @ vox)[:3].T
    np.testing.assert_allclose(surface_to_scanner(surface, info), scanner, atol=1e-5)
    np.testing.assert_allclose(scanner_to_surface(scanner, info), surface, atol=1e-5)


def test_lia_volume_reduces_to_c_ras_offset(tmp_path):
    """For a conformed (LIA) orig.mgz the general conversion is the plain c_ras shift."""
    lia = np.array([[-0.5, 0, 0, 30.0], [0, 0, 0.5, -20.0], [0, -0.5, 0, 25.0], [0, 0, 0, 1]])
    info = volume_info_from_mgz(_mgz(tmp_path / "orig.mgz", lia))
    pts = np.random.default_rng(1).uniform(-20, 20, (30, 3))
    np.testing.assert_allclose(surface_to_scanner(pts, info), pts + info["cras"], atol=1e-6)


def test_volume_info_round_trips_through_a_surface(tmp_path):
    """What s12b writes reads back as the same volume geometry."""
    mgz = _mgz(tmp_path / "orig.mgz", _oblique_affine())
    info = volume_info_from_mgz(mgz)
    surf = tmp_path / "lh.test"
    write_geometry(str(surf), np.zeros((3, 3)), np.array([[0, 1, 2]]), volume_info=info)
    _, _, meta = read_geometry(str(surf), read_metadata=True)
    for key in ("volume", "voxelsize", "xras", "yras", "zras", "cras"):
        np.testing.assert_allclose(np.asarray(meta[key], float), np.asarray(info[key], float), atol=1e-5)


@needs_ants
def test_ants_point_transform_uses_lps(tmp_path):
    """A +2 mm LPS x shift moves RAS points 2 mm toward -x (to the right)."""
    xfm = _itk_translation(tmp_path / "shift.txt", (2.0, 0.0, 0.0))
    pts = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 9.0]])
    moved = transform_points_ants(pts, xfm)
    np.testing.assert_allclose(moved, pts + [-2.0, 0.0, 0.0], atol=1e-4)


@needs_ants
def test_warp_places_template_in_subject_surface_ras(tmp_path):
    """Identity transform: same world position, re-expressed in the subject's frame."""
    tpl_mgz = _mgz(tmp_path / "tpl.mgz", np.diag([0.5, 0.5, 0.5, 1.0]))
    subj_mgz = _mgz(tmp_path / "orig.mgz", _oblique_affine())
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    tpl_surf = tmp_path / "lh.white"
    write_geometry(str(tpl_surf), verts, faces, volume_info=volume_info_from_mgz(tpl_mgz))
    xfm = _itk_translation(tmp_path / "identity.txt", (0.0, 0.0, 0.0))

    out = tmp_path / "lh.orig"
    world = warp_template_surface(tpl_surf, xfm, subj_mgz, out)

    tpl_world = surface_to_scanner(verts, volume_info_from_mgz(tpl_mgz))
    np.testing.assert_allclose(world, tpl_world, atol=1e-4)
    got, got_faces, meta = read_geometry(str(out), read_metadata=True)
    np.testing.assert_allclose(surface_to_scanner(got, meta), tpl_world, atol=1e-4)
    np.testing.assert_array_equal(got_faces, faces)
    np.testing.assert_allclose(meta["cras"], volume_info_from_mgz(subj_mgz)["cras"], atol=1e-6)


def test_fraction_in_mask_flags_a_shifted_surface(tmp_path):
    """Vertices are looked up in the mask through its surface RAS; off-grid counts as outside."""
    data = np.zeros((20, 20, 20), dtype=np.uint8)
    data[5:15, 5:15, 5:15] = 1
    mask = tmp_path / "mask.mgz"
    nib.save(nib.MGHImage(data, _oblique_affine()), str(mask))
    tkr = nib.load(str(mask)).header.get_vox2ras_tkr()
    faces = np.array([[0, 1, 2]])

    def frac(voxels):
        surf = tmp_path / "lh.orig.nofix"
        write_geometry(str(surf), nib.affines.apply_affine(tkr, np.asarray(voxels, float)), faces)
        return fraction_in_mask(surf, mask)

    assert frac([[6, 6, 6], [10, 10, 10], [14, 14, 14]]) == 1.0
    assert frac([[6, 6, 6], [10, 10, 10], [16, 10, 10]]) == pytest.approx(2 / 3)
    assert frac([[6, 6, 6], [10, 10, 10], [40, 10, 10]]) == pytest.approx(2 / 3)  # off the grid


def test_surface_qc_records_the_template_init(tmp_path):
    """surface_qc.json says which starting surface each hemisphere had."""
    import json

    from fastsurfer_surfrecon.pipeline import ReconSurfPipeline

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "lh.template_init.json").write_text(json.dumps({"warped_fraction_in_mask": 0.99}))
    fake = SimpleNamespace(
        config=SimpleNamespace(subject_id="sub-x"),
        sd=SimpleNamespace(surf_dir=tmp_path / "surf", scripts_dir=scripts),
        _thickness_summary=lambda hemi: None,
    )
    ReconSurfPipeline._write_surface_qc(fake, ["lh", "rh"])
    report = json.loads((scripts / "surface_qc.json").read_text())
    assert report["template_init"] == {"lh": {"warped_fraction_in_mask": 0.99}}
    assert "topology_fix" not in report


# --- labels ----------------------------------------------------------------------


def test_rip_label_is_within_minus_frozen(tmp_path):
    """--rip-label holds what is outside the label, so the label is the movable set."""
    coords = np.arange(30, dtype=float).reshape(10, 3)
    surf = tmp_path / "lh.orig"
    write_geometry(str(surf), coords, np.array([[0, 1, 2]]))
    frozen = tmp_path / "frozen.label"
    write_vertex_label(frozen, [2, 3, 4], coords, "frozen")
    cortex = tmp_path / "cortex.label"
    write_vertex_label(cortex, [0, 1, 2, 3, 7, 8], coords, "cortex")

    everywhere = tmp_path / "all.label"
    assert write_rip_label(everywhere, surf, frozen) == 7
    assert sorted(read_label(str(everywhere))) == [0, 1, 5, 6, 7, 8, 9]

    in_cortex = tmp_path / "cortex_movable.label"
    assert write_rip_label(in_cortex, surf, frozen, within=cortex) == 4
    assert sorted(read_label(str(in_cortex))) == [0, 1, 7, 8]


def test_shipped_v1_label_indexes_the_template_mesh():
    """The shipped V1 label is valid on the shipped white surface (same ico6 mesh)."""
    for hemi, n_v1 in (("lh", 4789), ("rh", 4714)):
        verts, _ = read_geometry(str(TEMPLATE / "surf" / f"{hemi}.white"))
        sphere, _ = read_geometry(str(TEMPLATE / "surf" / f"{hemi}.sphere"))
        v1 = read_label(str(TEMPLATE / "label" / f"{hemi}.V1.label"))
        assert len(verts) == len(sphere) == 40962
        assert len(v1) == n_v1 and v1.max() < len(verts) and len(set(v1)) == len(v1)


# --- stage gating ----------------------------------------------------------------


def _stage_cfg(**kw):
    base = dict(template_init=False, longitudinal=False, template_freeze_label="V1")
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "template_init, longitudinal, s12b_off, s08_s12_off",
    [
        (False, False, True, False),  # v3.0.0: tessellate
        (True, False, False, True),  # template-initialised run
        (True, True, True, True),  # timepoint of a template base: inherits (s00)
        (False, True, True, True),  # timepoint of a tessellated base: inherits (s00)
    ],
)
def test_exactly_one_geometry_source_runs(template_init, longitudinal, s12b_off, s08_s12_off):
    cfg = _stage_cfg(template_init=template_init, longitudinal=longitudinal)
    sd = SimpleNamespace()
    assert TemplateInit(cfg, sd, "lh").is_disabled() is s12b_off
    assert Tessellation(cfg, sd, "lh").is_disabled() is s08_s12_off
    assert TopologyFix(cfg, sd, "lh").is_disabled() is s08_s12_off


def test_stale_freeze_label_does_not_freeze_a_tessellated_mesh(tmp_path):
    """A leftover label must not rip vertices of a mesh it was not made for."""
    label_dir = tmp_path / "label"
    label_dir.mkdir()
    (label_dir / "lh.template.V1.label").write_text("#!ascii label\n1\n0 0 0 0 0\n")
    sd = SimpleNamespace(label_dir=label_dir, surf_dir=tmp_path / "surf")

    tessellated = TemplateInit(_stage_cfg(), sd, "lh")
    tessellated.hemi_label = lambda name: label_dir / f"lh.{name}"
    assert frozen_label(tessellated) is None

    templated = TemplateInit(_stage_cfg(template_init=True), sd, "lh")
    templated.hemi_label = lambda name: label_dir / f"lh.{name}"
    assert frozen_label(templated) == label_dir / "lh.template.V1.label"

    unfrozen = TemplateInit(_stage_cfg(template_init=True, template_freeze_label=None), sd, "lh")
    unfrozen.hemi_label = lambda name: label_dir / f"lh.{name}"
    assert frozen_label(unfrozen) is None


# --- config ----------------------------------------------------------------------


@pytest.fixture
def fs_home(tmp_path, monkeypatch):
    monkeypatch.setenv("FREESURFER_HOME", str(tmp_path / "fs"))
    (tmp_path / "fs").mkdir()
    return tmp_path


def _recon_cfg(tmp_path, **kw):
    return ReconSurfConfig.with_defaults(
        subject_id="sub-01", subjects_dir=tmp_path, atlas={"name": "ARM2"}, processing={}, verbose=0, **kw
    )


def test_template_init_defaults_to_off_with_the_shipped_template(fs_home):
    cfg = _recon_cfg(fs_home)
    assert cfg.template_init is False
    assert cfg.template_subject_dir == TEMPLATE.resolve()
    assert cfg.template_freeze_label == "V1"


def test_template_init_requires_a_transform(fs_home):
    with pytest.raises(ValidationError, match="template_xfm"):
        _recon_cfg(fs_home, template_init=True)
    with pytest.raises(ValidationError, match="not found"):
        _recon_cfg(fs_home, template_init=True, template_xfm=fs_home / "missing.nii.gz")


def test_template_init_requires_a_complete_template(fs_home):
    xfm = fs_home / "xfm.txt"
    xfm.write_text("x")
    incomplete = fs_home / "tpl"
    (incomplete / "surf").mkdir(parents=True)
    with pytest.raises(ValidationError, match="incomplete"):
        _recon_cfg(fs_home, template_init=True, template_xfm=xfm, template_subject_dir=incomplete)
    assert _recon_cfg(fs_home, template_init=True, template_xfm=xfm).template_init is True
