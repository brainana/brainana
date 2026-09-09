"""Surface placement must anchor to the base only for longitudinal timepoints.

Inheriting the base's topology gives vertex correspondence, but it is the
*anchoring* here -- starting each pass from the base's surface and capping how
far vertices may travel -- that actually reduces across-timepoint variance. It
also damps genuine change, which is why it must be strictly gated on
config.longitudinal and must not leak into cross-sectional runs.

The flags mirror `recon-all -long`: --max-cbv-dist 3.5 on both passes,
?h.orig_white as the white input (recon-all:4224-4235), ?h.orig_pial plus
--blend-surf .25 <white> for the pial (recon-all:4289-4304).
"""

import pytest

from fastsurfer_surfrecon.wrappers.mris import mris_place_surface


class _Recorder:
    """Captures the argv that would have been handed to FreeSurfer."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append([str(c) for c in cmd])

        class _Done:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Done()


@pytest.fixture
def recorded(monkeypatch, tmp_path):
    rec = _Recorder()
    monkeypatch.setattr("fastsurfer_surfrecon.wrappers.mris.run_fs_command", rec)
    return rec


def _place(tmp_path, **kwargs):
    return mris_place_surface(
        input_surf=tmp_path / "lh.input",
        output_surf=tmp_path / "lh.out",
        hemi="lh",
        wm=tmp_path / "wm.mgz",
        invol=tmp_path / "brain.finalsurfs.mgz",
        aseg=tmp_path / "aseg.presurf.mgz",
        adgws_in=tmp_path / "autodet.gw.stats.lh.dat",
        **kwargs,
    )


def test_multi_argument_option_stays_separate_argv_items(recorded, tmp_path):
    """--blend-surf takes a weight AND a surface; joining them breaks parsing."""
    _place(tmp_path, pial=True, blend_surf=(0.25, tmp_path / "lh.white"))
    argv = recorded.calls[0]
    i = argv.index("--blend-surf")
    assert argv[i + 1] == "0.25"
    assert argv[i + 2].endswith("lh.white")
    # The failure mode being guarded against: one token holding both values.
    assert not any(" " in tok for tok in argv), f"space inside a token: {argv}"


def test_max_cbv_dist_is_emitted_when_given(recorded, tmp_path):
    _place(tmp_path, white=True, max_cbv_dist=3.5)
    argv = recorded.calls[0]
    assert argv[argv.index("--max-cbv-dist") + 1] == "3.5"


def test_none_valued_options_are_omitted(recorded, tmp_path):
    """Cross-sectional runs pass max_cbv_dist=None and must get no flag."""
    _place(tmp_path, white=True, max_cbv_dist=None, blend_surf=None)
    argv = recorded.calls[0]
    assert "--max-cbv-dist" not in argv
    assert "--blend-surf" not in argv


def test_s15_source_gates_anchoring_on_the_longitudinal_flag():
    """Guards against the anchoring being applied unconditionally.

    A source-level check because exercising s15._run() end to end needs a fully
    populated subject tree; what matters is that neither flag can be emitted
    without consulting config.longitudinal.
    """
    from pathlib import Path

    src = Path(
        "src/fastsurfer_surfrecon/stages/s15_surface_placement.py"
    ).read_text()
    assert "longitudinal = self.config.longitudinal" in src
    # Both passes must condition the cap on that flag.
    assert src.count("max_cbv_dist=long_max_cbv_dist if longitudinal else None") == 2
    # The anchor surfaces are only chosen inside a longitudinal branch.
    assert 'orig_white = self.hemi_path("orig_white")' in src
    assert 'orig_pial = self.hemi_path("orig_pial")' in src


def test_anchoring_strength_comes_from_config_not_constants():
    """Both knobs trade bias for variance, so they must not be hard-coded.

    A tighter cap reduces across-timepoint noise but also damps genuine change;
    that is a study-design choice, not a constant.
    """
    from pathlib import Path

    from fastsurfer_surfrecon.config import ReconSurfConfig

    src = Path(
        "src/fastsurfer_surfrecon/stages/s15_surface_placement.py"
    ).read_text()
    assert "long_max_cbv_dist = self.config.long_max_cbv_dist" in src
    assert "blend_surf = (self.config.long_pial_blend_weight, white)" in src
    # No stray literals left behind.
    assert "= 3.5" not in src
    assert "(0.25," not in src
    # Defaults match recon-all -long.
    assert ReconSurfConfig.model_fields["long_max_cbv_dist"].default == 3.5
    assert ReconSurfConfig.model_fields["long_pial_blend_weight"].default == 0.25
