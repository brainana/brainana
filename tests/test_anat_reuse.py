"""Func-only runs on earlier anatomical outputs (--anat_from).

The functional workflow's anatomical inputs are rebuilt from a previous brainana output
directory: T1w brain, T1w <-> template transforms, atlas + LUT, fastsurfer subject ids, and
the resolved template as the registration reference.
"""

import json
from pathlib import Path

import pytest

from nhp_mri_prep.utils.anat_reuse import AnatReuseError, collect_anat_outputs, link_fastsurfer

SPACE = "NMT2Sym:res-05"
ROOT = Path(__file__).resolve().parents[1]


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    return p


def _anat(root: Path, sub: str, ses: str = "", xfm: bool = True, seg: bool = True) -> None:
    d = root / f"sub-{sub}" / (f"ses-{ses}" if ses else "") / "anat"
    prefix = f"sub-{sub}" + (f"_ses-{ses}" if ses else "")
    _touch(d / f"{prefix}_space-T1w_desc-preproc_T1w_brain.nii.gz")
    _touch(d / f"{prefix}_space-T1w_desc-preproc_T1w.nii.gz")
    if xfm:
        for name in (f"{prefix}_from-T1w_to-NMT2Sym_mode-image_xfm", f"{prefix}_from-NMT2Sym_to-T1w_mode-image_xfm"):
            _touch(d / f"{name}.json")  # sidecar sorts first; must not be picked
            _touch(d / f"{name}.nii.gz")
    if seg:
        _touch(d / f"{prefix}_space-T1w_desc-brain_atlasARM2.nii.gz")
        _touch(d / f"{prefix}_space-T1w_desc-brain_atlasARM2.tsv")


@pytest.fixture
def out_tree(tmp_path):
    root = tmp_path / "preprocessed"
    _anat(root, "01", "a")
    _anat(root, "02")  # subject-level (synthesized) anatomy
    _anat(root, "03", "b", xfm=False, seg=False)
    (root / "fastsurfer" / "sub-01_ses-a").mkdir(parents=True)
    (root / "fastsurfer" / "sub-02").mkdir(parents=True)
    (root / "dataset_description.json").write_text(json.dumps({"TemplateSource": {"OutputSpace": SPACE}}))
    return root


def test_collects_session_and_subject_level_anat(out_tree):
    found = collect_anat_outputs(out_tree, SPACE)
    rows = {(r["sub"], r["ses"]): r for r in found["anat"]}
    assert set(rows) == {("01", "a"), ("02", ""), ("03", "b")}
    r = rows[("01", "a")]
    assert r["bids_name"] == "sub-01_ses-a_T1w.nii.gz"
    assert r["xfm_fwd"].endswith("sub-01_ses-a_from-T1w_to-NMT2Sym_mode-image_xfm.nii.gz")
    assert r["xfm_inv"].endswith("sub-01_ses-a_from-NMT2Sym_to-T1w_mode-image_xfm.nii.gz")
    assert r["seg"].endswith("_atlasARM2.nii.gz") and r["lut"].endswith("_atlasARM2.tsv")
    assert rows[("02", "")]["bids_name"] == "sub-02_T1w.nii.gz"
    assert "xfm_fwd" not in rows[("03", "b")] and "seg" not in rows[("03", "b")]
    assert Path(found["template"]).name.startswith("tpl-NMT2Sym_res-05")
    assert {(f["sub"], f["ses"], f["id"]) for f in found["fastsurfer"]} == {("01", "a", "sub-01_ses-a"), ("02", "", "sub-02")}


def test_rejects_other_output_space(out_tree):
    with pytest.raises(AnatReuseError, match="output space"):
        collect_anat_outputs(out_tree, "NMT2Sym:res-1")


def test_rejects_dir_without_anat(tmp_path):
    (tmp_path / "sub-01" / "func").mkdir(parents=True)
    with pytest.raises(AnatReuseError, match="no '"):
        collect_anat_outputs(tmp_path, SPACE)
    with pytest.raises(AnatReuseError, match="not found"):
        collect_anat_outputs(tmp_path / "missing", SPACE)


def test_links_fastsurfer_subjects(out_tree, tmp_path):
    new_out = tmp_path / "new"
    (new_out / "fastsurfer" / "sub-02").mkdir(parents=True)  # already there: left alone
    linked = link_fastsurfer(out_tree, new_out, ["sub-01_ses-a", "sub-02"])
    assert linked == ["sub-01_ses-a"]
    assert (new_out / "fastsurfer" / "sub-01_ses-a").resolve() == (out_tree / "fastsurfer" / "sub-01_ses-a").resolve()
    assert not (new_out / "fastsurfer" / "sub-02").is_symlink()


def test_flag_is_declared_and_main_skips_anat_wf():
    assert "anat_from" in (ROOT / "known_flags.txt").read_text().split()
    assert "anat_from = null" in (ROOT / "nextflow.config").read_text()
    assert "--anat_from" in (ROOT / "USAGE.txt").read_text()
    main = (ROOT / "main.nf").read_text()
    branch = main[main.index("if (params.anat_from) {"): main.index("ANAT_WF()")]
    assert "collect_reused_anat.py" in branch and "FUNC_WF(" in branch
    assert "} else {" in branch  # ANAT_WF only runs on the other branch
