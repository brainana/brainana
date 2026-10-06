"""Discovery selects only magnitude anatomicals.

BIDS lets T1w/T2w carry ``part-<mag|phase|real|imag>``. A correctly named
``..._run-2_part-phase_T1w`` used to be picked up as a T1w: its distinct ``run``
flipped the session into multi-run synthesis, and the phase image was averaged
into the subject's T1w. Discovery (and the validator that mirrors it) must keep
only the magnitude image.

Neither discovery nor the validator opens a NIfTI, so the fixtures are empty
files.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from nhp_mri_prep.nextflow_scripts.discover_bids_for_nextflow import (
    _scan_bids_layout,
)
from nhp_mri_prep.steps.bids_discovery import discover_bids_dataset


def _make(root: Path, rel_paths: List[str]) -> Path:
    for rel in rel_paths:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (root / "dataset_description.json").write_text(
        '{"Name": "t", "BIDSVersion": "1.9.0"}'
    )
    return root


def _discover(root: Path, synthesis_level: str = "session"):
    return discover_bids_dataset(root, {"anat": {"synthesis_level": synthesis_level}})


def _t1w_jobs(anat_jobs):
    return [j for j in anat_jobs if j["suffix"] == "T1w"]


def _paths(job):
    return sorted(Path(p).name for p in job.get("file_paths") or [job["file_path"]])


def test_annexed_symlinked_niftis_are_discovered(tmp_path):
    """DataLad/git-annex datasets store every NIfTI as a symlink into .git/annex."""
    files = [
        "sub-01/ses-01/anat/sub-01_ses-01_run-1_T1w.nii.gz",
        "sub-01/ses-02/func/sub-01_ses-02_task-rest_run-1_bold.nii.gz",
    ]
    for i, rel in enumerate(files):
        obj = tmp_path / ".git" / "annex" / "objects" / f"MD5E-{i}.nii.gz"
        obj.parent.mkdir(parents=True, exist_ok=True)
        obj.touch()
        link = tmp_path / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(Path("../../..") / obj.relative_to(tmp_path))
    _make(tmp_path, [])

    anat, func = _discover(tmp_path)
    assert [_paths(j) for j in _t1w_jobs(anat)] == [
        ["sub-01_ses-01_run-1_T1w.nii.gz"]
    ]
    assert [Path(j["file_path"]).name for j in func] == [
        "sub-01_ses-02_task-rest_run-1_bold.nii.gz"
    ]


def test_symlinked_bids_root_is_discovered(tmp_path):
    real = _make(tmp_path / "real", ["sub-01/anat/sub-01_T1w.nii.gz"])
    link = tmp_path / "link"
    link.symlink_to(real)
    anat, _ = _discover(link)
    assert [_paths(j) for j in _t1w_jobs(anat)] == [["sub-01_T1w.nii.gz"]]


def test_phase_run_does_not_trigger_synthesis(tmp_path):
    _make(
        tmp_path,
        [
            "sub-01/ses-01/anat/sub-01_ses-01_acq-mp2rage_run-1_T1w.nii.gz",
            "sub-01/ses-01/anat/sub-01_ses-01_acq-mp2rage_run-2_part-phase_T1w.nii.gz",
        ],
    )
    anat, _ = _discover(tmp_path)
    t1w = _t1w_jobs(anat)
    assert len(t1w) == 1
    assert t1w[0]["needs_synthesis"] is False
    assert _paths(t1w[0]) == ["sub-01_ses-01_acq-mp2rage_run-1_T1w.nii.gz"]


def test_part_mag_kept_part_phase_dropped(tmp_path):
    _make(
        tmp_path,
        [
            "sub-01/ses-01/anat/sub-01_ses-01_run-1_part-mag_T1w.nii.gz",
            "sub-01/ses-01/anat/sub-01_ses-01_run-1_part-phase_T1w.nii.gz",
            "sub-01/ses-01/anat/sub-01_ses-01_run-1_part-real_T2w.nii.gz",
        ],
    )
    anat, _ = _discover(tmp_path)
    assert [_paths(j) for j in anat] == [["sub-01_ses-01_run-1_part-mag_T1w.nii.gz"]]


def test_phase_only_session_not_counted_for_cross_session_synthesis(tmp_path):
    _make(
        tmp_path,
        [
            "sub-01/ses-01/anat/sub-01_ses-01_run-1_T1w.nii.gz",
            "sub-01/ses-02/anat/sub-01_ses-02_run-1_part-phase_T1w.nii.gz",
        ],
    )
    anat, _ = _discover(tmp_path, synthesis_level="subject")
    t1w = _t1w_jobs(anat)
    assert len(t1w) == 1
    assert t1w[0].get("synthesis_scope") != "cross_session"
    assert _paths(t1w[0]) == ["sub-01_ses-01_run-1_T1w.nii.gz"]


def test_mp2rage_collection_and_unit1_not_selected(tmp_path):
    _make(
        tmp_path,
        [
            "sub-01/ses-01/anat/sub-01_ses-01_run-1_T1w.nii.gz",
            "sub-01/ses-01/anat/sub-01_ses-01_run-1_inv-1_MP2RAGE.nii.gz",
            "sub-01/ses-01/anat/sub-01_ses-01_run-1_inv-2_MP2RAGE.nii.gz",
            "sub-01/ses-01/anat/sub-01_ses-01_run-1_UNIT1.nii.gz",
        ],
    )
    anat, _ = _discover(tmp_path)
    assert [_paths(j) for j in anat] == [["sub-01_ses-01_run-1_T1w.nii.gz"]]


def test_single_frame_bold_is_still_selected(tmp_path):
    # A 1-frame bold can be a legitimate acquisition; discovery must not drop it.
    _make(tmp_path, ["sub-01/ses-01/func/sub-01_ses-01_task-rest_run-4_bold.nii.gz"])
    _, func = _discover(tmp_path)
    assert [Path(j["file_path"]).name for j in func] == [
        "sub-01_ses-01_task-rest_run-4_bold.nii.gz"
    ]


def test_validator_reports_non_magnitude_anat_as_not_processed(tmp_path):
    phase = "sub-01/ses-01/anat/sub-01_ses-01_run-2_part-phase_T1w.nii.gz"
    _make(
        tmp_path,
        [
            "sub-01/ses-01/anat/sub-01_ses-01_run-1_T1w.nii.gz",
            "sub-01/ses-01/anat/sub-01_ses-01_run-1_part-mag_T2w.nii.gz",
            phase,
            "sub-01/ses-01/func/sub-01_ses-01_task-rest_run-1_bold.nii.gz",
        ],
    )
    findings = _scan_bids_layout(tmp_path)
    skipped = [p for f in findings if f.code == "BIDS202" for p in f.paths]
    assert skipped == [phase]
    assert any("part-phase" in f.detail for f in findings if f.code == "BIDS202")
