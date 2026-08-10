"""Unit tests for structural BIDS input validation.

Regression coverage for the failure these checks exist to catch: a subject
directory whose files carry a *different* ``sub-`` label. pybids resolves that
conflict in favour of the directory and discards the filename entity, so
discovery never noticed, and the pipeline published ``sub-longmonk1/`` full of
``sub-longmonk_*`` files.

The validator only stats the tree - it never opens a NIfTI - so the fixtures
below are empty files.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import List
from unittest import mock

import pytest

from nhp_mri_prep.nextflow_scripts.discover_bids_for_nextflow import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    _format_finding,
    _scan_bids_layout,
    main,
    validate_bids,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _make(
    root: Path, rel_paths: List[str], *, dataset_description: bool = True
) -> Path:
    """Create empty files at ``rel_paths`` under ``root``."""
    for rel in rel_paths:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    if dataset_description:
        (root / "dataset_description.json").write_text('{"Name": "t"}')
    return root


def _codes(findings, severity=None):
    return sorted(
        f.code for f in findings if severity is None or f.severity == severity
    )


def _by_code(findings, code):
    matches = [f for f in findings if f.code == code]
    assert matches, f"expected a {code} finding, got {_codes(findings)}"
    return matches


# --- clean datasets must stay silent ----------------------------------------


def test_clean_session_dataset_has_no_findings(tmp_path):
    _make(tmp_path, ["sub-01/ses-001/anat/sub-01_ses-001_run-1_T1w.nii.gz"])
    assert _scan_bids_layout(tmp_path) == []


def test_clean_sessionless_dataset_has_no_findings(tmp_path):
    _make(
        tmp_path,
        ["sub-01/anat/sub-01_T1w.nii.gz", "sub-01/func/sub-01_task-rest_bold.nii.gz"],
    )
    assert _scan_bids_layout(tmp_path) == []


def test_clean_func_with_full_entities_has_no_findings(tmp_path):
    _make(tmp_path, ["sub-01/ses-001/func/sub-01_ses-001_task-rest_run-1_bold.nii.gz"])
    assert _scan_bids_layout(tmp_path) == []


def test_extra_entities_do_not_trip_validation(tmp_path):
    """acq-/desc-/run- are parsed but only sub-/ses- are checked."""
    _make(tmp_path, ["sub-01/anat/sub-01_acq-MPRAGE_run-1_T1w.nii.gz"])
    assert _scan_bids_layout(tmp_path) == []


def test_uncompressed_nifti_is_treated_like_nii_gz(tmp_path):
    _make(tmp_path, ["sub-01/anat/sub-01_T1w.nii"])
    assert _scan_bids_layout(tmp_path) == []


def test_bundled_example_dataset_is_clean():
    """The shipped demo dataset must never trip validation.

    This is the "do not break working datasets" barrier: ``docs/demo.rst`` walks
    users through processing this exact tree.
    """
    dataset = REPO_ROOT / "examples" / "dataset_example"
    if not dataset.is_dir():
        pytest.skip("bundled example dataset not present")
    assert _scan_bids_layout(dataset) == []


# --- the regression case ----------------------------------------------------


def test_long_test_layout_is_rejected(tmp_path):
    """Reproduces /mnt/DataDrive3/swap/test_brainana/raw/long_test exactly."""
    _make(
        tmp_path,
        [
            "sub-longmonk1/anat/sub-longmonk_ses-1_run-1_T1w.nii.gz",
            "sub-longmonk1/anat/sub-longmonk_ses-1_run-2_T1w.nii.gz",
            "sub-longmonk1/anat/sub-longmonk_ses-1_run-1_T2w.nii.gz",
            "sub-longmonk1/anat/sub-longmonk_ses-1_run-2_T2w.nii.gz",
            "sub-longmonk1/anat/sub-longmonk_ses-1_run-3_T2w.nii.gz",
        ],
        dataset_description=False,
    )
    findings = _scan_bids_layout(tmp_path)

    # One finding covering all five files, not five findings.
    mismatch = _by_code(findings, "BIDS101")
    assert len(mismatch) == 1
    assert mismatch[0].severity == SEVERITY_ERROR
    assert len(mismatch[0].paths) == 5

    assert _by_code(findings, "BIDS201")[0].severity == SEVERITY_WARNING
    assert _by_code(findings, "BIDS203")[0].severity == SEVERITY_WARNING

    with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        assert validate_bids(tmp_path) is False


def test_subject_mismatch_report_names_both_labels(tmp_path):
    _make(tmp_path, ["sub-longmonk1/anat/sub-longmonk_T1w.nii.gz"])
    err = io.StringIO()
    with redirect_stderr(err), redirect_stdout(io.StringIO()):
        validate_bids(tmp_path)
    text = err.getvalue()
    # The message is the deliverable: a user must be able to see which two labels
    # disagree without going back to the filesystem.
    assert "sub-longmonk1" in text
    assert "sub-longmonk" in text
    assert "BIDS101" in text


# --- subject identity -------------------------------------------------------


def test_underscore_in_subject_label_is_caught_as_mismatch(tmp_path):
    """``sub-pd0_2`` parses to ``sub='pd0'`` - the label breaks output naming."""
    _make(tmp_path, ["sub-pd0_2/anat/sub-pd0_2T1_run-1_T1w.nii.gz"])
    assert (
        _by_code(_scan_bids_layout(tmp_path), "BIDS101")[0].severity == SEVERITY_ERROR
    )


def test_hyphen_in_subject_label_is_clean(tmp_path):
    """False-positive guard for the test above: hyphens round-trip fine."""
    _make(tmp_path, ["sub-long-monk/anat/sub-long-monk_T1w.nii.gz"])
    assert _scan_bids_layout(tmp_path) == []


def test_missing_subject_entity_is_error(tmp_path):
    _make(tmp_path, ["sub-01/anat/T1w.nii.gz"])
    assert (
        _by_code(_scan_bids_layout(tmp_path), "BIDS102")[0].severity == SEVERITY_ERROR
    )


def test_distinct_mismatches_are_separate_findings(tmp_path):
    """Grouping is per defect, so two different label pairs must not merge."""
    _make(
        tmp_path,
        [
            "sub-01/anat/sub-98_T1w.nii.gz",
            "sub-02/anat/sub-99_T1w.nii.gz",
        ],
    )
    assert len(_by_code(_scan_bids_layout(tmp_path), "BIDS101")) == 2


# --- session identity -------------------------------------------------------


def test_session_dir_filename_mismatch_is_error(tmp_path):
    _make(tmp_path, ["sub-01/ses-A/anat/sub-01_ses-B_T1w.nii.gz"])
    assert (
        _by_code(_scan_bids_layout(tmp_path), "BIDS103")[0].severity == SEVERITY_ERROR
    )


def test_missing_session_entity_single_session_is_warning(tmp_path):
    _make(tmp_path, ["sub-01/ses-001/anat/sub-01_T1w.nii.gz"])
    findings = _scan_bids_layout(tmp_path)
    assert _by_code(findings, "BIDS204")[0].severity == SEVERITY_WARNING
    assert _codes(findings, SEVERITY_ERROR) == []


def test_missing_session_entity_multi_session_is_error(tmp_path):
    """Two sessions with no session entity collapse onto one output path."""
    _make(
        tmp_path,
        [
            "sub-01/ses-001/anat/sub-01_T1w.nii.gz",
            "sub-01/ses-002/anat/sub-01_T1w.nii.gz",
        ],
    )
    assert (
        _by_code(_scan_bids_layout(tmp_path), "BIDS104")[0].severity == SEVERITY_ERROR
    )


def test_session_entity_without_session_dir_is_warning(tmp_path):
    _make(tmp_path, ["sub-01/anat/sub-01_ses-1_T1w.nii.gz"])
    findings = _scan_bids_layout(tmp_path)
    assert _by_code(findings, "BIDS201")[0].severity == SEVERITY_WARNING
    assert _codes(findings, SEVERITY_ERROR) == []


# --- BIDS105: datatype dirs beside session dirs -----------------------------
#
# BIDS allows a subject directory to hold EITHER ses-*/ or datatype dirs, never
# both (schema rules.directories.raw.subject.subdirs is oneOf[session, datatype];
# spec: "Within the session subdirectory - or the subject subdirectory if no
# session subdirectories are present - are subdirectories named according to data
# type"). Discovery queries one session per pass, so the subject-level files are
# read by nobody and vanish from the run without comment.


def test_datatype_dir_beside_session_dirs_is_error(tmp_path):
    _make(
        tmp_path,
        [
            "sub-01/ses-001/anat/sub-01_ses-001_T1w.nii.gz",
            "sub-01/anat/sub-01_T1w.nii.gz",
        ],
    )
    finding = _by_code(_scan_bids_layout(tmp_path), "BIDS105")[0]
    assert finding.severity == SEVERITY_ERROR
    # Only the orphaned file is named - the properly-nested one is fine.
    assert finding.paths == ("sub-01/anat/sub-01_T1w.nii.gz",)


def test_datatype_beside_sessions_respects_the_gate(tmp_path):
    """A file discovery would never consume stays a warning, as everywhere else."""
    _make(
        tmp_path,
        [
            "sub-01/ses-001/anat/sub-01_ses-001_T1w.nii.gz",
            "sub-01/misc/whatever.nii.gz",
        ],
    )
    findings = _scan_bids_layout(tmp_path)
    assert _codes(findings, SEVERITY_ERROR) == []
    assert _by_code(findings, "BIDS202")


def test_datatype_beside_sessions_is_out_of_scope_with_session_filter(tmp_path):
    """A file under no session can never belong to an explicitly selected one."""
    _make(
        tmp_path,
        [
            "sub-01/ses-001/anat/sub-01_ses-001_T1w.nii.gz",
            "sub-01/anat/sub-01_T1w.nii.gz",
        ],
    )
    assert _codes(_scan_bids_layout(tmp_path, sessions=["001"]), SEVERITY_ERROR) == []
    assert _codes(_scan_bids_layout(tmp_path), SEVERITY_ERROR) == ["BIDS105"]


def test_bids105_suppresses_bids201_for_the_same_file(tmp_path):
    """BIDS201's fix - move it under a matching ses-*/ - is exactly BIDS105's fix."""
    _make(
        tmp_path,
        [
            "sub-01/ses-001/anat/sub-01_ses-001_T1w.nii.gz",
            "sub-01/anat/sub-01_ses-001_T1w.nii.gz",
        ],
    )
    codes = _codes(_scan_bids_layout(tmp_path))
    assert "BIDS105" in codes
    assert "BIDS201" not in codes


# --- the gate: only files discovery would select can raise an error ---------


def test_bold_in_anat_dir_is_warning_not_error(tmp_path):
    _make(tmp_path, ["sub-01/anat/sub-99_task-rest_bold.nii.gz"])
    findings = _scan_bids_layout(tmp_path)
    assert _codes(findings, SEVERITY_ERROR) == []
    assert _by_code(findings, "BIDS202")


def test_nifti_outside_datatype_dir_is_warning(tmp_path):
    """The ``sub-HJT/ynimg_*_PET_*.nii`` case - loose volumes beside real data."""
    _make(
        tmp_path,
        [
            "sub-HJT/ynimg_116163_PET_TEMPLATE_111_mmc4.nii",
            "sub-HJT/anat/sub-HJT_T1w.nii.gz",
        ],
    )
    findings = _scan_bids_layout(tmp_path)
    assert _codes(findings, SEVERITY_ERROR) == []
    assert _by_code(findings, "BIDS202")[0].paths == (
        "sub-HJT/ynimg_116163_PET_TEMPLATE_111_mmc4.nii",
    )


def test_unrecognized_suffix_in_anat_is_warning(tmp_path):
    """The ``histvol_uglyLH_grayscale_70um`` case: in anat/, but not a read suffix."""
    _make(tmp_path, ["sub-ugly/anat/histvol_uglyLH_grayscale_70um.nii.gz"])
    findings = _scan_bids_layout(tmp_path)
    assert _codes(findings, SEVERITY_ERROR) == []
    assert _by_code(findings, "BIDS202")


def test_unknown_datatype_dir_is_warning(tmp_path):
    _make(tmp_path, ["sub-01/misc/sub-99_T1w.nii.gz"])
    findings = _scan_bids_layout(tmp_path)
    assert _codes(findings, SEVERITY_ERROR) == []
    assert _by_code(findings, "BIDS202")


def test_known_but_unconsumed_datatype_names_the_modality(tmp_path):
    """A recognized BIDS datatype gets a more specific 'why' than a stray dir."""
    _make(tmp_path, ["sub-01/dwi/sub-99_dwi.nii.gz"])
    findings = _scan_bids_layout(tmp_path)
    assert _codes(findings, SEVERITY_ERROR) == []
    assert "dwi" in _by_code(findings, "BIDS202")[0].detail


def test_nifti_nested_below_datatype_dir_is_warning(tmp_path):
    """A datatype dir sits directly under sub-*/ or sub-*/ses-*/, nothing deeper."""
    _make(tmp_path, ["sub-01/anat/extra/sub-99_T1w.nii.gz"])
    findings = _scan_bids_layout(tmp_path)
    assert _codes(findings, SEVERITY_ERROR) == []
    assert _by_code(findings, "BIDS202")


def test_sidecars_are_not_scanned(tmp_path):
    """Only NIfTIs are inputs; a mislabelled .json must not abort a run."""
    _make(tmp_path, ["sub-01/anat/sub-01_T1w.nii.gz", "sub-01/anat/sub-99_T1w.json"])
    assert _scan_bids_layout(tmp_path) == []


# --- scoping ----------------------------------------------------------------


def test_nested_subject_dirs_are_ignored(tmp_path):
    """Mirrors ``_is_top_level_subject_path``: only top-level ``sub-*`` counts."""
    _make(
        tmp_path,
        [
            "sub-01/anat/sub-01_T1w.nii.gz",
            "derivatives/sub-01/anat/sub-99_T1w.nii.gz",
            "sub-01/badQC/sub-aaa/anat/sub-bbb_T1w.nii.gz",
        ],
    )
    findings = _scan_bids_layout(tmp_path)
    assert _codes(findings, SEVERITY_ERROR) == []


@pytest.mark.parametrize("wanted", [["01"], ["sub-01"]])
def test_subjects_filter_limits_findings(tmp_path, wanted):
    """A broken subject must not block subjects the user did not ask for."""
    _make(
        tmp_path,
        [
            "sub-01/anat/sub-01_T1w.nii.gz",
            "sub-02/anat/sub-99_T1w.nii.gz",
        ],
    )
    assert _scan_bids_layout(tmp_path, subjects=wanted) == []
    assert _codes(_scan_bids_layout(tmp_path), SEVERITY_ERROR) == ["BIDS101"]


def test_unknown_subject_filter_finds_nothing(tmp_path):
    """Discovery warns about the missing subject later; validation stays quiet."""
    _make(tmp_path, ["sub-01/anat/sub-01_T1w.nii.gz"])
    assert _scan_bids_layout(tmp_path, subjects=["99"]) == []


@pytest.mark.parametrize("wanted", [["001"], ["ses-001"]])
def test_sessions_filter_limits_findings(tmp_path, wanted):
    """A broken session must not block sessions the user did not ask for."""
    _make(
        tmp_path,
        [
            "sub-01/ses-001/anat/sub-01_ses-001_T1w.nii.gz",
            "sub-01/ses-002/anat/sub-01_ses-999_T1w.nii.gz",
        ],
    )
    assert _scan_bids_layout(tmp_path, sessions=wanted) == []
    assert _codes(_scan_bids_layout(tmp_path), SEVERITY_ERROR) == ["BIDS103"]


def test_session_filter_downgrades_multi_session_error(tmp_path):
    """With one session selected there is no second session to collide with."""
    _make(
        tmp_path,
        [
            "sub-01/ses-001/anat/sub-01_T1w.nii.gz",
            "sub-01/ses-002/anat/sub-01_T1w.nii.gz",
        ],
    )
    assert _codes(_scan_bids_layout(tmp_path), SEVERITY_ERROR) == ["BIDS104"]
    scoped = _scan_bids_layout(tmp_path, sessions=["001"])
    assert _codes(scoped, SEVERITY_ERROR) == []
    assert _by_code(scoped, "BIDS204")[0].severity == SEVERITY_WARNING


# --- the filter that reaches validation must be the *effective* one ---------
#
# run_brainana.sh forwards --subjects/--sessions only when the user typed them,
# and discover_bids_dataset() falls back to config bids_filtering otherwise. If
# validation does not apply the same fallback it scans files discovery will never
# touch, and a config-filtered run aborts on an out-of-scope subject.


def _run_main(tmp_path, dataset, config_body, argv_extra=()):
    """Drive main() end-to-end; return its exit code (0 when it does not abort)."""
    config = tmp_path / "config.yaml"
    config.write_text(config_body)
    argv = [
        "discover_bids_for_nextflow.py",
        "--bids_dir",
        str(dataset),
        "--output_dir",
        str(tmp_path / "out"),
        "--config_file",
        str(config),
        *argv_extra,
    ]
    with mock.patch.object(sys, "argv", argv):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                main()
            except SystemExit as exc:
                return exc.code or 0
    return 0


def test_config_subject_filter_reaches_validation(tmp_path):
    dataset = _make(
        tmp_path / "bids",
        [
            "sub-01/anat/sub-01_T1w.nii.gz",
            "sub-02/anat/sub-99_T1w.nii.gz",  # broken, but not requested
        ],
    )
    assert _run_main(tmp_path, dataset, 'bids_filtering:\n  subjects: ["01"]\n') == 0
    # Same dataset with no filter must still abort - proves the fixture is broken.
    assert _run_main(tmp_path, dataset, "bids_filtering:\n  subjects: null\n") == 1


def test_config_session_filter_reaches_validation(tmp_path):
    dataset = _make(
        tmp_path / "bids",
        [
            "sub-01/ses-001/anat/sub-01_ses-001_T1w.nii.gz",
            "sub-01/ses-002/anat/sub-01_ses-999_T1w.nii.gz",  # broken, not requested
        ],
    )
    assert _run_main(tmp_path, dataset, 'bids_filtering:\n  sessions: ["001"]\n') == 0
    assert _run_main(tmp_path, dataset, "bids_filtering:\n  sessions: null\n") == 1


def test_cli_filter_still_wins_over_config(tmp_path):
    dataset = _make(
        tmp_path / "bids",
        ["sub-01/anat/sub-01_T1w.nii.gz", "sub-02/anat/sub-99_T1w.nii.gz"],
    )
    code = _run_main(
        tmp_path,
        dataset,
        "bids_filtering:\n  subjects: null\n",
        argv_extra=["--subjects", "01"],
    )
    assert code == 0


# --- hidden files -----------------------------------------------------------


def test_appledouble_sidecar_is_ignored(tmp_path):
    """pybids ignores dotfiles, so a finding on one could only be a false positive.

    ``._sub-99_T1w.nii.gz`` is what macOS leaves beside a file copied over SMB. It
    parses as a different subject, so before this was skipped it aborted the run
    over a file nothing ever reads.
    """
    _make(
        tmp_path,
        ["sub-01/anat/sub-01_T1w.nii.gz", "sub-01/anat/._sub-99_T1w.nii.gz"],
    )
    assert _scan_bids_layout(tmp_path) == []


def test_hidden_directories_are_ignored(tmp_path):
    _make(
        tmp_path, ["sub-01/anat/sub-01_T1w.nii.gz", "sub-01/.hidden/sub-99_T1w.nii.gz"]
    )
    assert _scan_bids_layout(tmp_path) == []


# --- report mechanics -------------------------------------------------------


def test_findings_are_deterministically_ordered(tmp_path):
    _make(
        tmp_path,
        [
            "sub-01/anat/sub-99_T1w.nii.gz",
            "sub-02/anat/T1w.nii.gz",
            "sub-03/anat/sub-03_ses-1_T1w.nii.gz",
            "sub-04/misc/sub-04_T1w.nii.gz",
        ],
        dataset_description=False,
    )
    first = _scan_bids_layout(tmp_path)
    assert first == _scan_bids_layout(tmp_path)
    # Errors are reported before warnings.
    severities = [f.severity for f in first]
    assert severities == sorted(severities, key=lambda s: s != SEVERITY_ERROR)


def test_long_path_lists_are_truncated(tmp_path):
    """A 200-file defect must not bury the report under 200 lines."""
    _make(tmp_path, [f"sub-01/anat/sub-99_run-{i}_T1w.nii.gz" for i in range(15)])
    finding = _by_code(_scan_bids_layout(tmp_path), "BIDS101")[0]
    assert len(finding.paths) == 15
    rendered = _format_finding(finding)
    assert rendered.count("sub-99_run-") == 10
    assert "... 5 more" in rendered


def test_clean_dataset_reports_pass_and_returns_true(tmp_path):
    _make(tmp_path, ["sub-01/anat/sub-01_T1w.nii.gz"])
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(io.StringIO()):
        assert validate_bids(tmp_path) is True
    assert "BIDS structure check passed" in out.getvalue()


def test_validate_bids_still_fails_on_missing_subject_dirs(tmp_path):
    """Pre-existing behaviour: the wrong-directory mistake stays a clear error."""
    (tmp_path / "anat").mkdir()
    err = io.StringIO()
    with redirect_stderr(err), redirect_stdout(io.StringIO()):
        assert validate_bids(tmp_path) is False
    assert "does not look like a BIDS dataset" in err.getvalue()
