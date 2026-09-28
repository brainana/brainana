"""Unit tests for the QC report's ingest-findings section.

Ingest records what it repaired and what it could not in the derivative JSON
sidecars. A log line scrolls past and a sidecar is opened by nobody, so these
pin that the findings actually reach the per-subject HTML report — and that a
clean dataset still produces a report with no such section at all.
"""

import json

import pytest

from nhp_mri_prep.quality_control.data_findings import (
    collect_ingest_findings,
    has_findings,
    render_data_findings_content,
)


def _sidecar(subject_dir, name, payload):
    anat = subject_dir / "anat"
    anat.mkdir(parents=True, exist_ok=True)
    (anat / name).write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------


def test_clean_subject_yields_no_findings(tmp_path):
    _sidecar(
        tmp_path, "sub-01_T1w.json", {"Synthesized": False, "SkullStripped": False}
    )

    findings = collect_ingest_findings(tmp_path)

    assert findings == {"repaired": [], "unrepaired": []}
    assert not has_findings(findings)
    assert render_data_findings_content(findings) == ""


def test_missing_subject_dir_is_not_an_error(tmp_path):
    findings = collect_ingest_findings(tmp_path / "nope")

    assert findings == {"repaired": [], "unrepaired": []}


def test_unreadable_sidecar_is_skipped(tmp_path):
    anat = tmp_path / "anat"
    anat.mkdir(parents=True)
    (anat / "broken.json").write_text("{not json", encoding="utf-8")
    _sidecar(
        tmp_path, "sub-01_T1w.json", {"OrientationRecovered": ["assumed-LAS-centered"]}
    )

    findings = collect_ingest_findings(tmp_path)

    # One bad sidecar must not take the whole report down with it.
    assert [e["key"] for e in findings["repaired"]] == ["OrientationRecovered"]


def test_each_repair_key_is_collected(tmp_path):
    _sidecar(
        tmp_path,
        "sub-01_T1w.json",
        {
            "Sources": ["/bids/sub-01/anat/sub-01_T1w.nii.gz"],
            "Input4DCollapsed": ["squeezed"],
            "OrientationRecovered": ["assumed-LAS-centered"],
            "QformSformReconciled": ["qform-set-from-sform"],
        },
    )

    findings = collect_ingest_findings(tmp_path)

    assert [e["key"] for e in findings["repaired"]] == [
        "Input4DCollapsed",
        "OrientationRecovered",
        "QformSformReconciled",
    ]
    assert findings["repaired"][0]["files"] == ["sub-01_T1w.nii.gz"]


def test_unrepaired_warnings_are_collected(tmp_path):
    _sidecar(
        tmp_path,
        "sub-01_T1w.json",
        {
            "Sources": ["/bids/sub-01/anat/sub-01_T1w.nii.gz"],
            "InputHeaderWarnings": ["Spatial units are 'meter', not mm."],
        },
    )

    findings = collect_ingest_findings(tmp_path)

    assert findings["repaired"] == []
    assert len(findings["unrepaired"]) == 1
    assert "meter" in findings["unrepaired"][0]["message"]


def test_one_defect_across_runs_groups_into_one_entry(tmp_path):
    for run in ("1", "2"):
        _sidecar(
            tmp_path,
            f"sub-01_run-{run}_T1w.json",
            {
                "Sources": [f"/bids/sub-01_run-{run}_T1w.nii.gz"],
                "OrientationRecovered": ["assumed-LAS-centered"],
            },
        )

    findings = collect_ingest_findings(tmp_path)

    assert len(findings["repaired"]) == 1
    assert findings["repaired"][0]["files"] == [
        "sub-01_run-1_T1w.nii.gz",
        "sub-01_run-2_T1w.nii.gz",
    ]


def test_sidecar_without_sources_falls_back_to_its_own_name(tmp_path):
    _sidecar(tmp_path, "sub-01_T1w.json", {"Input4DCollapsed": ["mean"]})

    findings = collect_ingest_findings(tmp_path)

    assert findings["repaired"][0]["files"] == ["sub-01_T1w.json"]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


@pytest.mark.parametrize("findings", [None, {}, {"repaired": [], "unrepaired": []}])
def test_nothing_to_report_renders_nothing(findings):
    assert render_data_findings_content(findings) == ""
    assert not has_findings(findings)


def test_orientation_repair_always_carries_the_left_right_caveat(tmp_path):
    """The one finding here that can silently invalidate a result."""
    _sidecar(
        tmp_path, "sub-01_T1w.json", {"OrientationRecovered": ["assumed-LAS-centered"]}
    )

    html = render_data_findings_content(collect_ingest_findings(tmp_path))

    assert "left/right mirror" in html
    assert "Confirm handedness" in html


def test_repairs_alone_read_as_informational(tmp_path):
    _sidecar(tmp_path, "sub-01_T1w.json", {"Input4DCollapsed": ["squeezed"]})

    html = render_data_findings_content(collect_ingest_findings(tmp_path))

    assert 'class="status ok"' in html
    assert "Repaired automatically" in html
    assert "Not repaired" not in html


def test_unrepaired_findings_escalate_the_section_to_warn(tmp_path):
    _sidecar(
        tmp_path,
        "sub-01_T1w.json",
        {
            "Input4DCollapsed": ["squeezed"],
            "InputHeaderWarnings": ["Header pixdim disagrees with the affine."],
        },
    )

    html = render_data_findings_content(collect_ingest_findings(tmp_path))

    assert 'class="status warn"' in html
    assert "Action needed" in html
    # Both groups render, clearly separated.
    assert "Repaired automatically" in html
    assert "Not repaired" in html


def test_full_report_renders_the_section_and_its_nav_entry(tmp_path):
    """Pins the template wiring — a missing {DATA_FINDINGS_SECTION} key is a KeyError."""
    import logging

    from nhp_mri_prep.quality_control.reports import write_html_report

    _sidecar(
        tmp_path,
        "sub-01_T1w.json",
        {"OrientationRecovered": ["assumed-LAS-centered"]},
    )
    report_data = {
        "metadata": {
            "generation_time": "2026-01-01 00:00:00",
            "pipeline_name": "brainana",
            "version": "0.0.0",
            "working_directory": str(tmp_path),
            "subject_id": "01",
        },
        "configuration": {},
        "organized_snapshots": {
            "anatomical": {},
            "functional": {},
            "field_mapping": {},
            "summary": {},
        },
        "dataset_context": {"subject_file_counts": {}},
        "available_entities": {},
        "run_status": None,
        "data_findings": collect_ingest_findings(tmp_path),
    }

    out = tmp_path / "sub-01.html"
    write_html_report(report_data, out, logging.getLogger(__name__))
    html = out.read_text(encoding="utf-8")

    assert 'id="DataFindings"' in html
    assert 'href="#DataFindings"' in html
    assert "left/right mirror" in html

    # And a clean subject gets neither the section nor the nav entry.
    clean = tmp_path / "clean.html"
    write_html_report(
        {**report_data, "data_findings": {"repaired": [], "unrepaired": []}},
        clean,
        logging.getLogger(__name__),
    )
    clean_html = clean.read_text(encoding="utf-8")
    assert 'id="DataFindings"' not in clean_html
    assert 'href="#DataFindings"' not in clean_html


def test_findings_are_html_escaped(tmp_path):
    _sidecar(
        tmp_path,
        "sub-01_T1w.json",
        {"InputHeaderWarnings": ["<script>alert(1)</script>"]},
    )

    html = render_data_findings_content(collect_ingest_findings(tmp_path))

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
