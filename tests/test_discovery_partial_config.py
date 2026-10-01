"""A partial --config must not change what discovery does.

Discovery used to read the user's YAML as-is: validate_config() merged the
package defaults but its result was thrown away, and discovery then fell back
to its own defaults for every key the file omitted. For anat.synthesis_level
that fallback was "session" while defaults.yaml (and so the effective config
the pipeline records) says "subject" -- a partial config silently switched a
multi-session subject from one shared T1w to one per session.
"""

import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import pytest

from nhp_mri_prep.config.config_io import get_default_config
from nhp_mri_prep.nextflow_scripts.discover_bids_for_nextflow import main
from nhp_mri_prep.steps.bids_discovery import discover_bids_dataset


def _make(root: Path, rel_paths):
    for rel in rel_paths:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (root / "dataset_description.json").write_text(
        '{"Name": "t", "BIDSVersion": "1.9.0"}'
    )
    return root


MULTI_SESSION = [
    "sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz",
    "sub-01/ses-02/anat/sub-01_ses-02_T1w.nii.gz",
    "sub-01/ses-01/func/sub-01_ses-01_task-rest_bold.nii.gz",
]


def _run_main(tmp_path, dataset, config_body, argv_extra=()):
    config = tmp_path / "config.yaml"
    config.write_text(config_body)
    out = tmp_path / "out"
    argv = [
        "discover_bids_for_nextflow.py",
        "--bids_dir", str(dataset),
        "--output_dir", str(out),
        "--config_file", str(config),
        *argv_extra,
    ]  # fmt: skip
    with mock.patch.object(sys, "argv", argv):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                main()
            except SystemExit as exc:
                assert not exc.code, "discovery aborted"
    reports = out / "nextflow_reports"
    return (
        json.loads((reports / "anatomical_jobs.json").read_text()),
        json.loads((reports / "functional_jobs.json").read_text()),
    )


def _t1w(jobs):
    return [j for j in jobs if j["suffix"] == "T1w"]


def test_partial_config_uses_default_synthesis_level(tmp_path):
    assert get_default_config()["anat"]["synthesis_level"] == "subject"
    dataset = _make(tmp_path / "bids", MULTI_SESSION)
    # A config that says nothing about anat at all.
    anat, _ = _run_main(tmp_path, dataset, "general:\n  verbose: 1\n")
    t1w = _t1w(anat)
    assert len(t1w) == 1, "cross-session synthesis expected (defaults.yaml: subject)"
    assert t1w[0]["needs_synthesis"] is True


def test_explicit_session_level_still_honoured(tmp_path):
    dataset = _make(tmp_path / "bids", MULTI_SESSION)
    anat, _ = _run_main(tmp_path, dataset, 'anat:\n  synthesis_level: "session"\n')
    assert len(_t1w(anat)) == 2


def test_empty_sections_do_not_crash(tmp_path):
    dataset = _make(tmp_path / "bids", MULTI_SESSION)
    anat, _ = _run_main(tmp_path, dataset, "bids_filtering:\ngeneral:\n  verbose: 1\n")
    assert len(_t1w(anat)) == 1


def test_library_fallback_matches_defaults(tmp_path):
    """discover_bids_dataset() called directly with a config lacking the key."""
    dataset = _make(tmp_path / "bids", MULTI_SESSION)
    anat, _ = discover_bids_dataset(dataset, {"general": {"anat_only": True}})
    assert len(_t1w(anat)) == 1


@pytest.mark.parametrize("flag", [["--anat_only"], ["--anat_only", "true"]])
def test_cli_anat_only_skips_functional_discovery(tmp_path, flag):
    dataset = _make(tmp_path / "bids", MULTI_SESSION)
    _, func = _run_main(tmp_path, dataset, "general:\n  anat_only: false\n", flag)
    assert func == []
    _, func = _run_main(tmp_path, dataset, "general:\n  anat_only: false\n")
    assert len(func) == 1


@pytest.mark.parametrize("value", ["false", "0", "no"])
def test_cli_anat_only_false_overrides_yaml_true(tmp_path, value):
    """`--anat_only false` must win over YAML true, as it does in main.nf."""
    dataset = _make(tmp_path / "bids", MULTI_SESSION)
    _, func = _run_main(
        tmp_path, dataset, "general:\n  anat_only: true\n", ["--anat_only", value]
    )
    assert len(func) == 1
