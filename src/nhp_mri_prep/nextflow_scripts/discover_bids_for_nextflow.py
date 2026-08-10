#!/usr/bin/env python3
"""
BIDS discovery script for Nextflow pipeline.

This script runs BEFORE Nextflow starts to:
1. Validate the BIDS dataset structure (always; cannot be skipped)
2. Discover all anatomical and functional jobs
3. Print a summary of discovered jobs
4. Save JSON files for Nextflow to read

This ensures discovery completes before processing starts, allowing
Nextflow to show proper job counts in progress.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src/ to path for nhp_mri_prep imports (nextflow_scripts/ -> nhp_mri_prep -> src)
_src_dir = Path(__file__).resolve().parent.parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from nhp_mri_prep.config.config_io import get_nested_config_value, load_yaml_config
from nhp_mri_prep.config.config_validation import validate_config
from nhp_mri_prep.steps.bids_discovery import (
    _normalize_bids_id,
    _normalize_to_list,
    discover_bids_dataset,
)
from nhp_mri_prep.utils.bids import get_filename_stem, parse_bids_entities

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"

# Datatype directories the pipeline reads, and the suffixes it consumes in each.
# Mirrors the pybids filters in steps/bids_discovery.py (datatype="anat",
# suffix=["T1w", "T2w"] / datatype="func", suffix="bold").
_CONSUMED_SUFFIXES: Dict[str, set] = {"anat": {"T1w", "T2w"}, "func": {"bold"}}

# Recognized BIDS datatype directories. A NIfTI under any other directory is not
# discovered, but naming one here keeps the "will not be processed" message honest
# about *why* (unsupported modality vs. stray directory).
_KNOWN_DATATYPES = frozenset({"anat", "func", "fmap", "dwi", "perf", "swi", "pet"})

_NIFTI_EXTENSIONS = (".nii.gz", ".nii")


@dataclass(frozen=True)
class BidsFinding:
    """One layout defect, together with every file exhibiting it.

    Findings are grouped by defect rather than by file: five files sharing one
    subject-label mismatch are a single finding naming five paths, not five
    findings. ``paths`` are relative to the dataset root so the report reads
    identically against ``/input`` in Docker and against a host path locally.
    """

    code: str
    severity: str
    title: str
    detail: str
    fix: str
    paths: Tuple[str, ...] = field(default_factory=tuple)


def _iter_subject_niftis(subject_dir: Path):
    """Yield ``(file_path, session_label, datatype)`` for NIfTIs under one subject dir.

    ``session_label`` is the label of the ``ses-*`` path component, or None when the
    file is not under one. ``datatype`` is the immediate parent directory name when
    it sits at the expected depth, else None.

    Nested ``sub-*`` directories are pruned so the scan's scope matches discovery's:
    ``_is_top_level_subject_path()`` in steps/bids_discovery.py drops files whose
    first path component under the dataset root is not the subject directory, which
    is what excludes layouts like ``badQC/sub-aaa`` and ``derivatives/sub-01``.

    Hidden files and directories are skipped for the same parity reason: pybids
    ignores everything starting with a dot, so a finding raised on one could only
    ever be a false positive. That matters most for macOS AppleDouble sidecars -
    ``._sub-99_T1w.nii.gz`` copied alongside a valid file parses as a *different*
    subject and would otherwise abort the run over a file nothing ever reads.
    """
    for dirpath, dirnames, filenames in os.walk(subject_dir):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if not d.startswith(".")
            and not d.startswith("sub-")
            and not os.path.islink(os.path.join(dirpath, d))
        )
        rel = Path(dirpath).relative_to(subject_dir).parts
        if rel and rel[0].startswith("ses-"):
            session = rel[0][len("ses-") :]
            depth = rel[1:]
        else:
            session = None
            depth = rel
        # A datatype directory sits directly under sub-*/ or sub-*/ses-*/.
        datatype = depth[0] if len(depth) == 1 else None
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            if name.endswith(_NIFTI_EXTENSIONS):
                yield Path(dirpath) / name, session, datatype


def _scan_bids_layout(
    bids_dir: Path,
    subjects: Optional[List[str]] = None,
    sessions: Optional[List[str]] = None,
) -> List[BidsFinding]:
    """Cross-check every input file's *path* identity against its *filename* identity.

    This is the check pybids cannot make. ``BIDSLayout`` resolves ``subject`` and
    ``session`` from the directory names and silently discards a conflicting entity
    in the filename, so a file named ``sub-A_T1w.nii.gz`` sitting in ``sub-B/anat/``
    is reported by pybids as subject B with no complaint. Brainana then names output
    *directories* from that (directory-derived) job ID but output *basenames* from
    the original filename stem, publishing ``sub-B/anat/sub-A_*.nii.gz``. The scan
    therefore walks the tree itself and never consults pybids.

    It also rejects a second layout pybids accepts silently: a subject holding both
    ``ses-*/`` directories and datatype directories directly under ``sub-*/``. BIDS
    allows datatype directories at the subject level only when no session
    directories are present (schema ``rules.directories.raw.subject.subdirs`` is
    ``oneOf: [session, datatype]``), and discovery queries a fixed session per pass,
    so the subject-level files are read by nobody - see ``BIDS105``.

    Errors are raised only for files discovery would actually select — inside a
    consumed datatype directory, with a consumed suffix. A file outside that set is
    never read, so it cannot mislabel anything; it is reported as a warning instead
    so the silent skip at least becomes visible.

    Args:
        bids_dir: Path to the BIDS dataset root.
        subjects: Optional subject filter (``01`` and ``sub-01`` both accepted).
            Scoping matters: without it a single malformed subject would block
            every other subject in the dataset.
        sessions: Optional session filter, same normalization. Sessions outside the
            filter are not walked at all, so they cannot raise findings, and they
            are excluded from the "does this subject have several sessions?" count
            that decides ``BIDS104`` vs ``BIDS204``.

    Returns:
        Findings sorted deterministically by (severity, code, first path).
    """
    root = bids_dir.resolve()
    wanted = (
        {_normalize_bids_id(s, "sub-") for s in subjects}
        if subjects is not None
        else None
    )
    wanted_sessions = (
        {_normalize_bids_id(s, "ses-") for s in sessions}
        if sessions is not None
        else None
    )

    # defect key -> list of dataset-root-relative paths
    mismatched_subject: Dict[Tuple[str, str], List[str]] = {}
    missing_subject: List[str] = []
    mismatched_session: Dict[Tuple[str, str], List[str]] = {}
    missing_session_multi: List[str] = []
    missing_session_single: List[str] = []
    session_without_dir: List[str] = []
    datatype_beside_sessions: List[str] = []
    not_discoverable: Dict[str, List[str]] = {}

    for subject_dir in sorted(p for p in root.glob("sub-*") if p.is_dir()):
        dir_subject = subject_dir.name[len("sub-") :]
        if wanted is not None and dir_subject not in wanted:
            continue
        session_dirs = [p for p in subject_dir.glob("ses-*") if p.is_dir()]
        # BIDS105 keys on the layout as it exists on disk, not on the selection:
        # the subject is non-compliant whether or not this run reads every session.
        subject_uses_sessions = bool(session_dirs)
        n_session_dirs = len(
            [
                p
                for p in session_dirs
                if wanted_sessions is None or p.name[len("ses-") :] in wanted_sessions
            ]
        )

        for path, dir_session, datatype in _iter_subject_niftis(subject_dir):
            if (
                dir_session is not None
                and wanted_sessions is not None
                and dir_session not in wanted_sessions
            ):
                continue
            rel = str(path.relative_to(root))
            stem = get_filename_stem(path)
            entities = parse_bids_entities(stem)

            # --- the gate: is this a file discovery would actually select? ---
            if datatype not in _CONSUMED_SUFFIXES:
                reason = (
                    f"it is not inside an {'/'.join(sorted(_CONSUMED_SUFFIXES))} directory"
                    if datatype is None or datatype not in _KNOWN_DATATYPES
                    else f"Brainana does not process the '{datatype}' datatype"
                )
                not_discoverable.setdefault(reason, []).append(rel)
                continue
            suffix = stem.rsplit("_", 1)[-1]
            if suffix not in _CONSUMED_SUFFIXES[datatype]:
                consumed = ", ".join(sorted(_CONSUMED_SUFFIXES[datatype]))
                reason = (
                    f"'{suffix}' is not a suffix Brainana reads from "
                    f"{datatype}/ (expected: {consumed})"
                )
                not_discoverable.setdefault(reason, []).append(rel)
                continue

            # --- subject identity ---
            file_subject = entities.get("sub")
            if not file_subject:
                missing_subject.append(rel)
            elif file_subject != dir_subject:
                mismatched_subject.setdefault((dir_subject, file_subject), []).append(
                    rel
                )

            # --- session identity ---
            file_session = entities.get("ses")
            if dir_session is not None:
                if not file_session:
                    if n_session_dirs > 1:
                        missing_session_multi.append(rel)
                    else:
                        missing_session_single.append(rel)
                elif file_session != dir_session:
                    mismatched_session.setdefault(
                        (dir_session, file_session), []
                    ).append(rel)
            elif subject_uses_sessions and wanted_sessions is None:
                # A datatype directory beside ses-*/ dirs. Not valid BIDS, and
                # discovery never reads it. Reported instead of BIDS201 because
                # BIDS201's advice - move it under a matching ses-*/ - is exactly
                # this finding's fix, and the two together are just noise.
                #
                # Skipped entirely when an explicit session filter is in effect:
                # a file under no session can never belong to a selected session,
                # so it is out of scope for the run the same way an unselected
                # subject is.
                datatype_beside_sessions.append(rel)
            elif file_session:
                session_without_dir.append(rel)

    findings: List[BidsFinding] = []

    for (dir_subject, file_subject), paths in mismatched_subject.items():
        findings.append(
            BidsFinding(
                code="BIDS101",
                severity=SEVERITY_ERROR,
                title="Filename subject label does not match its subject directory",
                detail=(
                    f"directory says : sub-{dir_subject}\n"
                    f"filename says  : sub-{file_subject}\n"
                    "Output directories are named from the DIRECTORY and output\n"
                    "files from the FILENAME, so this would publish\n"
                    f"  sub-{dir_subject}/**/sub-{file_subject}_*.nii.gz\n"
                    "- a subject directory full of another subject's files."
                ),
                fix=(
                    f"Rename the {len(paths)} file(s) to start with 'sub-{dir_subject}_', "
                    f"or rename the directory to 'sub-{file_subject}/'."
                ),
                paths=tuple(sorted(paths)),
            )
        )

    if missing_subject:
        findings.append(
            BidsFinding(
                code="BIDS102",
                severity=SEVERITY_ERROR,
                title="Input filename has no 'sub-' entity",
                detail=(
                    "Output basenames are built from the filename's entities. With no "
                    "'sub-' entity every subject's outputs would be named "
                    "'sub-unknown_*', and under anat.synthesis_level: subject they "
                    "would all be written to the same 'sub-unknown/anat/' path."
                ),
                fix=(
                    "Rename each file so it starts with the label of the subject "
                    "directory it sits in, e.g. 'sub-01_T1w.nii.gz'."
                ),
                paths=tuple(sorted(missing_subject)),
            )
        )

    for (dir_session, file_session), paths in mismatched_session.items():
        findings.append(
            BidsFinding(
                code="BIDS103",
                severity=SEVERITY_ERROR,
                title="Filename session label does not match its session directory",
                detail=(
                    f"directory says : ses-{dir_session}\n"
                    f"filename says  : ses-{file_session}\n"
                    "As with the subject label, output directories follow the "
                    "directory and output filenames follow the filename, so the two "
                    "would disagree in the published tree."
                ),
                fix=(
                    f"Rename the {len(paths)} file(s) to use 'ses-{dir_session}_', "
                    f"or move them under 'ses-{file_session}/'."
                ),
                paths=tuple(sorted(paths)),
            )
        )

    if missing_session_multi:
        findings.append(
            BidsFinding(
                code="BIDS104",
                severity=SEVERITY_ERROR,
                title="Filename has no 'ses-' entity but the subject has several sessions",
                detail=(
                    "The published basename would carry no session, so files from "
                    "different sessions of the same subject collapse onto identical "
                    "names. Under the default anat.synthesis_level: subject they also "
                    "share one output path, and one session silently overwrites the "
                    "other."
                ),
                fix=(
                    "Add the session to each filename, matching its directory, "
                    "e.g. 'sub-01_ses-001_T1w.nii.gz'."
                ),
                paths=tuple(sorted(missing_session_multi)),
            )
        )

    if datatype_beside_sessions:
        findings.append(
            BidsFinding(
                code="BIDS105",
                severity=SEVERITY_ERROR,
                title="Data directory sits beside session directories, not inside one",
                detail=(
                    "BIDS allows a subject directory to contain EITHER session\n"
                    "directories OR datatype directories, never both:\n"
                    "  sub-01/ses-001/anat/   <- sessions in use\n"
                    "  sub-01/anat/           <- only valid with no ses-*/ at all\n"
                    "This subject has both. Discovery reads one session at a time, so\n"
                    "these files belong to no session it ever queries and are dropped\n"
                    "without being processed - the run would appear to succeed while\n"
                    "silently ignoring this input."
                ),
                fix=(
                    "Move each file into the 'ses-<label>/' directory it belongs to "
                    "(and add a matching 'ses-' entity to the filename), or, if the "
                    "subject really has one session, remove the 'ses-*/' level "
                    "entirely."
                ),
                paths=tuple(sorted(datatype_beside_sessions)),
            )
        )

    if missing_session_single:
        findings.append(
            BidsFinding(
                code="BIDS204",
                severity=SEVERITY_WARNING,
                title="Filename has no 'ses-' entity although it sits under a session directory",
                detail=(
                    "Only one session of this subject is being processed, so nothing "
                    "collides and processing is unaffected. The published filenames "
                    "will simply not record which session they came from."
                ),
                fix=(
                    "Optional: add the session to each filename, "
                    "e.g. 'sub-01_ses-001_T1w.nii.gz'."
                ),
                paths=tuple(sorted(missing_session_single)),
            )
        )

    if session_without_dir:
        findings.append(
            BidsFinding(
                code="BIDS201",
                severity=SEVERITY_WARNING,
                title="Filename has a 'ses-' entity but there is no 'ses-*/' directory level",
                detail=(
                    "BIDS expects sub-<label>/ses-<label>/<datatype>/ once sessions "
                    "are used. Brainana reads the session from the filename here, so "
                    "the outputs stay consistent, but the published tree gains a "
                    "'ses-*/' level the input does not have."
                ),
                fix=(
                    "Move each file into a matching 'ses-<label>/' directory under "
                    "its subject, or drop the 'ses-' entity from the filename."
                ),
                paths=tuple(sorted(session_without_dir)),
            )
        )

    for reason, paths in not_discoverable.items():
        findings.append(
            BidsFinding(
                code="BIDS202",
                severity=SEVERITY_WARNING,
                title="File will not be processed",
                detail=f"These NIfTI files are skipped because {reason}.",
                fix=(
                    "Ignore this if the files are extra data kept alongside the "
                    "dataset. Otherwise move them into an anat/ or func/ directory "
                    "and give them a suffix Brainana reads."
                ),
                paths=tuple(sorted(paths)),
            )
        )

    if not (bids_dir / "dataset_description.json").is_file():
        # Deliberately a WARNING, permanently. Brainana never reads the input's
        # dataset_description.json (it only writes one at the derivatives root), and
        # the great majority of NHP datasets in the wild do not ship one. Promoting
        # this to an error would block datasets that process perfectly today in
        # exchange for nothing.
        findings.append(
            BidsFinding(
                code="BIDS203",
                severity=SEVERITY_WARNING,
                title="No dataset_description.json at the dataset root",
                detail=(
                    "BIDS requires this file. Brainana does not read it, so the run "
                    "is unaffected, but other BIDS tools will reject the dataset."
                ),
                fix=(
                    'Create it with: {"Name": "<dataset name>", '
                    '"BIDSVersion": "1.8.0", "DatasetType": "raw"}'
                ),
            )
        )

    severity_rank = {SEVERITY_ERROR: 0, SEVERITY_WARNING: 1}
    # Sorted so the same dataset always yields the same report: glob/walk order is
    # filesystem-dependent, and an abort message that reorders between runs is very
    # hard to support.
    findings.sort(
        key=lambda f: (severity_rank[f.severity], f.code, f.paths[0] if f.paths else "")
    )
    return findings


def _format_finding(finding: BidsFinding, max_paths: int = 10) -> str:
    """Render one finding as an indented console block."""
    lines = [f"  [{finding.code}] {finding.title}"]
    lines += [f"    {line}" for line in finding.detail.splitlines()]
    lines.append(f"    Fix: {finding.fix}")
    if finding.paths:
        lines.append(f"    Affected files ({len(finding.paths)}):")
        for path in finding.paths[:max_paths]:
            lines.append(f"      {path}")
        if len(finding.paths) > max_paths:
            lines.append(f"      ... {len(finding.paths) - max_paths} more")
    return "\n".join(lines)


def _print_findings(findings: List[BidsFinding], bids_dir: Path) -> None:
    """Print the validation report.

    The whole report goes to a single stream - stderr when it aborts the run,
    stdout otherwise - so the sections cannot interleave out of order when the
    two streams are buffered differently (which they are as soon as the wrapper's
    output is piped or captured into a log).
    """
    errors = [f for f in findings if f.severity == SEVERITY_ERROR]
    warnings = [f for f in findings if f.severity == SEVERITY_WARNING]

    lines = ["", "=" * 60, "BIDS input validation", "=" * 60, f"Dataset: {bids_dir}"]

    if errors:
        lines.append(
            f"\nERRORS ({len(errors)}) - these would produce mislabelled output, "
            "or silently drop input."
        )
        for finding in errors:
            lines += ["", _format_finding(finding)]

    if warnings:
        lines.append(f"\nWARNINGS ({len(warnings)}) - the run continues; review these.")
        for finding in warnings:
            lines += ["", _format_finding(finding)]

    if errors:
        lines += [
            f"\nAborting: {_pluralize(len(errors), 'error')} in the BIDS input.",
            "",
        ]
    else:
        lines += ["\nINFO: BIDS structure check passed", ""]

    print("\n".join(lines), file=sys.stderr if errors else sys.stdout)


def validate_bids(
    bids_dir: Path,
    subjects: Optional[List[str]] = None,
    sessions: Optional[List[str]] = None,
) -> bool:
    """
    Dependency-free structural validation of the input BIDS dataset.

    This does not run the full bids-validator (the pipeline intentionally builds
    its BIDSLayout with ``validate=False`` to tolerate benign, especially
    macaque-specific, spec deviations). It checks the three things that actually
    break Brainana:

    1. The directory contains at least one ``sub-*/`` subject directory. This
       turns the common "pointed at the wrong directory" mistake into a clear,
       early error instead of a later, vaguer "No jobs discovered".
    2. Every discoverable file's ``sub-``/``ses-`` entities agree with the
       directories it sits in. A disagreement is silently resolved by pybids in
       favour of the directory, after which output directories and output
       filenames disagree - see :func:`_scan_bids_layout`.
    3. No subject mixes ``ses-*/`` directories with datatype directories at the
       subject level, a layout BIDS forbids and discovery silently skips.

    Args:
        bids_dir: Path to BIDS dataset
        subjects: Optional subject filter, matching the run's effective
            ``--subjects`` / ``bids_filtering.subjects``
        sessions: Optional session filter, matching the run's effective
            ``--sessions`` / ``bids_filtering.sessions``

    Returns:
        True if the dataset can be processed, False if any error was found
    """
    has_subject_dir = any(
        p.is_dir() and p.name.startswith("sub-") for p in bids_dir.iterdir()
    )
    if not has_subject_dir:
        print(
            f"ERROR: {bids_dir} does not look like a BIDS dataset: "
            f"no 'sub-*/' subject directories found.",
            file=sys.stderr,
        )
        return False

    findings = _scan_bids_layout(bids_dir, subjects, sessions)
    _print_findings(findings, bids_dir)
    return not any(f.severity == SEVERITY_ERROR for f in findings)


def _count_job_bids_inputs(job: Dict[str, Any]) -> int:
    """Count original BIDS NIfTI inputs represented by one discovery job."""
    fps = job.get("file_paths")
    if isinstance(fps, list) and fps:
        return len(fps)
    if job.get("file_path"):
        return 1
    return 0


def _sum_job_bids_inputs(jobs: List[Dict[str, Any]]) -> int:
    """Sum BIDS NIfTI inputs across a list of discovery jobs."""
    return sum(_count_job_bids_inputs(j) for j in jobs)


def _pluralize(n: int, singular: str, plural: str | None = None) -> str:
    """Return ``f'{n} {singular|plural}'`` with basic English pluralization."""
    if n == 1:
        return f"{n} {singular}"
    return f"{n} {plural if plural is not None else singular + 's'}"


def _print_modality_summary(
    label: str,
    modality_jobs: List[Dict[str, Any]],
    synthesis_type: str,
) -> None:
    """Print one anatomical modality line plus optional synthesis breakdown."""
    n_jobs = len(modality_jobs)
    n_inputs = _sum_job_bids_inputs(modality_jobs)
    print(
        f"  {label}: {_pluralize(n_inputs, 'BIDS file')} → {_pluralize(n_jobs, 'job')}"
    )

    synthesis_jobs = [
        j
        for j in modality_jobs
        if j.get("needs_synthesis", False) and j.get("synthesis_type") == synthesis_type
    ]
    if not synthesis_jobs:
        return

    cross_session = [
        j for j in synthesis_jobs if j.get("synthesis_scope") == "cross_session"
    ]
    within_session = [
        j for j in synthesis_jobs if j.get("synthesis_scope") == "within_session"
    ]
    if cross_session:
        print(
            f"    - Cross-session synthesis: "
            f"{_pluralize(len(cross_session), 'job')} "
            f"({_sum_job_bids_inputs(cross_session)} inputs)"
        )
    if within_session:
        print(
            f"    - Within-session synthesis: "
            f"{_pluralize(len(within_session), 'job')} "
            f"({_sum_job_bids_inputs(within_session)} inputs)"
        )
    # Complement of `synthesis_jobs` within this modality, rather than a count of
    # jobs without `needs_synthesis`: a job flagged for synthesis under a
    # *different* synthesis_type belongs to neither line above nor here, and
    # counting it as single would stop the sub-counts adding up to the job total.
    n_single = n_jobs - len(synthesis_jobs)
    if n_single > 0:
        print(f"    - Single (no synthesis): {_pluralize(n_single, 'job')}")


def print_summary(
    anat_jobs: List[Dict[str, Any]],
    func_jobs: List[Dict[str, Any]],
    anat_only: bool = False,
) -> None:
    """
    Print a summary of discovered jobs.

    Distinguishes original BIDS NIfTI inputs from post-synthesis processing
    jobs so multi-run anatomical synthesis is not misreported as a single file.

    Args:
        anat_jobs: List of anatomical job dictionaries
        func_jobs: List of functional job dictionaries
        anat_only: True when ``general.anat_only`` is set. Functional discovery
            is skipped entirely in that case, so zero counts would wrongly read
            as "this dataset has no BOLD data"; an explicit notice is printed
            instead.
    """
    print("\n" + "=" * 60)
    print("BIDS Discovery Summary")
    print("=" * 60)

    # Count subjects
    anat_subjects = sorted(set(j.get("subject_id") for j in anat_jobs))
    func_subjects = sorted(set(j.get("subject_id") for j in func_jobs))
    all_subjects = sorted(set(anat_subjects + func_subjects))

    print(f"\nSubjects: {len(all_subjects)}")
    if len(all_subjects) <= 10:
        print(f"  {', '.join(all_subjects)}")
    else:
        print(f"  {', '.join(all_subjects[:10])} ... ({len(all_subjects) - 10} more)")

    # Anatomical summary: BIDS inputs vs processing jobs after synthesis grouping
    anat_inputs = _sum_job_bids_inputs(anat_jobs)
    print("\nAnatomical data:")
    print(f"  BIDS inputs → processing jobs: {anat_inputs} → {len(anat_jobs)}")

    t1w_jobs = [j for j in anat_jobs if j.get("suffix") == "T1w"]
    t2w_jobs = [j for j in anat_jobs if j.get("suffix") == "T2w"]
    _print_modality_summary("T1w", t1w_jobs, synthesis_type="t1w")
    _print_modality_summary("T2w", t2w_jobs, synthesis_type="t2w")

    # Functional summary (BOLD is currently 1:1 with jobs)
    print("\nFunctional data:")
    if anat_only:
        print(
            "  Skipped: anat_only = true — functional data is not "
            "discovered or processed in this run."
        )
        print("\n")
        return

    bold_inputs = _sum_job_bids_inputs(func_jobs)
    print(f"  Processing jobs: {len(func_jobs)}")
    print(f"  BOLD: {_pluralize(bold_inputs, 'BIDS file')}")
    if func_jobs:
        func_tasks = sorted(set(j.get("task") for j in func_jobs if j.get("task")))
        if func_tasks:
            print(f"  Tasks: {', '.join(func_tasks)}")

    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Discover BIDS dataset for Nextflow pipeline"
    )
    parser.add_argument(
        "--bids_dir", type=Path, required=True, help="Path to BIDS dataset directory"
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--config_file",
        type=Path,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated list of subject IDs to filter",
    )
    parser.add_argument(
        "--sessions",
        type=str,
        default=None,
        help="Comma-separated list of session IDs to filter",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of task names to filter",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="Comma-separated list of run numbers to filter",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.bids_dir.exists():
        print(f"ERROR: BIDS directory not found: {args.bids_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.config_file.exists():
        print(f"ERROR: Config file not found: {args.config_file}", file=sys.stderr)
        sys.exit(1)

    # Load config (accepts tabs in indentation via normalization)
    try:
        config = load_yaml_config(args.config_file)
    except Exception as e:
        print(f"ERROR: Failed to load config file: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate the config up front so bad values (e.g. dof: 7, an unknown
    # skullstripping method, an out-of-range weight) fail fast here with a
    # clear message, rather than surfacing as an opaque error deep in a
    # process after minutes of compute. validate_config() merges the user
    # config over the defaults, so it checks the effective settings.
    try:
        validate_config(config)
    except (ValueError, TypeError) as e:
        print(f"ERROR: Invalid configuration in {args.config_file}:", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    # Parse filtering parameters
    subjects_list = None
    if args.subjects:
        subjects_list = [s.strip() for s in args.subjects.split(",")]

    sessions_list = None
    if args.sessions:
        sessions_list = [s.strip() for s in args.sessions.split(",")]

    tasks_list = None
    if args.tasks:
        tasks_list = [t.strip() for t in args.tasks.split(",")]

    runs_list = None
    if args.runs:
        runs_list = [r.strip() for r in args.runs.split(",")]

    # Validate BIDS dataset structure (always runs; there is no way to skip it).
    #
    # Scoped to the subjects/sessions this run will actually process, so one
    # malformed subject cannot block the rest of a dataset the user did not ask
    # for. That scope has to be the *effective* filter, not just the CLI flags:
    # discover_bids_dataset() falls back to config bids_filtering when a flag is
    # absent, and run_brainana.sh only forwards a flag the user typed. Resolving
    # the same fallback here keeps validation and discovery looking at one set of
    # files - otherwise a config-only filter validates the whole dataset and
    # aborts on a subject that was never going to be processed.
    bids_filtering = config.get("bids_filtering", {}) or {}
    validate_subjects = subjects_list or _normalize_to_list(
        bids_filtering.get("subjects")
    )
    validate_sessions = sessions_list or _normalize_to_list(
        bids_filtering.get("sessions")
    )
    if not validate_bids(args.bids_dir, validate_subjects, validate_sessions):
        sys.exit(1)

    # Create output directory
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Verify directory was created and is accessible
        if not args.output_dir.exists():
            print(
                f"ERROR: Failed to create output directory: {args.output_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not args.output_dir.is_dir():
            print(
                f"ERROR: Output path exists but is not a directory: {args.output_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        # Resolve to absolute path for clarity
        args.output_dir = args.output_dir.resolve()
    except (OSError, PermissionError) as e:
        print(
            f"ERROR: Failed to create output directory {args.output_dir}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create nextflow_reports subdirectory
    try:
        (args.output_dir / "nextflow_reports").mkdir(exist_ok=True)
        if not (args.output_dir / "nextflow_reports").exists():
            print("ERROR: Failed to create nextflow_reports directory", file=sys.stderr)
            sys.exit(1)
    except (OSError, PermissionError) as e:
        print(
            f"ERROR: Failed to create nextflow_reports directory: {e}", file=sys.stderr
        )
        sys.exit(1)

    # Discover jobs
    try:
        anat_jobs, func_jobs = discover_bids_dataset(
            bids_dir=args.bids_dir,
            config=config,
            subjects=subjects_list,
            sessions=sessions_list,
            tasks=tasks_list,
            runs=runs_list,
        )
    except Exception as e:
        print(f"ERROR: BIDS discovery failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Print summary. anat_only comes from the same config key discovery itself
    # uses to skip functional discovery, so the summary matches what ran.
    anat_only = bool(get_nested_config_value(config, "general.anat_only", False))
    print_summary(anat_jobs, func_jobs, anat_only=anat_only)

    # Save JSON files
    anat_json_path = args.output_dir / "nextflow_reports" / "anatomical_jobs.json"
    func_json_path = args.output_dir / "nextflow_reports" / "functional_jobs.json"

    with open(anat_json_path, "w") as f:
        json.dump(anat_jobs, f, indent=2)

    with open(func_json_path, "w") as f:
        json.dump(func_jobs, f, indent=2)

    # Verify files were written successfully
    if not anat_json_path.exists() or not func_json_path.exists():
        print("ERROR: Failed to write job list files", file=sys.stderr)
        sys.exit(1)

    print("INFO: Discovery complete. Saved job lists to:")
    print(f"  - {anat_json_path}")
    print(f"  - {func_json_path}")
    print(f"INFO: Output directory: {args.output_dir}")

    # Exit with error if no jobs found
    if not anat_jobs and not func_jobs:
        print("ERROR: No jobs discovered. Check that:", file=sys.stderr)
        print("  (1) The path is the BIDS dataset root.", file=sys.stderr)
        print(
            "  (2) It contains at least one subject with anat and/or func data in BIDS layout.",
            file=sys.stderr,
        )
        print(
            "  (3) Validate with https://bids-standard.github.io/bids-validator/ if unsure.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
