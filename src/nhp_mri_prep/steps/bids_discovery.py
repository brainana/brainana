"""
BIDS dataset discovery for Nextflow integration.

This module provides functions to discover BIDS datasets and return
job lists that can be used to create Nextflow channels.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from bids import BIDSLayout


logger = logging.getLogger(__name__)


def _normalize_to_list(value):
    """
    Normalize a value to a list of strings.

    Handles None, single values (int, str), lists, and comma/space-separated
    strings. Splits strings on commas and/or whitespace so config can use
    e.g. "sub-001, sub-006" or "sub-001 sub-006".

    Args:
        value: Value to normalize (can be None, int, str, or list)

    Returns:
        List of strings, or None if input was None
    """
    if value is None:
        return None
    if isinstance(value, list):
        # Convert all items in list to strings
        return [str(item) for item in value]
    s = str(value).strip()
    if not s:
        return None
    # Split on comma and/or whitespace; drop empty parts
    parts = [p.strip() for p in re.split(r"[,\s]+", s) if p.strip()]
    return parts if parts else None


def _normalize_bids_id(value, prefix):
    """
    Normalize a BIDS ID by removing prefix if present.

    Handles both 'prefix-032102' and '032102' formats.
    Preserves None values. Value should already be a string.

    Args:
        value: ID value (should be str, but handles int/None for safety)
        prefix: Prefix to remove (e.g., 'sub-', 'ses-', 'task-', 'run-')

    Returns:
        Normalized ID as string (without prefix), or None if input was None
    """
    if value is None:
        return None
    # Convert to string (should already be string from _normalize_to_list)
    value_str = str(value)
    if value_str.startswith(prefix):
        return value_str[len(prefix) :]
    return value_str


def _is_top_level_subject_path(
    bids_dir: Path,
    file_path: Path,
    subject_id: str,
) -> bool:
    """
    Return True iff the file lives under bids_dir/sub-{subject_id}/ (top-level subject only).

    Used to exclude nested subject dirs (e.g. bids_dir/badQC/sub-aaa). Paths are resolved
    so symlinks and relative components do not affect the check.
    """
    try:
        root = bids_dir.resolve()
        path = Path(file_path).resolve()
        rel = path.relative_to(root)
    except (ValueError, OSError):
        return False
    if not rel.parts:
        return False
    return rel.parts[0] == f"sub-{subject_id}"


def discover_bids_dataset(
    bids_dir: Path,
    config: Dict[str, Any],
    subjects: Optional[List[str]] = None,
    sessions: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    runs: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Discover BIDS dataset and return job lists for Nextflow.

    This function discovers all anatomical and functional files in a BIDS dataset
    and returns them as lists of job dictionaries that can be used to create
    Nextflow channels.

    Args:
        bids_dir: Path to BIDS dataset root directory
        config: Configuration dictionary (used for filtering)
        subjects: Optional list of subject IDs to include (None = all)
        sessions: Optional list of session IDs to include (None = all)
        tasks: Optional list of task names to include (None = all)
        runs: Optional list of run numbers to include (None = all)

    Returns:
        Tuple of (anatomical_jobs, functional_jobs)
        Each job is a dictionary with keys:
        - subject_id: Subject ID (without 'sub-' prefix)
        - session_id: Session ID (without 'ses-' prefix, or None)
        - file_path: Path to input file
        - modality: 'anat' or 'func'
        - suffix: BIDS suffix (e.g., 'T1w', 'bold')
        - task: Task name (for func, None for anat)
        - run: Run number (for func, None for anat)
        - entities: Full BIDS entities dictionary
        - needs_synthesis: Whether multiple T1w runs need synthesis (anat only)
    """
    bids_dir = Path(bids_dir)
    resolved_bids_dir = bids_dir.resolve()

    if not bids_dir.exists():
        raise FileNotFoundError(f"BIDS directory not found: {bids_dir}")

    # Initialize BIDS layout
    try:
        layout = BIDSLayout(
            str(bids_dir),
            validate=False,  # Skip validation for speed
            derivatives=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize BIDS layout: {e}")

    # Apply filtering from config if not overridden
    bids_filtering = config.get("bids_filtering", {})
    if subjects is None:
        subjects = bids_filtering.get("subjects")
    if sessions is None:
        sessions = bids_filtering.get("sessions")
    if tasks is None:
        tasks = bids_filtering.get("tasks")
    if runs is None:
        runs = bids_filtering.get("runs")

    # Normalize to lists and remove prefixes
    subjects = _normalize_to_list(subjects)
    sessions = _normalize_to_list(sessions)
    tasks = _normalize_to_list(tasks)
    runs = _normalize_to_list(runs)

    # Normalize IDs (remove prefixes if present)
    if subjects is not None:
        subjects = [_normalize_bids_id(s, "sub-") for s in subjects]
    if sessions is not None:
        sessions = [_normalize_bids_id(s, "ses-") for s in sessions]
    if tasks is not None:
        tasks = [_normalize_bids_id(t, "task-") for t in tasks]
    if runs is not None:
        runs = [_normalize_bids_id(r, "run-") for r in runs]

    # Get all subjects; keep only those with data under top-level sub-* (exclude e.g. badQC/sub-xxx)
    all_subjects = layout.get_subjects()
    top_level_subjects = [
        sub
        for sub in all_subjects
        if any(
            _is_top_level_subject_path(resolved_bids_dir, Path(f.path), sub)
            for f in layout.get(subject=sub, extension=[".nii.gz", ".nii"])
        )
    ]
    if subjects is None:
        target_subjects = top_level_subjects
    else:
        target_subjects = [s for s in subjects if s in top_level_subjects]
        if len(target_subjects) != len(subjects):
            missing = set(subjects) - set(target_subjects)
            logger.warning(
                f"Some requested subjects not found: {missing}. "
                f"If your config has unquoted numbers (e.g., '032102'), YAML may parse them incorrectly. "
                f'Please quote the values in your YAML config (e.g., subjects: "032102" or subjects: ["032102"]).'
            )

    anatomical_jobs = []
    functional_jobs = []

    # Check if we should skip functional discovery based on config
    general_config = config.get("general", {})
    anat_only = general_config.get("anat_only", False)

    # Get synthesis level from config
    anat_config = config.get("anat", {})
    synthesis_level = anat_config.get(
        "synthesis_level", "session"
    )  # Default to "session"

    # "session_longitudinal" discovers exactly like "session": one anatomical
    # per session. Its extra work -- the within-subject base template and the
    # base-seeded longitudinal reconstructions -- happens later, inside
    # SURF_RECON_WF, and is built from the per-session FreeSurfer trees rather
    # than from raw BIDS images. There is deliberately no third kind of
    # synthesis job here: emitting one would push the base through the
    # ANAT_SYNTHESIS -> CONFORM -> BIAS -> SKULLSTRIP chain, where it does not
    # belong, and a subject-level anatomical row alongside per-session rows
    # would then be picked up as the highest-priority functional reference and
    # silently redirect every functional session to the base.
    cross_session_synthesis = synthesis_level == "subject"

    # Discover anatomical files (always discover, regardless of anat_only setting)
    for sub in target_subjects:
        # Get sessions for this subject
        sub_sessions = layout.get_sessions(subject=sub)
        if not sub_sessions:
            sub_sessions = [None]  # No sessions in dataset

        # Filter sessions if specified
        if sessions is not None:
            sub_sessions = [s for s in sub_sessions if s in sessions]

        # For "subject" mode: collect all anatomical files across sessions first
        if cross_session_synthesis:
            # Collect all anatomical files across all sessions for this subject
            all_t1w_files = []  # List of (session, file) tuples
            all_t2w_files = []  # List of (session, file) tuples

            for ses in sub_sessions:
                anat_filters = {
                    "subject": sub,
                    "datatype": "anat",
                    "suffix": ["T1w", "T2w"],
                    "extension": [".nii.gz", ".nii"],
                }
                if ses:
                    anat_filters["session"] = ses

                anat_files = [
                    f
                    for f in layout.get(**anat_filters)
                    if _is_top_level_subject_path(resolved_bids_dir, Path(f.path), sub)
                ]
                if not anat_files:
                    continue

                t1w_files = [
                    f for f in anat_files if f.get_entities().get("suffix") == "T1w"
                ]
                t2w_files = [
                    f for f in anat_files if f.get_entities().get("suffix") == "T2w"
                ]

                for f in t1w_files:
                    all_t1w_files.append((ses, f))
                for f in t2w_files:
                    all_t2w_files.append((ses, f))

            # Group by modality and check if cross-session synthesis is needed
            # T1w processing
            if all_t1w_files:
                # Count sessions with T1w files
                t1w_sessions = set([ses for ses, f in all_t1w_files])

                if len(t1w_sessions) > 1:
                    # Cross-session synthesis: create subject-level synthesis job
                    t1w_file_list = [f for ses, f in all_t1w_files]
                    synthesis_job = {
                        "subject_id": sub,
                        "session_id": "",  # Empty string for subject-level
                        "file_paths": [f.path for f in t1w_file_list],
                        "modality": "anat",
                        "suffix": "T1w",
                        "task": None,
                        "run": None,
                        "entities": t1w_file_list[0].get_entities(),
                        "needs_synthesis": True,
                        "synthesis_type": "t1w",
                        "synthesis_scope": "cross_session",
                    }
                    anatomical_jobs.append(synthesis_job)
                else:
                    # Only one session has T1w: process within-session
                    ses = list(t1w_sessions)[0]
                    t1w_file_list = [f for s, f in all_t1w_files if s == ses]

                    # Check if multiple runs exist (needs within-session synthesis)
                    t1w_runs = [
                        f.get_entities().get("run")
                        for f in t1w_file_list
                        if f.get_entities().get("run")
                    ]
                    t1w_needs_synthesis = len(set(t1w_runs)) > 1 if t1w_runs else False

                    if t1w_needs_synthesis:
                        synthesis_job = {
                            "subject_id": sub,
                            "session_id": ses,
                            "file_paths": [f.path for f in t1w_file_list],
                            "modality": "anat",
                            "suffix": "T1w",
                            "task": None,
                            "run": None,
                            "entities": t1w_file_list[0].get_entities(),
                            "needs_synthesis": True,
                            "synthesis_type": "t1w",
                            "synthesis_scope": "within_session",
                        }
                        anatomical_jobs.append(synthesis_job)
                    else:
                        # Single T1w file - create regular job
                        for t1w_file in t1w_file_list:
                            entities = t1w_file.get_entities()
                            job = {
                                "subject_id": sub,
                                "session_id": ses,
                                "file_path": t1w_file.path,
                                "modality": "anat",
                                "suffix": "T1w",
                                "task": None,
                                "run": entities.get("run"),
                                "entities": entities,
                                "needs_synthesis": False,
                            }
                            anatomical_jobs.append(job)

            # T2w processing
            if all_t2w_files:
                # Count sessions with T2w files
                t2w_sessions = set([ses for ses, f in all_t2w_files])

                # Check if T1w exists (for T2w→T1w registration)
                # Check if subject-level T1w exists
                has_t1w_subject_level = any(
                    job.get("synthesis_scope") == "cross_session"
                    and job.get("suffix") == "T1w"
                    and job.get("subject_id") == sub
                    for job in anatomical_jobs
                )

                # Also check if any session has T1w (for within-session registration)
                t1w_sessions_for_t2w = (
                    set([ses for ses, f in all_t1w_files]) if all_t1w_files else set()
                )

                if len(t2w_sessions) > 1:
                    # Cross-session synthesis: create subject-level synthesis job
                    t2w_file_list = [f for ses, f in all_t2w_files]
                    t2w_synthesis_job = {
                        "subject_id": sub,
                        "session_id": "",  # Empty string for subject-level
                        "file_paths": [f.path for f in t2w_file_list],
                        "modality": "anat",
                        "suffix": "T2w",
                        "task": None,
                        "run": None,
                        "entities": t2w_file_list[0].get_entities(),
                        "needs_synthesis": True,
                        "synthesis_type": "t2w",
                        "synthesis_scope": "cross_session",
                        "needs_t1w_registration": has_t1w_subject_level
                        or len(t1w_sessions_for_t2w) > 0,
                    }
                    anatomical_jobs.append(t2w_synthesis_job)
                else:
                    # Only one session has T2w: process within-session
                    ses = list(t2w_sessions)[0]
                    t2w_file_list = [f for s, f in all_t2w_files if s == ses]
                    has_t1w_in_session = ses in t1w_sessions_for_t2w

                    # Check if multiple runs exist (needs within-session synthesis)
                    t2w_runs = [
                        f.get_entities().get("run")
                        for f in t2w_file_list
                        if f.get_entities().get("run")
                    ]
                    t2w_needs_synthesis = len(set(t2w_runs)) > 1 if t2w_runs else False

                    if t2w_needs_synthesis:
                        t2w_synthesis_job = {
                            "subject_id": sub,
                            "session_id": ses,
                            "file_paths": [f.path for f in t2w_file_list],
                            "modality": "anat",
                            "suffix": "T2w",
                            "task": None,
                            "run": None,
                            "entities": t2w_file_list[0].get_entities(),
                            "needs_synthesis": True,
                            "synthesis_type": "t2w",
                            "synthesis_scope": "within_session",
                            "needs_t1w_registration": has_t1w_in_session
                            or has_t1w_subject_level,
                        }
                        anatomical_jobs.append(t2w_synthesis_job)
                    else:
                        # Single T2w file - create regular job
                        for t2w_file in t2w_file_list:
                            entities = t2w_file.get_entities()
                            job = {
                                "subject_id": sub,
                                "session_id": ses,
                                "file_path": t2w_file.path,
                                "modality": "anat",
                                "suffix": "T2w",
                                "task": None,
                                "run": entities.get("run"),
                                "entities": entities,
                                "needs_synthesis": False,
                                "needs_t1w_registration": has_t1w_in_session
                                or has_t1w_subject_level,
                            }
                            anatomical_jobs.append(job)

        else:
            # "session" mode: process each session individually (current behavior)
            for ses in sub_sessions:
                # Find anatomical files
                anat_filters = {
                    "subject": sub,
                    "datatype": "anat",
                    "suffix": ["T1w", "T2w"],
                    "extension": [".nii.gz", ".nii"],
                }
                if ses:
                    anat_filters["session"] = ses

                anat_files = [
                    f
                    for f in layout.get(**anat_filters)
                    if _is_top_level_subject_path(resolved_bids_dir, Path(f.path), sub)
                ]

                if not anat_files:
                    continue

                # Group T1w and T2w files by run to detect multi-run cases
                t1w_files = [
                    f for f in anat_files if f.get_entities().get("suffix") == "T1w"
                ]
                t2w_files = [
                    f for f in anat_files if f.get_entities().get("suffix") == "T2w"
                ]

                # Check if multiple T1w runs exist (needs synthesis)
                t1w_runs = [
                    f.get_entities().get("run")
                    for f in t1w_files
                    if f.get_entities().get("run")
                ]
                t1w_needs_synthesis = len(set(t1w_runs)) > 1 if t1w_runs else False

                # Check if multiple T2w runs exist (needs synthesis)
                t2w_runs = [
                    f.get_entities().get("run")
                    for f in t2w_files
                    if f.get_entities().get("run")
                ]
                t2w_needs_synthesis = len(set(t2w_runs)) > 1 if t2w_runs else False

                # Create job for each anatomical file
                # For T1w: if multiple runs, create synthesis job first
                if t1w_files:
                    if t1w_needs_synthesis:
                        # Create synthesis job
                        synthesis_job = {
                            "subject_id": sub,
                            "session_id": ses,
                            "file_paths": [f.path for f in t1w_files],
                            "modality": "anat",
                            "suffix": "T1w",
                            "task": None,
                            "run": None,
                            "entities": t1w_files[0].get_entities(),
                            "needs_synthesis": True,
                            "synthesis_type": "t1w",
                        }
                        anatomical_jobs.append(synthesis_job)
                    else:
                        # Single T1w file - create regular job
                        for t1w_file in t1w_files:
                            entities = t1w_file.get_entities()
                            job = {
                                "subject_id": sub,
                                "session_id": ses,
                                "file_path": t1w_file.path,
                                "modality": "anat",
                                "suffix": "T1w",
                                "task": None,
                                "run": entities.get("run"),
                                "entities": entities,
                                "needs_synthesis": False,
                            }
                            anatomical_jobs.append(job)

                # T2w files: if multiple runs, create synthesis job; otherwise individual jobs
                # Check if T2w should be registered to T1w (if T1w exists in same session)
                has_t1w_in_session = len(t1w_files) > 0
                if t2w_files:
                    if t2w_needs_synthesis:
                        # Create synthesis job for T2w
                        t2w_synthesis_job = {
                            "subject_id": sub,
                            "session_id": ses,
                            "file_paths": [f.path for f in t2w_files],
                            "modality": "anat",
                            "suffix": "T2w",
                            "task": None,
                            "run": None,
                            "entities": t2w_files[0].get_entities(),
                            "needs_synthesis": True,
                            "synthesis_type": "t2w",
                            "needs_t1w_registration": has_t1w_in_session,
                        }
                        anatomical_jobs.append(t2w_synthesis_job)
                    else:
                        # Single T2w file - create regular job
                        for t2w_file in t2w_files:
                            entities = t2w_file.get_entities()
                            job = {
                                "subject_id": sub,
                                "session_id": ses,
                                "file_path": t2w_file.path,
                                "modality": "anat",
                                "suffix": "T2w",
                                "task": None,
                                "run": entities.get("run"),
                                "entities": entities,
                                "needs_synthesis": False,
                                "needs_t1w_registration": has_t1w_in_session,  # Flag for special T2w processing
                            }
                            anatomical_jobs.append(job)

    # Discover functional files (skip if anat_only is True)
    if not anat_only:
        for sub in target_subjects:
            sub_sessions = layout.get_sessions(subject=sub)
            if not sub_sessions:
                sub_sessions = [None]

            if sessions is not None:
                sub_sessions = [s for s in sub_sessions if s in sessions]

            for ses in sub_sessions:
                # Find functional files
                func_filters = {
                    "subject": sub,
                    "datatype": "func",
                    "suffix": "bold",
                    "extension": [".nii.gz", ".nii"],
                }
                if ses:
                    func_filters["session"] = ses
                if tasks:
                    func_filters["task"] = tasks
                if runs:
                    func_filters["run"] = runs

                func_files = [
                    f
                    for f in layout.get(**func_filters)
                    if _is_top_level_subject_path(resolved_bids_dir, Path(f.path), sub)
                ]

                # Create job for each functional file
                for func_file in func_files:
                    entities = func_file.get_entities()
                    job = {
                        "subject_id": sub,
                        "session_id": ses,
                        "file_path": func_file.path,
                        "modality": "func",
                        "suffix": "bold",
                        "task": entities.get("task"),
                        "run": entities.get("run"),
                        "entities": entities,
                        "needs_synthesis": False,
                    }
                    functional_jobs.append(job)
    else:
        logger.info("Skipping functional discovery (anat_only = True)")

    logger.info(
        f"Discovered {len(anatomical_jobs)} anatomical jobs and {len(functional_jobs)} functional jobs"
    )

    return anatomical_jobs, functional_jobs
