"""
BIDS (Brain Imaging Data Structure) utilities for nhp_mri_prep.

This module provides functions for parsing and working with BIDS-compliant
file names and directory structures.
"""

import re
from pathlib import Path
from typing import Dict, Optional, Union, Any
from dataclasses import dataclass
import json


# Standard BIDS entity order. 'others' is a placeholder: any entity not in this
# list is emitted at the 'others' position (move 'others' to control custom entity order).
BIDS_ENTITY_ORDER = [
    "sub",
    "ses",
    "task",
    "acq",
    "ce",
    "dir",
    "rec",
    "run",
    "echo",
    "flip",
    "inv",
    "mt",
    "part",
    "recording",
    "others",
    "space",
    "split",
    "desc",
]

# Final `_`-separated BIDS suffix tokens before extension (MRI anat/func/dwi).
# Not exhaustive for every BIDS derivative; unknown tails are left unchanged.
# Tokens are sorted longest-first when building _BIDS_MODALITY_SUFFIXES so
# e.g. `_boldref` beats `_bold`, `_T2w` beats `_T2`.
# See BIDS MRI filename patterns: https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/01-magnetic-resonance-imaging-data.html
_BIDS_MODALITY_SUFFIX_TOKENS: tuple[str, ...] = ("T1w", "T2w", "bold", "boldref")
_BIDS_MODALITY_SUFFIXES: tuple[str, ...] = tuple(
    f"_{t}" for t in sorted(_BIDS_MODALITY_SUFFIX_TOKENS, key=len, reverse=True)
)

# BIDS ``space`` key-value segments in filename stems (``_space-<value>``).
_BIDS_SPACE_RE = re.compile(r"_space-[a-zA-Z0-9-]+")


def _strip_space_entities(stem: str) -> str:
    """Remove every ``_space-<value>`` segment from a filename stem."""
    return _BIDS_SPACE_RE.sub("", stem)


def _strip_modality_suffix(stem: str) -> str:
    """Remove one trailing recognized BIDS MRI modality suffix from a filename stem."""
    for suffix in _BIDS_MODALITY_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _suffix_has_space_entity(suffix: str) -> bool:
    """True if *suffix* adds a BIDS ``space`` token (``space-...`` or ``_space-...``)."""
    return suffix.startswith("space-") or "_space-" in suffix


def replace_bids_space(stem: str, new_space: str) -> str:
    """Replace (or insert) the ``space-{}`` entity in a BIDS filename stem.

    Removes any existing ``_space-*`` segments, then inserts
    ``_space-{new_space}`` at the canonical BIDS position (highest priority
    anchor wins):

    1. Before ``_desc-``  — e.g. ``sub-01_desc-brain_mask``
                           → ``sub-01_space-T1w_desc-brain_mask``
    2. Before the trailing modality suffix (``_boldref``, ``_bold``,
       ``_T1w``, ``_T2w``) — e.g. ``sub-01_T2w``
                           → ``sub-01_space-scanner_T2w``
    3. Appended           — no recognised anchor present.
    """
    clean = _strip_space_entities(stem)
    space_tag = f"_space-{new_space}"

    # 1. Insert before _desc- (space precedes desc in BIDS entity order)
    if "_desc-" in clean:
        return clean.replace("_desc-", f"{space_tag}_desc-", 1)

    # 2. Insert before the trailing modality suffix
    base = _strip_modality_suffix(clean)
    if base != clean:
        return f"{base}{space_tag}{clean[len(base):]}"

    # 3. No known anchor — append
    return f"{clean}{space_tag}"


def get_filename_stem(file_path: Union[str, Path]) -> str:
    """
    Extract the filename stem (without extensions) from a file path.

    Handles multiple extensions like .nii.gz, .nii, .gz properly.
    This function preserves the exact input structure instead of reconstructing
    BIDS entities, avoiding potential mismatches.

    Args:
        file_path: Path to the file

    Returns:
        Filename without extensions

    Examples:
        >>> get_filename_stem("sub-01_ses-pre_T1w.nii.gz")
        "sub-01_ses-pre_T1w"
        >>> get_filename_stem("/path/to/sub-01_task-rest_bold.nii")
        "sub-01_task-rest_bold"
    """
    file_path = Path(file_path)
    stem = file_path.name

    # Remove extensions in order of preference
    for ext in [".nii.gz", ".nii", ".gz"]:
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    return stem


# Data-file extensions a derivative sidecar may pair with, longest-match first so that
# compound extensions (e.g. ``.surf.gii``, ``.nii.gz``) are stripped whole.
_SIDECAR_DATA_EXTENSIONS = [
    ".nii.gz",
    ".surf.gii",
    ".shape.gii",
    ".label.gii",
    ".func.gii",
    ".gii",
    ".nii",
    ".mat",
    ".h5",
    ".txt",
    ".tsv",
]


def create_bids_sidecar_filename(image_filename: Union[str, Path]) -> str:
    """Return the JSON sidecar basename paired with a derivative data file.

    Follows the BIDS pairing rule used by fMRIPrep: the sidecar basename is identical to
    the data file's basename with only the data extension replaced by ``.json`` (all BIDS
    entities preserved). Only the basename is returned — callers write it next to the data
    file.

    Examples:
        >>> create_bids_sidecar_filename('sub-01_space-NMT2Sym_desc-preproc_T1w.nii.gz')
        'sub-01_space-NMT2Sym_desc-preproc_T1w.json'
        >>> create_bids_sidecar_filename('sub-01_from-T1w_to-template_mode-image_xfm.txt')
        'sub-01_from-T1w_to-template_mode-image_xfm.json'
        >>> create_bids_sidecar_filename('sub-01_hemi-L_desc-cortex_mask.label.gii')
        'sub-01_hemi-L_desc-cortex_mask.json'
    """
    name = Path(image_filename).name
    for ext in _SIDECAR_DATA_EXTENSIONS:
        if name.endswith(ext):
            return name[: -len(ext)] + ".json"
    # Unknown/extensionless: fall back to replacing the final suffix, else append.
    stem = Path(name).stem
    return (stem if stem != name else name) + ".json"


def parse_bids_entities(filename: str) -> Dict[str, str]:
    """
    Parse all BIDS entities from a filename.

    This function extracts ALL key-value pairs that follow the BIDS naming
    convention (key-value) from a filename, without being limited to a
    predefined set of entities. This ensures we capture both standard
    and custom BIDS entities.

    Args:
        filename: BIDS filename to parse (can be full path or just filename)

    Returns:
        Dictionary mapping entity keys to values

    Examples:
        >>> parse_bids_entities("sub-01_ses-pre_task-rest_run-1_bold.nii.gz")
        {'sub': '01', 'ses': 'pre', 'task': 'rest', 'run': '1'}

        >>> parse_bids_entities("sub-032097_ses-001_run-1_desc-brain_T1w.nii.gz")
        {'sub': '032097', 'ses': '001', 'run': '1', 'desc': 'brain'}
    """
    filename = Path(filename).name

    entities = {}

    # This captures ALL key-value pairs, not just predefined ones
    pattern = r"([a-zA-Z]+)-([a-zA-Z0-9-]+)"
    matches = re.findall(pattern, filename)

    for entity, value in matches:
        entities[entity] = value

    return entities


def create_bids_filename(
    entities: Dict[str, str], suffix: str, extension: str = ".nii.gz"
) -> str:
    """
    Create a BIDS-compliant filename from entities dictionary.

    Args:
        entities: Dictionary of BIDS entities (key-value pairs)
        suffix: BIDS suffix (e.g., 'T1w', 'bold', 'desc-brain_T1w')
        extension: File extension (default: '.nii.gz')

    Returns:
        BIDS-compliant filename

    Examples:
        >>> create_bids_filename({'sub': '01', 'ses': 'pre'}, 'T1w')
        'sub-01_ses-pre_T1w.nii.gz'

        >>> create_bids_filename({'sub': '01', 'run': '1'}, 'desc-brain_T1w')
        'sub-01_run-1_desc-brain_T1w.nii.gz'
    """
    # Use the standard BIDS entity order; 'others' is a placeholder for any
    # entity not in the list (e.g. qcq, site-specific keys).
    standard_entities = set(BIDS_ENTITY_ORDER) - {"others"}
    components = []

    for entity in BIDS_ENTITY_ORDER:
        if entity == "others":
            # Emit all custom entities (not in standard list) in sorted order
            custom = sorted(set(entities.keys()) - standard_entities)
            for k in custom:
                components.append(f"{k}-{entities[k]}")
        elif entity in entities:
            components.append(f"{entity}-{entities[entity]}")

    # Join components and add suffix and extension
    filename = "_".join(components)
    if filename:
        filename += f"_{suffix}{extension}"
    else:
        filename = f"{suffix}{extension}"

    return filename


def create_bids_output_filename(
    original_file_path: Union[str, Path],
    suffix: str,
    modality: str,
    extension: str = ".nii.gz",
) -> str:
    """
    Create a BIDS-compliant output filename from an original input filename.

    This function mimics the behavior of the old anat2template.py workflow:
    1. Gets the filename stem from the original file
    2. Removes the modality suffix (e.g., '_T1w', '_bold')
    3. Adds the new suffix and modality back

    This preserves the exact input structure including any non-standard entities.

    Args:
        original_file_path: Path to the original input file
        suffix: New BIDS suffix (e.g., 'desc-preproc', 'desc-brain', 'space-NMT2Sym_desc-preproc')
        modality: Modality to append (e.g., 'T1w', 'T2w', 'bold')
        extension: File extension (default: '.nii.gz')

    Returns:
        BIDS-compliant output filename

    When *suffix* introduces a ``space`` entity (``space-...`` or contains ``_space-``),
    any existing ``_space-*`` segments are removed from the prefix first so names do not
    stack two space entities.

    Examples:
        >>> create_bids_output_filename('sub-032309_ses-001_T1w.nii.gz', 'desc-preproc', 'T1w')
        'sub-032309_ses-001_desc-preproc_T1w.nii.gz'

        >>> create_bids_output_filename('sub-01_ses-pre_task-rest_bold.nii.gz', 'desc-preproc', 'bold')
        'sub-01_ses-pre_task-rest_desc-preproc_bold.nii.gz'

        >>> create_bids_output_filename(
        ...     'sub-032309_space-scanner_T2w.nii.gz', 'space-T1w_desc-preproc', 'T2w')
        'sub-032309_space-T1w_desc-preproc_T2w.nii.gz'

        >>> create_bids_output_filename(
        ...     'sub-032309_space-scanner_T2w.nii.gz', 'desc-preproc', 'T2w')
        'sub-032309_space-scanner_desc-preproc_T2w.nii.gz'
    """
    # Get the filename stem from the original file (e.g., 'sub-032309_ses-001_T1w')
    original_stem = get_filename_stem(original_file_path)

    # Remove the modality suffix from the stem (e.g., 'sub-032309_ses-001_T1w' -> 'sub-032309_ses-001')
    # This matches the old behavior: bids_prefix_wo_modality = bids_prefix.replace(f"_{modality}", "")
    bids_prefix_wo_modality = original_stem.replace(f"_{modality}", "")

    # If the replacement didn't work (modality not found), try boldref fallback
    if bids_prefix_wo_modality == original_stem:
        if modality == "boldref" and "_bold" in original_stem:
            bids_prefix_wo_modality = original_stem.replace("_bold", "")
        # else: fallback — use original_stem as-is

    # Avoid ``..._space-A_space-B_...``: when this step's suffix introduces a new
    # ``space`` entity, drop any existing ``_space-*`` from the prefix first.
    if _suffix_has_space_entity(suffix):
        bids_prefix_wo_modality = _strip_space_entities(bids_prefix_wo_modality)

    # Create the new filename: prefix + suffix + modality + extension
    # This matches: f"{bids_prefix_wo_modality}_desc-preproc_{modality}.nii.gz"
    output_filename = f"{bids_prefix_wo_modality}_{suffix}_{modality}{extension}"

    return output_filename


def create_synthesized_bids_filename(
    original_file: Path,
    modality: str,
    is_subject_level: bool,
    synthesized: bool,
) -> tuple[str, str]:
    """
    Basename and downstream path for outputs of the anatomical synthesis step.

    The returned **basename** (first element) always includes a ``space`` entity so
    the linked output file matches BIDS derivative naming:

    - **T1w:** ``space-scanner`` (native T1w acquisition space)
    - **T2w:** ``space-T2wScanner`` (native T2w acquisition space)

    The **downstream path** (second element) uses the same directory layout but a
    basename **without** the space entity, so later steps can key off the same
    template stem as pre-synthesis inputs.

    - **Synthesized:** Only ``sub`` and optional ``ses`` (omitted when subject-level);
      run/acq/etc. are dropped. Intended for merged images.
    - **Passthrough:** Uses :func:`create_bids_output_filename` so the stem (minus
      modality) is preserved verbatim (run, acq, entity order, non-standard keys).
      Inserts the modality-appropriate space before the modality. When subject-level,
      the ``_ses-<value>`` segment is removed from the stem first.

    Args:
        original_file: Template input path (typically the first input)
        modality: Modality suffix (e.g., "T1w", "T2w")
        is_subject_level: True when the channel has no session (subject-level job)
        synthesized: True if a merge was performed, False if single-file passthrough

    Returns:
        ``(bids_filename_with_space_scanner, bids_path_for_downstream_wo_space)``
    """
    parsed = parse_bids_entities(original_file.name)
    if modality == "T2w":
        space_suffix = "space-T2wScanner"
        space_entity_value = "T2wScanner"
    else:
        space_suffix = "space-scanner"
        space_entity_value = "scanner"

    if synthesized:
        out_entities: Dict[str, str] = {
            "sub": parsed.get("sub", "unknown"),
        }
        if not is_subject_level and "ses" in parsed:
            out_entities["ses"] = parsed["ses"]
        out_entities["space"] = space_entity_value
        bids_filename = create_bids_filename(
            entities=out_entities,
            suffix=modality,
            extension=".nii.gz",
        )
    else:
        stem = get_filename_stem(original_file)
        if is_subject_level:
            stem = re.sub(r"_ses-[a-zA-Z0-9-]+", "", stem)
        synthetic = Path(stem + ".nii.gz")
        bids_filename = create_bids_output_filename(
            synthetic,
            suffix=space_suffix,
            modality=modality,
            extension=".nii.gz",
        )

    downstream_basename = (
        _strip_space_entities(get_filename_stem(bids_filename)) + ".nii.gz"
    )

    sub_id = parsed.get("sub", "unknown")
    if is_subject_level:
        bids_path_for_downstream = f"sub-{sub_id}/anat/{downstream_basename}"
    else:
        bids_path_for_downstream = str(original_file.parent / downstream_basename)

    return bids_filename, bids_path_for_downstream


def longitudinal_bids_name(bids_name: Union[str, Path], kind: str) -> str:
    """Filename the base template or a longitudinal timepoint publishes under.

    The two longitudinal streams write derivatives, atlases and QC figures beside
    the cross-sectional ones, into directories that carry no extra level to
    separate them -- QC figures in particular all land in ``sub-<id>/figures``.
    So the *filename* is the only thing keeping them apart, and it has to be
    derived once, here, rather than by regex at each call site.

    Args:
        bids_name: A session's BIDS name. May be an absolute path: callers get
            this from ``create_synthesized_bids_filename``, whose session-level
            branch returns a full path, and an earlier regex-based version of
            this logic was anchored with ``^`` and therefore never fired at all.
        kind: ``"base"`` or ``"long"``.

    Returns:
        A bare filename (no directory), which is what every consumer wants --
        ``create_bids_output_filename`` and ``get_bids_prefix`` both take the
        basename anyway.

    Raises:
        ValueError: If *kind* is unknown, or *bids_name* has no ``sub`` entity.

    Examples:
        >>> longitudinal_bids_name('sub-01_ses-a_T1w.nii.gz', 'base')
        'sub-01_acq-base_T1w.nii.gz'
        >>> longitudinal_bids_name('sub-01_ses-a_acq-mprage_run-1_T1w.nii.gz', 'long')
        'sub-01_ses-a_acq-long_run-1_T1w.nii.gz'
    """
    if kind not in ("base", "long"):
        raise ValueError(f"kind must be 'base' or 'long', got {kind!r}")

    name = Path(str(bids_name)).name
    parsed = parse_bids_entities(name)
    if "sub" not in parsed:
        raise ValueError(
            f"Cannot derive a longitudinal name from {bids_name!r}: no sub- entity."
        )

    modality = next(
        (t for t in _BIDS_MODALITY_SUFFIX_TOKENS if name.endswith(f"_{t}.nii.gz")),
        "T1w",
    )

    if kind == "base":
        # Only sub. The base spans every session, so ses -- and any session-level
        # entity such as run- or rec- -- would be inherited from whichever session
        # happened to sort first, which makes the base's name depend on an
        # ordering that means nothing. Keeping it to sub also makes this identical
        # to the acq-base prefix the base's derivatives already publish under, so
        # the two name families agree by construction rather than by coincidence.
        entities = {"sub": parsed["sub"], "acq": "base"}
    else:
        # Everything the session carried, minus derivative-only entities, with acq
        # *replaced* rather than prepended -- appending would give two acq- values
        # for input named sub-XX_ses-Y_acq-mprage_T1w. create_bids_filename re-emits
        # through BIDS_ENTITY_ORDER, so acq lands before run with no regex.
        entities = {k: v for k, v in parsed.items() if k not in ("space", "desc")}
        entities["acq"] = "long"

    return create_bids_filename(entities, suffix=modality, extension=".nii.gz")


def get_bids_prefix(
    bids_name: Union[str, Path],
    run_identifier: Optional[str] = None,
    session_level: bool = False,
) -> str:
    """
    Generate BIDS prefix for output files, handling session-level vs run-level naming.

    This function provides a consistent way to generate BIDS prefixes across the pipeline,
    especially for within-session coregistration scenarios where:
    - Session-level processing: Keep only sub/ses entities (remove task/run/acq)
    - Run-level processing: Preserve all entities from original template

    Args:
        bids_name: Original BIDS filename or path
        run_identifier: Run identifier string (empty/None for session-level)
        session_level: Force session-level even if run_identifier is provided

    Returns:
        BIDS prefix without trailing modality suffix (e.g. ``_T1w``, ``_bold``,
        ``_boldref``) when that suffix is one of the recognized MRI tokens in
        ``_BIDS_MODALITY_SUFFIX_TOKENS``.

    Examples:
        >>> # Session-level: removes run-specific entities
        >>> get_bids_prefix('sub-01_ses-001_task-rest_run-1_bold.nii.gz', run_identifier='')
        'sub-01_ses-001'

        >>> # Run-level: preserves all entities
        >>> get_bids_prefix('sub-01_ses-001_task-rest_run-1_bold.nii.gz', run_identifier='task-rest_run-1')
        'sub-01_ses-001_task-rest_run-1'

        >>> # Run-level: strips anat suffixes too
        >>> get_bids_prefix('sub-01_ses-001_T1w.nii.gz', run_identifier='x')
        'sub-01_ses-001'

        >>> # Force session-level
        >>> get_bids_prefix('sub-01_ses-001_task-rest_run-1_bold.nii.gz', session_level=True)
        'sub-01_ses-001'
    """
    # Determine if we should use session-level naming
    is_session_level = (
        session_level or not run_identifier or run_identifier.strip() == ""
    )

    if is_session_level:
        # Session-level: keep only sub and ses entities
        parsed = parse_bids_entities(str(bids_name))
        filtered_entities = {}
        if "sub" in parsed:
            filtered_entities["sub"] = parsed["sub"]
        if "ses" in parsed:
            filtered_entities["ses"] = parsed["ses"]

        # Create prefix without suffix
        prefix = create_bids_filename(filtered_entities, "", extension="")
        # Remove trailing underscore if present
        return prefix.rstrip("_")
    else:
        # Run-level: preserve all entities from original template
        original_stem = get_filename_stem(bids_name)
        return _strip_modality_suffix(original_stem)


def find_bids_metadata(
    nifti_path: Union[str, Path], dataset_dir: Union[str, Path]
) -> Optional[Dict[str, Any]]:
    """
    Find and load BIDS sidecar JSON metadata for a NIfTI file using hierarchical search.

    BIDS inheritance principle: JSON files at higher levels in the hierarchy
    are inherited by files at lower levels, with more specific files taking precedence.

    Search order:
    1. Exact match in same directory (most specific)
    2. Session level (if applicable)
    3. Subject level
    4. Dataset root level (most general)

    Args:
        nifti_path: Path to the NIfTI file
        dataset_dir: Root directory of the BIDS dataset

    Returns:
        Dictionary containing merged metadata from all applicable JSON files,
        or None if no metadata found
    """
    nifti_path = Path(nifti_path)
    dataset_dir = Path(dataset_dir)

    # Extract BIDS entities from filename
    entities = parse_bids_entities(nifti_path.name)

    general_name = (
        nifti_path.name.split("_")[-1]
        .replace(".nii.gz", ".json")
        .replace(".nii", ".json")
    )

    # Generate potential JSON filenames at different hierarchy levels
    json_candidates = []

    # 1. Exact match in same directory (highest priority)
    exact_json = nifti_path.with_suffix("").with_suffix(".json")
    if exact_json.exists():
        json_candidates.append(exact_json)

    # 2. Session level JSON (if session exists)
    if entities.get("ses"):
        ses_dir = dataset_dir / f"sub-{entities['sub']}" / f"ses-{entities['ses']}"

        # Look for modality-specific JSONs
        if "task" in entities:
            # task-specific JSON
            ses_task_json = (
                ses_dir / nifti_path.parent.name / f"task-{entities['task']}_bold.json"
            )
            if ses_task_json.exists():
                json_candidates.append(ses_task_json)

        # General modality JSON at session level (e.g. T1w.json, bold.json)
        if nifti_path.parent.name in ["func", "anat", "dwi", "fmap"]:
            ses_general_json = ses_dir / nifti_path.parent.name / general_name
            if ses_general_json.exists():
                json_candidates.append(ses_general_json)

    # 3. Subject level JSON
    sub_dir = dataset_dir / f"sub-{entities['sub']}"

    if "task" in entities:
        # task-specific JSON at subject level
        sub_task_json = (
            sub_dir / nifti_path.parent.name / f"task-{entities['task']}_bold.json"
        )
        if sub_task_json.exists():
            json_candidates.append(sub_task_json)

    # General modality JSON at subject level
    if nifti_path.parent.name in ["func", "anat", "dwi", "fmap"]:
        sub_general_json = sub_dir / nifti_path.parent.name / general_name
        if sub_general_json.exists():
            json_candidates.append(sub_general_json)

    # 4. Dataset root level JSON (lowest priority)
    if "task" in entities:
        # task-specific JSON at dataset level
        dataset_task_json = dataset_dir / f"task-{entities['task']}_bold.json"
        if dataset_task_json.exists():
            json_candidates.append(dataset_task_json)

    # General modality JSON at dataset level
    dataset_general_json = dataset_dir / general_name
    if dataset_general_json.exists():
        json_candidates.append(dataset_general_json)

    # Load and merge JSON files (most general first, most specific last)
    merged_metadata = {}

    # Reverse the list to start with most general (dataset level) and end with most specific (exact match)
    for json_file in reversed(json_candidates):
        try:
            with open(json_file, "r") as f:
                metadata = json.load(f)
                merged_metadata.update(
                    metadata
                )  # More specific files override general ones
        except (json.JSONDecodeError, IOError):
            # Log warning but continue with other files
            continue

    return merged_metadata if merged_metadata else None


@dataclass
class BIDSFile:
    """Represents a BIDS file with metadata."""

    path: str
    sub: str
    ses: Optional[str] = None
    run: Optional[str] = None
    task: Optional[str] = None
    acq: Optional[str] = None
    modality: Optional[str] = None
    suffix: Optional[str] = None
    extension: Optional[str] = None
    entities: Optional[Dict[str, str]] = None
