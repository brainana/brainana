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


# Canonical emission order for brainana output names. 'others' is a placeholder:
# any entity not in this list is emitted at the 'others' position (move 'others'
# to control custom entity order).
#
# Every entity brainana itself writes must appear here. An entity that is absent
# falls into the 'others' slot, which sits *before* space/desc -- so a name that
# round-trips through parse_bids_entities + create_bids_filename would come back
# reordered. That is why ``stat`` and ``hemi`` are listed: without them a tSNR map
# rebuilt from its own entities became ``_stat-tsnr_desc-preproc_`` instead of
# ``_desc-preproc_stat-tsnr_``.
#
# Positions are the ones brainana and docs/outputs.rst already publish, NOT the
# abstract BIDS order, because published names are a consumer contract:
#   * ``hemi`` after ``space`` -- atlas-<n>_space-fsnative_hemi-L_<prefix>.func.gii,
#     which brainana-viewer discovers by that exact shape.
#   * ``stat`` after ``desc`` -- <prefix>_space-T1w_desc-preproc_stat-tsnr_boldmap.
#   * ``atlas`` first -- atlas-<n>_space-<sp>_sub-<id>.nii.gz. Subject-last is a
#     declared deviation (see DECLARED_DEVIATIONS); leading ``atlas-`` is what the
#     viewer and utils/templates.py glob on.
# See docs_temp/update_instruction/name_guideline.md before moving any of them.
BIDS_ENTITY_ORDER = [
    "atlas",
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
    "from",
    "to",
    "mode",
    "space",
    "res",
    "den",
    "hemi",
    "split",
    "desc",
    "stat",
]

# What each entity means, and who is allowed to write it. The rule this table
# encodes: brainana copies identity entities from the raw data and never sets
# them; every concern of its own gets exactly one named slot.
#
# This is the table that rules out the bug it was written for. The longitudinal
# stream used to mark its products with ``acq-base`` / ``acq-long``, but ``acq``
# is an identity entity -- the same dev-test dataset carries a real ``acq-test``
# from raw data. So a stream marker and an acquisition label were indistinguishable,
# a timepoint's own ``acq`` was destroyed, and the QC report (which groups figures
# by identity entities) read one session as two. The marker now lives in ``space``,
# which is what the distinction actually is: a base-seeded reconstruction is in
# base space.
ENTITY_ROLES: Dict[str, str] = {
    # identity -- inherited from the raw file. brainana copies, never sets.
    "sub": "identity",
    "ses": "identity",
    "task": "identity",
    "acq": "identity",
    "ce": "identity",
    "dir": "identity",
    "rec": "identity",
    "run": "identity",
    "echo": "identity",
    "flip": "identity",
    "inv": "identity",
    "mt": "identity",
    "part": "identity",
    "recording": "identity",
    # frame -- which reference frame the data are in. Written by brainana.
    "space": "frame",
    "from": "frame",
    "to": "frame",
    "mode": "frame",
    "res": "frame",
    "den": "frame",
    "hemi": "frame",
    # variant -- which processing variant this is. Written by brainana, once.
    "desc": "variant",
    "split": "variant",
    # product -- what kind of product this is. Written by brainana.
    "atlas": "product",
    "stat": "product",
}

IDENTITY_ROLE = "identity"

# Deviations from BIDS that brainana publishes deliberately, with the consumer
# that depends on each. validate_output_name() reports these as declared rather
# than as violations. Do not "fix" one without changing its consumer first.
DECLARED_DEVIATIONS: Dict[str, str] = {
    "brain_suffix_tail": (
        "A '_brain' token after the modality suffix "
        "(<prefix>_space-T1w_desc-preproc_T1w_brain.nii.gz). Not a BIDS suffix, so "
        "no BIDS parser can read it back -- including this repo's own input "
        "validator. Kept because brainana-viewer's data contract names it in the "
        "base-volume fallback chain, and docs/outputs.rst publishes it."
    ),
    "atlas_subject_last": (
        "Atlas outputs lead with 'atlas-<name>' and carry 'sub-' last "
        "(atlas-ARM1_space-T1w_sub-01_ses-001.nii.gz). Kept because atlas discovery "
        "in brainana-viewer, utils/templates.py and steps/anatomical.py all match on "
        "the leading 'atlas-' token."
    ),
    "unknown_raw_entity": (
        "An entity brainana does not know, inherited verbatim from the raw data "
        "(e.g. 'test-xxx'). Preserved rather than dropped: dropping it would "
        "collapse two runs that differ only by that entity onto one filename, and "
        "publishDir runs with overwrite:false, so that is a silently missing file."
    ),
}

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


# One ``key-value`` token. Values admit '.' so ``res-0.5`` survives as one token.
_ENTITY_TOKEN_RE = re.compile(r"([a-zA-Z]+)-([A-Za-z0-9.-]+)")

# Extensions brainana publishes, longest-first so '.surf.gii' beats '.gii'. Used by
# validate_output_name to find the suffix; get_filename_stem handles the NIfTI ones.
_PUBLISHED_EXTENSIONS: tuple[str, ...] = tuple(
    sorted(
        (
            ".surf.gii",
            ".func.gii",
            ".shape.gii",
            ".label.gii",
            ".gii",
            ".png",
            ".svg",
            ".json",
            ".tsv",
            ".csv",
            ".mat",
            ".h5",
            ".bib",
            ".md",
            ".dat",
            ".txt",
            ".html",
            ".mgz",
            ".mgh",
            ".lta",
        ),
        key=len,
        reverse=True,
    )
)


def _suffix_entity_keys(suffix: str) -> list[str]:
    """Entity keys a *suffix* introduces, in the order it introduces them.

    ``'space-T1w_desc-preproc'`` -> ``['space', 'desc']``; ``'T1w'`` -> ``[]``.
    """
    keys = []
    for token in suffix.split("_"):
        match = _ENTITY_TOKEN_RE.fullmatch(token)
        if match and match.group(1) not in keys:
            keys.append(match.group(1))
    return keys


def _strip_entities(stem: str, keys) -> str:
    """Remove every ``_<key>-<value>`` segment for *keys* from a filename stem.

    The leading ``_`` is required, so a stem-initial entity (``sub-01_...``,
    ``atlas-ARM1_...``) is never eaten.
    """
    for key in keys:
        stem = re.sub(rf"_{re.escape(key)}-[A-Za-z0-9.-]+", "", stem)
    return stem


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
            # Emit unknown entities in the order the caller gave them, which for a
            # dict from parse_bids_entities is the order they appeared in the
            # filename. Sorting them here instead made the two naming paths
            # disagree: create_bids_output_filename preserves an inherited
            # entity's position, this one alphabetised it.
            for k in entities:
                if k not in standard_entities:
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

    Any entity the *suffix* introduces is removed from the prefix first, so a name
    never stacks the same entity twice (``space-A_space-B``, ``desc-a_desc-b``).

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

    # Drop the input's own suffix. A recognized modality token at the *end* of the
    # stem is stripped as a whole token, longest-first, so a '_boldref' input asked
    # for modality 'bold' loses '_boldref' rather than the '_bold' inside it --
    # which is what turned 'sub-01_ses-001_boldref' into 'sub-01_ses-001ref'.
    stripped = _strip_modality_suffix(original_stem)
    if stripped != original_stem:
        bids_prefix_wo_modality = stripped
    else:
        # No recognized trailing token (e.g. the declared '_T1w_brain' tail, whose
        # last token is '_brain'). Fall back to the historical replace so those
        # names keep the shape docs/outputs.rst publishes.
        bids_prefix_wo_modality = original_stem.replace(f"_{modality}", "")

    # Never stack an entity the suffix is about to set. Previously this de-duplicated
    # ``space`` only, which is why a ``desc-preproc`` input given a ``desc-func2target``
    # suffix produced two ``desc-`` tokens -- and parse_bids_entities keeps only the
    # last, so the round trip silently dropped ``preproc``.
    bids_prefix_wo_modality = _strip_entities(
        bids_prefix_wo_modality, _suffix_entity_keys(suffix)
    )

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

    The marker is the ``space`` entity, because that is what the distinction is: a
    base-seeded reconstruction lives in the subject's base space, which is already
    the documented reason the functional and fsnative-atlas streams keep using the
    cross-sectional trees. It used to be ``acq``, which was wrong twice over --
    ``acq`` is an identity entity inherited from the raw data, so a timepoint's own
    ``acq-mprage`` was destroyed and two acquisitions in one session collapsed onto
    one filename; and consumers that group by identity entities (the QC report)
    read one session as two.

    Examples:
        >>> longitudinal_bids_name('sub-01_ses-a_T1w.nii.gz', 'base')
        'sub-01_space-base_T1w.nii.gz'
        >>> longitudinal_bids_name('sub-01_ses-a_acq-mprage_run-1_T1w.nii.gz', 'long')
        'sub-01_ses-a_acq-mprage_run-1_space-base_T1w.nii.gz'
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
        # to the space-base prefix the base's derivatives already publish under, so
        # the two name families agree by construction rather than by coincidence.
        entities = {"sub": parsed["sub"], "space": "base"}
        return create_bids_filename(entities, suffix=modality, extension=".nii.gz")

    # Every identity entity the session carried, including its own acq-, minus the
    # derivative-only desc. Only space is replaced, and space is brainana's to
    # write -- routed through derive_output_name so that if this ever moves to
    # another slot, the identity-entity check catches it here rather than in a
    # published tree.
    return derive_output_name(
        name,
        suffix=modality,
        set_entities={"space": "base"},
        drop_entities=("desc",),
        extension=".nii.gz",
    )


class NamingConflictError(ValueError):
    """An output name would overwrite an inherited entity, or carry one twice.

    Raised rather than silently resolved because both outcomes are data loss that
    nothing downstream can detect: an overwritten identity entity erases what the
    scanner recorded, and a duplicated key is dropped by parse_bids_entities (it
    keeps the last), so the name no longer round-trips.
    """


def derive_output_name(
    source: Union[str, Path],
    *,
    suffix: str,
    set_entities: Optional[Dict[str, str]] = None,
    drop_entities: tuple[str, ...] = (),
    extension: str = ".nii.gz",
) -> str:
    """Build one output name from an input name plus the entities this step sets.

    The single constructor for brainana output names. It copies the source's
    entities, applies *set_entities*, and emits the result in
    ``BIDS_ENTITY_ORDER`` -- refusing, rather than resolving, the two ways a name
    loses information:

    * setting an ``identity`` entity (``ENTITY_ROLES``) that the source already
      carries with a different value. Identity entities describe what was
      acquired; brainana copies them and has nothing to say about them. This is
      the check that rejects the ``acq-base`` / ``acq-long`` marker the
      longitudinal stream used to write over a timepoint's real ``acq-mprage``.
    * emitting any key twice.

    Args:
        source: The input name or path whose entities are inherited. Only the
            basename is read.
        suffix: The BIDS suffix, one token (``'T1w'``, ``'bold'``, ``'mask'``).
            Entities belong in *set_entities*, not here.
        set_entities: Entities this step writes, e.g.
            ``{"space": "base", "desc": "preproc"}``.
        drop_entities: Entity keys to remove from the inherited set. Use it for
            an entity that stops being true of the output -- a session-level
            product dropping ``run``, say.
        extension: File extension, ``'.nii.gz'`` by default.

    Returns:
        A bare filename, in canonical entity order.

    Raises:
        NamingConflictError: On an identity-entity conflict, or a duplicate key.
        ValueError: If *suffix* is empty or carries an entity token.

    Examples:
        >>> derive_output_name('sub-01_ses-a_run-1_T1w.nii.gz',
        ...                    suffix='T1w', set_entities={'space': 'base'})
        'sub-01_ses-a_run-1_space-base_T1w.nii.gz'

        >>> derive_output_name('sub-01_ses-a_space-NMT2Sym_desc-preproc_bold.nii.gz',
        ...                    suffix='bold', set_entities={'desc': 'func2target'})
        'sub-01_ses-a_space-NMT2Sym_desc-func2target_bold.nii.gz'

        >>> derive_output_name('sub-01_ses-a_acq-mprage_T1w.nii.gz',
        ...                    suffix='T1w', set_entities={'acq': 'long'})
        Traceback (most recent call last):
            ...
        nhp_mri_prep.utils.bids.NamingConflictError: ...
    """
    if not suffix:
        raise ValueError("suffix is required; an output name needs a BIDS suffix.")
    if _suffix_entity_keys(suffix):
        raise ValueError(
            f"suffix {suffix!r} carries an entity token. Pass entities in "
            "set_entities so they can be ordered and conflict-checked."
        )

    name = Path(str(source)).name
    entities = dict(parse_bids_entities(name))
    for key in drop_entities:
        entities.pop(key, None)

    for key, value in (set_entities or {}).items():
        current = entities.get(key)
        if (
            current is not None
            and current != value
            and ENTITY_ROLES.get(key) == IDENTITY_ROLE
        ):
            raise NamingConflictError(
                f"Refusing to set {key}-{value} on {name!r}: it already carries "
                f"{key}-{current}, and {key!r} is an identity entity inherited from "
                f"the raw data. Pick a slot brainana owns (see ENTITY_ROLES), or "
                f"drop it explicitly via drop_entities if it is genuinely no longer true."
            )
        entities[key] = value

    return create_bids_filename(entities, suffix=suffix, extension=extension)


def validate_output_name(name: Union[str, Path]) -> list[str]:
    """Check one published name against the output-naming contract.

    Returns a list of violation strings -- empty when the name is clean. The
    deliberate deviations in ``DECLARED_DEVIATIONS`` are not reported: they are
    published on purpose and consumers depend on them.

    Checks:

    1. No entity key appears twice (``parse_bids_entities`` keeps only the last,
       so a repeated key is silently dropped information).
    2. Entities are emitted in ``BIDS_ENTITY_ORDER`` -- or, for the atlas tree,
       in its own declared shape (``atlas-<name>_space-<frame>_...``).
    3. Exactly one suffix token.

    It checks *shape*, not meaning: it cannot tell that a marker went into the
    wrong slot, only that the result is well formed. Slot choice is enforced where
    names are built, by ``derive_output_name``.

    Args:
        name: A published filename or path. Only the basename is checked.

    Returns:
        Violations, most structural first. Declared deviations are omitted.

    Examples:
        >>> validate_output_name('sub-01_ses-a_space-T1w_desc-preproc_T1w.nii.gz')
        []
        >>> validate_output_name('sub-01_space-NMT2Sym_desc-preproc_desc-func2target_bold.png')
        ["duplicate entity 'desc' (values: preproc, func2target); parse keeps only the last"]
        >>> # declared: subject-last atlas names, and the '_brain' suffix tail
        >>> validate_output_name('atlas-ARM1_space-T1w_sub-01.nii.gz')
        []
    """
    basename = Path(str(name)).name
    stem = get_filename_stem(basename)
    # Strip the extension explicitly rather than cutting at the last dot: an entity
    # value may contain one (res-0.5), and '.surf.gii' is two segments.
    for dotted in _PUBLISHED_EXTENSIONS:
        if stem.endswith(dotted):
            stem = stem[: -len(dotted)]
            break

    tokens = stem.split("_")
    entity_tokens = [(t, _ENTITY_TOKEN_RE.fullmatch(t)) for t in tokens]
    keys = [m.group(1) for _, m in entity_tokens if m]
    suffix_tokens = [t for t, m in entity_tokens if not m]

    violations: list[str] = []

    # 1. duplicate keys
    seen: Dict[str, list] = {}
    for _, match in entity_tokens:
        if match:
            seen.setdefault(match.group(1), []).append(match.group(2))
    for key, vals in seen.items():
        if len(vals) > 1:
            violations.append(
                f"duplicate entity {key!r} (values: {', '.join(vals)}); "
                "parse keeps only the last"
            )

    # 2. canonical order. The atlas tree publishes sub- last on purpose, so a name
    # led by atlas- is exempt from the general order check
    # (DECLARED_DEVIATIONS['atlas_subject_last']) -- but the convention has a shape
    # of its own, and it is load-bearing: every atlas consumer, brainana-viewer
    # included, discovers these by 'atlas-<name>_space-*'. So check that instead.
    # Without this the exemption was a hole: a rename that moved space- to the end
    # of an atlas name passed validation and was caught only by diffing a real
    # output tree against the previous run.
    if stem.startswith("atlas-"):
        if "space" in keys and keys[:2] != ["atlas", "space"]:
            violations.append(
                f"atlas name must read atlas-<name>_space-<frame>_...: got "
                f"{'_'.join(keys)}"
            )
    else:
        others = BIDS_ENTITY_ORDER.index("others")
        positions = [
            BIDS_ENTITY_ORDER.index(k) if k in BIDS_ENTITY_ORDER else others
            for k in keys
        ]
        if positions != sorted(positions):
            violations.append(
                f"entities out of canonical order: {'_'.join(keys)} "
                "(see BIDS_ENTITY_ORDER)"
            )

    # 3. one suffix token. A trailing '_brain' is declared
    # (DECLARED_DEVIATIONS['brain_suffix_tail']).
    if len(suffix_tokens) > 1:
        declared_tail = (
            len(suffix_tokens) == 2
            and suffix_tokens[-1] == "brain"
            and suffix_tokens[0] in _BIDS_MODALITY_SUFFIX_TOKENS
        )
        if not declared_tail:
            violations.append(
                f"multi-token suffix {'_'.join(suffix_tokens)!r}; "
                "a BIDS name carries exactly one suffix"
            )

    # An entity brainana does not know is not checked further: it came from the raw
    # data and is declared (DECLARED_DEVIATIONS['unknown_raw_entity']). That every
    # entity brainana *does* write has a role is an invariant of the two tables,
    # pinned by tests/test_output_naming_contract.py rather than re-checked here.

    return violations


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

    Note:
        The session-level branch drops ``space``. A caller naming a product whose
        frame is not the session's own (the longitudinal base) must therefore take
        the frame from the source separately -- see ``anat_backproject_atlases``,
        which reads ``space`` off *bids_name* and puts it in the one ``space`` slot
        its names carry.

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
        for key in ("sub", "ses"):
            if key in parsed:
                filtered_entities[key] = parsed[key]

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
