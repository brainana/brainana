"""
Nextflow-specific utility functions for nhp_mri_prep.

This module provides utilities that are commonly used in Nextflow process scripts.
"""

from pathlib import Path
import gzip
import json
import logging
import os
import shutil
import sys
from typing import Callable, Dict, Any, Optional, Union

from .bids import get_filename_stem
from ..config.config_io import load_yaml_config
from .logger import LOG_DATEFMT, LOG_FORMAT


def ensure_stderr_logging_if_unconfigured(level: int = logging.INFO) -> None:
    """
    If the root logger has no handlers (typical in ``python3 <<'EOF'`` Nextflow tasks),
    configure ``logging.basicConfig`` so ``logger.info`` from library code reaches stderr.

    No-op when logging was already configured (e.g. CLI scripts or tests).
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        stream=sys.stderr,
    )


def _file_starts_with_gzip_magic(path: Path) -> bool:
    """Return True if file begins with gzip magic (0x1f 0x8b)."""
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def create_output_link(source_file, target_file):
    """
    Create symlink from source to target, fallback to copy if symlink fails.

    This function is used in Nextflow processes to avoid duplicating large files
    between the work directory and process output directory. Nextflow's publishDir
    will follow the symlink and copy the actual file content to the final output.

    IMPORTANT: Resolves source symlinks to the original non-symlink file before
    creating a new symlink. This prevents deep symlink chains.

    Symlinks can fail with "Operation not supported" (errno 95 / EOPNOTSUPP) when
    the target directory is on a filesystem that does not support them, such as:
    - Docker bind-mounts from a Windows host
    - NFS or network shares mounted with nosymlink
    - FAT32, exFAT, or other non-Unix filesystems

    In those cases the function falls back to copying the file.

    Args:
        source_file: Path to source file (typically in work/ directory)
        target_file: Path to target file (in process output directory)

    Returns:
        None (creates symlink or copies file)
    """
    source_path = Path(source_file)
    target_path = Path(target_file)

    # Resolve source to actual file (follows symlink chain to original file)
    # This prevents creating symlinks to symlinks (deep symlink chains)
    source_resolved = source_path.resolve(strict=True)

    # BIDS / nibabel expect .nii.gz files to be gzip-compressed. Symlinking or copying
    # raw .nii bytes to a *.nii.gz name (e.g. after template normalization) breaks readers.
    if str(target_path).endswith(".nii.gz") and not _file_starts_with_gzip_magic(
        source_resolved
    ):
        try:
            same_file = source_resolved == target_path.resolve(strict=True)
        except OSError:
            same_file = False
        if same_file:
            with open(source_resolved, "rb") as f_in:
                raw = f_in.read()
            target_path.unlink(missing_ok=False)
            with gzip.open(target_path, "wb", compresslevel=6) as gz_out:
                gz_out.write(raw)
        else:
            if target_path.exists() or target_path.is_symlink():
                target_path.unlink()
            with open(source_resolved, "rb") as f_in:
                with gzip.open(target_path, "wb", compresslevel=6) as gz_out:
                    shutil.copyfileobj(f_in, gz_out)
        return

    # Guard: if source and target are the same underlying file, creating a symlink
    # would produce a self-referential or no-op link.  Replace any existing symlink
    # at the target with a real copy to leave a valid file in place, then return.
    # NOTE: This does NOT fix Nextflow "Missing output file" errors for pass-through
    # steps.  Nextflow excludes output files by relative path in the work directory,
    # so a copy at the same path as a staged input is still excluded.  Pass-through
    # outputs must be written to a different path (e.g. an nf_out/ subdirectory).
    try:
        target_resolved = target_path.resolve(strict=True)
        if source_resolved == target_resolved:
            if target_path.is_symlink():
                target_path.unlink()
                shutil.copy2(str(source_resolved), str(target_path))
            return
    except OSError:
        pass  # target doesn't exist yet — proceed normally

    # Remove target if it exists
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()

    try:
        # Calculate relative path from target's parent to resolved source
        # This ensures the symlink works even if work directories are moved
        source_rel = os.path.relpath(str(source_resolved), str(target_path.parent))
        os.symlink(source_rel, str(target_path))
    except Exception:
        # Symlink not possible: filesystem doesn't support symlinks (EOPNOTSUPP),
        # cross-device link, or any other failure. Fall back to copy.
        shutil.copy2(str(source_resolved), str(target_path))


def load_config(config_file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    This is a convenience wrapper for loading YAML config files in Nextflow processes.

    Args:
        config_file_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration values

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML parsing fails
    """
    config_path = Path(config_file_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_yaml_config(config_path)
    return config or {}


def config_section(config: Dict[str, Any], dotted: str) -> Dict[str, Any]:
    """A nested config section as a dict, whatever the YAML actually held.

    ``config.get("anat", {}).get("surface_reconstruction", {})`` looks safe and is
    not: a key written with an empty value -- ``longitudinal:`` on a line of its
    own -- parses as ``None``, not ``{}``, so the next ``.get()`` raises
    ``AttributeError`` from inside a Nextflow task. Under
    ``errorStrategy 'ignore'`` that surfaces as nothing happening at all.

    Args:
        config: Full configuration dictionary.
        dotted: Dotted path to the section, e.g.
            ``"anat.surface_reconstruction.longitudinal"``.

    Returns:
        The section, or ``{}`` if any level is missing, null, or not a dict.
    """
    node: Any = config
    for key in dotted.split("."):
        if not isinstance(node, dict):
            return {}
        node = node.get(key)
    return node if isinstance(node, dict) else {}


def config_value(
    config: Dict[str, Any],
    dotted: str,
    default: Any = None,
    cast: Optional[Callable[[Any], Any]] = None,
) -> Any:
    """A single config value, treating an explicit null as "not set".

    ``dict.get(key, default)`` returns ``None`` for a key present with an empty
    value, so ``float(cfg.get("max_cbv_dist", 3.5))`` raises ``TypeError`` on
    ``max_cbv_dist:``. Callers want the default there, not a traceback.

    Args:
        config: Full configuration dictionary.
        dotted: Dotted path to the value.
        default: Returned when the key is absent or null.
        cast: Applied to a non-null value, e.g. ``float``. Never applied to
            ``default``, so a ``None`` default survives.

    Returns:
        The cast value, or ``default``.
    """
    parent, _, leaf = dotted.rpartition(".")
    section = config_section(config, parent) if parent else config
    value = section.get(leaf) if isinstance(section, dict) else None
    if value is None:
        return default
    return cast(value) if cast is not None else value


def detect_modality(bids_naming_template: Union[str, Path]) -> str:
    """
    Detect anatomical modality (T1w or T2w) from BIDS naming template filename.

    Args:
        bids_naming_template: Path to BIDS file (used as naming template)

    Returns:
        Modality string: 'T1w' or 'T2w' (defaults to 'T1w' if not detected)
    """
    original_stem = get_filename_stem(bids_naming_template)
    modality = "T1w"  # default
    if "_T2w" in original_stem or original_stem.endswith("_T2w"):
        modality = "T2w"
    elif "_T1w" in original_stem or original_stem.endswith("_T1w"):
        modality = "T1w"
    return modality


def save_metadata(
    metadata_dict: Dict[str, Any], output_path: Union[str, Path] = "metadata.json"
) -> None:
    """
    Save metadata dictionary to JSON file.

    Args:
        metadata_dict: Dictionary containing metadata to save
        output_path: Path to output JSON file (default: 'metadata.json')
    """
    output_file = Path(output_path)
    with open(output_file, "w") as f:
        json.dump(metadata_dict, f, indent=2)


def normalize_session_id(session_id_raw: Optional[str]) -> Optional[str]:
    """
    Normalize session ID from Nextflow.

    Handles various representations of empty/null session IDs:
    - None
    - Empty string ""
    - Whitespace-only strings
    - String "null" (Nextflow may pass "null" as a string when session_id is empty/null in Groovy)

    Args:
        session_id_raw: Raw session ID from Nextflow

    Returns:
        Normalized session ID string, or None if empty/null
    """
    if not session_id_raw:
        return None

    session_id = session_id_raw.strip()
    if not session_id or session_id.lower() == "null":
        return None

    return session_id
