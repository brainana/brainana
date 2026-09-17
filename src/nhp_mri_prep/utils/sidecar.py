"""
JSON sidecar generation for brainana derivatives.

Writes BIDS-style ``.json`` sidecars next to preprocessed derivative files (fMRIPrep
field vocabulary) and a dataset-level ``dataset_description.json``. The distinguishing
brainana field is ``TemplateSource``, which records how the output space was resolved —
a bundled spec (e.g. ``NMT2Sym:res-05``) or a user-supplied custom template file path.

brainana has no global provenance graph, so ``Sources`` is populated per-process from the
inputs/transforms a step actually consumed; it is not a stitched end-to-end ancestry.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..version import get_version
from .bids import create_bids_sidecar_filename
from .logger import get_logger
from .nextflow import save_metadata
from .templates import is_custom_template_path, space_label_for

logger = get_logger(__name__)

# The BIDS release the derivatives declare. Kept equal to the version
# validate_bids tells users to put in their *input* dataset_description.json
# (discover_bids_for_nextflow.py), so brainana asks nothing of its inputs that its
# own output does not state.
BIDS_VERSION = "1.8.0"


def _generated_by() -> List[Dict[str, str]]:
    """The BIDS ``GeneratedBy`` block identifying this pipeline + version."""
    return [{"Name": "brainana", "Version": get_version()}]


# Registration-engine token (as recorded in step metadata) -> display name for the
# transform sidecar ``GeneratedBy``. Tokens come from operations/registration.py
# (ants_register) and operations/preprocessing.py (conform_to_template rigid_method).
ENGINE_DISPLAY = {
    "flirt": "FLIRT",
    "sitk": "SimpleITK",
    "simpleitk": "SimpleITK",
    "fireants": "FireANTs",
    "ants": "ANTs",
    "antspyx": "ANTsPy",
}


@lru_cache(maxsize=None)
def _engine_version(engine: str) -> Optional[str]:
    """Best-effort version string for a registration engine, or None.

    In-process package versions (SimpleITK/antspyx/FireANTs) are cheap; FLIRT/ANTs use
    the existing external-tool version helpers. Any failure returns None (Version omitted).
    Cached so a task pays at most one lookup per engine.
    """
    key = str(engine).strip().lower()
    try:
        if key in ("sitk", "simpleitk"):
            import SimpleITK

            return str(SimpleITK.__version__)
        if key == "antspyx":
            import ants

            return str(ants.__version__)
        if key == "fireants":
            from importlib.metadata import version

            return str(version("fireants"))
        if key in ("flirt", "ants"):
            from ..environment import extract_version_number, get_command_version

            cmd = "flirt -version" if key == "flirt" else "antsRegistration --version"
            return extract_version_number(get_command_version(cmd)) or None
    except Exception:
        return None
    return None


def engine_generated_by(engine: str) -> List[Dict[str, str]]:
    """``GeneratedBy`` block naming the real registration engine (+ version if resolvable).

    Unknown/empty tokens fall back to the brainana block so a sidecar is never left
    without provenance.
    """
    if not engine:
        return _generated_by()
    key = str(engine).strip().lower()
    name = ENGINE_DISPLAY.get(key)
    if name is None:
        return _generated_by()
    entry: Dict[str, str] = {"Name": name}
    ver = _engine_version(key)
    if ver:
        entry["Version"] = ver
    return [entry]


def template_source_block(
    output_space: Optional[str], resolved_template_path: Optional[Union[str, Path]]
) -> Dict[str, Any]:
    """Build the ``TemplateSource`` record for a template-space derivative.

    Args:
        output_space: The ``template.output_space`` value as given — a bundled spec
                      (``NMT2Sym:res-05``) or a custom template file path.
        resolved_template_path: Absolute path the spec/path resolved to (may be None if
                      the caller could not resolve it).
    """
    block: Dict[str, Any] = {
        "OutputSpace": str(output_space) if output_space else None,
        "TemplatePath": str(resolved_template_path) if resolved_template_path else None,
        "Custom": is_custom_template_path(output_space),
    }
    return block


def _filename_is_template_space(image_name: str, template_label: str) -> bool:
    """Whether ``image_name`` carries the template label as a space/from/to entity.

    Matches ``space-<label>``, ``from-<label>`` or ``to-<label>`` (transform files encode
    the target space in ``from-``/``to-``). Native spaces (``space-T1w``, ``space-scanner``,
    ``space-bold``) never match the template label unless the user explicitly requested them.
    """
    if not template_label:
        return False
    pattern = rf"(?:^|_)(?:space|from|to)-{re.escape(template_label)}(?:_|\.|$)"
    return re.search(pattern, image_name) is not None


def write_derivative_sidecar(
    image_path: Union[str, Path],
    *,
    output_space: Optional[str] = None,
    resolved_template_path: Optional[Union[str, Path]] = None,
    sources: Optional[List[str]] = None,
    skull_stripped: Optional[bool] = None,
    roi_type: Optional[str] = None,
    task_name: Optional[str] = None,
    repetition_time: Optional[float] = None,
    slice_timing_corrected: Optional[bool] = None,
    include_template_source: Optional[bool] = None,
    engine: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a JSON sidecar next to a derivative data file and return its path.

    The sidecar basename mirrors ``image_path`` (data extension swapped for ``.json``).
    ``TemplateSource`` is included when the file lives in the output template space — by
    default this is auto-detected from the filename (``space-``/``from-``/``to-`` matching
    the resolved template label); pass ``include_template_source`` to force it on/off.

    ``engine`` names the registration backend that produced a transform file (``flirt``,
    ``sitk``, ``fireants``, ``ants``, ``antspyx``) — for xfm sidecars ``GeneratedBy`` then
    names that engine instead of brainana. When omitted/unknown, ``GeneratedBy`` is brainana.

    Only non-None fields are written, so native-space files naturally get the minimal
    fMRIPrep set (e.g. ``SkullStripped``/``Type``/``Sources``) without a template block.
    """
    image_path = Path(image_path)
    sidecar_path = image_path.parent / create_bids_sidecar_filename(image_path)

    fields: Dict[str, Any] = {}

    template_label = space_label_for(output_space)
    if include_template_source is None:
        include_template_source = _filename_is_template_space(
            image_path.name, template_label
        )
    if include_template_source:
        fields["SpatialReference"] = template_label
        fields["TemplateSource"] = template_source_block(
            output_space, resolved_template_path
        )

    if skull_stripped is not None:
        fields["SkullStripped"] = bool(skull_stripped)
    if roi_type is not None:
        fields["Type"] = roi_type
    if task_name is not None:
        fields["TaskName"] = task_name
    if repetition_time is not None:
        fields["RepetitionTime"] = repetition_time
    if slice_timing_corrected is not None:
        fields["SliceTimingCorrected"] = bool(slice_timing_corrected)
    if sources:
        fields["Sources"] = list(sources)
    if extra:
        fields.update(extra)

    fields["GeneratedBy"] = engine_generated_by(engine) if engine else _generated_by()

    save_metadata(fields, sidecar_path)
    logger.debug(f"System: wrote derivative sidecar - {sidecar_path}")
    return sidecar_path


def bold_timeseries_fields(
    bids_name: Union[str, Path], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Sidecar kwargs (``repetition_time``/``slice_timing_corrected``) for a 4D preproc BOLD.

    Reads the raw BOLD json for ``bids_name`` (a full path into the raw dataset) and mirrors
    the slice-timing-correction gate the pipeline itself applies, so the sidecar reflects what
    actually happened to this run. Returns a splat-ready dict for ``write_derivative_sidecar``;
    ``repetition_time`` is omitted when the raw metadata has no TR.
    """
    from ..config.bids_adapter import _is_valid_slice_timing_data
    from .bids import find_bids_metadata

    p = Path(bids_name)
    dataset_dir = next(
        (parent.parent for parent in p.parents if parent.name.startswith("sub-")), None
    )
    if dataset_dir is None:
        dataset_dir = p.parent.parent.parent.parent  # legacy session-layout fallback
    meta = find_bids_metadata(p, dataset_dir) or {}

    tr = meta.get("RepetitionTime")
    # Default True to match the pipeline gate (defaults.yaml func.slice_timing_correction
    # .enabled = true, and func_slice_timing_correction defaults it to True); otherwise a
    # config missing the key would mis-record STC as not applied.
    stc_enabled = bool(
        config.get("func", {}).get("slice_timing_correction", {}).get("enabled", True)
    )
    fields: Dict[str, Any] = {
        "slice_timing_corrected": stc_enabled
        and _is_valid_slice_timing_data(meta.get("SliceTiming"), tr)
    }
    if tr is not None:
        fields["repetition_time"] = tr
    return fields


def write_dataset_description(
    output_dir: Union[str, Path],
    *,
    output_space: Optional[str] = None,
    resolved_template_path: Optional[Union[str, Path]] = None,
    name: str = "brainana derivatives",
) -> Path:
    """Write ``dataset_description.json`` at the derivatives root and return its path.

    Records the pipeline (``GeneratedBy``) and the run's template source once for the
    whole dataset.

    ``BIDSVersion`` is required at a dataset root, and the pipeline says so to its
    users -- ``validate_bids`` tells them their *input* dataset needs it. Omitting
    it from brainana's own output while declaring ``DatasetType: derivative`` left
    the output failing a rule the pipeline enforces on its inputs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dd_path = output_dir / "dataset_description.json"

    description: Dict[str, Any] = {
        "Name": name,
        "BIDSVersion": BIDS_VERSION,
        "DatasetType": "derivative",
        "GeneratedBy": _generated_by(),
        "TemplateSource": template_source_block(output_space, resolved_template_path),
    }

    save_metadata(description, dd_path)
    logger.debug(f"System: wrote dataset_description.json - {dd_path}")
    return dd_path
