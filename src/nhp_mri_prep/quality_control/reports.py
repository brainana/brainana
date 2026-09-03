"""
Quality Control Report Generation

This module generates comprehensive HTML reports summarizing preprocessing results,
including embedded snapshots, processing parameters, and quality metrics.
"""

import os
import json
import logging
import html
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Iterator
import re

from ..utils.bids import parse_bids_entities, BIDS_ENTITY_ORDER
from ..config.config_io import get_nested_config_value
from .run_status import render_run_status_content
from .data_findings import (
    collect_ingest_findings,
    has_findings,
    render_data_findings_content,
)


# Configuration constants
SNAPSHOT_MAPPINGS = {
    "conformFullFOV": {
        "key": "conform_fullfov_overlay",
        "description": "Conform to template space — full field of view",
    },
    "conform": {"key": "conform_overlay", "description": "Conform to template space"},
    "biascorrect": {
        "key": "bias_correction_comparison",
        "description": "Bias field correction",
    },
    "atlasSegmentation": {
        "key": "atlas_segmentation_overlay",
        "description": "Atlas segmentation",
    },
    "anat2template": {
        "key": "anat2template_registration_overlay",
        "description": "Structural to template registration",
    },
    "func2anat": {
        "key": "func2anat_registration_overlay",
        "description": "Functional to anatomical registration",
    },
    "func2target": {
        "key": "func2target_registration_overlay",
        "description": "Functional to target registration",
    },
    "T2w2T1w": {
        "key": "T2w2T1w_registration_overlay",
        "description": "T2w to T1w coregistration",
    },
    "T2w2template": {
        "key": "T2w2template_registration_overlay",
        "description": "T2w to template registration",
    },
    "T1wT2wCombined": {
        "key": "t1wt2w_combined_comparison",
        "description": "T1wT2wCombined comparison",
    },
    "sescoreg": {
        "key": "func_coreg_overlay",
        "description": "Within-session functional coregistration",
    },
    "tSNR": {"key": "tsnr_boldmap", "description": "tSNR map"},
    "motion": {"key": "motion_parameters", "description": "Motion parameters"},
    "confounds": {
        "key": "confounds_timeseries",
        "description": "Confound regressors",
    },
    "surfReconTissueSeg": {
        "key": "surf_recon_tissue_seg_overlay",
        "description": "Surface reconstruction tissue segmentation",
    },
    "corticalSurfAndMeasures": {
        "key": "cortical_surf_and_measures_overlay",
        "description": "Cortical surface and measures",
    },
    "skullstrip": {"key": "skullstrip_overlay", "description": "Skullstripping"},
}

# The motion and confounds figures are stacked on the same frame axis and shade the same frames, so
# they share this sentence verbatim rather than keeping two copies that could drift apart.
_FRAME_SHADING_CAPTION = (
    "Gray bands mark non-steady-state (dummy) volumes at the start of the run; red bands mark "
    "frames flagged as motion outliers by the FD / standardized-DVARS thresholds. Both are "
    "indicator regressors only, NO volumes are removed from the BOLD data."
)

# Figure descriptions shown above the figure (same font style as "Get figure file"); first letter auto-capitalized.
# Key by desc; for 'conform' use (desc, modality) because anatomical vs functional differ.
# A "<br>" starts a new line in the rendered caption; these strings are injected as HTML.
FIGURE_DESCRIPTIONS = {
    "conformFullFOV": (
        "rigid registered T1w at its full field of view (underlaid); template space (contour)."
        "<br>Lavender box marks the field of view every downstream step actually uses."
    ),
    "conform": {
        "anatomical": (
            "lavender box of the above full-field-of-view figure, enlarged. "
            "Every downstream step is built from this."
        ),
        "functional": "rigid registered BOLD (underlaid); target space (contour)",
    },
    # The anatomical caption above is written as the second half of a pair. When the
    # full-FOV figure is not in the report -- no expansion was needed, its inputs were
    # missing, or rendering it failed -- the back-reference would cite a figure that is
    # not there, so the conform figure describes itself instead.
    "conformStandalone": "rigid registered T1w (underlaid); template space (contour)",
    "anat2template": "registered T1w (underlaid); template space (contour)",
    "atlasSegmentation": "ARM2: CHARM level 2 parcellation in cortex and SARM level 2 parcellation in subcortex",
    "surfReconTissueSeg": "White surface (blue contour); pial surface (red contour)",
    "T2w2T1w": "rigid registered T2w (underlaid); T1w space (contour)",
    "T2w2template": "registered T2w (underlaid); template space (contour)",
    "func2anat": "registered BOLD (underlaid); T1w space (contour)",
    "func2target": "registered BOLD (underlaid); target space (contour)",
    "sescoreg": "within-session func run coregistration",
    "tSNR": "session-average temporal SNR map (volume; surface projection if available)",
    "motion": (
        "Rigid-body head motion estimated by motion correction: rotations (degrees) and "
        "translations (mm) about the x, y, and z axes."
        f"<br>{_FRAME_SHADING_CAPTION}"
    ),
    "confounds": (
        "Confound regressors: global signal (GS), CSF, white matter (WM), DVARS, and framewise "
        "displacement (FD). CSF and WM are included only when T1w is available; DVARS and FD are "
        "undefined at frame 0."
        f"<br>{_FRAME_SHADING_CAPTION}"
    ),
}

# Consecutive figure pairs that are two views of a SINGLE processing step. The second figure of a
# pair is rendered without the divider rule that normally separates steps. Keyed by snapshot_type,
# and matched only when the two are actually adjacent, so a pair broken up by a missing figure
# still gets its normal separator.
SAME_STEP_FIGURE_PAIRS = {
    # Full-FOV conform and the cropped conform are one conform step: the second is the first's
    # lavender box, enlarged.
    ("conform_fullfov_overlay", "conform_overlay"),
}

SNAPSHOT_ORDER = [
    "conform_fullfov_overlay",
    "conform_overlay",
    "skullstrip_overlay",
    "atlas_segmentation_overlay",
    "bias_correction_comparison",
    "anat2template_registration_overlay",
    "T2w2T1w_registration_overlay",
    "t1wt2w_combined_comparison",
    "T2w2template_registration_overlay",
    "surf_recon_tissue_seg_overlay",
    "cortical_surf_and_measures_overlay",
    "func_coreg_overlay",  # Within-session coregistration (appears before run-specific snapshots)
    "tsnr_boldmap",
    "func2anat_registration_overlay",  # Functional to anatomical (intermediate step in sequential transforms)
    "func2target_registration_overlay",
    "motion_parameters",
    "confounds_timeseries",
]

SNAPSHOT_ORDER_INDEX = {key: index for index, key in enumerate(SNAPSHOT_ORDER)}

# Full sequential stage labels used in anat-to-template registration sentences.
_XFM_STAGE_LABELS: Dict[str, str] = {
    "translation": "translation-only",
    "rigid": "rigid",
    "affine": "translation, rigid, and affine",
    "syn": "translation, rigid, affine, and non-linear (SyN)",
}

# Abbreviated transform labels used in parenthetical func coregistration sentences.
_XFM_COREG_LABELS: Dict[str, str] = {
    "translation": "translation",
    "rigid": "rigid",
    "affine": "rigid and affine",
    "syn": "rigid, affine, and SyN",
}

# APA 7 references for the Methods section. Key = exact in-text citation (Author et al., YEAR).
# Only references whose key appears in the methods body are included in the report.
# Full reference: docs_temp/paper/methods_reference.md
_METHODS_REFERENCE_MAP = {
    "Avants et al., 2008": (
        "Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2008). Symmetric diffeomorphic "
        "image registration with cross-correlation: Evaluating automated labeling of elderly and "
        "neurodegenerative brain. Medical Image Analysis, 12(1), 26–41. "
        "https://doi.org/10.1016/j.media.2007.06.004"
    ),
    "Cox, 1996": (
        "Cox, R. W. (1996). AFNI: Software for analysis and visualization of functional magnetic "
        "resonance neuroimages. Computers and Biomedical Research, 29(3), 162–173. "
        "https://doi.org/10.1006/cbmr.1996.0014"
    ),
    "Cox & Hyde, 1997": (
        "Cox, R. W., & Hyde, J. S. (1997). Software tools for analysis and visualization of fMRI "
        "data. NMR in Biomedicine, 10(4–5), 171–178. "
        "https://doi.org/10.1002/(SICI)1099-1492(199706/08)10:4/5<171::AID-NBM453>3.0.CO;2-L"
    ),
    "Dale et al., 1999": (
        "Dale, A. M., Fischl, B., & Sereno, M. I. (1999). Cortical surface-based analysis: "
        "Segmentation and surface reconstruction. NeuroImage, 9(2), 179–194. "
        "https://doi.org/10.1006/nimg.1998.0395"
    ),
    "Henschel et al., 2020": (
        "Henschel, L., Conjeti, S., Estrada, S., Diers, K., Fischl, B., & Reuter, M. (2020). "
        "FastSurfer: A fast and accurate deep learning based neuroimaging pipeline. "
        "NeuroImage, 219, 117012. https://doi.org/10.1016/j.neuroimage.2020.117012"
    ),
    "Jenkinson et al., 2002": (
        "Jenkinson, M., Bannister, P., Brady, M., & Smith, S. (2002). Improved optimization for the "
        "robust and accurate linear registration and motion correction of brain images. "
        "NeuroImage, 17(2), 825–841. https://doi.org/10.1006/nimg.2002.1132"
    ),
    "Jena et al., 2024": (
        "Jena, R., Chaudhari, P., & Gee, J. C. (2024). FireANTs: Adaptive Riemannian optimization "
        "for multi-scale diffeomorphic registration. Nature Communications."
    ),
    "Jena et al., 2026": (
        "Jena, R., Zope, V., Chaudhari, P., & Gee, J. C. (2026). A scalable distributed framework for "
        "multimodal GigaVoxel image registration. The Fourteenth International Conference on Learning "
        "Representations. https://openreview.net/forum?id=8dLexnao2h"
    ),
    "Jung et al., 2021": (
        "Jung, B., Taylor, P. A., Seidlitz, J., Suber, A., Donahue, C. J., Coalson, T., Glasser, "
        "M. F., Shafer, A. T., Van Essen, D. C., Dienes, T., Earl, E., Feczko, E., Fair, D. A., & "
        "Donahue, J. N. (2021). A comprehensive macaque fMRI pipeline and hierarchical atlas. "
        "NeuroImage, 235, 117997. https://doi.org/10.1016/j.neuroimage.2021.117997"
    ),
    "Tustison et al., 2010": (
        "Tustison, N. J., Avants, B. B., Cook, P. A., Zheng, Y., Egan, A., Yushkevich, P. A., & "
        "Gee, J. C. (2010). N4ITK: Improved N3 bias correction. IEEE Transactions on Medical "
        "Imaging, 29(6), 1310–1320. https://doi.org/10.1109/TMI.2010.2046908"
    ),
    "Wang et al., 2021": (
        "Wang, X., Li, X., & Xu, T. (2021). U-net model for brain extraction: Trained on humans for "
        "transfer to non-human primates. NeuroImage, 235, 118001. "
        "https://doi.org/10.1016/j.neuroimage.2021.118001"
    ),
}


def _cited_references(methods_body: str) -> List[str]:
    """Return full reference strings for all citation keys that appear in methods_body, sorted alphabetically by first author."""
    seen: set = set()
    for match in re.findall(r"\(([^)]+)\)", methods_body):
        for part in match.split(";"):
            key = part.strip()
            if key in _METHODS_REFERENCE_MAP:
                seen.add(key)
    refs = [_METHODS_REFERENCE_MAP[k] for k in seen]
    return sorted(refs)


def _iter_leaf_snapshots(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield leaf dicts in a nested snapshot hierarchy (each has a ``path`` key)."""
    for value in data.values():
        if isinstance(value, dict):
            if "path" in value:
                yield value
            else:
                yield from _iter_leaf_snapshots(value)


def _anat_suffix(
    filename: str, entities: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Return ``'T1w'``, ``'T2w'``, or ``None`` from filename ending and BIDS entities."""
    ent = entities or {}
    if filename.endswith("_T2w.png") or ent.get("suffix") == "T2w":
        return "T2w"
    if filename.endswith("_T1w.png") or ent.get("suffix") == "T1w":
        return "T1w"
    return None


def _snapshot_sort_key(snapshot: Dict[str, Any]) -> tuple:
    """Sort key: QC figure order, then T1w before T2w, then filename."""
    base_order = SNAPSHOT_ORDER_INDEX.get(snapshot.get("snapshot_type", ""), 999)
    filename = snapshot.get("filename", "")
    suf = _anat_suffix(filename, snapshot.get("entities", {}))
    modality_order = 1 if suf == "T2w" else 0
    return (base_order, modality_order, filename)


def _snapshot_sort_key_hierarchy_item(item: tuple) -> tuple:
    """Like ``_snapshot_sort_key`` but third component is the hierarchy dict key (stable ordering)."""
    name, snapshot_info = item
    filename = Path(snapshot_info["path"]).name
    b, m, _ = _snapshot_sort_key(
        {
            "snapshot_type": snapshot_info.get("snapshot_type", ""),
            "entities": snapshot_info.get("entities", {}),
            "filename": filename,
        }
    )
    return (b, m, name)


class BidsEntityProcessor:
    """Handles all BIDS entity processing operations."""

    @staticmethod
    def extract_entities_from_snapshots(data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract all unique BIDS entity combinations from snapshot hierarchy."""
        entities_list = []

        def collect_entities(level_data: Dict[str, Any]) -> None:
            for value in level_data.values():
                if isinstance(value, dict):
                    if "entities" in value:
                        entities = {
                            k: v
                            for k, v in value["entities"].items()
                            if k not in ["sub", "desc", "space"]
                        }
                        if entities and entities not in entities_list:
                            entities_list.append(entities)
                    else:
                        collect_entities(value)

        collect_entities(data)
        return sorted(
            entities_list, key=lambda x: [x.get(k, "") for k in BIDS_ENTITY_ORDER]
        )

    @staticmethod
    def create_display_text(entities: Dict[str, str]) -> str:
        """Create human-readable display text from BIDS entities."""
        parts = []
        for entity in BIDS_ENTITY_ORDER:
            if entity in entities:
                if entity == "ses":
                    parts.append(
                        f'session <span class="bids-entity">{entities[entity]}</span>'
                    )
                elif entity == "task":
                    parts.append(
                        f'task <span class="bids-entity">{entities[entity]}</span>'
                    )
                elif entity == "run":
                    parts.append(
                        f'run <span class="bids-entity">{entities[entity]}</span>'
                    )
                else:
                    parts.append(
                        f'{entity} <span class="bids-entity">{entities[entity]}</span>'
                    )

        return ", ".join(parts)

    @staticmethod
    def clean_header_id(text: str) -> str:
        """Clean text to create valid HTML ID."""
        clean_text = re.sub(r"<[^>]+>", "", text)
        clean_text = re.sub(r"[^a-zA-Z0-9-]", "-", clean_text.lower())
        return re.sub(r"-+", "-", clean_text).strip("-")


class SnapshotProcessor:
    """Handles snapshot discovery, parsing, and organization."""

    @staticmethod
    def discover_and_parse(
        snapshot_dir: Path,
        logger: logging.Logger,
        provided_paths: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Discover and parse all snapshots in one pass."""
        if provided_paths:
            snapshot_files = {name: Path(path) for name, path in provided_paths.items()}
            logger.info(f"QC: using {len(snapshot_files)} provided snapshot paths")
        else:
            png_files = list(snapshot_dir.glob("*.png"))
            logger.info(f"QC: auto-discovered {len(png_files)} PNG files")
            snapshot_files = {
                SnapshotProcessor._create_key(f.name): f for f in png_files
            }

        snapshots = {}
        available_entities = {key: set() for key in BIDS_ENTITY_ORDER if key != "sub"}

        # Decided over the whole set before captioning any single figure: the anatomical
        # conform caption points at the full-FOV figure, which is optional.
        has_full_fov_figure = any(
            parse_bids_entities(path.name).get("desc") == "conformFullFOV"
            for path in snapshot_files.values()
        )

        for name, path in snapshot_files.items():
            entities = parse_bids_entities(path.name)

            # Collect available entities
            for entity_key in BIDS_ENTITY_ORDER:
                if entity_key != "sub" and entity_key in entities:
                    available_entities[entity_key].add(entities[entity_key])

            desc = entities.get("desc", "")
            mapping = SNAPSHOT_MAPPINGS.get(desc, {})
            snapshot_type = mapping.get("key", desc)

            # Determine modality first
            modality = SnapshotProcessor._determine_modality(path.name)

            # Customize description based on modality for conform snapshots
            description = mapping.get("description", "")
            if desc == "conform" and modality == "functional":
                description = "Conform to target space"

            # Figure description (underlaid/contour text) for QC report
            figure_desc_entry = FIGURE_DESCRIPTIONS.get(desc)
            if isinstance(figure_desc_entry, dict):
                figure_description = figure_desc_entry.get(modality, "")
            else:
                figure_description = (
                    figure_desc_entry if isinstance(figure_desc_entry, str) else ""
                )
            if (
                desc == "conform"
                and modality == "anatomical"
                and not has_full_fov_figure
            ):
                figure_description = FIGURE_DESCRIPTIONS["conformStandalone"]

            # Store the filename separately for reliable path construction
            snapshots[name] = {
                "path": str(path),
                "filename": path.name,
                "entities": entities,
                "modality": modality,
                "description": description,
                "snapshot_type": snapshot_type,
                "figure_description": figure_description,
            }

        # Convert sets to sorted lists
        for key in available_entities:
            available_entities[key] = sorted(available_entities[key])

        logger.info(f"QC: parsed {len(snapshots)} snapshots")
        return {"snapshots": snapshots, "available_entities": available_entities}

    @staticmethod
    def _create_key(filename: str) -> str:
        """Create snapshot key from filename."""
        entities = parse_bids_entities(filename)
        desc = entities.get("desc", "")
        mapping = SNAPSHOT_MAPPINGS.get(desc, {})
        base_name = mapping.get("key", desc)

        key_parts = [base_name]
        for entity_key in BIDS_ENTITY_ORDER:
            if entity_key != "sub" and entity_key in entities:
                key_parts.append(f"{entity_key}-{entities[entity_key]}")

        # Add modality suffix to avoid key collisions between T1w/T2w
        suf = _anat_suffix(filename, entities)
        if suf == "T1w":
            key_parts.append("T1w")
        elif suf == "T2w":
            key_parts.append("T2w")
        elif filename.endswith("_bold.png") or filename.endswith("_boldref.png"):
            key_parts.append("bold")

        return "_".join(key_parts)

    @staticmethod
    def _determine_modality(name: str) -> str:
        """Determine modality from filename."""
        # Check for functional first (bold or boldref suffix), as functional files can contain
        # space-T1w or space-T2w entities which would otherwise be misclassified
        if (
            name.lower().endswith("_bold.png")
            or "_bold.png" in name.lower()
            or name.lower().endswith("_boldref.png")
            or "_boldref.png" in name.lower()
        ):
            return "functional"
        # Check for anatomical in suffix position (e.g., _T1w.png, _T2w.png)
        # This avoids false positives from space-T1w or space-T2w entities
        elif name.endswith("_T1w.png") or name.endswith("_T2w.png"):
            return "anatomical"
        # Fallback: check if T1w/T2w appears as a suffix pattern (before extension)
        elif "_T1w." in name or "_T2w." in name:
            return "anatomical"
        elif "fmap" in name.lower():
            return "field_mapping"
        else:
            return "summary"

    @staticmethod
    def organize_by_hierarchy(
        snapshots: Dict[str, Any],
        snapshot_dir: Path,
        report_path: Path,
        logger: logging.Logger,
    ) -> Dict[str, Any]:
        """Organize snapshots by BIDS hierarchy."""
        organized = {
            "anatomical": {},
            "functional": {},
            "field_mapping": {},
            "summary": {},
        }

        # Calculate relative path from report parent to snapshot directory
        # snapshot_dir is the published path: /full/path/to/output/sub-XXX/figures
        # report_path might be relative (work directory) or absolute (published)
        # We need the published report path for correct relative path calculation
        # Derive it from snapshot_dir: if snapshot_dir is /path/to/output/sub-XXX/figures,
        # then published report is /path/to/output/sub-XXX.html

        snapshot_dir_str = str(snapshot_dir)
        report_path_str = str(report_path)

        # If report_path is relative, derive published path from snapshot_dir
        if not os.path.isabs(report_path_str):
            # snapshot_dir is like: /path/to/output/sub-XXX/figures
            # published report is: /path/to/output/sub-XXX.html
            snapshot_dir_path = Path(snapshot_dir_str)
            # Get parent (sub-XXX) and then parent again (output directory)
            output_dir = snapshot_dir_path.parent.parent
            report_filename = Path(report_path_str).name  # e.g., "sub-XXX.html"
            published_report_path = output_dir / report_filename
            report_parent = str(published_report_path.parent)
        else:
            # report_path is already absolute (published path)
            report_parent = str(Path(report_path_str).parent)

        # Calculate relative path using string paths (don't resolve to avoid work directory issues)
        try:
            report_to_snapshot_dir = os.path.relpath(snapshot_dir_str, report_parent)
        except ValueError:
            # If paths are on different drives (Windows) or can't be made relative,
            # extract the relative portion manually
            # Both paths should share a common prefix up to the output directory
            report_parent_parts = Path(report_parent).parts
            snapshot_dir_parts = Path(snapshot_dir_str).parts

            # Find common prefix
            common_parts = []
            for r_part, s_part in zip(report_parent_parts, snapshot_dir_parts):
                if r_part == s_part:
                    common_parts.append(r_part)
                else:
                    break

            # Calculate relative path: go up from report_parent, then down to snapshot_dir
            up_levels = len(report_parent_parts) - len(common_parts)
            down_parts = snapshot_dir_parts[len(common_parts) :]

            if up_levels > 0 and down_parts:
                report_to_snapshot_dir = os.path.join(
                    *([".."] * up_levels + list(down_parts))
                )
            elif down_parts:
                report_to_snapshot_dir = os.path.join(*down_parts)
            else:
                report_to_snapshot_dir = "."

        for name, snapshot_info in snapshots.items():
            # Get filename (stored separately for reliability, or extract from path)
            filename = snapshot_info.get("filename", Path(snapshot_info["path"]).name)

            # Construct relative path: from report_parent to snapshot_dir, then filename
            if report_to_snapshot_dir == ".":
                relative_path = filename
            else:
                relative_path = os.path.join(report_to_snapshot_dir, filename)

            snapshot_data = {
                "path": relative_path,
                "entities": snapshot_info["entities"],
                "description": snapshot_info["description"],
                "snapshot_type": snapshot_info["snapshot_type"],
                "figure_description": snapshot_info.get("figure_description", ""),
            }

            modality = snapshot_info["modality"]
            SnapshotProcessor._add_to_hierarchy(
                organized[modality], snapshot_data, snapshot_info["entities"], name
            )

        SnapshotProcessor._sort_hierarchy(organized)
        logger.info("QC: organized snapshots by BIDS hierarchy with relative paths")
        return organized

    @staticmethod
    def _add_to_hierarchy(root: Dict, snapshot_data: Dict, entities: Dict, name: str):
        """Add snapshot to hierarchical structure."""
        current = root
        for entity_key in BIDS_ENTITY_ORDER:
            if entity_key != "sub" and entity_key in entities:
                value = entities[entity_key]
                if value not in current:
                    current[value] = {}
                current = current[value]
        current[name] = snapshot_data

    @staticmethod
    def _sort_hierarchy(data: Dict):
        """Sort hierarchical structure recursively."""
        for key, value in data.items():
            if isinstance(value, dict) and "path" not in value:
                if any(
                    isinstance(v, dict) and "snapshot_type" in v for v in value.values()
                ):
                    # Sort by snapshot order, with special handling for T1w/T2w modality order
                    sorted_items = sorted(
                        value.items(), key=_snapshot_sort_key_hierarchy_item
                    )
                else:
                    # For non-snapshot items, ensure T1w comes before T2w
                    def _nonsnap_sort_key(item):
                        name, _ = item
                        if name.endswith("_T1w.png") or "_T1w" in name:
                            return (0, name)
                        if name.endswith("_T2w.png") or "_T2w" in name:
                            return (1, name)
                        return (2, name)

                    sorted_items = sorted(value.items(), key=_nonsnap_sort_key)

                data[key] = dict(sorted_items)
                SnapshotProcessor._sort_hierarchy(data[key])


class HtmlGenerator:
    """Handles all HTML generation operations."""

    @staticmethod
    def create_navigation_menu(
        organized_snapshots: Dict[str, Any],
        has_run_status: bool = False,
        has_data_findings: bool = False,
    ) -> str:
        """Create navigation menu (flat top bar; multi-group modalities get a dropdown)."""
        nav_items = ['<a href="#Summary">Summary</a>']

        if has_run_status:
            nav_items.append('<a href="#Status">Run status</a>')

        if has_data_findings:
            nav_items.append('<a href="#DataFindings">Data findings</a>')

        # Add modality sections with dropdowns if they have content
        for modality, title in [
            ("anatomical", "Structural"),
            ("functional", "Functional"),
            ("field_mapping", "B₀ field mapping"),
        ]:
            if organized_snapshots[modality]:
                section_prefix = modality
                groups = HtmlGenerator._group_snapshots_by_entities(
                    organized_snapshots[modality], section_prefix
                )
                # Anatomical returns two-level dict (ses/run -> modality -> list); nav uses top-level keys
                group_keys = list(groups.keys())
                if len(group_keys) > 1:
                    dropdown_items = []
                    for group_key in group_keys:
                        nav_id = f"{section_prefix}-{BidsEntityProcessor.clean_header_id(group_key)}"
                        dropdown_items.append(f'<a href="#{nav_id}">{group_key}</a>')
                    dropdown_content = "\n".join(dropdown_items)
                    nav_items.append(
                        f"""<details class="nav-dd"><summary>{title} ▾</summary>
<div class="menu">
{dropdown_content}
</div></details>"""
                    )
                else:
                    nav_items.append(f'<a href="#{modality.title()}">{title}</a>')

        nav_items.extend(
            [
                '<a href="#About">About</a>',
                '<a href="#Methods">Methods</a>',
            ]
        )

        return "\n".join(nav_items)

    @staticmethod
    def create_section(section_id: str, title: str, content: str) -> str:
        """Create a section with header and content."""
        return f'<div id="{section_id}"><h1 class="section">{title}</h1>{content}</div>'

    @staticmethod
    def create_prose_section(section_id: str, title: str, content: str) -> str:
        """Create a section whose heading and body share the prose measure."""
        return (
            f'<div id="{section_id}">'
            f'<div class="prose"><h1 class="section">{title}</h1>{content}</div>'
            f"</div>"
        )

    @staticmethod
    def create_summary_section(
        report_data: Dict[str, Any], config: Dict[str, Any]
    ) -> str:
        """Create summary section."""
        subject_id = report_data["metadata"]["subject_id"]
        organized = report_data["organized_snapshots"]

        dataset_context = report_data.get("dataset_context", {})
        if "subject_file_counts" in dataset_context:
            subject_file_counts = dict(dataset_context["subject_file_counts"])
        elif "job_file_counts" in dataset_context:
            t1w_count = dataset_context["job_file_counts"]["anatomical"]
            func_count = dataset_context["job_file_counts"]["functional"]
            subject_file_counts = {
                "t1w": t1w_count,
                "t2w": 0,
                "t1w_processed": t1w_count,
                "t2w_processed": 0,
                "functional": func_count,
            }
        else:
            anat_counts = HtmlGenerator._count_anatomical_by_modality(
                organized["anatomical"]
            )
            func_count = HtmlGenerator._count_unique_images(organized["functional"])
            subject_file_counts = {
                "t1w": anat_counts["t1w"],
                "t2w": anat_counts["t2w"],
                "t1w_processed": anat_counts["t1w"],
                "t2w_processed": anat_counts["t2w"],
                "functional": func_count,
            }

        subject_file_counts.setdefault(
            "t1w_processed", subject_file_counts.get("t1w", 0)
        )
        subject_file_counts.setdefault(
            "t2w_processed", subject_file_counts.get("t2w", 0)
        )

        structural_rows = HtmlGenerator._structural_images_summary_rows(
            subject_file_counts
        )
        func_count = subject_file_counts.get("functional", 0)

        # Standard output space: template.output_space (e.g. "NMT2Sym:res-05") -> display template name
        output_space_raw = (
            config.get("template", {}).get("output_space")
            or config.get("output_spaces")
            or "NMT2Sym"
        )
        # Show the output_space as given: the full spec incl. resolution (e.g.
        # "NMT2Sym:res-05") for a bundled template, or the raw file path for a custom one.
        output_space_display = str(output_space_raw) if output_space_raw else "NMT2Sym"

        # Surface reconstruction: show "Run by Brainana" if this report includes surface reconstruction QC
        has_surf = HtmlGenerator._has_surface_recon_snapshots(organized["anatomical"])
        freesurfer_text = "Run by Brainana" if has_surf else "Not applicable"

        rows = (
            [("Subject ID", html.escape(str(subject_id)))]
            + structural_rows
            + [
                ("Functional images", html.escape(str(func_count))),
                ("Output spaces", html.escape(str(output_space_display))),
                ("Surface reconstruction", html.escape(str(freesurfer_text))),
            ]
        )
        kv = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
        content = (
            f'<dl class="kv">{kv}</dl>'
            '<p class="note">Detailed processing parameters: '
            '<a href="nextflow_reports/config.yaml" target="_blank">'
            "<code>./nextflow_reports/config.yaml</code></a>.</p>"
        )

        return HtmlGenerator.create_section("Summary", "Summary", content)

    @staticmethod
    def create_status_section(report_data: Dict[str, Any]) -> str:
        """Create run-status section (pass / pass-with-warnings / early abort).

        Renders nothing when no run status is available (e.g. report generated
        outside the pipeline), so the report degrades gracefully.
        """
        run_status = report_data.get("run_status")
        content = render_run_status_content(run_status)
        if not content:
            return ""
        return HtmlGenerator.create_section("Status", "Run status", content)

    @staticmethod
    def create_data_findings_section(report_data: Dict[str, Any]) -> str:
        """Create the ingest-normalization findings section.

        Deliberately separate from the run-status tier: that tier answers "did the
        pipeline execute", and demoting a green run to "pass with warnings" because
        a header had odd units would train people to ignore the badge. Renders
        nothing when the subject's inputs were well-formed.
        """
        content = render_data_findings_content(report_data.get("data_findings"))
        if not content:
            return ""
        return HtmlGenerator.create_section("DataFindings", "Data findings", content)

    @staticmethod
    def create_modality_section(
        section_id: str, data: Dict[str, Any], title: str = None
    ) -> str:
        """Create modality section with snapshots."""
        if not data:
            return ""

        if title is None:
            title = section_id

        content = HtmlGenerator._render_snapshots(data, section_id.lower())
        return HtmlGenerator.create_section(section_id, title, content)

    @staticmethod
    def _group_chips(group_key: str) -> str:
        """Render a group label as BIDS-style chips (e.g. `ses-001` `run-1` `task-rest`).

        `group_key` is the display text built by BidsEntityProcessor.create_display_text
        (entity values wrapped in <span class="bids-entity">). The odd subject-level group
        ("sub-level") renders as a single readable chip.
        """
        plain = re.sub(r"<[^>]+>", "", group_key).strip()
        if plain.lower() == "sub-level":
            return '<span class="gchip">subject-level</span>'
        abbr = {"session": "ses", "run": "run", "task": "task", "acquisition": "acq"}
        chips = []
        for part in plain.split(", "):
            part = part.strip()
            if not part:
                continue
            bits = part.split(" ", 1)
            if len(bits) == 2:
                key, val = bits
                tag = abbr.get(key.lower(), key.lower())
                chips.append(
                    f'<span class="gchip">{html.escape(tag)}-{html.escape(val)}</span>'
                )
            else:
                chips.append(f'<span class="gchip">{html.escape(part)}</span>')
        return "".join(chips)

    @staticmethod
    def _render_snapshots(data: Dict[str, Any], section_prefix: str) -> str:
        """Render snapshots with grouping."""
        html_parts = []

        # Group snapshots by BIDS entities (two-level for anatomical: ses/run then T1w/T2w)
        snapshot_groups = HtmlGenerator._group_snapshots_by_entities(
            data, section_prefix
        )

        def render_snapshot_blocks(snapshots: List[Dict[str, Any]]) -> None:
            previous_type = ""
            for snapshot_data in snapshots:
                snapshot_id = (
                    f"{section_prefix}-{snapshot_data['filename'].replace('.', '-')}"
                )
                title = snapshot_data.get("description", snapshot_data["filename"])
                path = snapshot_data["path"]
                fig_desc = snapshot_data.get("figure_description", "")
                if fig_desc:
                    fig_desc = fig_desc[0].upper() + fig_desc[1:]
                cap = f'<div class="cap">{fig_desc}</div>' if fig_desc else ""
                # Two views of one processing step run without a rule between them, so the report
                # does not read as two separate steps.
                snapshot_type = snapshot_data.get("snapshot_type", "")
                fig_class = (
                    "fig fig-same-step"
                    if (previous_type, snapshot_type) in SAME_STEP_FIGURE_PAIRS
                    else "fig"
                )
                previous_type = snapshot_type
                html_parts.append(
                    f"""<div class="{fig_class}" id="{snapshot_id}">
<p class="ftitle"><a href="{path}" target="_blank">{title}</a></p>{cap}
<a href="{path}" target="_blank"><img class="svg-reportlet" src="{path}" /></a>
</div>"""
                )

        # Two-level structure (anatomical): ses/run -> T1w/T2w -> snapshots
        first_val = (
            next(iter(snapshot_groups.values()), None) if snapshot_groups else None
        )
        two_level = isinstance(first_val, dict)
        if two_level:
            for group_key, modality_dict in snapshot_groups.items():
                if group_key:
                    header_id = f"{section_prefix}-{BidsEntityProcessor.clean_header_id(group_key)}"
                    html_parts.append(
                        f'<h2 class="group-head" id="{header_id}">{HtmlGenerator._group_chips(group_key)}</h2>'
                    )
                for modality in ("T1w", "T2w"):
                    if modality in modality_dict:
                        html_parts.append(f'<div class="eyebrow">{modality}</div>')
                        render_snapshot_blocks(modality_dict[modality])
        else:
            for group_key, snapshots in snapshot_groups.items():
                if group_key:
                    header_id = f"{section_prefix}-{BidsEntityProcessor.clean_header_id(group_key)}"
                    html_parts.append(
                        f'<h2 class="group-head" id="{header_id}">{HtmlGenerator._group_chips(group_key)}</h2>'
                    )
                render_snapshot_blocks(snapshots)

        return "\n".join(html_parts)

    @staticmethod
    def _group_snapshots_by_entities(
        data: Dict[str, Any], section_prefix: str = ""
    ) -> Dict[str, Any]:
        """Group snapshots by BIDS entities. For anatomical (section_prefix=='anatomical'),
        returns two-level Dict[ses/run_key, Dict[modality, list]]; otherwise flat Dict[group_key, list].
        """
        all_snapshots = []
        for value in _iter_leaf_snapshots(data):
            filename = Path(value["path"]).name
            entities = parse_bids_entities(filename)
            all_snapshots.append(
                {
                    "filename": filename,
                    "path": value["path"],
                    "entities": entities,
                    "description": value.get("description", ""),
                    "snapshot_type": value.get("snapshot_type", ""),
                    "figure_description": value.get("figure_description", ""),
                }
            )

        # Anatomical: two-level grouping (ses/run -> T1w/T2w -> snapshots)
        if section_prefix == "anatomical":
            groups = {}
            for snapshot in all_snapshots:
                entities = snapshot["entities"]
                entities_no_suffix = {
                    k: v
                    for k, v in entities.items()
                    if k not in ["sub", "desc", "space", "suffix"]
                }
                base_group_key = (
                    BidsEntityProcessor.create_display_text(entities_no_suffix)
                    or "sub-level"
                )
                filename = snapshot.get("filename", "")
                # T1wT2wCombined comparison is shown under T2w (between T2w2T1w and T2w2template)
                if snapshot.get("snapshot_type") == "t1wt2w_combined_comparison":
                    modality = "T2w"
                else:
                    suf = _anat_suffix(filename, entities)
                    modality = suf if suf is not None else "T1w"
                if base_group_key not in groups:
                    groups[base_group_key] = {}
                if modality not in groups[base_group_key]:
                    groups[base_group_key][modality] = []
                groups[base_group_key][modality].append(snapshot)
            for base_key in groups:
                for mod in groups[base_key]:
                    groups[base_key][mod].sort(key=_snapshot_sort_key)

            # Sort top-level by session
            def anat_group_sort(item):
                group_name, modality_dict = item
                session_value = ""
                for snap_list in modality_dict.values():
                    if snap_list:
                        session_value = snap_list[0].get("entities", {}).get("ses", "")
                        break
                return (session_value or "", group_name)

            return dict(sorted(groups.items(), key=anat_group_sort))

        # Flat grouping for functional / field_mapping
        groups = {}
        for snapshot in all_snapshots:
            entities = {
                k: v
                for k, v in snapshot["entities"].items()
                if k not in ["sub", "desc", "space"]
            }
            base_group_key = (
                BidsEntityProcessor.create_display_text(entities) or "sub-level"
            )
            filename = snapshot.get("filename", "")
            ent = snapshot.get("entities", {})
            # T1wT2wCombined comparison is shown under T2w
            if snapshot.get("snapshot_type") == "t1wt2w_combined_comparison":
                group_key = (
                    "T2w" if base_group_key == "sub-level" else f"{base_group_key} T2w"
                )
            else:
                suf = _anat_suffix(filename, ent)
                if suf == "T1w":
                    group_key = (
                        "T1w"
                        if base_group_key == "sub-level"
                        else f"{base_group_key} T1w"
                    )
                elif suf == "T2w":
                    group_key = (
                        "T2w"
                        if base_group_key == "sub-level"
                        else f"{base_group_key} T2w"
                    )
                else:
                    group_key = base_group_key
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(snapshot)

        for group_key in groups:
            groups[group_key].sort(key=_snapshot_sort_key)

        def group_sort_key(group_item):
            group_name, snapshots = group_item
            session_value = (
                snapshots[0].get("entities", {}).get("ses", "") if snapshots else ""
            )
            has_task_or_run = any(
                ("task" in s.get("entities", {})) or ("run" in s.get("entities", {}))
                for s in snapshots
            )
            # Session-level groups (e.g., "session 001") should appear before
            # task/run-specific groups within the same session.
            group_level_order = 1 if has_task_or_run else 0
            return (session_value or "", group_level_order, group_name)

        return dict(sorted(groups.items(), key=group_sort_key))

    @staticmethod
    def _count_unique_images(data: Dict[str, Any]) -> int:
        """Count unique functional BOLD runs from organized snapshots (fallback).

        Excludes ``desc`` (QC/processing step), ``sub`` (report subject), and
        ``space`` — derivative QC filenames use ``space`` for the reference
        grid (e.g. func2anat vs func2target), not a distinct acquisition.

        Only snapshots with ``task`` and/or ``run`` are counted so session-level
        QC (within-session coregistration, tSNR, etc.) is not treated as an
        extra BOLD acquisition. Prefer :func:`_count_func_jobs_from_discovery`
        when ``functional_jobs.json`` is available.
        """
        if not data:
            return 0

        unique_images = set()
        _exclude_from_func_identity = frozenset({"desc", "sub", "space"})

        for value in _iter_leaf_snapshots(data):
            if "entities" not in value:
                continue
            if value.get("snapshot_type") == "func_coreg_overlay":
                continue
            entities = value["entities"]
            if "task" not in entities and "run" not in entities:
                continue
            image_id = tuple(
                sorted(
                    (k, v)
                    for k, v in entities.items()
                    if k not in _exclude_from_func_identity
                )
            )
            unique_images.add(image_id)
        return len(unique_images)

    @staticmethod
    def _count_anatomical_by_modality(data: Dict[str, Any]) -> Dict[str, int]:
        """Count unique T1w and T2w images from organized anatomical snapshots."""
        t1w_ids: set = set()
        t2w_ids: set = set()

        if data:
            for value in _iter_leaf_snapshots(data):
                if "entities" not in value:
                    continue
                entities = value["entities"]
                image_id = tuple(
                    sorted(
                        (k, v) for k, v in entities.items() if k not in ["desc", "sub"]
                    )
                )
                filename = Path(value["path"]).name
                if _anat_suffix(filename, entities) == "T2w":
                    t2w_ids.add(image_id)
                else:
                    # T1w or unspecified anatomical (_T1w.png or other)
                    t1w_ids.add(image_id)
        return {"t1w": len(t1w_ids), "t2w": len(t2w_ids)}

    @staticmethod
    def _structural_images_summary_rows(
        subject_file_counts: Dict[str, Any],
    ) -> List[tuple]:
        """Summary (key, value) rows for structural images.

        Compact "Structural images: N T1w, N T2w" when acquired == processed; otherwise
        one "<mod> images" row per modality showing acquired vs after-synthesis counts.
        """
        t1a = subject_file_counts.get("t1w")
        t2a = subject_file_counts.get("t2w")
        if t1a == "N/A" or t2a == "N/A":
            return [("Structural images", "N/A")]

        t1a_i = int(t1a) if t1a is not None else 0
        t2a_i = int(t2a) if t2a is not None else 0
        t1p_i = int(subject_file_counts.get("t1w_processed", t1a_i) or 0)
        t2p_i = int(subject_file_counts.get("t2w_processed", t2a_i) or 0)

        if (t1a_i == t1p_i) and (t2a_i == t2p_i):
            return [("Structural images", f"{t1a_i} T1w, {t2a_i} T2w")]

        rows: List[tuple] = []
        for label, acquired, processed in (
            ("T1w", t1a_i, t1p_i),
            ("T2w", t2a_i, t2p_i),
        ):
            if not acquired and not processed:
                continue
            value = f"Acquired {acquired}"
            if acquired != processed:
                value += f" · After synthesis {processed}"
            rows.append((f"{label} images", value))
        return rows

    @staticmethod
    def _has_surface_recon_snapshots(data: Dict[str, Any]) -> bool:
        """Return True if any anatomical snapshot is from surface reconstruction QC."""
        if not data:
            return False

        for value in _iter_leaf_snapshots(data):
            if "snapshot_type" not in value:
                continue
            st = (value.get("snapshot_type") or "").lower()
            if "surf" in st or "cortical" in st:
                return True
        return False

    @staticmethod
    def create_about_section(report_data: Dict[str, Any]) -> str:
        """Create about section."""
        metadata = report_data["metadata"]
        content = f"""<p>This report was generated by <strong>{metadata["pipeline_name"]}</strong> version <strong>{metadata["version"]}</strong>.</p>
<p>Generated on: {metadata["generation_time"]}</p>"""
        return HtmlGenerator.create_prose_section("About", "About", content)

    @staticmethod
    def _conform_sentence(
        data_label: str, space_label: str, skull_enabled: bool
    ) -> str:
        """Return the conform-to-space sentence, shared between T1w and fMRI preprocessing."""
        if skull_enabled:
            return (
                f"The {data_label} was conformed to {space_label} to ensure better performance "
                "of the subsequent steps: first, initial skullstripping was performed using a CNN "
                "model fine-tuned from DeepBet (Wang et al., 2021), then rigid registration to the "
                f"{space_label} brain was performed with FLIRT (FSL; Jenkinson et al., 2002)."
            )
        return (
            f"The {data_label} was conformed to {space_label} via rigid registration with FLIRT "
            "(FSL; Jenkinson et al., 2002) to ensure better performance of the subsequent steps."
        )

    @staticmethod
    def _build_anat_methods_paragraph(config: Dict[str, Any], has_t2w: bool) -> tuple:
        """Build anatomical preprocessing paragraphs dynamically from config.

        Returns:
            (t1w_text, t2w_text): t2w_text is None when has_t2w is False.
        """
        # --- T1w paragraph ---
        t1w = [
            "T1w images were preprocessed as follows. "
            "When multiple T1w images existed per session or subject, a single synthesized T1w was "
            "created by rigid coregistration to the first image using ANTs "
            "(Avants et al., 2008) and averaging in reference space."
        ]

        skull_enabled = bool(
            get_nested_config_value(
                config, "anat.skullstripping_segmentation.enabled", True
            )
        )

        if get_nested_config_value(config, "anat.conform.enabled", True):
            t1w.append(
                HtmlGenerator._conform_sentence("T1w", "template space", skull_enabled)
            )

        if skull_enabled:
            t1w.append(
                "Brain tissue segmentation and brain mask generation were performed using a "
                "CNN fine-tuned from FastSurfer one (Henschel et al., 2020) "
                "and trained on macaque brain atlases (CHARM/SARM level 2; Jung et al., 2021)."
            )

        if get_nested_config_value(config, "anat.bias_correction.enabled", True):
            t1w.append(
                "The T1w was corrected for intensity non-uniformity with "
                "N4BiasFieldCorrection (Tustison et al., 2010), "
                "using the brain mask to restrict the correction."
            )

        xfm_type = (
            get_nested_config_value(
                config, "registration.anat2template_xfm_type", "syn"
            )
            or "syn"
        ).lower()
        stage = _XFM_STAGE_LABELS.get(xfm_type, xfm_type)
        reg_sentence = (
            f"Volume-based spatial registration to the template was performed through "
            f"{stage} registration with antsRegistration (ANTs; Avants et al., 2008)."
        )
        if xfm_type == "syn":
            reg_sentence += (
                " For the non-linear stage, FireANTs (Jena et al., 2024; Jena et al., 2026) "
                "was used when available."
            )
        t1w.append(reg_sentence)

        if get_nested_config_value(config, "anat.surface_reconstruction.enabled", True):
            t1w.append(
                "Cortical surface reconstruction was performed using a modified FastSurfer pipeline "
                "(Henschel et al., 2020) adapted for non-human primates, based on the FreeSurfer "
                "surface reconstruction framework (Dale et al., 1999)."
            )

        # --- T2w paragraph (only when T2w data is present) ---
        if not has_t2w:
            return " ".join(t1w), None

        t2w = [
            "As with the T1w, when multiple T2w images existed per session or subject, a single "
            "synthesized T2w was created."
        ]

        t2w.append(
            "The T2w was rigidly coregistered to the T1w space using ANTs (Avants et al., 2008)."
        )

        return " ".join(t1w), " ".join(t2w)

    @staticmethod
    def _build_func_methods_paragraph(config: Dict[str, Any]) -> str:
        """Build the functional preprocessing paragraph dynamically from config."""
        anat_only = bool(get_nested_config_value(config, "general.anat_only", False))
        if anat_only:
            return "Functional data preprocessing was not performed (anatomical-only mode)."

        sentences = ["fMRI data were preprocessed as follows."]

        if get_nested_config_value(
            config, "func.slice_timing_correction.enabled", True
        ):
            sentences.append(
                "Slice timing correction was applied using AFNI 3dTshift (Cox, 1996; Cox & Hyde, 1997)."
            )

        motion_enabled = bool(
            get_nested_config_value(config, "func.motion_correction.enabled", True)
        )
        if motion_enabled:
            sentences.append(
                "Head motion correction was performed with mcflirt (FSL; Jenkinson et al., 2002)."
            )

        despike_enabled = bool(
            get_nested_config_value(config, "func.despike.enabled", False)
        )
        if despike_enabled:
            sentences.append(
                "Despiking was applied using AFNI 3dDespike (Cox, 1996; Cox & Hyde, 1997) "
                "to reduce the impact of extreme timepoints."
            )

        if get_nested_config_value(config, "func.coreg_runs_within_session", True):
            sentences.append(
                "When multiple fMRI runs existed within a session, within-session coregistration was "
                "performed using ANTs (Avants et al., 2008) by registering each run's mean image to a reference run."
            )

        func_skull = bool(
            get_nested_config_value(config, "func.skullstripping.enabled", True)
        )
        if get_nested_config_value(config, "func.conform.enabled", True):
            if func_skull:
                sentences.append(
                    "The fMRI mean image was conformed to target space to improve downstream alignment: "
                    "first, initial skullstripping was performed using a CNN model fine-tuned from DeepBet "
                    "(Wang et al., 2021); then the image was rigidly registered to the target using FLIRT "
                    "(FSL; Jenkinson et al., 2002). The same conform transform was then applied to the full 4D BOLD series."
                )
            else:
                sentences.append(
                    "The fMRI mean image was conformed to target space via rigid registration with FLIRT "
                    "(FSL; Jenkinson et al., 2002), and the same conform transform was then applied to the full 4D BOLD series."
                )
        elif func_skull:
            sentences.append(
                "The fMRI data was skullstripped using a CNN model fine-tuned from DeepBet (Wang et al., 2021)."
            )

        func2anat_xfm = (
            get_nested_config_value(config, "registration.func2anat_xfm_type", "syn")
            or "syn"
        ).lower()
        xfm_desc = _XFM_COREG_LABELS.get(func2anat_xfm, func2anat_xfm)
        if func2anat_xfm == "syn":
            sentences.append(
                "The mean fMRI data was registered to the selected anatomical reference using "
                "ANTs (rigid and affine; Avants et al., 2008); for non-linear registration, FireANTs "
                "(Jena et al., 2024; Jena et al., 2026) was used. The resulting transforms were applied "
                "to the full 4D BOLD and brain mask in sequence."
            )
        else:
            sentences.append(
                f"The mean fMRI data was registered to the selected anatomical reference using "
                f"ANTs ({xfm_desc}; Avants et al., 2008). The resulting transforms were applied "
                f"to the full 4D BOLD and brain mask in sequence."
            )

        skip_steps = []
        if motion_enabled:
            skip_steps.append("motion correction")
        if despike_enabled:
            skip_steps.append("despiking")
        if skip_steps:
            sentences.append(
                f"Runs with fewer than 15 volumes skipped {' and '.join(skip_steps)}; "
            )

        return " ".join(sentences)

    @staticmethod
    def create_methods_section(report_data: Dict[str, Any]) -> str:
        """Create methods section with fMRIPrep-style boilerplate (methods and references), structured with headings and lists."""
        meta = report_data.get("metadata", {})
        version = meta.get("version", "unknown")
        config = report_data.get("configuration", {}) or {}

        # Detect whether T2w data was actually processed for this subject
        dataset_context = report_data.get("dataset_context", {})
        if "subject_file_counts" in dataset_context:
            sfc = dataset_context["subject_file_counts"]
            has_t2w = (
                max(int(sfc.get("t2w", 0) or 0), int(sfc.get("t2w_processed", 0) or 0))
                > 0
            )
        else:
            anat_counts = HtmlGenerator._count_anatomical_by_modality(
                report_data.get("organized_snapshots", {}).get("anatomical", {})
            )
            has_t2w = anat_counts.get("t2w", 0) > 0

        parts = []

        # Intro paragraph
        intro = (
            "Results included in this manuscript come from preprocessing performed using "
            f"<b>brainana {html.escape(version)}</b>."
        )
        parts.append(f'<p class="methods-intro">{intro}</p>')

        t1w_text = ""
        t2w_text = None
        func_text = ""

        # Anatomical section — built directly to support optional T2w subheadings
        parts.append('<h3 class="methods-subtitle">Anatomical data preprocessing</h3>')
        if isinstance(config, dict):
            t1w_text, t2w_text = HtmlGenerator._build_anat_methods_paragraph(
                config, has_t2w
            )
            if has_t2w:
                parts.append('<h4 class="methods-subsubtitle">T1w preprocessing</h4>')
            parts.append(f"<p>{html.escape(t1w_text)}</p>")
            if has_t2w and t2w_text:
                parts.append('<h4 class="methods-subsubtitle">T2w preprocessing</h4>')
                parts.append(f"<p>{html.escape(t2w_text)}</p>")

        # Functional section
        parts.append('<h3 class="methods-subtitle">Functional data preprocessing</h3>')
        if isinstance(config, dict):
            func_text = HtmlGenerator._build_func_methods_paragraph(config)
            parts.append(f"<p>{html.escape(func_text)}</p>")

        # References: only those cited in the methods text above
        methods_body = " ".join([intro, t1w_text, t2w_text or "", func_text])
        refs_list = _cited_references(methods_body)
        parts.append('<h3 class="methods-subtitle">References</h3>')
        items = "".join(f"<li>{html.escape(ref)}</li>" for ref in refs_list)
        parts.append(f'<ul class="methods-refs">{items}</ul>')

        content = '<div class="methods-structured">\n' + "\n".join(parts) + "\n</div>"
        return HtmlGenerator.create_prose_section("Methods", "Methods", content)


def _resolve_nextflow_reports_dir(
    snapshot_dir: Path, report_path: Path
) -> Optional[Path]:
    """Return nextflow_reports/ if anatomical_jobs.json exists (Brainana output root)."""
    candidates: List[Path] = []
    snap = snapshot_dir.resolve()
    if snap.name == "figures" and snap.parent.name.startswith("sub-"):
        candidates.append(snap.parent.parent / "nextflow_reports")
    candidates.append(report_path.parent.resolve() / "nextflow_reports")
    seen: set = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if (c / "anatomical_jobs.json").is_file():
            return c
    return None


def _count_anat_inputs_from_jobs(
    jobs: List[Dict[str, Any]], subject_id: str, suffix: str
) -> int:
    """Count original anatomical NIfTI inputs for a subject from discovery job list."""
    n = 0
    for job in jobs:
        if job.get("subject_id") != subject_id or job.get("suffix") != suffix:
            continue
        fps = job.get("file_paths")
        if isinstance(fps, list) and fps:
            n += len(fps)
        elif job.get("file_path"):
            n += 1
    return n


def _subject_has_anatomical_job(jobs: List[Dict[str, Any]], subject_id: str) -> bool:
    return any(j.get("subject_id") == subject_id for j in jobs)


def _count_func_jobs_from_discovery(jobs: List[Dict[str, Any]], subject_id: str) -> int:
    """Count discovered functional BOLD jobs for one subject (exact ``subject_id``)."""
    return sum(1 for job in jobs if job.get("subject_id") == subject_id)


def generate_qc_report(
    snapshot_dir: Union[str, Path],
    report_path: Union[str, Path],
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    snapshot_paths: Optional[Dict[str, str]] = None,
    dataset_context: Optional[Dict[str, Any]] = None,
    run_status: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, str]:
    """Generate comprehensive HTML quality control report."""
    snapshot_dir, report_path = Path(snapshot_dir), Path(report_path)

    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        # Discover and parse snapshots
        snapshot_data = SnapshotProcessor.discover_and_parse(
            snapshot_dir, logger, snapshot_paths
        )

        # Organize snapshots by hierarchy
        organized_snapshots = SnapshotProcessor.organize_by_hierarchy(
            snapshot_data["snapshots"], snapshot_dir, report_path, logger
        )

        # Build report metadata
        subject_id_match = re.search(r"sub-(\w+)", report_path.name)
        subject_id = subject_id_match.group(1) if subject_id_match else None

        anat_proc = HtmlGenerator._count_anatomical_by_modality(
            organized_snapshots["anatomical"]
        )
        func_snap = HtmlGenerator._count_unique_images(
            organized_snapshots["functional"]
        )
        subject_file_counts: Dict[str, Any] = {
            "t1w": anat_proc["t1w"],
            "t2w": anat_proc["t2w"],
            "t1w_processed": anat_proc["t1w"],
            "t2w_processed": anat_proc["t2w"],
            "functional": func_snap,
        }

        nfr = _resolve_nextflow_reports_dir(snapshot_dir, report_path)
        if subject_id and nfr is not None:
            jobs_path = nfr / "anatomical_jobs.json"
            try:
                with open(jobs_path, encoding="utf-8") as jf:
                    anat_jobs: List[Dict[str, Any]] = json.load(jf)
                if _subject_has_anatomical_job(anat_jobs, subject_id):
                    subject_file_counts["t1w"] = _count_anat_inputs_from_jobs(
                        anat_jobs, subject_id, "T1w"
                    )
                    subject_file_counts["t2w"] = _count_anat_inputs_from_jobs(
                        anat_jobs, subject_id, "T2w"
                    )
                    logger.info(
                        "QC: using anatomical input counts from discovery for subject %s (%s)",
                        subject_id,
                        jobs_path,
                    )
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(
                    "QC: could not read %s for report summary counts: %s", jobs_path, e
                )

            func_jobs_path = nfr / "functional_jobs.json"
            try:
                with open(func_jobs_path, encoding="utf-8") as jf:
                    func_jobs: List[Dict[str, Any]] = json.load(jf)
                subject_file_counts["functional"] = _count_func_jobs_from_discovery(
                    func_jobs, subject_id
                )
                logger.info(
                    "QC: using functional input counts from discovery for subject %s (%s)",
                    subject_id,
                    func_jobs_path,
                )
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(
                    "QC: could not read %s for report summary counts: %s",
                    func_jobs_path,
                    e,
                )

        merged_context = dict(dataset_context or {})
        user_sfc = merged_context.pop("subject_file_counts", None)
        if user_sfc:
            subject_file_counts.update(user_sfc)
        subject_file_counts.setdefault("t1w_processed", anat_proc["t1w"])
        subject_file_counts.setdefault("t2w_processed", anat_proc["t2w"])
        subject_file_counts.setdefault("functional", func_snap)
        merged_context["subject_file_counts"] = subject_file_counts

        from nhp_mri_prep.version import get_version

        # Ingest-normalization findings live in the published derivative sidecars.
        # snapshot_dir is <output_dir>/sub-XXX/figures, so its parent is the subject
        # directory — the same layout assumption organize_by_hierarchy already makes.
        data_findings = collect_ingest_findings(snapshot_dir.parent, logger)

        report_data = {
            "metadata": {
                "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pipeline_name": "brainana",
                "version": get_version(),
                "working_directory": str(report_path.parent),
                "subject_id": subject_id,
            },
            "configuration": config,
            "organized_snapshots": organized_snapshots,
            "dataset_context": merged_context,
            "available_entities": snapshot_data["available_entities"],
            "run_status": run_status,
            "data_findings": data_findings,
        }

        # Generate HTML report
        _generate_html_report(report_data, report_path, logger)

        logger.info(f"QC: report generated successfully - {report_path}")
        return {"html_report": str(report_path)}

    except Exception as e:
        logger.error(f"QC: report generation failed - {str(e)}")
        raise RuntimeError(f"Quality control report generation failed: {str(e)}")


def _generate_html_report(
    report_data: Dict[str, Any], report_path: Path, logger: logging.Logger
) -> None:
    """Generate HTML report file."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate all sections
    navigation = HtmlGenerator.create_navigation_menu(
        report_data["organized_snapshots"],
        bool(report_data.get("run_status")),
        has_findings(report_data.get("data_findings")),
    )
    summary = HtmlGenerator.create_summary_section(
        report_data, report_data["configuration"]
    )
    status = HtmlGenerator.create_status_section(report_data)
    data_findings = HtmlGenerator.create_data_findings_section(report_data)
    anatomical = HtmlGenerator.create_modality_section(
        "Anatomical", report_data["organized_snapshots"]["anatomical"], "Structural"
    )
    functional = HtmlGenerator.create_modality_section(
        "Functional", report_data["organized_snapshots"]["functional"]
    )
    field_mapping = HtmlGenerator.create_modality_section(
        "FieldMapping",
        report_data["organized_snapshots"]["field_mapping"],
        "B₀ field mapping",
    )
    about = HtmlGenerator.create_about_section(report_data)
    methods = HtmlGenerator.create_methods_section(report_data)

    # Create complete HTML
    html_content = _create_html_template().format(
        STYLE=_REPORT_CSS,
        SCRIPT=_REPORT_JS,
        SUBJECT=report_path.stem,
        NAVIGATION_MENU=navigation,
        SUMMARY_SECTION=summary,
        STATUS_SECTION=status,
        DATA_FINDINGS_SECTION=data_findings,
        ANATOMICAL_SECTION=anatomical,
        FUNCTIONAL_SECTION=functional,
        FIELD_MAPPING_SECTION=field_mapping,
        ABOUT_SECTION=about,
        METHODS_SECTION=methods,
        GENERATION_TIME=report_data["metadata"]["generation_time"],
        VERSION=report_data["metadata"]["version"],
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Output: HTML report written - {report_path}")


# brainana design system — single source of truth for the QC report styling.
# Inlined (not linked) so each report stays portable; the IBM Plex webfont loads from
# Google Fonts with a system-sans fallback when offline.
_REPORT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{
  --bn-bg:#fffeee; --bn-surface:#fffddd; --bn-inset:#ffffff;
  --bn-ink:#1b1b16; --bn-text:#333333; --bn-muted:#6b6651;
  --bn-border:#e6e1cd; --bn-border-mid:#cdc9b0; --bn-code-bg:#f7f6ef;
  --bn-accent:#fff27a; --bn-link:#8a7a00; --bn-link-hover:#6f6300;
  --bn-ok:#3f6b46; --bn-ok-bg:#eef3ea; --bn-ok-border:#cdddc2;
  --bn-fail:#9c4636; --bn-fail-bg:#f8ece8; --bn-fail-border:#e7c8bd;
  --bn-warn:#8a6400; --bn-warn-bg:#fdf3d6; --bn-warn-border:#e6d8a0;
  --bn-r-card:10px; --bn-r-inset:8px; --bn-r-chip:6px; --bn-r-pill:999px;
  --bn-font:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --bn-mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  --bn-fs:16px; --bn-lh:1.5; --bn-lh-heading:1.25; --bn-space-p:16px; --bn-measure:960px;
  --bn-shadow-pop:0 8px 28px rgba(60,50,0,.14); --bar-h:54px;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;font-family:var(--bn-font);color:var(--bn-text);background:var(--bn-inset);
  font-size:var(--bn-fs);line-height:var(--bn-lh);-webkit-font-smoothing:antialiased}
a{color:var(--bn-link);text-decoration:none}a:hover{color:var(--bn-link-hover);text-decoration:underline}
code{font-family:var(--bn-mono);font-size:85%;background:var(--bn-code-bg);padding:.2em .4em;border-radius:var(--bn-r-chip)}
.topbar{position:fixed;top:0;left:0;right:0;z-index:20;height:var(--bar-h);display:flex;align-items:center;
  gap:8px;padding:0 24px;background:rgba(255,255,255,.93);backdrop-filter:blur(8px);border-bottom:1px solid var(--bn-border)}
.topbar .brand{font-weight:700;font-size:14px;color:var(--bn-ink);background:var(--bn-accent);
  padding:3px 9px;border-radius:6px;white-space:nowrap;letter-spacing:.01em}
.topbar .brand-sep{color:var(--bn-border-mid);font-weight:400;margin:0 1px}
.topbar .subject{font-weight:700;font-size:16px;color:var(--bn-ink);white-space:nowrap}
.topbar nav{display:flex;align-items:center;gap:2px;margin-left:16px;flex-wrap:wrap}
.topbar nav a{color:var(--bn-text);font-size:14px;padding:6px 10px;border-radius:6px}
.topbar nav a:hover{background:var(--bn-surface);text-decoration:none}
.nav-dd{position:relative}
.nav-dd>summary{list-style:none;cursor:pointer;font-size:14px;padding:6px 10px;border-radius:6px;color:var(--bn-text);white-space:nowrap}
.nav-dd>summary::-webkit-details-marker{display:none}
.nav-dd>summary:hover,.nav-dd[open]>summary{background:var(--bn-surface)}
.nav-dd .menu{position:absolute;top:calc(100% + 6px);left:0;min-width:260px;max-width:560px;width:max-content;max-height:72vh;
  overflow:auto;background:var(--bn-inset);border:1px solid var(--bn-border-mid);border-radius:10px;box-shadow:var(--bn-shadow-pop);padding:6px}
.nav-dd .menu a{display:block;color:var(--bn-text);font-size:14px;padding:7px 10px;border-radius:6px;white-space:nowrap;line-height:1.35}
.nav-dd .menu a:hover{background:var(--bn-surface);text-decoration:none}
.bids-entity{font-family:var(--bn-mono);font-size:.85em;background:var(--bn-surface);border:1px solid var(--bn-border-mid);
  border-radius:var(--bn-r-chip);padding:0 5px;color:var(--bn-text)}
main{max-width:1012px;margin:0 auto;padding:calc(var(--bar-h) + 32px) 48px 140px}
main p{margin:0 0 var(--bn-space-p)}
h1.section{font-size:1.75em;font-weight:600;line-height:var(--bn-lh-heading);color:var(--bn-ink);
  margin:56px 0 16px;padding-bottom:.3em;border-bottom:2px solid var(--bn-border-mid);scroll-margin-top:calc(var(--bar-h) + 16px)}
h1.section:first-of-type{margin-top:0}
.group-head{display:flex;flex-wrap:wrap;align-items:center;gap:7px;margin:32px 0 0;padding-top:18px;
  border-top:1px solid var(--bn-border);scroll-margin-top:calc(var(--bar-h) + 16px)}
h1.section + .group-head{border-top:none;padding-top:0;margin-top:20px}
.group-head .gchip{font-family:var(--bn-mono);font-size:13px;font-weight:600;color:var(--bn-ink);
  background:var(--bn-surface);border:1px solid var(--bn-border-mid);border-radius:var(--bn-r-chip);padding:2px 9px}
.eyebrow{font-size:12px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--bn-link);margin:24px 0 0}
.fig{padding:22px 0 20px;border-top:1px solid var(--bn-border);scroll-margin-top:calc(var(--bar-h) + 16px)}
.group-head + .fig,.eyebrow + .fig{border-top:none}
/* Second view of a single step: no rule, and tucked closer to the figure it belongs with. */
.fig.fig-same-step{border-top:none;padding-top:6px}
.fig .ftitle{font-size:1.05em;font-weight:600;margin:0;line-height:var(--bn-lh-heading)}.fig .ftitle a{color:var(--bn-ink)}
.fig .cap{color:var(--bn-muted);font-size:.9em;margin:4px 0 0}
.fig img{display:block;width:100%;max-width:960px;height:auto;margin-top:14px;border:none;border-radius:var(--bn-r-inset);background:#0c0c0c}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:8px 24px;margin:0;font-size:1em}
.kv dt{color:var(--bn-muted)}.kv dd{margin:0;font-weight:500}
.note{color:var(--bn-muted);font-size:.9em;margin-top:16px}
.prose{max-width:var(--bn-measure)}.prose p{margin:0 0 var(--bn-space-p)}
.methods-subtitle{font-size:1.15em;font-weight:600;line-height:var(--bn-lh-heading);margin:24px 0 8px}
.methods-subsubtitle{font-size:1em;font-weight:600;margin:16px 0 6px}
.methods-intro{margin:0 0 var(--bn-space-p)}
.methods-refs{margin:8px 0 var(--bn-space-p) 0;padding-left:2em;color:var(--bn-muted);font-size:.9em}
.methods-refs li{margin-bottom:6px}
.status{border:1px solid var(--bn-border-mid);border-left:3px solid var(--bn-border-mid);border-radius:var(--bn-r-card);padding:16px 20px;margin:16px 0;background:var(--bn-inset)}
.status.ok{border-left-color:var(--bn-ok)}
.status.warn{border-left-color:var(--bn-warn)}
.status.fail{border-left-color:var(--bn-fail)}
.status .headline{display:flex;align-items:center;gap:8px;font-weight:600;font-size:1.05em;margin:0 0 6px;color:var(--bn-ink)}
.status .fgroup{margin:14px 0 6px;font-size:.8em;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:var(--bn-muted)}
.status .findings{list-style:none;padding:0;margin:0}
.status .findings li{margin:0 0 12px;padding-left:14px;border-left:2px solid var(--bn-border);font-size:.95em;color:var(--bn-text)}
.status .findings li:last-child{margin-bottom:0}
.status .fnote{margin-top:4px;color:var(--bn-muted);font-size:.92em;line-height:1.5}
.status .ffiles{margin-top:6px;font-size:.88em;color:var(--bn-muted);overflow-wrap:anywhere}
.status .badge{font-size:11px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;padding:2px 8px;border-radius:var(--bn-r-pill);border:1px solid currentColor;color:var(--bn-muted)}
.status.ok .badge{color:var(--bn-ok)}.status.warn .badge{color:var(--bn-warn)}.status.fail .badge{color:var(--bn-fail)}
.status p{margin:8px 0;font-size:.95em;color:var(--bn-text)}
.status .meta{display:flex;flex-wrap:wrap;gap:6px 18px;list-style:none;padding:0;margin:10px 0 0;font-size:.9em;color:var(--bn-muted)}
.status .meta li{min-width:0}.status .meta li:not(:has(code)){white-space:nowrap}
.status .meta li:has(code){flex:1 1 100%;white-space:normal;overflow-wrap:anywhere;word-break:break-word}
.status .meta code{word-break:break-all}.status .meta b{color:var(--bn-text);font-weight:600}
.status .errlog{margin:14px 0 0;padding:16px;background:var(--bn-code-bg);border:1px solid var(--bn-border);border-radius:var(--bn-r-inset);
  color:var(--bn-text);font-family:var(--bn-mono);font-size:12.5px;line-height:1.45;white-space:pre-wrap;word-break:break-word;max-height:42vh;overflow:auto}
"""

# Closes any open nav dropdown on outside-click (capture phase) or Escape. No framework.
_REPORT_JS = """
<script>
document.addEventListener('click', function (e) {
  document.querySelectorAll('details.nav-dd[open]').forEach(function (d) {
    if (!d.contains(e.target)) d.open = false;
  });
}, true);
document.addEventListener('keydown', function (e) {
  if (e.key === 'Escape') document.querySelectorAll('details.nav-dd[open]').forEach(function (d) { d.open = false; });
});
</script>
"""


def _create_html_template() -> str:
    """Create base HTML template."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="generator" content="brainana {VERSION}" />
<title>{SUBJECT}</title>
<style type="text/css">{STYLE}</style>
</head>
<body>
<header class="topbar"><span class="brand">brainana</span><span class="brand-sep">|</span><span class="subject">{SUBJECT}</span><nav>
{NAVIGATION_MENU}
</nav></header>
<main>
{SUMMARY_SECTION}
{STATUS_SECTION}
{DATA_FINDINGS_SECTION}
{ANATOMICAL_SECTION}
{FUNCTIONAL_SECTION}
{FIELD_MAPPING_SECTION}
{ABOUT_SECTION}
{METHODS_SECTION}
</main>
{SCRIPT}
</body>
</html>"""
