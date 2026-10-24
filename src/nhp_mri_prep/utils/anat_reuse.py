"""Reuse the anatomical outputs of an earlier brainana run (``--anat_from``).

The functional workflow needs six anatomical inputs per T1w session: the brain-extracted
T1w, the T1w <-> template transforms, the registration reference (the resolved output
template), the surface-reconstruction subject id, and the T1w-space atlas with its LUT.
All but the reference are published files, so a func-only run can rebuild them from an
earlier output directory instead of rerunning the anatomical workflow.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

BRAIN_SUFFIX = "_space-T1w_desc-preproc_T1w_brain.nii.gz"


class AnatReuseError(RuntimeError):
    """The earlier output directory cannot supply the anatomical inputs."""


def _output_space_in(path: Path) -> Optional[str]:
    try:
        return (json.loads(path.read_text()).get("TemplateSource") or {}).get(
            "OutputSpace"
        )
    except (OSError, ValueError, AttributeError):
        return None


def previous_output_space(anat_dir: Union[str, Path]) -> Optional[str]:
    """``TemplateSource.OutputSpace`` of the earlier run.

    Read from a T1w -> template transform sidecar first: it was written by the registration
    that made the transforms, whereas dataset_description.json is rewritten at the start of
    every run into that directory. Falls back to dataset_description.json.
    """
    anat_dir = Path(anat_dir)
    for sidecar in sorted(
        anat_dir.glob("sub-*/**/anat/*_from-T1w_to-*_mode-image_xfm.json")
    ):
        space = _output_space_in(sidecar)
        if space:
            return space
    dd = anat_dir / "dataset_description.json"
    return _output_space_in(dd) if dd.is_file() else None


XFM_EXTENSIONS = (
    ".nii.gz",
    ".h5",
    ".mat",
)  # as ANAT_REGISTRATION emits them (not the .json sidecars)


def _one(paths: List[Path]) -> Optional[Path]:
    paths = sorted(p for p in paths if p.name.endswith(XFM_EXTENSIONS))
    return paths[0] if paths else None


def collect_anat_outputs(
    anat_dir: Union[str, Path],
    output_space: str,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Find the anatomical inputs of the functional workflow in an earlier output dir.

    Returns ``{"template": path, "anat": [row, ...], "fastsurfer": [row, ...]}``. Each anat
    row has ``sub``/``ses`` (no prefixes; ``ses`` is "" for subject-level anatomy), ``brain``,
    ``bids_name`` (the T1w basename the anatomical workflow names outputs from), and, when
    published, ``xfm_fwd``/``xfm_inv``/``seg``/``lut``. Fastsurfer rows are ``sub``/``ses``/``id``.
    Raises AnatReuseError when the directory has no T1w brain, used another output space, or
    is ``output_dir`` itself. Subjects without anatomy are not an error: the functional
    workflow registers them to the template, as in a full run.
    """
    from .templates import resolve_template, space_label_for

    anat_dir = Path(anat_dir)
    if not anat_dir.is_dir():
        raise AnatReuseError(f"--anat_from directory not found: {anat_dir}")
    if output_dir is not None and Path(output_dir).resolve() == anat_dir.resolve():
        raise AnatReuseError(
            "--anat_from is the same directory as --output_dir; write the functional-only "
            "run to a new output directory"
        )
    previous = previous_output_space(anat_dir)
    if previous is not None and previous != output_space:
        raise AnatReuseError(
            f"--anat_from outputs are in output space '{previous}', this run uses "
            f"'{output_space}'; the T1w->template transforms would not match"
        )
    label = space_label_for(output_space)

    rows: List[Dict[str, Any]] = []
    for brain in sorted(anat_dir.glob(f"sub-*/**/anat/*{BRAIN_SUFFIX}")):
        rel = brain.relative_to(anat_dir).parts
        sub = rel[0][len("sub-") :]
        ses = rel[1][len("ses-") :] if rel[1].startswith("ses-") else ""
        prefix = brain.name[: -len(BRAIN_SUFFIX)]
        d = brain.parent
        row: Dict[str, Any] = {
            "sub": sub,
            "ses": ses,
            "brain": str(brain),
            "bids_name": f"{prefix}_T1w.nii.gz",
        }
        fwd = _one(list(d.glob(f"{prefix}_from-T1w_to-{label}_mode-image_xfm.*")))
        inv = _one(list(d.glob(f"{prefix}_from-{label}_to-T1w_mode-image_xfm.*")))
        if fwd and inv:
            row.update(xfm_fwd=str(fwd), xfm_inv=str(inv))
        for seg in sorted(d.glob(f"{prefix}_space-T1w_desc-brain_atlas*.nii.gz")):
            lut = seg.with_name(seg.name[: -len(".nii.gz")] + ".tsv")
            if lut.is_file():
                row.update(seg=str(seg), lut=str(lut))
                break
        rows.append(row)
    if not rows:
        raise AnatReuseError(
            f"no '*{BRAIN_SUFFIX}' under {anat_dir}/sub-*/[ses-*/]anat"
        )

    fastsurfer = []
    fs_root = anat_dir / "fastsurfer"
    if fs_root.is_dir():
        for d in sorted(
            p for p in fs_root.iterdir() if p.is_dir() and p.name.startswith("sub-")
        ):
            m = re.match(r"sub-([^_]+)(?:_ses-([^_]+))?", d.name)
            if m:
                fastsurfer.append(
                    {"sub": m.group(1), "ses": m.group(2) or "", "id": d.name}
                )

    return {
        "template": str(resolve_template(output_space)),
        "anat": rows,
        "fastsurfer": fastsurfer,
    }


def link_fastsurfer(
    anat_dir: Union[str, Path], output_dir: Union[str, Path], ids: List[str]
) -> List[str]:
    """Link earlier fastsurfer subject dirs into this run's output (the tSNR surface
    projection reads ``<output_dir>/fastsurfer/<id>``). Existing entries are left alone.
    """
    linked = []
    dst_root = Path(output_dir) / "fastsurfer"
    for sid in ids:
        src, dst = Path(anat_dir) / "fastsurfer" / sid, dst_root / sid
        if src.is_dir() and not dst.exists() and not dst.is_symlink():
            dst_root.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src.resolve(), target_is_directory=True)
            linked.append(sid)
    return linked
