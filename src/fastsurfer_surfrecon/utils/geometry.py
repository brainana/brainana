"""Voxel-grid comparisons, for checks that a volume did not move.

Deliberately nibabel + numpy only. This module is reached from
``nhp_mri_prep.steps.surface_longitudinal``, which every
``ANAT_SURFACE_RECONSTRUCTION`` task imports for ``collect_base_inputs`` -- so it
must not live under ``fastsurfer_surfrecon.io``, whose ``__init__`` eagerly
imports ``image.py`` and with it SimpleITK.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import nibabel as nib
import numpy as np

__all__ = ["volume_geometry", "volume_affine", "describe_geometry_mismatch"]


def volume_geometry(path) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    """The volume's shape and voxel sizes, rounded so float noise does not differ."""
    img = nib.load(str(path))
    zooms = tuple(round(float(z), 4) for z in img.header.get_zooms()[:3])
    # int(), not the header's own type: MGH headers hand back numpy int32, which
    # json.dumps refuses -- and these end up in provenance files.
    return tuple(int(n) for n in img.shape[:3]), zooms


def volume_affine(path) -> np.ndarray:
    """The full voxel-to-world mapping.

    Shape and zooms alone would accept a translation or an axis flip, which is
    precisely what "the grid moved" means for a transform that targets it.
    """
    return np.asarray(nib.load(str(path)).affine, dtype=float)


def describe_geometry_mismatch(
    a: Path,
    b: Path,
    *,
    labels: Sequence[str] = ("a", "b"),
    atol: float = 1e-4,
) -> Optional[str]:
    """Compare two volumes' grids; return a human-readable diff, or None if equal.

    Returns a string rather than raising so each caller can pick its own
    exception type -- ``StageOutputError`` inside a pipeline stage,
    ``RuntimeError`` in the step layer -- without this module having to know
    about either.

    Args:
        a: First volume.
        b: Second volume.
        labels: What to call each volume in the diff. Use the *role* each plays
            ("this timepoint, resampled through the LTA"), not the filename --
            the reader can see the filenames, what they need is which is which.
        atol: Absolute tolerance for the affine comparison.

    Returns:
        None when shape, zooms and affine all agree; otherwise a multi-line
        description naming every quantity that differs.
    """
    shape_a, zooms_a = volume_geometry(a)
    shape_b, zooms_b = volume_geometry(b)
    affine_a, affine_b = volume_affine(a), volume_affine(b)

    differs = []
    if shape_a != shape_b:
        differs.append(f"shape: {shape_a} vs {shape_b}")
    if zooms_a != zooms_b:
        differs.append(f"zooms: {zooms_a} vs {zooms_b}")
    if affine_a.shape != affine_b.shape or not np.allclose(
        affine_a, affine_b, atol=atol
    ):
        differs.append(f"affine (atol={atol:g})")
    if not differs:
        return None

    return (
        "Differs in " + ", ".join(differs) + ".\n"
        f"  {labels[0]}: {a}\n"
        f"    shape={shape_a} zooms={zooms_a}\n"
        f"    affine=\n{np.array2string(affine_a, precision=4)}\n"
        f"  {labels[1]}: {b}\n"
        f"    shape={shape_b} zooms={zooms_b}\n"
        f"    affine=\n{np.array2string(affine_b, precision=4)}"
    )
