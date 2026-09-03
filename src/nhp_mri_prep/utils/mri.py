"""
MRI processing utilities for nhp_mri_prep.

This module provides utilities for common MRI processing operations.
"""

import logging
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from nibabel.orientations import aff2axcodes


# ---------------------------------------------------------------------------
# Dimensionality normalization (anatomical images)
# ---------------------------------------------------------------------------

# NIfTI intent codes whose trailing dimension encodes vector/tensor components
# rather than repeated acquisitions. Collapsing these would corrupt the data —
# ANTs displacement fields are (X, Y, Z, 1, 3) with intent 1007.
_NON_SCALAR_INTENT_CODES = frozenset(
    {
        1004,  # NIFTI_INTENT_GENMATRIX
        1005,  # NIFTI_INTENT_SYMMATRIX
        1006,  # NIFTI_INTENT_DISPVECT
        1007,  # NIFTI_INTENT_VECTOR
        1008,  # NIFTI_INTENT_POINTSET
        1009,  # NIFTI_INTENT_TRIANGLE
        1010,  # NIFTI_INTENT_QUATERNION
        2003,  # NIFTI_INTENT_RGB_VECTOR
    }
)


def as_3d_image(
    img: nib.Nifti1Image,
    logger: Optional[logging.Logger] = None,
) -> Tuple[nib.Nifti1Image, str]:
    """Return a 3-D view of a scalar anatomical image, plus what was done to it.

    Some scanners and DICOM converters emit anatomicals with a trailing singleton
    frame axis, e.g. ``(144, 144, 60, 1)`` with ``dim[0] == 4``. Such images are
    geometrically 3-D but break any code that assumes ``ndim == 3`` — notably
    ``nibabel.processing.resample_to_output``, which raises outright.

    Args:
        img: Loaded NIfTI image.
        logger: Logger instance (optional).

    Returns:
        ``(image, action)`` where *action* is one of:

        - ``"unchanged"``: input was already 3-D; the **same object** is returned,
          so no data is read, no dtype changes, and nothing is re-quantized.
        - ``"squeezed"``: trailing singleton dimension(s) dropped losslessly
          (affine, qform/sform codes, pixdim and on-disk dtype all preserved).
        - ``"mean"``: a genuine multi-volume 4-D image collapsed by averaging over
          the last axis, written as float32.

    Raises:
        ValueError: If the image carries a vector/tensor intent code, has fewer
            than 3 dimensions, or still has more than 4 dimensions after
            squeezing (e.g. a ``(X, Y, Z, 1, 3)`` displacement field).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    intent_code = int(img.header["intent_code"])
    if intent_code in _NON_SCALAR_INTENT_CODES:
        raise ValueError(
            f"Refusing to collapse an image with non-scalar intent code {intent_code} "
            f"(shape {img.shape}); its trailing axis holds vector/tensor components, "
            f"not repeated volumes."
        )

    if len(img.shape) < 3:
        raise ValueError(f"Expected a 3-D or 4-D image, got shape {img.shape}")

    if len(img.shape) == 3:
        return img, "unchanged"

    # squeeze_image only trims *trailing* singleton axes, so a (X, Y, Z, 1, 3)
    # vector field survives it untouched (and is rejected by the guard below).
    squeezed = nib.funcs.squeeze_image(img)
    if len(squeezed.shape) == 3:
        logger.info(f"Collapsed singleton frame axis: {img.shape} -> {squeezed.shape}")
        return squeezed, "squeezed"

    if len(squeezed.shape) > 4:
        raise ValueError(
            f"Cannot collapse a {len(squeezed.shape)}-D image (shape {img.shape}) to 3-D"
        )

    # Genuine multi-volume 4-D: average over the frame axis. float32 rather than
    # the source dtype — re-quantizing the mean of N int16 volumes back to int16
    # would lose precision for no benefit (cf. calculate_func_tmean above).
    logger.warning(
        f"Multi-volume anatomical {squeezed.shape} detected; averaging over the "
        f"last axis to obtain a 3-D image."
    )
    mean_data = np.mean(squeezed.get_fdata(), axis=-1).astype(np.float32)
    out = nib.Nifti1Image(mean_data, squeezed.affine, squeezed.header)
    # Nifti1Image honours the header's dtype, so an int16 source header would
    # silently truncate the float32 mean unless we override it here.
    out.header.set_data_dtype(np.float32)
    out.header.set_zooms(squeezed.header.get_zooms()[:3])
    return out, "mean"


def _reject_in_place(imagef: Path, outputf: Path) -> None:
    """Refuse to write a normalized image over its own source file.

    Not a style rule — on an uncompressed ``.nii`` this is a process-killing bug,
    not an exception. nibabel mmaps the ArrayProxy, ``np.asanyarray(img.dataobj)``
    hands back a view onto that mapping rather than a copy, and ``nib.save`` then
    truncates the file the view points at. The next page fault raises SIGBUS, which
    no ``except`` can catch: the interpreter dies with exit 135 and no traceback,
    so Nextflow reports a task that vanished. ``.nii.gz`` happens to survive it
    (gzip cannot mmap), which is what makes it dangerous — an all-``.nii.gz`` test
    suite stays green while real ``.nii`` datasets die.
    """
    try:
        same = imagef.resolve() == outputf.resolve()
    except OSError:
        same = False
    if same:
        raise ValueError(
            f"Refusing to normalize {imagef.name} onto itself ({outputf}); "
            f"the output path must differ from the input path."
        )


def ensure_3d(
    imagef: Union[str, Path],
    outputf: Optional[Union[str, Path]] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Path, str]:
    """Guarantee a 3-D image on disk, writing a new file only when needed.

    Wrapper around :func:`as_3d_image` for the path-in/path-out callers in this
    codebase. A 3-D input is a no-op: the input path is returned and nothing is
    written, so datasets that were already well-formed stay byte-for-byte
    identical.

    For anatomicals at ingest use :func:`normalize_anat_input` instead — it applies
    this plus the orientation and geometry repairs in a single read/write. This
    function remains for callers that only need the dimensionality guarantee
    (``conform_to_template``, which also handles functional temporal means).

    The input file is never modified in place — Nextflow stages task inputs
    read-only, so a rewrite must go somewhere else. When a rewrite is needed and
    *outputf* is None this raises, forcing callers to name a destination rather
    than defaulting to somewhere unsafe (a read-only BIDS tree, or a task-dir
    root where a stray ``*.nii.gz`` would be picked up by an output glob).

    Args:
        imagef: Input NIfTI image path.
        outputf: Destination for the normalized image. Only used — and only
            required — when the input is not already 3-D.
        logger: Logger instance (optional).

    Returns:
        ``(path, action)``; see :func:`as_3d_image` for the action values. For
        ``"unchanged"`` the returned path is *imagef*.

    Raises:
        ValueError: If a rewrite is needed but *outputf* is None, if *outputf*
            names the input file itself, or for any reason listed in
            :func:`as_3d_image`.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    imagef = Path(imagef)
    if outputf is not None:
        _reject_in_place(imagef, Path(outputf))

    src = nib.load(str(imagef))
    img, action = as_3d_image(src, logger)

    if action == "unchanged":
        return imagef, action

    if outputf is None:
        raise ValueError(
            f"{imagef.name} has shape {src.shape} and needs collapsing to 3-D, "
            f"but no output path was given."
        )

    outputf = Path(outputf)
    outputf.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(outputf))
    logger.info(f"Output: 3-D normalized image written - {outputf}")
    return outputf, action


# ---------------------------------------------------------------------------
# Geometry normalization and header inspection (anatomical images)
# ---------------------------------------------------------------------------


def as_oriented_image(
    img: nib.Nifti1Image,
    logger: Optional[logging.Logger] = None,
) -> Tuple[nib.Nifti1Image, str]:
    """Guarantee a stored orientation, returning the image and what was done to it.

    A NIfTI with ``qform_code == 0`` and ``sform_code == 0`` carries no spatial
    orientation at all. Readers do not fail on this — they each *invent* a
    fallback, and they do not invent the same one:

    - nibabel and FSL use the header's base affine: LAS with the origin at the
      centre of the voxel grid.
    - ITK (and therefore ANTs) uses an identity direction matrix in its own LPS
      world, with the origin at the *corner* of the grid.

    So an orientation-less input silently means different geometry to different
    steps of the same pipeline. Two files that are genuinely on one grid come out
    disagreeing by an axis flip and a translation of half the field of view, and
    images of *different* sizes get corner-aligned rather than centre-aligned.

    This makes the fallback explicit instead of implicit. The affine written is
    ``header.get_base_affine()`` — precisely what nibabel and FSL already assume —
    so nothing about the pipeline's current numeric behaviour changes; the point
    is that ITK now reads the same geometry as everyone else. Only geometry
    metadata changes: the voxel array is never resampled or reoriented. It is
    re-encoded on write, though, so a source carrying ``scl_slope``/``scl_inter``
    can shift by one quantization step in its stored integers (the scaled values,
    and plain unscaled data, round-trip exactly).

    An image that already declares an orientation is a no-op: the **same object**
    is returned, so well-formed data is never rewritten.

    Note:
        This recovers a *convention*, not ground truth. The assumed affine puts
        +x at the subject's left; if the acquisition actually ran the other way,
        the result is a left/right mirror, which no downstream rigid or affine
        registration can undo and which is invisible on inspection. Confirm
        handedness against an external record before trusting hemisphere-wise
        results from a dataset that needed this repair.

    Args:
        img: Loaded NIfTI image.
        logger: Logger instance (optional).

    Returns:
        ``(image, action)`` where *action* is one of:

        - ``"unchanged"``: input already had a qform or sform.
        - ``"assumed-<code>-centered"`` (e.g. ``"assumed-LAS-centered"``): the
          base affine was stamped into both qform and sform with code 2
          (scanner anat).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    header = img.header
    if int(header["qform_code"]) > 0 or int(header["sform_code"]) > 0:
        return img, "unchanged"

    # The affine nibabel/FSL already fall back to for this header: LAS, origin at
    # the centre of the voxel grid. Derived rather than hand-built so the written
    # geometry is by construction the one the pipeline is already using.
    affine = header.get_base_affine()
    orientation = get_image_orientation_from_affine(affine)

    logger.warning(
        f"Data: image has qform_code=0 and sform_code=0 (no stored orientation). "
        f"Assuming {orientation} with a centred origin, the convention nibabel "
        f"and FSL already fall back to for this header. Verify left/right against "
        f"an external record before trusting hemisphere-wise results."
    )

    # Pass the source header through so descrip/xyzt_units/dtype survive, then
    # override the geometry explicitly — the Nifti1Image constructor picks its own
    # qform/sform codes, and we want code 2 (scanner anat) on both, deterministically.
    out = nib.Nifti1Image(np.asanyarray(img.dataobj), affine, header)
    out.header.set_data_dtype(header.get_data_dtype())
    out.set_qform(affine, code=2)
    out.set_sform(affine, code=2)
    return out, f"assumed-{orientation}-centered"


def reconcile_qform_sform(
    img: nib.Nifti1Image,
    logger: Optional[logging.Logger] = None,
) -> Tuple[nib.Nifti1Image, str]:
    """Make a disagreeing qform and sform agree, resolving toward the sform.

    A NIfTI may store its geometry twice, and nothing enforces that the two agree.
    When they disagree, which one a reader believes is a matter of that reader's
    policy — so the same file means two different grids to two different tools:

    - nibabel and FSL prefer the **sform**, and so does every other step in this
      pipeline (``img.affine`` is the sform whenever ``sform_code > 0``).
    - FastSurfer's ``check_affine_in_nifti`` (``fastsurfer_nn/data_loader/conform.py``)
      detects the mismatch and resolves it toward the **qform**, overwriting the
      sform with it before segmentation runs.

    So an unreconciled mismatch means brainana registers against one geometry and
    segments against another, with only a warning buried in the FastSurfer log to
    say so. Writing the sform into both makes every reader agree, and because the
    sform is what this pipeline already reads, it changes no current numeric
    behaviour. The sform's own code is preserved rather than forced to a constant —
    it carries meaning (1 scanner, 2 aligned, 4 MNI) that is worth keeping.

    Only geometry metadata changes; the voxel array is never resampled (see
    :func:`as_oriented_image` for the one caveat on re-encoding). An image whose
    forms already agree — or which has only one of them — is a no-op and the
    **same object** is returned.

    Args:
        img: Loaded NIfTI image.
        logger: Logger instance (optional).

    Returns:
        ``(image, action)`` where *action* is ``"unchanged"`` or
        ``"qform-set-from-sform"``.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    header = img.header
    q_code = int(header["qform_code"])
    s_code = int(header["sform_code"])

    # Nothing to reconcile unless both are actually present. The neither-present
    # case belongs to as_oriented_image; a single present form is unambiguous.
    if q_code <= 0 or s_code <= 0:
        return img, "unchanged"

    sform = img.get_sform()
    if np.allclose(sform, img.get_qform(), atol=1e-3):
        return img, "unchanged"

    logger.warning(
        f"Data: qform and sform disagree (qform_code={q_code}, sform_code={s_code}). "
        f"Resolving toward the sform, which nibabel, FSL and the rest of this "
        f"pipeline already read; without this, FastSurfer would resolve toward the "
        f"qform and segment against a different grid than registration used."
    )

    out = nib.Nifti1Image(np.asanyarray(img.dataobj), sform, header)
    out.header.set_data_dtype(header.get_data_dtype())
    out.set_sform(sform, code=s_code)
    out.set_qform(sform, code=s_code)
    return out, "qform-set-from-sform"


def inspect_header(img: nib.Nifti1Image) -> list:
    """Report header defects that have no safe automatic repair. Read-only.

    These are the cases where guessing would be worse than reporting. Rescaling a
    metre-unit header or rewriting pixdim to match an affine could just as easily
    turn a recoverable dataset into confidently wrong output, so the findings are
    surfaced — in the log, the JSON sidecar and the QC report — and the call is
    left to whoever knows the acquisition.

    Args:
        img: Loaded NIfTI image, after any repairs have been applied.

    Returns:
        A list of human-readable finding strings; empty when the header is sane.
    """
    findings = []
    header = img.header

    # NIfTI treats unset spatial units as mm, so only an explicit non-mm unit is a
    # finding. nibabel does not convert: get_zooms() returns raw pixdim whatever
    # the unit says, so every voxel size in this pipeline would be off by the unit
    # factor while ITK/ANTs reads it correctly.
    xyz_unit = header.get_xyzt_units()[0]
    if xyz_unit not in ("mm", "unknown"):
        findings.append(
            f"Spatial units are '{xyz_unit}', not mm. nibabel does not rescale "
            f"pixdim, so brainana reads voxel sizes "
            f"{tuple(round(float(z), 6) for z in header.get_zooms()[:3])} verbatim "
            f"while ITK/ANTs converts them to mm. Convert the header to mm before "
            f"processing."
        )

    # Same comparison FastSurfer's check_affine_in_nifti makes before it aborts,
    # done here so the problem surfaces at ingest rather than deep in segmentation.
    vox_affine = np.sqrt((img.affine[:3, :3] * img.affine[:3, :3]).sum(0))
    vox_header = np.asarray(header.get_zooms()[:3], dtype=float)
    if not np.allclose(vox_affine, vox_header, atol=1e-3):
        findings.append(
            f"Header pixdim {tuple(round(float(v), 4) for v in vox_header)} "
            f"disagrees with the affine's voxel scale "
            f"{tuple(round(float(v), 4) for v in vox_affine)}. The header is "
            f"self-inconsistent; FastSurfer aborts on this during segmentation."
        )

    return findings


def normalize_anat_input(
    imagef: Union[str, Path],
    outputf: Union[str, Path],
    logger: Optional[logging.Logger] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Normalize one anatomical at ingest: one read, one write, one report.

    Applies every repair an anatomical needs before any other step touches it,
    then reports what could not be repaired. Composing the repairs in memory
    rather than chaining path-in/path-out helpers means a file needing three
    repairs still costs a single load and a single save.

    Repairs applied, in order:

    1. :func:`as_3d_image` — drop a trailing singleton frame axis, or mean-collapse
       a genuine multi-volume anatomical.
    2. :func:`as_oriented_image` — stamp an orientation when the header stores none.
    3. :func:`reconcile_qform_sform` — resolve a qform/sform disagreement toward
       the sform.
    4. Container format — the output is always gzipped, so no downstream step
       handles a raw ``.nii``.

    Then :func:`inspect_header` records what is wrong but not safely fixable.

    A well-formed, already-gzipped input is a true no-op: *imagef* is returned and
    nothing is written, so such datasets stay byte-for-byte identical.

    Args:
        imagef: Input NIfTI image path.
        outputf: Destination. The extension is forced to ``.nii.gz``; the stem is
            preserved, because the synthesis step parses BIDS entities out of it.
        logger: Logger instance (optional).

    Returns:
        ``(path, report)``. *report* has keys ``dim``, ``orientation``, ``geometry``
        (each an action string, ``"unchanged"`` when the repair did not fire) and
        ``warnings`` (a possibly-empty list from :func:`inspect_header`).

    Raises:
        ValueError: If *outputf* names *imagef* itself, or for any reason listed in
            :func:`as_3d_image`.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    from .bids import get_filename_stem

    imagef = Path(imagef)
    outputf = Path(outputf)
    outputf = outputf.parent / f"{get_filename_stem(outputf)}.nii.gz"
    _reject_in_place(imagef, outputf)

    src = nib.load(str(imagef))
    img, dim_action = as_3d_image(src, logger)
    img, orient_action = as_oriented_image(img, logger)
    img, geom_action = reconcile_qform_sform(img, logger)
    warnings = inspect_header(img)

    for finding in warnings:
        logger.warning(f"Data: {imagef.name} - {finding}")

    report = {
        "dim": dim_action,
        "orientation": orient_action,
        "geometry": geom_action,
        "warnings": warnings,
    }

    repaired = any(
        action != "unchanged" for action in (dim_action, orient_action, geom_action)
    )
    # A raw .nii is rewritten even when nothing else changed: it is the format that
    # makes an accidental read-write of the same path fatal (see _reject_in_place),
    # and normalizing it here means no later step has to think about it.
    needs_gzip = not str(imagef).endswith(".nii.gz")

    if not repaired and not needs_gzip:
        return imagef, report

    outputf.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(outputf))
    logger.info(f"Output: normalized anatomical written - {outputf}")
    return outputf, report


def calculate_func_tmean(
    funcf: str,
    outputf: str,
    logger: logging.Logger,
) -> Dict[str, str]:
    """Calculate the temporal mean of a functional image (nibabel, FSL-free).

    Equivalent to ``fslmaths <funcf> -Tmean <outputf>``: averages over the last
    (time) axis for 4D inputs; a 3D input is written through unchanged. Implemented
    with nibabel/numpy so it needs no FSL binary.

    Args:
        funcf: Input functional file
        outputf: Output mean file path
        logger: Logger instance

    Returns:
        Dictionary with output file path
    """
    try:
        img = nib.load(str(funcf))
        data = img.get_fdata()
        mean_data = data.mean(axis=-1) if data.ndim == 4 else data
        out_img = nib.Nifti1Image(mean_data.astype(np.float32), img.affine, img.header)
        out_img.header.set_data_dtype(np.float32)
        nib.save(out_img, str(outputf))
        logger.info(f"Output: mean functional image created successfully - {outputf}")
    except Exception as e:
        logger.error(f"Step: mean functional image creation failed - {str(e)}")
        raise

    return {"output": outputf}


def apply_brain_mask(
    imagef: Union[str, Path],
    maskf: Union[str, Path],
    outputf: Union[str, Path],
    logger: Optional[logging.Logger] = None,
    binarize: bool = True,
) -> Dict[str, str]:
    """Mask an image by a (brain) mask using nibabel (FSL-free).

    Equivalent to ``fslmaths <image> -mas <mask> <output>``: zeroes out voxels where the
    mask is zero, preserving the image's affine/header/dtype. Handles a 3D mask applied
    to a 4D image by broadcasting over the time axis. ``binarize`` mirrors FSL's ``-mas``
    semantics (mask treated as ``|mask| > 0``); set False to multiply by raw mask values.

    Args:
        imagef: Image to be masked (3D or 4D)
        maskf: Mask image (3D)
        outputf: Output file path
        logger: Logger instance (optional)
        binarize: Treat the mask as boolean (|mask| > 0); default True.

    Returns:
        Dictionary with the output file path under key 'output'.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    img = nib.load(str(imagef))
    mask_img = nib.load(str(maskf))
    data = img.get_fdata()
    mask = mask_img.get_fdata()
    mask = (np.abs(mask) > 0).astype(data.dtype) if binarize else mask

    if data.ndim == 4 and mask.ndim == 3:
        masked = data * mask[..., np.newaxis]
    else:
        masked = data * mask

    out_img = nib.Nifti1Image(
        masked.astype(img.get_data_dtype()), img.affine, img.header
    )
    nib.save(out_img, str(outputf))
    logger.info(f"Output: masked image created successfully - {outputf}")
    return {"output": str(outputf)}


def get_image_shape(
    imagef: str,
    logger: Optional[logging.Logger] = None,
) -> list:
    """Get image shape.

    Args:
        imagef: Input image file
        logger: Logger instance

    Returns:
        List of integers representing image dimensions [x, y, z, t]
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Read the shape via nibabel (FSL/AFNI-free). Mirrors ``3dinfo -n4`` by always
    # returning 4 dims [x, y, z, t] (t = 1 for 3D), without loading voxel data.
    try:
        shape = list(nib.load(str(imagef)).shape)
        while len(shape) < 4:
            shape.append(1)
        shape = [int(d) for d in shape[:4]]
        logger.info(f"Data: image shape - {shape}")
        return shape
    except Exception as e:
        logger.error(f"Step: image shape retrieval failed - {e}")
        raise RuntimeError(f"Image shape retrieval failed: {e}")


def shape_to_ants_input_type(shape: list) -> int:
    """Map image shape to ANTs antsApplyTransforms -e (input-image-type) value.

    ANTs -e options: 0=scalar, 1=vector, 2=tensor, 3=time-series, 4=multichannel,
    5=five-dimensional. For 3D/4D images: 3D or 4D with last dim 1 -> 0 (scalar);
    4D with last dim > 1 -> 3 (time-series).

    Args:
        shape: Image dimensions from get_image_shape [x, y, z, t] or [x, y, z]

    Returns:
        Integer 0-5 for antsApplyTransforms -e
    """
    if len(shape) < 3:
        return 0
    if len(shape) == 3:
        return 0
    # len >= 4: check 4th dimension
    if shape[3] <= 1:
        return 0  # scalar (3D or pseudo-3D)
    return 3  # time-series (4D)


def get_image_resolution(
    imagef: str,
    logger: Optional[logging.Logger] = None,
) -> list:
    """Get image resolution.

    Args:
        imagef: Input image file
        logger: Logger instance

    Returns:
        List of floats representing image resolution [x, y, z]
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Read voxel sizes via nibabel (FSL/AFNI-free). Equivalent to ``3dinfo -ad3``
    # (absolute |dx|,|dy|,|dz| in mm); magnitudes are orientation-convention independent.
    try:
        zooms = nib.load(str(imagef)).header.get_zooms()
        resolution = [abs(float(z)) for z in zooms[:3]]
        logger.info(f"Data: image resolution - {resolution}")
        return resolution
    except Exception as e:
        logger.error(f"Step: image resolution retrieval failed - {e}")
        raise RuntimeError(f"Image resolution retrieval failed: {e}")


def get_opposite_orientation(direction: str) -> str:
    """
    Get the opposite anatomical direction.

    Args:
        direction: One of 'R', 'L', 'A', 'P', 'S', 'I'

    Returns:
        Opposite direction (e.g., 'R' -> 'L', 'A' -> 'P')
    """
    opposites = {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}
    return opposites.get(direction, direction)


def get_image_orientation_from_affine(
    affine: Union[np.ndarray, nib.spatialimages.SpatialImage],
) -> str:
    """Get orientation code from an affine matrix.

    Args:
        affine: Affine matrix (4x4 numpy array) or nibabel image object

    Returns:
        Orientation code string (e.g., 'RAS', 'LPI', 'RPS')

    Raises:
        ValueError: If aff2axcodes returns unexpected type
        AttributeError: If affine cannot be extracted from input
    """
    # Extract affine matrix if input is a nibabel image
    if isinstance(affine, nib.spatialimages.SpatialImage):
        affine_matrix = affine.affine
    else:
        affine_matrix = affine

    # Get orientation code from affine matrix
    axes_codes = nib.aff2axcodes(affine_matrix)
    if not isinstance(axes_codes, (list, tuple)):
        raise ValueError(
            f"aff2axcodes returned unexpected type: {type(axes_codes)}, value: {axes_codes}"
        )

    orientation_code = "".join(axes_codes)
    return orientation_code


# Module-level cache for rotation matrices
_N90_ROTATIONS_3X3 = None
_N90_ROTATIONS_4X4 = None


def _generate_n90_rotations() -> list[np.ndarray]:
    """Generate all 24 unique n×90° rotation matrices (all rotational symmetries of a cube).

    For 90° increments, cos and sin values are always 0, 1, or -1, so we hardcode them
    for efficiency and clarity. Results are cached at module level.

    This generates all 24 possible orientations reachable by n×90° rotations.
    The 24 rotations are all proper rotations (det=1) that map basis vectors to ±basis vectors.

    Returns:
        List of 24 unique 3x3 rotation matrices
    """
    global _N90_ROTATIONS_3X3

    if _N90_ROTATIONS_3X3 is not None:
        return _N90_ROTATIONS_3X3

    # Base rotations around principal axes
    Rx90 = np.array(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64
    )  # 90° around X
    Rx180 = np.array(
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64
    )  # 180° around X
    Rx270 = np.array(
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64
    )  # 270° around X

    Ry90 = np.array(
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64
    )  # 90° around Y
    Ry180 = np.array(
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64
    )  # 180° around Y
    Ry270 = np.array(
        [[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64
    )  # 270° around Y

    Rz90 = np.array(
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64
    )  # 90° around Z
    Rz180 = np.array(
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64
    )  # 180° around Z
    Rz270 = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64
    )  # 270° around Z

    # Generate all combinations systematically
    # Start with identity and single-axis rotations
    rotations = [np.eye(3, dtype=np.float64)]
    rotations.extend([Rx90, Rx180, Rx270, Ry90, Ry180, Ry270, Rz90, Rz180, Rz270])

    # Generate composite rotations: all combinations of two rotations
    base_rots = [Rx90, Rx180, Rx270, Ry90, Ry180, Ry270, Rz90, Rz180, Rz270]
    for r1 in base_rots:
        for r2 in base_rots:
            composite = r1 @ r2
            rotations.append(composite)

    # Remove duplicates by comparing matrices (with tolerance for floating point)
    unique_rotations = []
    seen = set()
    for rot in rotations:
        # Round to handle any floating point issues, then convert to tuple
        rot_rounded = np.round(rot, decimals=10)
        rot_tuple = tuple(rot_rounded.flatten())
        if rot_tuple not in seen:
            seen.add(rot_tuple)
            unique_rotations.append(rot)

    # Should have exactly 24 rotations
    if len(unique_rotations) != 24:
        # If we don't have exactly 24, log a warning but continue
        import logging

        logging.warning(f"Expected 24 unique rotations, found {len(unique_rotations)}")

    _N90_ROTATIONS_3X3 = unique_rotations
    return unique_rotations


def _get_n90_rotations_4x4() -> list[np.ndarray]:
    """Get cached 4x4 rotation matrices for affine transformations.

    Returns:
        List of 4x4 rotation matrices (cached)
    """
    global _N90_ROTATIONS_4X4

    if _N90_ROTATIONS_4X4 is not None:
        return _N90_ROTATIONS_4X4

    rotations_3x3 = _generate_n90_rotations()
    rotations_4x4 = []

    for rot_3x3 in rotations_3x3:
        rot_4x4 = np.eye(4, dtype=rot_3x3.dtype)
        rot_4x4[:3, :3] = rot_3x3
        rotations_4x4.append(rot_4x4)

    _N90_ROTATIONS_4X4 = rotations_4x4
    return rotations_4x4


# Cross product coefficients: basis(type1) × basis(type2) = coeff * basis(type3)
# In RAS space: RL=X, AP=Y, SI=Z
# Standard cross products: X × Y = Z, Y × Z = X, Z × X = Y (positive)
# Reverse order: Y × X = -Z, Z × Y = -X, X × Z = -Y (negative)
_CROSS_COEFFS = {
    ("RL", "AP", "SI"): 1,  # R × A = S
    ("AP", "SI", "RL"): 1,  # A × S = R
    ("SI", "RL", "AP"): 1,  # S × R = A
    ("AP", "RL", "SI"): -1,  # A × R = I (=-S)
    ("SI", "AP", "RL"): -1,  # S × A = L (=-R)
    ("RL", "SI", "AP"): -1,  # R × S = P (=-A)
}

# Unit vectors for each direction
_DIRECTION_VECTORS = {
    "R": np.array([1, 0, 0]),
    "L": np.array([-1, 0, 0]),
    "A": np.array([0, 1, 0]),
    "P": np.array([0, -1, 0]),
    "S": np.array([0, 0, 1]),
    "I": np.array([0, 0, -1]),
}


def _get_handedness(orientation: str) -> int:
    """Determine if an orientation code is right-handed or left-handed.

    Right-handed: axis0 × axis1 = axis2 (e.g., RAS, PRS, LSA)
    Left-handed: axis0 × axis1 = -axis2 (e.g., PSR, RAI)

    Args:
        orientation: 3-letter orientation code (e.g., 'RAS', 'PSR')

    Returns:
        +1 for right-handed, -1 for left-handed
    """
    v0 = _DIRECTION_VECTORS[orientation[0]]
    v1 = _DIRECTION_VECTORS[orientation[1]]
    v2 = _DIRECTION_VECTORS[orientation[2]]

    cross = np.cross(v0, v1)

    if np.allclose(cross, v2):
        return 1  # right-handed
    elif np.allclose(cross, -v2):
        return -1  # left-handed
    else:
        raise ValueError(f"Invalid orientation code: {orientation}")


def _determine_third_orientation(
    dir1_idx: int,
    dir2_idx: int,
    dir3_idx: int,
    dir1: str,
    dir2: str,
    dir1_type: str,
    dir2_type: str,
) -> str:
    """Determine the third orientation direction using right-hand rule.

    Given two orientations and their positions, determines the third orientation
    that satisfies the right-hand rule: R × A = S (where × is cross product).

    The function is general and works for any combination:
    - A/P and S/I → determines R/L
    - A/P and R/L → determines S/I
    - S/I and R/L → determines A/P

    Args:
        dir1_idx: Axis index (0, 1, or 2) where first direction is located
        dir2_idx: Axis index (0, 1, or 2) where second direction is located
        dir3_idx: Axis index (0, 1, or 2) where third direction should be determined
        dir1: First direction (e.g., 'A', 'P', 'S', 'I', 'R', 'L')
        dir2: Second direction (e.g., 'A', 'P', 'S', 'I', 'R', 'L')
        dir1_type: Type of first direction ('AP', 'SI', or 'RL')
        dir2_type: Type of second direction ('AP', 'SI', or 'RL')

    Returns:
        The third direction that satisfies right-hand rule

    Examples:
        >>> # RAS: given A(1) and S(2), determine R at position 0
        >>> _determine_third_orientation(1, 2, 0, 'A', 'S', 'AP', 'SI')
        'R'

        >>> # PRS: given R(1) and S(2), determine P at position 0
        >>> _determine_third_orientation(1, 2, 0, 'R', 'S', 'RL', 'SI')
        'P'
    """
    # Right-handed cycles follow the pattern: (0,1,2), (1,2,0), (2,0,1)
    # These have even parity (determinant = +1)
    # Left-handed cycles: (0,2,1), (1,0,2), (2,1,0) have odd parity (determinant = -1)
    right_handed_cycles = {(0, 1, 2), (1, 2, 0), (2, 0, 1)}

    # Check if the axis order (dir3_idx, dir1_idx, dir2_idx) forms a right-handed cycle
    cycle = (dir3_idx, dir1_idx, dir2_idx)
    parity = 1 if cycle in right_handed_cycles else -1

    # Convert directions to signs: A/S/R = +1, P/I/L = -1
    dir1_sign = 1 if dir1 in ["A", "S", "R"] else -1
    dir2_sign = 1 if dir2 in ["A", "S", "R"] else -1

    # Determine which type of direction we need to return
    all_types = {"AP", "SI", "RL"}
    dir3_type = (all_types - {dir1_type, dir2_type}).pop()

    # Get cross product coefficient for (type at dir3_idx, type at dir1_idx, type at dir2_idx)
    # This accounts for the natural cross product direction between axis types
    coeff = _CROSS_COEFFS.get((dir3_type, dir1_type, dir2_type), 1)

    # Calculate third direction sign using the right-hand rule:
    # For a right-handed coordinate system: axis0 × axis1 = axis2
    # Combined formula: dir3_sign = parity × coeff × dir2_sign × dir1_sign
    dir3_sign = parity * coeff * dir2_sign * dir1_sign

    # Map sign to direction based on type
    if dir3_type == "RL":
        return "R" if dir3_sign > 0 else "L"
    elif dir3_type == "AP":
        return "A" if dir3_sign > 0 else "P"
    else:  # SI
        return "S" if dir3_sign > 0 else "I"


def correct_affine_for_mismatch_orientation(
    affine: Union[np.ndarray, nib.spatialimages.SpatialImage],
    real_A_is_actually_labeled_as: str,
    real_S_is_actually_labeled_as: str,
) -> np.ndarray:
    """Correct affine for orientation mismatch.

    The mismatch is caused by physical misorientation of the brain (n×90° rotation).
    This function finds the unique rotation that corrects the affine matrix.

    Args:
        affine: Affine matrix (4x4 numpy array) or nibabel image object
        real_A_is_actually_labeled_as: Where the real/physical Anterior is currently
            labeled in the image. For example, 'L' means the brain's anterior appears
            at the L-labeled position.
        real_S_is_actually_labeled_as: Where the real/physical Superior is currently
            labeled in the image. For example, 'A' means the brain's superior appears
            at the A-labeled position.

    Returns:
        Corrected affine matrix (4x4 numpy array)

    Example:
        If original orientation is RAS but real A is at L and real S is at S:
        1. RAS means x+=R, y+=A, z+=S
        2. From parameters: real A at L → real P at R; real S at S → real I at I
        3. Build target: x+ has real P → P; z+ has real S → S; y+ = ? (right-hand rule)
        4. Right-hand rule: P × ? = S → ? = R
        5. Result: PRS
    """
    # Extract affine matrix if input is a nibabel image
    if isinstance(affine, nib.spatialimages.SpatialImage):
        affine_matrix = affine.affine.copy()
    else:
        affine_matrix = affine.copy()

    _OPPOSITES = {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}

    real_A_dir = real_A_is_actually_labeled_as.upper()
    real_S_dir = real_S_is_actually_labeled_as.upper()

    # Step 1: Get current orientation code (e.g., RAS → ['R', 'A', 'S'])
    current_axes = list(aff2axcodes(affine_matrix))
    current_orientation = "".join(current_axes)

    # Step 2: Build mapping of where each real direction is labeled
    # From the two parameters, we can deduce all six directions:
    #   real_A at X → real_P at opposite(X)
    #   real_S at Y → real_I at opposite(Y)
    #   real_R and real_L are determined by right-hand rule later
    _OPPOSITES[real_A_dir]
    _OPPOSITES[real_S_dir]

    # Step 3: For each axis position, determine the correct label
    # The rule: if real X is at position labeled Y, then position Y should become X
    target_axes = [None, None, None]

    # Place A/P: real A is at real_A_dir, real P is at real_P_dir
    for i, label in enumerate(current_axes):
        if label == real_A_dir or label == _OPPOSITES[real_A_dir]:
            # This position contains real A or real P
            if label == real_A_dir:
                target_axes[i] = "A"
            else:
                target_axes[i] = "P"

    # Place S/I: real S is at real_S_dir, real I is at real_I_dir
    for i, label in enumerate(current_axes):
        if label == real_S_dir or label == _OPPOSITES[real_S_dir]:
            # This position contains real S or real I
            if label == real_S_dir:
                target_axes[i] = "S"
            else:
                target_axes[i] = "I"

    # Step 4: Determine handedness of original orientation
    # We want to preserve the handedness after correction
    handedness = _get_handedness(current_orientation)

    # Step 5: Use right/left-hand rule for the third axis (R/L)
    # Find the position that's still None
    third_pos = target_axes.index(None)

    # Find A/P and S/I positions and directions
    ap_pos = None
    si_pos = None
    ap_dir = None
    si_dir = None
    for i, d in enumerate(target_axes):
        if d in ["A", "P"]:
            ap_pos, ap_dir = i, d
        elif d in ["S", "I"]:
            si_pos, si_dir = i, d

    # Determine R/L using right-hand rule first
    rl_dir = _determine_third_orientation(
        ap_pos, si_pos, third_pos, ap_dir, si_dir, "AP", "SI"
    )

    # If original is left-handed, flip the result to preserve handedness
    if handedness == -1:
        rl_dir = _OPPOSITES[rl_dir]

    target_axes[third_pos] = rl_dir

    target_orientation = "".join(target_axes)

    # Step 6: Find the n×90° rotation that achieves target orientation
    rotations_4x4 = _get_n90_rotations_4x4()

    for rot_4x4 in rotations_4x4:
        test_affine = rot_4x4 @ affine_matrix
        test_axes = list(aff2axcodes(test_affine))
        test_orientation = "".join(test_axes)

        if test_orientation == target_orientation:
            return test_affine

    raise ValueError(
        f"Could not find n×90° rotation to transform orientation "
        f"{current_orientation} to {target_orientation}. "
        f"This may indicate the mismatch parameters are incorrect."
    )


# ---------------------------------------------------------------------------
# NIfTI image padding / cropping utilities
# ---------------------------------------------------------------------------
# NOTE: preprocessing.py has similar inline padding logic (percentage-based)
# that could be refactored to use pad_image() below.


def pad_image(
    imagef: Union[str, Path],
    outputf: Union[str, Path],
    pad_left: Union[np.ndarray, list],
    pad_right: Optional[Union[np.ndarray, list]] = None,
    logger: Optional[logging.Logger] = None,
    dtype: Optional[np.dtype] = None,
) -> None:
    """Zero-pad a NIfTI image and update the affine to keep physical coords consistent.

    Handles 3-D scalar images as well as higher-dimensional data (e.g. 4-D
    time-series or multi-component vector fields).  Only the first three
    (spatial) dimensions are padded.

    Args:
        imagef: Input NIfTI image path.
        outputf: Output path for the padded image.
        pad_left: Per-dimension left-side padding, shape ``(3,)``.
        pad_right: Per-dimension right-side padding, shape ``(3,)``.
            If *None*, symmetric padding is used (same as *pad_left*).
        logger: Optional logger instance.
        dtype: Optional dtype to cast to *before* padding.  ``get_fdata()`` always
            returns float64, so a large padded grid costs 8 bytes/voxel by default;
            passing e.g. ``np.float32`` halves the peak allocation.  *None* keeps the
            historical float64 behaviour.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    pad_left = np.asarray(pad_left, dtype=int)
    pad_right = (
        np.asarray(pad_right, dtype=int) if pad_right is not None else pad_left.copy()
    )

    img = nib.load(str(imagef))
    data = img.get_fdata()
    affine = img.affine.copy()
    header = img.header.copy()

    # Build pad_width — spatial dims get padding, extra dims (channels etc.) do not
    pad_width = [(int(left), int(right)) for left, right in zip(pad_left, pad_right)]
    for _ in range(len(data.shape) - 3):
        pad_width.append((0, 0))

    if dtype is not None:
        data = data.astype(dtype, copy=False)

    padded_data = np.pad(data, pad_width, mode="constant", constant_values=0)

    # Shift the affine origin so that the original voxels keep the same world coords
    pad_shift_world = affine[:3, :3] @ (-pad_left.astype(float))
    new_affine = affine.copy()
    new_affine[:3, 3] = affine[:3, 3] + pad_shift_world

    new_img = nib.Nifti1Image(padded_data, new_affine, header)
    if dtype is not None:
        new_img.header.set_data_dtype(dtype)
    nib.save(new_img, str(outputf))

    logger.info(
        f"Padded image from {list(data.shape[:3])} to {list(padded_data.shape[:3])} "
        f"(left={[int(v) for v in pad_left]}, right={[int(v) for v in pad_right]})"
    )


def pad_image_to_min_size(
    imagef: Union[str, Path],
    min_size: int,
    outputf: Union[str, Path],
    logger: Optional[logging.Logger] = None,
) -> Optional[np.ndarray]:
    """Zero-pad a NIfTI image so every spatial dimension is >= *min_size*.

    Symmetric zero-padding is applied to each dimension that is smaller than
    *min_size*.  When the deficit is odd, the extra voxel goes to the right
    side.

    Args:
        imagef: Input NIfTI image path.
        min_size: Minimum required size for each spatial dimension.
        outputf: Output path for the padded image.  Only written when padding
            is actually needed.
        logger: Optional logger instance.

    Returns:
        Per-side **left** padding as ``np.ndarray`` of shape ``(3,)``, or
        *None* if all dimensions were already >= *min_size* (no file written).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    img = nib.load(str(imagef))
    spatial_shape = np.array(img.shape[:3])

    if np.all(spatial_shape >= min_size):
        logger.debug(f"No padding needed: all dims {list(spatial_shape)} >= {min_size}")
        return None

    pad_total = np.maximum(min_size - spatial_shape, 0)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    pad_image(imagef, outputf, pad_left, pad_right, logger)

    return pad_left


def crop_image_to_original(
    imagef: Union[str, Path],
    ref_imagef: Union[str, Path],
    pad_left: np.ndarray,
    outputf: Union[str, Path],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Crop a padded NIfTI image back to match a reference image's grid.

    Reverses the effect of :func:`pad_image_to_min_size` by slicing out the
    padded voxels and restoring the reference image's affine.  Works for both
    scalar images and multi-component vector fields (e.g. ANTs warp fields).

    In-place operation is supported (``imagef == outputf``).

    Args:
        imagef: Padded image to crop.
        ref_imagef: Original (unpadded) reference image — its shape and affine
            define the target grid.
        pad_left: Per-dimension left padding that was applied (returned by
            :func:`pad_image_to_min_size`).
        outputf: Output path for the cropped image.
        logger: Optional logger instance.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    img = nib.load(str(imagef))
    ref = nib.load(str(ref_imagef))

    # Force eager load so in-place overwrite is safe
    data = np.asarray(img.dataobj)
    ref_shape = ref.shape[:3]
    pad_left = np.asarray(pad_left, dtype=int)

    # Crop spatial dims; preserve any extra dims (vector components, time, …)
    slices = [slice(int(p), int(p) + int(s)) for p, s in zip(pad_left, ref_shape)]
    for _ in range(len(data.shape) - 3):
        slices.append(slice(None))

    cropped_data = data[tuple(slices)]

    # Preserve intent codes / data-type metadata from the padded image header,
    # but use the reference affine so the output matches the original grid.
    new_header = img.header.copy()
    cropped_img = nib.Nifti1Image(cropped_data, ref.affine, new_header)
    nib.save(cropped_img, str(outputf))

    logger.info(
        f"Cropped image from {list(data.shape[:3])} back to {list(cropped_data.shape[:3])}"
    )


def write_inner_box_mask(
    grid_imagef: Union[str, Path],
    outputf: Union[str, Path],
    pad_left: Union[np.ndarray, list],
    inner_shape: Union[np.ndarray, list, tuple],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Write a binary mask marking a sub-box of an enlarged grid.

    Used by the conform QC snapshot to outline the cropped (target-FOV) region on
    top of the full-FOV underlay.  Shipping the box as a *mask image on the same
    grid* rather than as voxel coordinates means it travels through the same
    reorientation and slicing as the underlay, so the plotting layer needs no
    coordinate maths of its own.

    Args:
        grid_imagef: Image defining the output grid (the enlarged reference).
        outputf: Output path for the uint8 mask.
        pad_left: Per-dimension left padding that produced the enlarged grid — the
            lower corner of the box, shape ``(3,)``.
        inner_shape: Spatial shape of the box (the original, unpadded grid).
        logger: Optional logger instance.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    grid = nib.load(str(grid_imagef))
    shape = tuple(int(v) for v in grid.shape[:3])
    pad_left = np.asarray(pad_left, dtype=int)
    inner_shape = np.asarray(inner_shape, dtype=int)

    mask = np.zeros(shape, dtype=np.uint8)
    slices = tuple(
        slice(int(p), min(int(p) + int(n), int(s)))
        for p, n, s in zip(pad_left, inner_shape, shape)
    )
    mask[slices] = 1

    mask_img = nib.Nifti1Image(mask, grid.affine)
    mask_img.header.set_data_dtype(np.uint8)
    mask_img.header.set_xyzt_units("mm", "sec")
    nib.save(mask_img, str(outputf))

    logger.info(
        f"FOV box mask written: box {[int(v) for v in inner_shape]} at offset "
        f"{[int(v) for v in pad_left]} "
        f"within grid {list(shape)}"
    )
