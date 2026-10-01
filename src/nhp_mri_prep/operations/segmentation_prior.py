"""Template prior for the CNN brain segmentation: where brain should be.

The segmentation network drops brain it sees as too dark -- the far side of a
surface receive coil, a lobe under a strong bias field -- with high
confidence, and the mask it returns can look plausible by volume. The
template, registered to the subject, says where brain should be. This module
only measures; it never edits a segmentation:

- :func:`search_region` -- the N4 fitting region: the template brain, placed
  by conform's rigid alignment alone and widened, within the head. It
  contains the brain whatever its size, excludes the air behind a thin scalp,
  and excludes the background of MP2RAGE UNI images (brighter than brain, so
  no intensity threshold alone separates head from air there).
- :func:`register_template_prior` -- the template head registered to the
  subject, with its brain mask and atlas warped along.
- :func:`missed_brain` -- how much brain-like tissue the template places
  outside the CNN mask, and the Dice between the two.

The CNN always segments in the NMT2Sym grid it was trained on, so the prior is
the bundled NMT2Sym template whatever ``template.output_space`` is.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

PRIOR_TEMPLATE_HEAD = "tpl-NMT2Sym_res-05_T1w.nii.gz"
PRIOR_TEMPLATE_BRAINMASK = "tpl-NMT2Sym_res-05_brainmask.nii.gz"
PRIOR_ATLAS = "atlas-{atlas}_space-NMT2Sym_res-05.nii.gz"

# The rigidly placed template brain widened by this much holds every brain:
# 10 mm held 100% of every final brain mask across 28 subjects of 65-136 cm^3
# (6 mm already held 99.7-100%).
SEARCH_REGION_MARGIN_MM = 15.0
# Head mask: voxels above this fraction of the central-box brain reference.
HEAD_THRESHOLD_FRACTION = 0.1
# Missed tissue must lie this deep inside the warped template brain, so a
# registration error at the brain edge is not counted.
MISSED_CORE_MM = 1.0
# Missed tissue must fall inside this percentile range of the CNN brain.
MISSED_INTENSITY_PERCENTILES = (2.0, 99.5)
# Opening radius: thin slivers a registration error leaves along the ventral
# surface are not counted; a missed gyrus or lobe is.
MISSED_OPENING_MM = 1.0
# Below this Dice between the CNN mask and the warped template brain the
# registration is taken as unreliable. Correct segmentations register at
# 0.94-0.98; a small brain the template did not shrink onto at 0.87-0.90.
MIN_TEMPLATE_DICE = 0.92


def _template_file(name: str) -> Path:
    from fastsurfer_nn.inference.segmentation import _resolve_template_path

    return _resolve_template_path(name)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


def head_mask(data: np.ndarray) -> np.ndarray:
    """Tissue: largest component above 10% of brain intensity, closed and hole-filled.

    Excludes the air around the head. In MP2RAGE UNI images the background is
    brighter than brain and this keeps the whole field of view, which is why
    :func:`search_region` also bounds the region by the template.
    """
    from fastsurfer_nn.data_loader.conform import center_reference

    mask = _largest_component(data > HEAD_THRESHOLD_FRACTION * center_reference(data))
    mask = ndi.binary_closing(mask, iterations=6)
    for axis in range(mask.ndim):
        mask = np.stack(
            [ndi.binary_fill_holes(s) for s in np.moveaxis(mask, axis, 0)], axis
        )
    return mask


def search_region(image_path: Union[str, Path]) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """Where the brain can be: the placed template brain + margin, within the head.

    The conformed image is already rigidly aligned to NMT2Sym, so the template
    brain mask resampled by world coordinates lands on the subject's brain up
    to scale and shape; ``SEARCH_REGION_MARGIN_MM`` absorbs both (a container,
    not an outline). Intersecting with :func:`head_mask` drops the air it
    would otherwise reach past a thin scalp -- behind the occipital pole, air
    in the N4 fit darkens the pole and the CNN drops it -- while on UNI images,
    whose head mask is the whole field of view, the template bound keeps the
    bright background out.
    """
    from nibabel.processing import resample_from_to

    img = nib.load(str(image_path))
    brain = nib.load(str(_template_file(PRIOR_TEMPLATE_BRAINMASK)))
    placed = np.asanyarray(
        resample_from_to(brain, (img.shape[:3], img.affine), order=0).dataobj
    ) > 0
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    region = ndi.distance_transform_edt(~placed, sampling=zooms) <= SEARCH_REGION_MARGIN_MM
    region &= head_mask(np.asanyarray(img.dataobj).astype(np.float32))
    return region, img


@dataclass
class TemplatePrior:
    """The template brain and atlas warped to the subject's grid."""

    brain_prob: np.ndarray  # warped template brain mask, linear interpolation (0-1)
    labels: np.ndarray  # warped atlas, nearest neighbour
    engine: Optional[str] = None


def register_template_prior(
    image_path: Union[str, Path],
    work_dir: Union[str, Path],
    atlas_name: str,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> TemplatePrior:
    """Register the template head to ``image_path`` and warp the brain mask and atlas."""
    from fastsurfer_nn.inference.segmentation import _resolve_atlas_path

    from .registration import ants_apply_transforms, ants_register

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    reg_config = {
        "registration": dict(config.get("registration") or {}),
        "general": dict(config.get("general") or {}),
    }
    reg = ants_register(
        fixedf=str(image_path),
        movingf=str(_template_file(PRIOR_TEMPLATE_HEAD)),
        working_dir=str(work_dir),
        output_prefix="template_to_subject",
        config=reg_config,
        logger=logger,
        xfm_type="syn",
        compute_inverse=False,
        enable_fireants=bool(reg_config["registration"].get("enable_fireants", True)),
    )
    warped = {}
    for key, src, interp in (
        ("brain", _template_file(PRIOR_TEMPLATE_BRAINMASK), "Linear"),
        ("atlas", _resolve_atlas_path(PRIOR_ATLAS.format(atlas=atlas_name)), "NearestNeighbor"),
    ):
        name = f"template_{key}_in_subject.nii.gz"
        ants_apply_transforms(
            movingf=str(src),
            moving_type=0,
            interpolation=interp,
            outputf_name=name,
            fixedf=str(image_path),
            working_dir=str(work_dir),
            transformf=reg["forward_transform"],
            logger=logger,
            generate_tmean=False,
        )
        warped[key] = np.asanyarray(nib.load(str(work_dir / name)).dataobj)
    return TemplatePrior(
        brain_prob=np.asarray(warped["brain"], dtype=np.float32),
        labels=np.rint(warped["atlas"]).astype(np.int32),
        engine=reg.get("engine") if isinstance(reg, dict) else None,
    )


def missed_brain(
    mask: np.ndarray,
    image: np.ndarray,
    prior: TemplatePrior,
    zooms: Tuple[float, float, float],
) -> Dict[str, Any]:
    """Brain the template places outside the CNN mask, and the overall agreement.

    Counted: voxels deeper than ``MISSED_CORE_MM`` inside the warped template
    brain, labelled by the warped atlas (the template brain mask also covers
    the pituitary and olfactory region, which the atlas does not), outside the
    CNN mask, within the CNN brain's intensity range, thicker than
    ``MISSED_OPENING_MM`` and connected to the CNN mask.
    """
    mask = mask > 0
    template = prior.brain_prob > 0.5
    zooms = tuple(float(z) for z in zooms)
    vox_cm3 = float(np.prod(zooms)) / 1000.0

    depth = ndi.distance_transform_edt(template, sampling=zooms)
    lo, hi = np.percentile(image[mask], MISSED_INTENSITY_PERCENTILES)
    missed = (
        (depth > MISSED_CORE_MM)
        & (prior.labels != 0)
        & ~mask
        & (image >= lo)
        & (image <= hi)
    )
    radius = max(1, int(round(MISSED_OPENING_MM / min(zooms))))
    missed = ndi.binary_opening(missed, ndi.generate_binary_structure(3, 1), iterations=radius)
    lab, _ = ndi.label(missed | mask)
    attached = np.unique(lab[mask])
    missed &= np.isin(lab, attached[attached > 0])

    denom = mask.sum() + template.sum()
    dice = float(2 * (mask & template).sum() / denom) if denom else 0.0
    return {
        "TemplateDice": round(dice, 3),
        "MissedCm3": round(float(missed.sum()) * vox_cm3, 2),
        "Reliable": dice >= MIN_TEMPLATE_DICE,
    }
