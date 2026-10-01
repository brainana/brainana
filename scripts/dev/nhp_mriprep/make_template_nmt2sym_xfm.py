"""Build template_zoo/template/xfm/: rigid transforms from each bundled template to NMT2Sym.

The template prior, the V1 white-matter fix and the template surface register NMT2Sym
material to the subject with FireANTs, which needs the pair roughly aligned first
(see src/nhp_mri_prep/operations/nmt2sym_frame.py). When conform aligned the subject
to MEBRAINS, D99 or Yerkes19, these files supply that alignment.

For each template: an ANTs translation -> rigid -> affine registration of its brain to
the NMT2Sym res-05 brain (mutual information, centre-of-mass start), then the rigid
part of that affine (polar decomposition), with the translation chosen so the NMT2Sym
brain centroid lands where the affine puts it. Only the rigid is used: FireANTs
refines scale and shape, and a template-to-template warp would bias the prior.

Writes, per template X:
  from-X_to-NMT2Sym_mode-image_desc-rigid_xfm.txt   ITK text, LPS. Resamples X into
      NMT2Sym; as a point map it takes NMT2Sym points to X points (ANTs convention).
  from-X_to-NMT2Sym_mode-image_desc-rigid_xfm.json  how it was made, the affine it came
      from, and brain-mask Dice with NMT2Sym (identity / rigid / affine).

Run from the repo root with the venv active (needs ANTs):
    python scripts/dev/nhp_mriprep/make_template_nmt2sym_xfm.py [MEBRAINS D99 Yerkes19]
"""

import json
import logging
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from nibabel.processing import resample_from_to

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from nhp_mri_prep.config.config_io import get_default_config  # noqa: E402
from nhp_mri_prep.operations.nmt2sym_frame import (  # noqa: E402
    XFM_DIR,
    lps_to_ras,
    table_path,
    write_itk_affine,
)
from nhp_mri_prep.operations.registration import ants_register  # noqa: E402

ZOO = REPO / "template_zoo" / "template"
NMT_BRAIN = ZOO / "NMT2Sym" / "tpl-NMT2Sym_res-05_T1w_brain.nii.gz"
NMT_MASK = ZOO / "NMT2Sym" / "tpl-NMT2Sym_res-05_brainmask.nii.gz"
TEMPLATES = {  # brain image to register, mask to score (MEBRAINS res-05 has no mask)
    "MEBRAINS": ("MEBRAINS/tpl-MEBRAINS_res-05_T1w_brain", "MEBRAINS/tpl-MEBRAINS_res-05_T1w_brain"),
    "D99": ("D99/tpl-D99_res-05_T1w_brain", "D99/tpl-D99_res-05_brainmask"),
    "Yerkes19": ("Yerkes19/tpl-Yerkes19_res-05_T1w_brain", "Yerkes19/tpl-Yerkes19_res-05_brainmask"),
}

log = logging.getLogger("make_template_nmt2sym_xfm")


def lps_point_map(composite: Path) -> np.ndarray:
    """4x4 LPS point map (NMT2Sym -> template) of an affine ANTs composite."""
    tx = sitk.ReadTransform(str(composite))
    probe = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], float)
    moved = np.array([tx.TransformPoint(tuple(p)) for p in probe])
    out = np.eye(4)
    out[:3, :3] = ((moved[1:] - moved[0]) / 10).T
    out[:3, 3] = moved[0]
    return out


def rigid_part(affine: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Closest rotation to ``affine``, agreeing with it at ``centre`` (all LPS)."""
    u, _, vt = np.linalg.svd(affine[:3, :3])
    rot = u @ vt
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = affine[:3, :3] @ centre + affine[:3, 3] - rot @ centre
    return out


def dice_in_nmt(mask_file: Path, nmt_to_template_ras: np.ndarray) -> float:
    """Dice of the template mask, placed in NMT2Sym world by the map, with NMT2Sym's."""
    img = nib.load(str(mask_file))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    # template voxel -> template world -> NMT2Sym world
    moved = nib.Nifti1Image((data > 0).astype(np.uint8), np.linalg.inv(nmt_to_template_ras) @ img.affine)
    ref = nib.load(str(NMT_MASK))
    a = np.asanyarray(resample_from_to(moved, ref, order=0).dataobj) > 0
    b = np.asanyarray(ref.dataobj) > 0
    return round(float(2 * (a & b).sum() / (a.sum() + b.sum())), 3)


def main(names):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    XFM_DIR.mkdir(parents=True, exist_ok=True)
    nmt_mask = nib.load(str(NMT_MASK))
    centroid_ras = nib.affines.apply_affine(
        nmt_mask.affine, np.argwhere(np.asanyarray(nmt_mask.dataobj) > 0).mean(0)
    )
    centroid_lps = centroid_ras * [-1, -1, 1]
    ants_version = subprocess.run(
        ["antsRegistration", "--version"], capture_output=True, text=True
    ).stdout.split("\n")[0].strip()

    for name in names:
        brain, mask = (ZOO / f"{p}.nii.gz" for p in TEMPLATES[name])
        with tempfile.TemporaryDirectory(prefix=f"{name}_to_nmt2sym_") as tmp:
            result = ants_register(
                fixedf=str(NMT_BRAIN),
                movingf=str(brain),
                working_dir=tmp,
                output_prefix=f"{name}_to_NMT2Sym",
                config=get_default_config(),
                logger=log,
                xfm_type="affine",
                compute_inverse=False,
                enable_fireants=False,
            )
            affine_lps = lps_point_map(Path(result["forward_transform"]))
        rigid_lps = rigid_part(affine_lps, centroid_lps)
        out = write_itk_affine(table_path(name), rigid_lps)

        dice = {
            "identity": dice_in_nmt(mask, np.eye(4)),
            "rigid": dice_in_nmt(mask, lps_to_ras(rigid_lps)),
            "affine": dice_in_nmt(mask, lps_to_ras(affine_lps)),
        }
        scales = np.linalg.svd(affine_lps[:3, :3], compute_uv=False)
        record = {
            "Description": (
                f"Rigid {name} -> NMT2Sym (ITK text, LPS). mode-image: resamples {name} "
                f"into NMT2Sym; as a point map it takes NMT2Sym points to {name} points."
            ),
            "Method": (
                "ANTs translation->rigid->affine (MI, centre-of-mass start) of "
                f"{brain.name} to {NMT_BRAIN.name}; rigid = polar decomposition of the "
                "affine, agreeing with it at the NMT2Sym brain centroid"
            ),
            "Software": ants_version,
            "Date": date.today().isoformat(),
            "AffineLPS": np.round(affine_lps, 6).tolist(),
            "AffineScales": np.round(scales, 4).tolist(),
            "TranslationMm": round(float(np.linalg.norm(rigid_lps[:3, 3])), 2),
            "BrainMaskDiceWithNMT2Sym": dice,
        }
        out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
        log.info(f"{name}: {out.name}  Dice {dice}  scales {np.round(scales, 3)}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(TEMPLATES))
