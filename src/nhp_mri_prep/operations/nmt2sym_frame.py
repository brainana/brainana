"""Where a subject image sits relative to NMT2Sym world, and a header-only move into it.

Several anatomical steps register NMT2Sym material to the subject: the
segmentation's template prior and its N4 search region, the V1 white-matter fix
and the template surface prior. They register with FireANTs, which refines a
nearly aligned pair but does not reliably recover a large translation or
rotation. The pipeline's own registration gets its coarse alignment from conform
(a rigid alignment to the output template). These steps target NMT2Sym whatever
the output space is, so they need the subject roughly in NMT2Sym world first:

- **identity**: conform aligned the subject to NMT2Sym or NMT2Asym, which share
  NMT2Sym's world frame. Nothing changes.
- **table**: conform aligned the subject to another bundled template (MEBRAINS,
  D99, Yerkes19). Those use an AC-type origin about 25 mm from NMT2's, and D99
  is also pitched about 10 degrees. A rigid template-to-NMT2Sym transform,
  computed once offline (``scripts/dev/nhp_mriprep/make_template_nmt2sym_xfm.py``),
  is read from ``template_zoo/template/xfm/``.
- **rigid**: a custom template, or conform off (scanner coordinates). A rigid
  registration of the subject's brain to the NMT2Sym brain, with the same backend
  as conform (``anat.conform.rigid_method``: FLIRT, or SimpleITK in Brainana Lite).

The result is applied by rewriting only the image header (:func:`reframe`), so
every voxel array, and every result warped back onto it, stays on the subject's
own voxel grid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import nibabel as nib
import numpy as np

from ..utils.nextflow import config_value
from ..utils.templates import is_custom_template_path

logger = logging.getLogger(__name__)

NMT2SYM_BRAIN_SPEC = "NMT2Sym:res-05"  # resolves to tpl-NMT2Sym_res-05_T1w_brain
# Templates that share NMT2Sym's world frame (NMT2Asym: brain-mask Dice 0.991
# with no transform at all, the same as after an affine registration).
SAME_FRAME_FAMILIES = ("NMT2Sym", "NMT2Asym")
XFM_DIR = Path(__file__).resolve().parents[3] / "template_zoo" / "template" / "xfm"
# The rigid registration needs room around the brain: the conformed image is
# cropped tightly to the template grid, and SimpleITK refused such an input
# ("fewer than 4 pixels along direction 2").
RIGID_PAD_VOXELS = 20

_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])  # RAS <-> LPS (its own inverse)


@dataclass(frozen=True)
class Nmt2SymFrame:
    """Subject world -> NMT2Sym world (4x4, RAS mm), and where it came from."""

    to_nmt2sym: np.ndarray
    source: str  # "identity" | "table" | "rigid"
    detail: str = ""

    @property
    def is_identity(self) -> bool:
        return self.source == "identity"

    def provenance(self) -> Dict[str, Any]:
        """For sidecars and step metadata."""
        record: Dict[str, Any] = {"Source": self.source}
        if self.detail:
            record["Detail"] = self.detail
        if not self.is_identity:
            offset = self.to_nmt2sym[:3, 3]
            record["TranslationMm"] = round(float(np.linalg.norm(offset)), 1)
        return record


def frame_source(config: Dict[str, Any]) -> tuple[str, Optional[str]]:
    """How the subject relates to NMT2Sym world: ("identity"|"table"|"rigid", family)."""
    if not config_value(config, "anat.conform.enabled", True):
        return "rigid", None
    spec = str(config_value(config, "template.output_space", "NMT2Sym:res-05"))
    if is_custom_template_path(spec):
        return "rigid", None
    family = spec.split(":")[0]
    if family in SAME_FRAME_FAMILIES:
        return "identity", family
    if table_path(family).exists():
        return "table", family
    return "rigid", family


def table_path(family: str) -> Path:
    return XFM_DIR / f"from-{family}_to-NMT2Sym_mode-image_desc-rigid_xfm.txt"


# --- ITK text transforms -------------------------------------------------------
#
# ITK/ANTs transforms work in LPS and, like an image transform, map points of the
# fixed space to the moving space. "from-X_to-NMT2Sym_mode-image" resamples X
# into NMT2Sym, so as a point map it takes NMT2Sym points to X points.


def write_itk_affine(path: Union[str, Path], lps_point_map: np.ndarray) -> Path:
    """Write a 4x4 LPS point map as an ITK text AffineTransform (centre 0)."""
    m = np.asarray(lps_point_map, dtype=float)
    params = " ".join(f"{v:.10g}" for v in [*m[:3, :3].ravel(), *m[:3, 3]])
    path = Path(path)
    path.write_text(
        "#Insight Transform File V1.0\n#Transform 0\n"
        "Transform: AffineTransform_double_3_3\n"
        f"Parameters: {params}\nFixedParameters: 0 0 0\n"
    )
    return path


def read_itk_affine(path: Union[str, Path]) -> np.ndarray:
    """Read an ITK text affine/rigid transform as a 4x4 LPS point map."""
    params = fixed = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("Parameters:"):
            params = [float(v) for v in line.split(":", 1)[1].split()]
        elif line.startswith("FixedParameters:"):
            fixed = [float(v) for v in line.split(":", 1)[1].split()]
    if params is None or len(params) != 12:
        raise ValueError(f"{path}: not a 3-D ITK affine transform")
    rot = np.array(params[:9]).reshape(3, 3)
    centre = np.array(fixed if fixed else [0.0, 0.0, 0.0])
    out = np.eye(4)
    out[:3, :3] = rot
    # ITK: p' = R (p - c) + c + t
    out[:3, 3] = np.array(params[9:]) + centre - rot @ centre
    return out


def lps_to_ras(m: np.ndarray) -> np.ndarray:
    """A 4x4 point map in LPS as the same map in RAS (and back: the flip is its own inverse)."""
    return _LPS @ m @ _LPS


ras_to_lps = lps_to_ras


def load_table(family: str) -> np.ndarray:
    """Template-``family`` world -> NMT2Sym world (RAS), from the bundled rigid."""
    nmt2sym_to_family = lps_to_ras(read_itk_affine(table_path(family)))
    return np.linalg.inv(nmt2sym_to_family)


# --- runtime rigid -------------------------------------------------------------


def _padded(img: nib.Nifti1Image, pad: int) -> nib.Nifti1Image:
    data = np.asarray(img.dataobj, dtype=np.float32)
    out = np.zeros(tuple(n + 2 * pad for n in data.shape[:3]), np.float32)
    out[pad:-pad, pad:-pad, pad:-pad] = data
    affine = img.affine.copy()
    affine[:3, 3] = nib.affines.apply_affine(img.affine, [-pad] * 3)
    return nib.Nifti1Image(out, affine)


def rigid_to_nmt2sym(
    brain_file: Union[str, Path],
    work_dir: Union[str, Path],
    config: Dict[str, Any],
    log: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Subject world -> NMT2Sym world (RAS) from a rigid of a skull-stripped brain.

    Same backend as conform (``anat.conform.rigid_method``): FLIRT, or SimpleITK.
    """
    from .sitk_rigid_registration import fsl_mat_to_world_affine
    from ..utils.templates import resolve_template

    log = log or logger
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    nmt2sym = resolve_template(NMT2SYM_BRAIN_SPEC)
    moving = work_dir / "brain_padded.nii.gz"
    nib.save(_padded(nib.load(str(brain_file)), RIGID_PAD_VOXELS), str(moving))

    method = str(config_value(config, "anat.conform.rigid_method", "flirt"))
    if method == "sitk":
        from .sitk_rigid_registration import sitk_config_for_modality, sitk_register

        sitk_register(
            fixedf=Path(nmt2sym),
            movingf=moving,
            work_dir=work_dir,
            output_prefix="to_nmt2sym",
            sitk_config=sitk_config_for_modality("anat"),
            modality="anat",
        )
        nmt2sym_to_subject_lps = np.loadtxt(str(work_dir / "to_nmt2sym.world.mat"))
    else:
        from .registration import flirt_config_for_modality, flirt_register

        result = flirt_register(
            fixedf=nmt2sym,
            movingf=str(moving),
            working_dir=str(work_dir),
            output_prefix="to_nmt2sym",
            config=flirt_config_for_modality("anat"),
            logger=log,
            dof=6,
        )
        nmt2sym_to_subject_lps = fsl_mat_to_world_affine(
            result["forward_transform"], nmt2sym, str(moving)
        )
    return np.linalg.inv(lps_to_ras(nmt2sym_to_subject_lps))


def nmt2sym_frame(
    config: Dict[str, Any],
    work_dir: Union[str, Path],
    brain_file: Optional[Union[str, Path]] = None,
    log: Optional[logging.Logger] = None,
) -> Nmt2SymFrame:
    """Where this run's anatomical images sit relative to NMT2Sym world.

    ``brain_file`` (a skull-stripped image in the subject's world) is needed only
    for the runtime rigid; identity and table never look at the image.
    Raises when the rigid is needed and fails.
    """
    log = log or logger
    source, family = frame_source(config)
    if source == "identity":
        return Nmt2SymFrame(np.eye(4), "identity", family or "")
    if source == "table":
        log.info(f"[NMT2Sym frame] {family} output space: bundled rigid {table_path(family).name}")
        return Nmt2SymFrame(load_table(family), "table", table_path(family).name)
    if brain_file is None:
        raise ValueError("a skull-stripped brain is needed to register to NMT2Sym")
    method = str(config_value(config, "anat.conform.rigid_method", "flirt"))
    why = "conform off" if not config_value(config, "anat.conform.enabled", True) else (
        f"output space {family}" if family else "custom template"
    )
    log.info(f"[NMT2Sym frame] {why}: rigid ({method}) of the brain to NMT2Sym")
    matrix = rigid_to_nmt2sym(brain_file, work_dir, config, log)
    return Nmt2SymFrame(matrix, "rigid", method)


def reframe(
    image_file: Union[str, Path], frame: Nmt2SymFrame, out_file: Union[str, Path]
) -> Path:
    """Header-only copy of ``image_file`` placed in NMT2Sym world (same voxels).

    Identity returns ``image_file`` itself. Anything registered to the copy and
    resampled onto it lands on the original voxel grid.
    """
    if frame.is_identity:
        return Path(image_file)
    img = nib.load(str(image_file))
    affine = frame.to_nmt2sym @ img.affine
    out = nib.Nifti1Image(np.asanyarray(img.dataobj), affine, img.header.copy())
    out.set_qform(affine, 1)
    out.set_sform(affine, 1)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(out_file))
    return out_file
