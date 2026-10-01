"""surface_qc.json flags a pial surface that collapsed onto the white surface.

A field report had one hemisphere's pial 0.35 mm from its white surface and
nothing in the outputs said so. Across 912 PRIME-DE hemispheres the lowest
median thickness was 1.42 mm, so a median under 1.0 mm is flagged.
"""

from types import SimpleNamespace

import nibabel.freesurfer.io as fsio
import numpy as np

from fastsurfer_surfrecon.pipeline import ReconSurfPipeline


def _summary(tmp_path, values):
    surf = tmp_path / "surf"
    surf.mkdir(exist_ok=True)
    fsio.write_morph_data(str(surf / "lh.thickness"), np.asarray(values, np.float32))
    fake = SimpleNamespace(
        sd=SimpleNamespace(surf_dir=surf),
        COLLAPSED_THICKNESS_MM=ReconSurfPipeline.COLLAPSED_THICKNESS_MM,
    )
    return ReconSurfPipeline._thickness_summary(fake, "lh")


def test_normal_cortex_not_flagged(tmp_path):
    s = _summary(tmp_path, [0.0] * 100 + [1.8] * 900 + [0.4] * 10)
    assert s["collapsed"] is False
    assert s["median_mm"] == 1.8  # medial-wall zeros excluded


def test_collapsed_pial_flagged(tmp_path):
    s = _summary(tmp_path, [0.35] * 1000)
    assert s["collapsed"] is True
    assert s["frac_below_0.5mm"] == 1.0


def test_missing_thickness_is_none(tmp_path):
    fake = SimpleNamespace(sd=SimpleNamespace(surf_dir=tmp_path), COLLAPSED_THICKNESS_MM=1.0)
    assert ReconSurfPipeline._thickness_summary(fake, "lh") is None
