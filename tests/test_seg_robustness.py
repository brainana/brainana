"""Segmentation-input robustness: rescale cap, label islands, second pass.

* conform() maps the 99.9th percentile of *all* voxels to 255. With non-brain
  tissue far brighter than brain (a surface receive coil) the brain was
  squeezed into a few grey levels and the CNN returned a 7 cm^3 "brain". The
  cap must leave normal images byte-identical and lift the squeezed case.
* Small detached label fragments became blobs and topological defects.
* An undersized first-pass mask triggers a second pass on an N4-corrected input.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fastsurfer_nn.data_loader.conform import (
    CENTER_CAP,
    conform,
    getscale,
    rescale_cap_report,
)
from fastsurfer_nn.postprocessing.postseg_utils import relabel_small_islands


def _head(bright_rim: float | None = None, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.zeros((64, 64, 64), np.float32)
    a[16:48, 16:48, 16:48] = rng.normal(100, 10, (32, 32, 32)).clip(1)
    a[8:56, 8:56, 8:14] = rng.normal(60, 5, (48, 48, 6)).clip(1)  # scalp
    if bright_rim is not None:
        a[4:8, 4:60, 4:60] = bright_rim  # coil-side fat, far brighter than brain
    return a


# ---------------------------------------------------------------- rescale cap


def test_cap_leaves_normal_image_identical():
    data = _head()
    img = nib.Nifti1Image(data, np.eye(4))
    old = np.asanyarray(conform(img, rescale_center_cap=None).dataobj)
    new = np.asanyarray(conform(img).dataobj)
    assert np.array_equal(old, new)
    assert rescale_cap_report(data)["applied"] is False


def test_cap_lifts_squeezed_brain():
    data = _head(bright_rim=5000.0)
    img = nib.Nifti1Image(data, np.eye(4))
    old = np.asanyarray(conform(img, rescale_center_cap=None).dataobj)
    new = np.asanyarray(conform(img).dataobj)
    brain = (slice(20, 44),) * 3
    old_p99 = np.percentile(old[brain], 99)
    new_p99 = np.percentile(new[brain], 99)
    assert old_p99 < 40, "fixture must reproduce the squeezed case"
    # Brain p99 lands near 255 / CAP (~100 grey levels), not at 255.
    assert 255 / CENTER_CAP * 0.8 < new_p99 < 255 / CENTER_CAP * 1.3
    report = rescale_cap_report(data)
    assert report["applied"] is True and report["ratio"] > CENTER_CAP


def test_getscale_default_is_classic():
    """getscale() itself only caps when asked (mri_convert parity by default)."""
    data = _head(bright_rim=5000.0)
    assert getscale(data, 0, 255) == getscale(data, 0, 255, center_cap=None)
    assert getscale(data, 0, 255, center_cap=CENTER_CAP)[1] > getscale(data, 0, 255)[1]


# -------------------------------------------------------------- label islands


def _seg():
    seg = np.zeros((40, 40, 40), np.int16)
    seg[5:35, 5:35, 5:35] = 3  # cortex-ish label filling the box
    seg[10:30, 10:30, 10:30] = 2  # "WM" inside
    seg[20, 20, 20] = 3  # 1-voxel island of 3 inside 2
    seg[12:14, 12:14, 12:14] = 7  # label 7: its only component (8 vox)
    return seg


def test_islands_disabled_by_default_value():
    seg = _seg()
    out, n, nvox = relabel_small_islands(seg, (0.5, 0.5, 0.5), 0.0)
    assert np.array_equal(out, seg) and n == 0 and nvox == 0


def test_small_island_takes_neighbour_label():
    seg = _seg()
    out, n, nvox = relabel_small_islands(seg, (0.5, 0.5, 0.5), 2.5)
    assert out[20, 20, 20] == 2
    assert n == 1 and nvox == 1


def test_largest_component_of_a_label_is_never_removed():
    """Label 7 is tiny everywhere; its only (largest) piece must survive."""
    seg = _seg()
    out, _, _ = relabel_small_islands(seg, (0.5, 0.5, 0.5), 100.0)
    assert (out == 7).sum() == 8


def test_threshold_is_a_volume_not_a_voxel_count():
    seg = _seg()
    seg[25:27, 25:27, 25:27] = 3  # 8-voxel island of 3
    # 8 voxels at 0.5 mm = 1 mm^3 -> removed at 2.5 mm^3
    out, _, _ = relabel_small_islands(seg, (0.5, 0.5, 0.5), 2.5)
    assert out[25, 25, 25] == 2
    # 8 voxels at 1 mm = 8 mm^3 -> kept at 2.5 mm^3
    out, _, _ = relabel_small_islands(seg, (1.0, 1.0, 1.0), 2.5)
    assert out[25, 25, 25] == 3


# ------------------------------------------------ mask check and second pass


def _nifti(path: Path, data, zooms=(1.0, 1.0, 1.0)):
    nib.save(nib.Nifti1Image(data, np.diag([*zooms, 1.0])), str(path))
    return path


@pytest.fixture
def seg_env(tmp_path, monkeypatch):
    """apply_segmentation's helper with the CNN and N4 mocked out."""
    from nhp_mri_prep.operations import preprocessing as pp

    image = _nifti(tmp_path / "t1w.nii.gz", _head())
    out_dir = tmp_path / "fastsurfercnn_output"
    out_dir.mkdir()
    state = {"inputs": [], "mask_side": [20, 40]}

    def fake_run_segmentation(input_image, output_dir, **_):
        state["inputs"].append(Path(input_image).name)
        side = state["mask_side"][len(state["inputs"]) - 1]
        mask = np.zeros((64, 64, 64), np.uint8)
        mask[:side, :side, :side] = 1
        return {"brain_mask": str(_nifti(Path(output_dir) / "mask.nii.gz", mask))}

    def fake_bias_correction(imagef, working_dir, output_name, maskf, **_):
        out = Path(working_dir) / output_name
        out.write_bytes(Path(imagef).read_bytes())
        return {"imagef_bias_corrected": out}

    monkeypatch.setattr(pp, "run_segmentation", fake_run_segmentation)
    monkeypatch.setattr(pp, "bias_correction", fake_bias_correction)
    # Template brain: 40^3 voxels of 1 mm = 64 cm^3
    monkeypatch.setattr(pp, "_template_brain_volume_mm3", lambda cfg: 40.0**3)

    def run(n4_cfg):
        first = fake_run_segmentation(image, out_dir)
        return pp._anat_segmentation_checks(
            image_path=image,
            result=first,
            segmentation_kwargs={"input_image": image, "output_dir": out_dir},
            temp_output_dir=out_dir,
            work_dir=tmp_path,
            fscnn_cfg={"pre_inference_n4": n4_cfg},
            config={},
            logger=__import__("logging").getLogger("test"),
        )

    return state, run, out_dir


def test_undersized_mask_triggers_second_pass(seg_env):
    state, run, out_dir = seg_env
    state["mask_side"] = [20, 40]  # 8 cm^3 then 64 cm^3 vs 64 cm^3 template
    _, qc = run({"enabled": "auto"})
    assert state["inputs"] == ["t1w.nii.gz", "pre_inference_n4.nii.gz"]
    assert qc["SegmentationPasses"] == 2
    assert qc["Pass1"]["MaskUndersized"] is True
    assert qc["MaskUndersized"] is False
    assert qc["PreInferenceN4"]["Trigger"] == "undersized mask"
    assert (out_dir / "pass1" / "mask.nii.gz").exists()


def test_normal_mask_single_pass(seg_env):
    state, run, _ = seg_env
    state["mask_side"] = [40]
    _, qc = run({"enabled": "auto"})
    assert state["inputs"] == ["t1w.nii.gz"]
    assert qc["SegmentationPasses"] == 1 and qc["MaskUndersized"] is False


def test_disabled_never_runs_second_pass(seg_env):
    state, run, _ = seg_env
    state["mask_side"] = [20]
    _, qc = run({"enabled": False})
    assert state["inputs"] == ["t1w.nii.gz"]
    assert qc["MaskUndersized"] is True  # still reported


def test_forced_second_pass(seg_env):
    state, run, _ = seg_env
    state["mask_side"] = [40, 40]
    _, qc = run({"enabled": True})
    assert qc["SegmentationPasses"] == 2
    assert qc["PreInferenceN4"]["Trigger"] == "forced"
