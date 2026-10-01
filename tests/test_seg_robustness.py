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


def test_failed_checks_keep_the_first_pass(seg_env, monkeypatch):
    """An N4 or prior error is recorded, never a failed segmentation step."""
    from nhp_mri_prep.operations import preprocessing as pp

    state, run, _ = seg_env
    state["mask_side"] = [20]

    def broken_bias_correction(**_):
        raise RuntimeError("N4 broke")

    monkeypatch.setattr(pp, "bias_correction", broken_bias_correction)
    result, qc = run({"enabled": True})
    assert state["inputs"] == ["t1w.nii.gz"]
    assert qc["SegmentationPasses"] == 1
    assert "N4 broke" in qc["SegmentationCheckFailed"]
    assert Path(result["brain_mask"]).exists()


def test_failed_second_pass_restores_the_first(seg_env):
    state, run, out_dir = seg_env
    state["mask_side"] = [20]  # the fake CNN has no second mask: pass 2 raises
    result, qc = run({"enabled": True})
    assert qc["PreInferenceN4"]["Pass2Kept"] is False
    assert qc["PreInferenceN4"]["Error"]
    assert qc["MaskVolumeCm3"] == round(20**3 / 1000, 2)
    assert Path(result["brain_mask"]) == out_dir / "mask.nii.gz"
    assert np.asanyarray(nib.load(result["brain_mask"]).dataobj).sum() == 20**3
    assert not (out_dir / "pass1").exists()


def test_second_pass_reports_the_cap_on_the_image_it_segmented(
    seg_env, tmp_path, monkeypatch
):
    """Pass 2 segments the N4 image: the top-level cap fields must describe it."""
    from nhp_mri_prep.operations import preprocessing as pp

    state, run, _ = seg_env
    # A coil-lit rim 10x brighter than the brain: the cap binds on the original.
    head = np.zeros((64, 64, 64), np.float32)
    head[4:60, 4:60, 4:60] = 1000.0
    head[16:48, 16:48, 16:48] = 100.0
    _nifti(tmp_path / "t1w.nii.gz", head)

    def flattening_n4(imagef, working_dir, output_name, maskf, **_):
        out = Path(working_dir) / output_name
        _nifti(out, np.where(head > 0, 100.0, 0.0).astype(np.float32))
        return {"imagef_bias_corrected": out}

    monkeypatch.setattr(pp, "bias_correction", flattening_n4)
    state["mask_side"] = [20, 40]
    _, qc = run({"enabled": "auto"})

    assert qc["SegmentationPasses"] == 2
    assert qc["Pass1"]["IntensityCapApplied"] is True
    assert qc["IntensityCapApplied"] is False


# ------------------------------------------------ template prior


def _prior(brain, labels):
    from nhp_mri_prep.operations.segmentation_prior import TemplatePrior

    return TemplatePrior(brain_prob=brain.astype(np.float32), labels=labels.astype(np.int32))


def _cube(lo, hi, shape=(48, 48, 48)):
    m = np.zeros(shape, bool)
    m[lo:hi, lo:hi, lo:hi] = True
    return m


def test_missed_brain_counts_a_dropped_slab():
    from nhp_mri_prep.operations.segmentation_prior import missed_brain

    template = _cube(8, 40)
    mask = template.copy()
    mask[8:40, 8:40, 30:40] = False
    stats = missed_brain(mask, np.where(template, 100.0, 0.0),
                         _prior(template, np.where(template, 7, 0)), (1.0, 1.0, 1.0))
    assert stats["MissedCm3"] > 0 and stats["Reliable"] is False


def test_missed_brain_ignores_dark_and_unlabelled_voxels():
    from nhp_mri_prep.operations.segmentation_prior import missed_brain

    template = _cube(8, 40)
    mask = template.copy()
    mask[8:40, 8:40, 30:40] = False
    dark = missed_brain(mask, np.where(mask, 100.0, 1.0),
                        _prior(template, np.where(template, 7, 0)), (1.0, 1.0, 1.0))
    unlabelled = missed_brain(mask, np.where(template, 100.0, 0.0),
                              _prior(template, np.where(mask, 7, 0)), (1.0, 1.0, 1.0))
    assert dark["MissedCm3"] == 0 and unlabelled["MissedCm3"] == 0


def test_agreeing_mask_is_reliable_and_misses_nothing():
    from nhp_mri_prep.operations.segmentation_prior import missed_brain

    template = _cube(8, 40)
    stats = missed_brain(template, np.where(template, 100.0, 0.0),
                         _prior(template, np.where(template, 7, 0)), (1.0, 1.0, 1.0))
    assert stats == {"TemplateDice": 1.0, "MissedCm3": 0.0, "Reliable": True}


def test_search_region_holds_the_brain_and_drops_the_air():
    from nhp_mri_prep.operations.segmentation_prior import (
        PRIOR_TEMPLATE_BRAINMASK, PRIOR_TEMPLATE_HEAD, _template_file, search_region,
    )

    head_img = nib.load(str(_template_file(PRIOR_TEMPLATE_HEAD)))
    brain = np.asanyarray(nib.load(str(_template_file(PRIOR_TEMPLATE_BRAINMASK))).dataobj) > 0
    region, _ = search_region(_template_file(PRIOR_TEMPLATE_HEAD))
    air = np.asanyarray(head_img.dataobj) < 1e-3
    air &= ~__import__("scipy.ndimage", fromlist=["x"]).binary_dilation(~air, iterations=3)
    # The fit region may miss dark surface voxels open to the outside (hole
    # filling cannot reach them); the fitted field still covers the image.
    assert region[brain].mean() > 0.99
    assert region.sum() > brain.sum()
    assert not region[air].any()


@pytest.fixture
def prior_env(seg_env, monkeypatch):
    """seg_env with a CNN that writes labels, a mocked prior and a recording V1 fix."""
    from fastsurfer_nn.inference import segmentation as fseg
    from nhp_mri_prep.operations import preprocessing as pp
    from nhp_mri_prep.operations import segmentation_prior as sp

    state, run, out_dir = seg_env
    template = np.zeros((64, 64, 64), bool)
    template[:44, :44, :44] = True
    calls, roi_calls = [], []

    def fake_register(image_path, work_dir, atlas_name, config, logger):
        calls.append(Path(image_path).name)
        return _prior(template, np.where(template, 7, 0))

    def fake_region(image_path):
        img = nib.load(str(image_path))
        return np.ones(img.shape[:3], bool), img

    monkeypatch.setattr(sp, "register_template_prior", fake_register)
    monkeypatch.setattr(sp, "search_region", fake_region)
    monkeypatch.setattr(fseg, "apply_roi_wm_fix",
                        lambda seg_results, input_image, atlas_name, **kw: roi_calls.append(Path(input_image).name))
    real_seg = pp.run_segmentation

    def seg_with_labels(input_image, output_dir, **kw):
        res = real_seg(input_image=input_image, output_dir=output_dir, **kw)
        mask = np.asanyarray(nib.load(res["brain_mask"]).dataobj)
        res["segmentation"] = str(_nifti(Path(output_dir) / "segmentation.nii.gz", (mask * 3).astype(np.int16)))
        res["hemimask"] = str(_nifti(Path(output_dir) / "mask_hemi.nii.gz", mask.astype(np.int16)))
        res["atlas_name"] = "ARM2"
        res["input_image"] = str(input_image)
        return res

    monkeypatch.setattr(pp, "run_segmentation", seg_with_labels)
    image = out_dir.parent / "t1w.nii.gz"

    def run_prior(n4_cfg, prior_cfg):
        first = seg_with_labels(input_image=image, output_dir=out_dir)
        return pp._anat_segmentation_checks(
            image_path=image,
            result=first,
            segmentation_kwargs={"input_image": image, "output_dir": out_dir},
            temp_output_dir=out_dir,
            work_dir=out_dir.parent,
            fscnn_cfg={"pre_inference_n4": n4_cfg, "template_prior": prior_cfg},
            config={},
            logger=__import__("logging").getLogger("test"),
            roi_fix={"roi_name": "V1"},
        )

    return state, run_prior, calls, roi_calls


def test_agreeing_first_pass_runs_once(prior_env):
    state, run_prior, calls, roi_calls = prior_env
    state["mask_side"] = [44]
    _, qc = run_prior({"enabled": "auto"}, {"enabled": True})
    assert state["inputs"] == ["t1w.nii.gz"]
    assert calls == ["pre_inference_n4.nii.gz"]
    assert qc["TemplatePrior"]["MissedCm3"] == 0 and qc["TemplatePrior"]["Reliable"] is True
    assert roi_calls == ["t1w.nii.gz"]  # V1 fix once, on pass 1


def test_small_brain_is_not_a_failure_when_the_prior_agrees(prior_env, monkeypatch):
    """Under 80% of the template volume, but the registered template agrees."""
    from nhp_mri_prep.operations import preprocessing as pp

    state, run_prior, _, _ = prior_env
    monkeypatch.setattr(pp, "_template_brain_volume_mm3", lambda cfg: 120.0**3)
    state["mask_side"] = [44]
    _, qc = run_prior({"enabled": "auto"}, {"enabled": True})
    assert qc["MaskUndersized"] is True and qc["SegmentationPasses"] == 1


def test_undersized_fallback_without_prior(prior_env):
    state, run_prior, calls, _ = prior_env
    state["mask_side"] = [20, 40]
    _, qc = run_prior({"enabled": "auto"}, {"enabled": False})
    assert calls == [] and qc["PreInferenceN4"]["Trigger"] == "undersized mask"


def test_missed_brain_triggers_second_pass_and_keeps_the_better_one(prior_env):
    state, run_prior, _, roi_calls = prior_env
    state["mask_side"] = [38, 42]  # pass 2 agrees better with the template
    result, qc = run_prior({"enabled": "auto"}, {"enabled": True})
    assert qc["PreInferenceN4"]["Trigger"] == "template prior"
    assert qc["PreInferenceN4"]["Pass2Kept"] is True
    assert roi_calls == ["pre_inference_n4.nii.gz"]  # V1 fix once, on pass 2
    assert np.asanyarray(nib.load(result["brain_mask"]).dataobj).sum() == 42**3


def test_worse_second_pass_is_rejected(prior_env):
    state, run_prior, _, roi_calls = prior_env
    state["mask_side"] = [38, 30]  # pass 2 agrees worse
    result, qc = run_prior({"enabled": "auto"}, {"enabled": True})
    assert qc["PreInferenceN4"]["Pass2Kept"] is False
    assert qc["MaskVolumeCm3"] == round(38**3 / 1000, 2)
    assert np.asanyarray(nib.load(result["brain_mask"]).dataobj).sum() == 38**3
    assert roi_calls == ["t1w.nii.gz"]
    assert (Path(result["brain_mask"]).parent / "pass2_rejected" / "mask.nii.gz").exists()


def test_undersized_mask_with_a_small_miss_triggers(prior_env, monkeypatch):
    """Two weak signals together: under 80% of the template, and a small miss."""
    from nhp_mri_prep.operations import preprocessing as pp
    from nhp_mri_prep.operations import segmentation_prior as sp

    state, run_prior, _, _ = prior_env
    monkeypatch.setattr(sp, "missed_brain", lambda mask, image, prior, zooms: {
        "TemplateDice": 0.95, "MissedCm3": 0.4, "Reliable": True})
    monkeypatch.setattr(pp, "_template_brain_volume_mm3", lambda cfg: 120.0**3)
    state["mask_side"] = [44, 44]
    _, qc = run_prior({"enabled": "auto"}, {"enabled": True})
    assert qc["PreInferenceN4"]["Trigger"] == "undersized mask + template prior"

    state["inputs"].clear()
    state["mask_side"] = [44]
    monkeypatch.setattr(pp, "_template_brain_volume_mm3", lambda cfg: 40.0**3)
    _, qc = run_prior({"enabled": "auto"}, {"enabled": True})
    assert qc["SegmentationPasses"] == 1  # same miss, normal-sized mask
