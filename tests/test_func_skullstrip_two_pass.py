"""Func skull strip in two passes: N4 inside the dilated pass-1 mask, then the network again.

The func tmean's whole-field N4 leaves a receive-coil gradient, and the network drops the
dark lobe. The second pass runs on an N4 image fitted inside the first mask (dilated, within
the field of view); the registration input becomes that image inside the final mask.
"""

import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from nhp_mri_prep.operations import preprocessing as pp

FLOOR = 13.5  # conform pads with a constant floor, not 0
SHAPE = (40, 40, 40)


def _nifti(path: Path, data, zooms=(1.0, 1.0, 1.0)) -> Path:
    nib.save(nib.Nifti1Image(np.asarray(data, np.float32), np.diag([*zooms, 1.0])), str(path))
    return path


def _cube(lo, hi):
    m = np.zeros(SHAPE, bool)
    m[lo:hi, lo:hi, lo:hi] = True
    return m


@pytest.fixture
def two_pass(tmp_path, monkeypatch):
    """func_skullstrip_two_pass with the network and N4 mocked out."""
    data = np.full(SHAPE, FLOOR, np.float32)
    data[2:38, 2:38, 2:38] = 50.0  # acquired field of view
    data[10:30, 10:30, 10:30] = 100.0
    image = _nifti(tmp_path / "boldref.nii.gz", data)
    state = {"masks": [_cube(12, 28), _cube(11, 29)], "inputs": [], "n4": [], "n4_fail": False}

    def fake_skullstrip(imagef, modal, working_dir, output_name, config=None, logger=None):
        state["inputs"].append(Path(imagef).name)
        wd = Path(working_dir); wd.mkdir(parents=True, exist_ok=True)
        mask = state["masks"][len(state["inputs"]) - 1]
        mask_p = _nifti(wd / "brain_mask.nii.gz", mask)
        brain_p = _nifti(wd / output_name, nib.load(str(imagef)).get_fdata() * mask)
        return {"imagef_skullstripped": brain_p, "brain_mask": mask_p}

    def fake_bias_correction(imagef, working_dir, modal, output_name, config, logger=None, maskf=None):
        if state["n4_fail"]:
            raise RuntimeError("N4 exploded")
        state["n4"].append({"modal": modal, "maskf": maskf, "cfg": config[modal]["bias_correction"],
                            "region": nib.load(str(maskf)).get_fdata() > 0})
        out = _nifti(Path(working_dir) / output_name, nib.load(str(imagef)).get_fdata() * 2)
        return {"imagef_bias_corrected": out}

    monkeypatch.setattr(pp, "apply_skullstripping", fake_skullstrip)
    monkeypatch.setattr(pp, "bias_correction", fake_bias_correction)

    def run():
        return pp.func_skullstrip_two_pass(image, tmp_path / "work", "func_brain.nii.gz",
                                           config={"func": {}}, logger=logging.getLogger("test"))

    return state, run, data


def test_second_pass_runs_on_masked_n4(two_pass):
    state, run, _ = two_pass
    res = run()
    assert state["inputs"] == ["boldref.nii.gz", "func_n4_in_brain.nii.gz"]
    (call,) = state["n4"]
    assert call["modal"] == "func" and call["maskf"] is not None
    assert call["cfg"]["bspline_fitting"] == pp.FUNC_N4_BSPLINE == 80
    assert call["cfg"]["rescale_mean_to_100"] is False
    qc = res["skullstrip_qc"]
    assert qc["SkullstripPasses"] == 2
    assert np.array_equal(nib.load(str(res["brain_mask"])).get_fdata() > 0, _cube(11, 29))
    assert res["imagef_bias_corrected"].name == "func_n4_in_brain.nii.gz"


def test_n4_region_is_dilated_mask_within_fov(two_pass):
    state, run, data = two_pass
    run()
    region = state["n4"][0]["region"]
    mask1 = _cube(12, 28)
    assert region[mask1].all()
    assert region[12 - 6, 20, 20] and not region[12 - 7, 20, 20]  # 6 mm margin at 1 mm voxels
    assert not region[data <= FLOOR].any()  # never outside the acquired field of view


def test_registration_input_is_n4_inside_final_mask(two_pass):
    _, run, data = two_pass
    res = run()
    brain = nib.load(str(res["imagef_skullstripped"])).get_fdata()
    final = _cube(11, 29)
    assert np.allclose(brain[final], data[final] * 2)  # the fake N4 doubles intensities
    assert not brain[~final].any()


def test_growth_warning(two_pass, caplog):
    state, run, _ = two_pass
    state["masks"] = [_cube(12, 28), _cube(11, 29)]  # 16^3 -> 18^3: +42%
    with caplog.at_level(logging.WARNING):
        qc = run()["skullstrip_qc"]
    assert qc["MaskGrowthWarning"] is True and qc["Pass2GrowthPct"] > 10
    assert any("QC: func skull strip pass 2 added" in r.message for r in caplog.records)


def test_no_warning_for_small_growth(two_pass, caplog):
    state, run, _ = two_pass
    second = _cube(12, 28)
    second[28, 12:28, 12:28] = True  # one extra face: +6%
    state["masks"] = [_cube(12, 28), second]
    with caplog.at_level(logging.WARNING):
        qc = run()["skullstrip_qc"]
    assert qc["MaskGrowthWarning"] is False
    assert 0 < qc["Pass2GrowthPct"] < 10
    assert not any("QC:" in r.message for r in caplog.records)


def test_n4_failure_keeps_pass1(two_pass):
    state, run, _ = two_pass
    state["n4_fail"] = True
    res = run()
    qc = res["skullstrip_qc"]
    assert qc["SkullstripPasses"] == 1 and "N4 exploded" in qc["PreInferenceN4"]["Error"]
    assert res["imagef_bias_corrected"] is None
    assert np.array_equal(nib.load(str(res["brain_mask"])).get_fdata() > 0, _cube(12, 28))


def test_empty_pass1_skips_pass2(two_pass):
    state, run, _ = two_pass
    state["masks"] = [np.zeros(SHAPE, bool)]
    res = run()
    assert state["inputs"] == ["boldref.nii.gz"] and state["n4"] == []
    assert res["skullstrip_qc"]["SkullstripPasses"] == 1


def test_process_publishes_biascorrect_boldref():
    text = (Path(__file__).resolve().parents[1] / "modules" / "functional.nf").read_text()
    block = text[text.index("process FUNC_COMPUTE_BRAIN_MASK {"):]
    block = block[: block.index("\nprocess ")]
    assert "space-bold_desc-biascorrect_boldref}.{nii.gz,json}" in block
    assert '_space-bold_desc-biascorrect_boldref.nii.gz"' in block
    assert "extra=skullstrip_qc" in block
    # publishDir only publishes declared outputs: the image must be one (optional: pass-1 fallback)
    outputs = block[block.index("output:"): block.index("script:")]
    assert 'path "*_space-bold_desc-biascorrect_boldref.nii.gz", optional: true' in outputs
