"""Skull strips that find no brain: component selection, the hard failure, reporting.

PRIME-DE carmenlyon sub-12: the conform-stage network saw a brain-sized blob in
the background noise (578k voxels, mean probability 0.61) next to the brain
(552k voxels, 0.95). Keeping the largest component kept the blob, conform cropped
to air, the main skull strip returned an empty mask, and every later step ran on
noise until surface reconstruction crashed.
"""

import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from nhp_mri_prep.operations import preprocessing
from nhp_mri_prep.operations.preprocessing import (
    EMPTY_BRAIN_MASK_EXIT_CODE,
    EmptyBrainMaskError,
    _require_nonempty_mask,
)
from nhp_mri_prep.quality_control.run_status import (
    failed_task_names,
    render_run_status_content,
)
from nhp_skullstrip_nn.utils.morphology import select_confident_component

REPO = Path(__file__).resolve().parent.parent


def _two_blobs(faint_size, confident_size, faint_p=0.6, confident_p=0.95):
    """A faint cube and a confident cube, apart, as (label, prob)."""
    prob = np.zeros((60, 30, 30), np.float32)
    prob[2 : 2 + faint_size, 2:12, 2:12] = faint_p
    prob[40 : 40 + confident_size, 2:12, 2:12] = confident_p
    return prob > 0.5, prob


def test_confident_component_beats_larger_faint_one():
    label, prob = _two_blobs(faint_size=15, confident_size=10)
    mask, info = select_confident_component(label, prob)
    assert mask[45, 5, 5] and not mask[5, 5, 5]
    assert info["by"] == "core"
    assert info["kept_voxels"] == 1000 and info["largest_voxels"] == 1500


def test_largest_confident_component_kept_when_both_confident():
    label, prob = _two_blobs(faint_size=15, confident_size=10, faint_p=0.95)
    mask, _ = select_confident_component(label, prob)
    assert mask[5, 5, 5] and not mask[45, 5, 5]


def test_falls_back_to_largest_without_confident_voxels():
    label, prob = _two_blobs(faint_size=15, confident_size=10, confident_p=0.7)
    mask, info = select_confident_component(label, prob)
    assert mask[5, 5, 5] and not mask[45, 5, 5]
    assert info["by"] == "size"


def test_kept_component_voxels_unchanged():
    label, prob = _two_blobs(faint_size=15, confident_size=10)
    prob[40:50, 2:12, 2:4] = 0.55  # a faint rim on the confident blob stays in
    label = prob > 0.5
    mask, _ = select_confident_component(label, prob)
    assert mask.sum() == label[40:50].sum()


def test_empty_label():
    label = np.zeros((5, 5, 5), bool)
    mask, info = select_confident_component(label, label.astype(np.float32))
    assert not mask.any() and info["by"] == "empty"


def _mask_file(tmp_path, n_voxels, zoom=0.5):
    data = np.zeros((100, 100, 100), np.uint8)
    data.reshape(-1)[:n_voxels] = 1
    path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(data, np.diag([zoom, zoom, zoom, 1])), str(path))
    return path


@pytest.fixture
def template_92_5(monkeypatch):
    monkeypatch.setattr(preprocessing, "_template_brain_volume_mm3", lambda: 92_500.0)


def test_empty_mask_raises(tmp_path, template_92_5):
    import logging

    with pytest.raises(EmptyBrainMaskError, match="conform"):
        _require_nonempty_mask(_mask_file(tmp_path, 0), logging.getLogger("t"))


def test_undersized_but_real_mask_passes(tmp_path, template_92_5):
    import logging

    # 60 cm^3 at 0.5 mm: undersized (warned elsewhere), not empty
    _require_nonempty_mask(_mask_file(tmp_path, 480_000), logging.getLogger("t"))


def test_nextflow_ignores_the_empty_mask_exit_code():
    config = (REPO / "nextflow.config").read_text()
    block = config[config.index("withName: 'ANAT_SKULLSTRIPPING' {") :]
    block = block[: block.index("\n    }\n")]
    strategy = re.search(r"errorStrategy = \{(.*)\}", block).group(1)
    assert f"task.exitStatus == {EMPTY_BRAIN_MASK_EXIT_CODE} ? 'ignore'" in strategy

    module = (REPO / "modules" / "anatomical.nf").read_text()
    process = module[module.index("process ANAT_SKULLSTRIPPING {") :]
    process = process[: process.index("\nprocess ")]
    assert "except EmptyBrainMaskError" in process
    assert "sys.exit(EMPTY_BRAIN_MASK_EXIT_CODE)" in process


def test_ignored_skull_strip_does_not_continue_on_dummies():
    """An ignored session must leave the mask channel, not fall back to the dummy,
    and bias correction (hence registration onwards) must read the kept channel."""
    wf = (REPO / "workflows" / "anatomical_workflow.nf").read_text()
    skull = wf[wf.index("if (anat_skullstripping_enabled) {") :]
    skull = skull[: skull.index("\n    }\n")]
    assert ".filter { sub, ses, mask -> mask != null }" in skull
    assert "real_mask ?: mask_list[0]" not in skull
    assert "anat_after_conform_kept = anat_after_conform.join(skull_stripped_keys" in skull

    bias = wf[wf.index("// BIAS CORRECTION") : wf.index("// PUBLISH PHASE 1 OUTPUTS")]
    code = "\n".join(l for l in bias.splitlines() if not l.strip().startswith("//"))
    assert "anat_after_conform_kept" in code
    assert not re.search(r"\banat_after_conform\b", code)


def test_skull_strip_inference_runs_without_autograd():
    source = (
        REPO / "src" / "nhp_skullstrip_nn" / "inference" / "prediction.py"
    ).read_text()
    assert "@torch.no_grad()\ndef predict_volumes(" in source


def _trace(tmp_path, rows):
    path = tmp_path / "trace.txt"
    lines = ["task_id\thash\tname\tstatus\texit"]
    lines += [f"{i}\tab/cd\t{name}\t{status}\t-" for i, (name, status) in enumerate(rows)]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_failed_task_names_skips_retried_successes(tmp_path):
    trace = _trace(
        tmp_path,
        [
            ("ANAT_WF:ANAT_SKULLSTRIPPING (12_01)", "FAILED"),
            ("ANAT_WF:ANAT_REGISTRATION (03_01)", "FAILED"),
            ("ANAT_WF:ANAT_REGISTRATION (03_01)", "COMPLETED"),
            ("ANAT_WF:ANAT_CONFORM (12_01)", "COMPLETED"),
        ],
    )
    assert failed_task_names(trace) == ["ANAT_SKULLSTRIPPING (12_01)"]


def test_failed_task_names_tolerates_missing_trace(tmp_path):
    assert failed_task_names(str(tmp_path / "absent.txt")) == []
    assert failed_task_names(None) == []


def test_warning_banner_lists_failed_tasks(tmp_path):
    trace = _trace(tmp_path, [("ANAT_WF:ANAT_SKULLSTRIPPING (12_01)", "FAILED")])
    html = render_run_status_content(
        {"success": True, "ignored_count": 1, "trace_file": trace}
    )
    assert "<code>ANAT_SKULLSTRIPPING (12_01)</code>" in html
    assert "optional" not in html
