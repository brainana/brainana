"""Nextflow GPU scheduling: every task is CPU unless a GPU step takes a slot.

The pipeline's CPU mode relies on CUDA_VISIBLE_DEVICES="" reaching every task.
Four processes once exported a GPU only when they held a token and never hid
it otherwise, so with `--gpus all` plus general.gpu_device: -1 their Python
still initialised CUDA. GPU steps now take a slot at run time
(bin/brainana_gpu_slot.sh), so the GPU id never reaches the task script and
the task hash, and a failed task cannot leak its slot. These checks keep the
default-deny beforeScript, the slot table and the GPU processes consistent,
and exercise the slot helper itself.
"""

import os
import re
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
CONFIG = (REPO / "nextflow.config").read_text()
MODULE_FILES = sorted((REPO / "modules").glob("*.nf"))
WORKFLOW_FILES = sorted((REPO / "workflows").glob("*.nf")) + [REPO / "main.nf"]
SLOT_HELPER = REPO / "bin" / "brainana_gpu_slot.sh"

GPU_PROCESSES = {
    "ANAT_CONFORM",
    "ANAT_SKULLSTRIPPING",
    "ANAT_SURFACE_BASE_ATLAS",
    "ANAT_REGISTRATION",
    "FUNC_COMPUTE_CONFORM",
    "FUNC_COMPUTE_BRAIN_MASK",
    "FUNC_COMPUTE_REGISTRATION",
}


def _processes():
    for path in MODULE_FILES:
        text = path.read_text()
        for match in re.finditer(r"^process (\w+) \{", text, re.M):
            nxt = text.find("\nprocess ", match.end())
            yield match.group(1), text[match.start() : nxt if nxt != -1 else len(text)]


def _before_script():
    start = CONFIG.index("beforeScript = {")
    return CONFIG[start : CONFIG.index("\n    }\n", start)]


def test_before_script_hides_gpus_by_default():
    body = _before_script()
    assert 'export CUDA_VISIBLE_DEVICES=""' in body
    assert "export CUDA_DEVICE_ORDER=PCI_BUS_ID" in body
    assert "export BRAINANA_DEVICE=cpu" in body
    assert "export BRAINANA_GPU_SLOTS=" in body
    assert "export BRAINANA_GPU_LOCK_DIR=" in body


def test_gpu_processes_take_a_slot_and_nothing_else():
    blocks = dict(_processes())
    for name in GPU_PROCESSES:
        block = blocks[name]
        assert re.search(r"^    val use_gpu\b", block, re.M), name
        assert 'if [ "${use_gpu}" = "true" ]; then' in block, name
        assert "source brainana_gpu_slot.sh" in block, name


def test_no_process_sets_a_gpu_or_hashes_gpu_ids():
    """The GPU a task draws must not appear in its script (it would enter the hash)."""
    for name, block in _processes():
        assert "gpu_id" not in block, name
        assert "gpu_token" not in block, name
        assert "params.gpu_count" not in block, name
        assert not re.search(r"export CUDA_VISIBLE_DEVICES=\S", block), name


def test_only_gpu_processes_use_slots():
    for name, block in _processes():
        if name not in GPU_PROCESSES:
            assert "brainana_gpu_slot.sh" not in block, name


def test_token_queue_is_gone():
    for path in WORKFLOW_FILES:
        text = path.read_text()
        assert "gpu_queue" not in text, path.name
        assert "DataflowQueue" not in text, path.name


def test_within_session_coreg_takes_no_slot():
    blocks = dict(_processes())
    assert "use_gpu" not in blocks["FUNC_WITHIN_SES_COREG"]


def test_docker_gpus_follow_forced_cpu():
    assert "def dockerRunOpts = (dockerEnabled && gpuEnabled)" in CONFIG
    assert "gpuEnabled = gpuCount > 0 && !gpuForcedCpu" in CONFIG


def _run_task(tmp_path, slots, body, **popen):
    env = dict(os.environ)
    env.update(
        PATH=f"{SLOT_HELPER.parent}:{env['PATH']}",
        BRAINANA_GPU_SLOTS=slots,
        BRAINANA_GPU_LOCK_DIR=str(tmp_path / "locks"),
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    script = f"source brainana_gpu_slot.sh\n{body}"
    return subprocess.Popen(
        ["bash", "-ue", "-c", script],
        env=env,
        stdout=subprocess.PIPE,
        text=True,
        **popen,
    )


def test_slot_helper_cpu_mode_leaves_gpu_hidden(tmp_path):
    out = _run_task(tmp_path, "", 'echo "cvd=${CUDA_VISIBLE_DEVICES-unset}"').communicate()[0]
    assert "cvd=unset" in out


def test_slot_helper_assigns_distinct_slots(tmp_path):
    holder = _run_task(tmp_path, "3 5", 'echo "gpu=$CUDA_VISIBLE_DEVICES"; sleep 3')
    time.sleep(1)
    second = _run_task(tmp_path, "3 5", 'echo "gpu=$CUDA_VISIBLE_DEVICES dev=$BRAINANA_DEVICE"')
    out2 = second.communicate(timeout=30)[0]
    out1 = holder.communicate(timeout=30)[0]
    assert "gpu=3" in out1
    assert "gpu=5 dev=cuda" in out2


def test_slot_helper_waits_then_takes_a_released_slot(tmp_path):
    holder = _run_task(tmp_path, "7", "sleep 2")
    time.sleep(0.5)
    waiter = _run_task(tmp_path, "7", 'echo "gpu=$CUDA_VISIBLE_DEVICES"')
    holder.wait(timeout=30)
    out = waiter.communicate(timeout=30)[0]
    assert "gpu=7" in out
    assert "waited" in out


def test_slot_table_interleaves_gpus():
    """First-free slot assignment only spreads load if slots alternate GPUs."""
    assert "maxJobsPerGpu.times { slotGpus.each { id -> gpuSlotIds << id } }" in CONFIG
