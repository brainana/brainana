# Device management (CPU / GPU): developer guide

How brainana decides where each step runs, and the rules to follow when you add or
change a step. The short reason after each rule is the failure it prevents; most of
them have happened.

**Supported:** the Docker image on Linux, macOS (Docker Desktop, Intel and Apple
silicon, CPU only) and Windows (Docker Desktop with WSL2), plus local
`run_brainana.sh --no_docker` runs on Linux. NVIDIA GPUs only.
**Not supported:** native macOS / Apple MPS, Apptainer/Singularity, SLURM or other
multi-host executors. Code must not assume them, and nothing may schedule MPS.

---

## 1. Rules

1. **A task sees no GPU unless it is a GPU step.** The global `beforeScript` hides every
   GPU. Only a GPU step re-exposes one, and only one, by taking a slot.
   *Why:* a CPU task that merely asks `torch.cuda.is_available()` loads the CUDA driver,
   which aborted CPU runs on Windows/WSL2 (exit 134).
2. **Nextflow decides the device; Python follows.** `general.gpu_device` is interpreted by
   `nextflow.config` and `workflows/param_resolver.groovy`. Python reads the decision from
   the task's environment (`BRAINANA_DEVICE`, `CUDA_VISIBLE_DEVICES`), not from the config.
   *Why:* two layers interpreting the same setting drifted apart (an index that named a GPU
   the task could not see; a `-1` that the scheduler honoured and a step ignored).
3. **One device policy.** Every device choice in Python goes through `resolve_device()`.
   *Why:* three resolvers once gave three different answers for `"cuda"` (CPU, crash, GPU).
4. **Nothing that varies between identical runs goes into a task script or input.** That
   includes the GPU id, `params.gpu_count` and freshly written placeholder files.
   *Why:* Nextflow's `-resume` hashes the script text and its inputs (placeholder files by
   modification time), and every such value silently re-runs the task and everything after it.
5. **Threads and memory follow the task's allocation.** Use `task.cpus` (via
   `OMP_NUM_THREADS`), and never request more than the executor limits.
   *Why:* a fixed 8 torch threads in a 2-CPU task oversubscribed the host, and a 3-CPU request
   with `NXF_MAX_CPUS=2` could never be scheduled.
6. **Running out of GPU memory costs speed, not the step.** GPU work that could run out of
   memory is wrapped in `run_with_cpu_fallback()`.

## 2. How it works

```
nextflow.config (parse time)
  nvidia-smi --list-gpus      ─► gpuCount
  nvidia-smi memory.free      ─► maxJobsPerGpu = free VRAM / perJobVramMiB (1..4)
  user config gpu_device      ─► gpuForcedCpu, gpuIndex ─► gpuEnabled ─► docker --gpus all, maxForks
                                                        └─► gpuSlotIds (GPUs interleaved: 0 1 0 1 ...)
  beforeScript (every task)   ─► CUDA_VISIBLE_DEVICES=""  BRAINANA_DEVICE=cpu
                                 CUDA_DEVICE_ORDER=PCI_BUS_ID  OMP_NUM_THREADS=task.cpus ...
                                 BRAINANA_GPU_SLOTS  BRAINANA_GPU_LOCK_DIR=<work_dir>/.gpu_slots
                                 (CPU-mode run: unset LD_PRELOAD)

main.nf / workflows (run start)
  paramResolver.resolveUseGpu / resolveGpuIds / registrationUsesGpu
  GPU step? ─► process input `val use_gpu` = true | false

task script (GPU step, use_gpu = true)
  source brainana_gpu_slot.sh ─► flock a free slot (waits if none), export
                                 CUDA_VISIBLE_DEVICES=<gpu> BRAINANA_DEVICE=cuda;
                                 released by the kernel when the task ends

python
  resolve_device(spec) ─► [Device] cuda:0 | cpu ─► metadata.json "device"
  run_with_cpu_fallback(fn, device, ...) ─► on CUDA OOM: fn(cpu) once, recorded
```

### Where things live

| What | Where |
|---|---|
| GPU detection, `--gpus all`, slot table, `beforeScript`, CPU/memory caps | `nextflow.config` (`gpuCount`, `maxJobsPerGpu`, `perJobVramMiB`, `gpuEnabled`, `gpuSlotIds`, `capCpus`, `capMemory`, `retryMemory`) |
| Which steps run on a GPU; start-up validation of `gpu_device` | `workflows/param_resolver.groovy`: `parseGpuDevice`, `resolveUseGpu`, `resolveGpuIds`, `registrationUsesGpu` |
| Slot acquisition | `bin/brainana_gpu_slot.sh` (`bin/` is on every task's `PATH`) |
| Device policy (the implementation) | `src/fastsurfer_nn/utils/gpu_utils.py` |
| Device policy (public name) | `src/nhp_mri_prep/utils/gpu_device.py` re-exports it |
| Adapters | `fastsurfer_nn.utils.common.find_device` (FastSurfer's `ValueError` contract, view-aggregation memory check), `nhp_skullstrip_nn.utils.gpu` |
| Thread default | `src/fastsurfer_nn/utils/threads.py` `get_num_threads()` |
| Device record in `metadata.json` | `nhp_mri_prep.utils.nextflow.save_metadata()` |
| Config validation | `nhp_mri_prep.config.config_validation.validate_gpu_device()` |
| Container limits | `entrypoint.sh` `clamp_executor_limits` (clamps `NXF_MAX_CPUS`/`NXF_MAX_MEMORY` to the cgroup / VM) |

The implementation sits in `fastsurfer_nn` because `nhp_mri_prep` imports
`fastsurfer_nn` when the package loads; putting it in `nhp_mri_prep` makes a circular
import. Keep it there, and import it in new code from either name.

### `general.gpu_device`

| Value | Meaning |
|---|---|
| `auto` (default), `cuda`, `gpu` | GPU steps run on any detected GPU, else on the CPU |
| `-1`, `cpu` | everything on the CPU; containers are not given the GPU |
| `N`, `"N"`, `cuda:N` | every GPU step on physical GPU `N` (`nvidia-smi` numbering); the run stops at start-up if it does not exist |

Rejected at config load: anything else, including `mps`. Outside the pipeline
(standalone Python, Lite), `resolve_device()` applies the same values directly. There an
explicit GPU that cannot be served raises `RuntimeError`, and `auto` never picks MPS.

### Steps that run on a GPU

| Process | Uses the GPU for | Takes a slot when |
|---|---|---|
| ANAT_CONFORM, FUNC_COMPUTE_CONFORM | skull-strip network before the rigid conform | GPU scheduling is on |
| ANAT_SKULLSTRIPPING | segmentation network (and FireANTs in the V1 WM fix) | GPU scheduling is on |
| FUNC_COMPUTE_BRAIN_MASK | skull-strip network | GPU scheduling is on |
| ANAT_SURFACE_BASE_ATLAS | segmentation network on the longitudinal base | GPU scheduling is on |
| ANAT_REGISTRATION | FireANTs SyN | on, `registration.enable_fireants`, `anat2template_xfm_type: syn` |
| FUNC_COMPUTE_REGISTRATION | FireANTs SyN | on, `enable_fireants`, `func2anat` or `func2template` is `syn` |

Everything else, including FUNC_WITHIN_SES_COREG (always rigid; FireANTs is SyN only),
runs on the CPU and never sees a GPU.

## 3. Recipes

### Adding a CPU step

Nothing device-related to do. The `beforeScript` already hides the GPU. Give the process
a `withName` resource block in `nextflow.config`, wrapping fixed values:
`cpus = { capCpus(2) }`, `memory = { capMemory(8.GB) }` or `{ retryMemory(8.GB, task.attempt) }`.
Leaving them unwrapped makes the task unschedulable on a small Docker Desktop VM.

### Adding a GPU step

1. **Process** (`modules/*.nf`): add an input `val use_gpu`, and at the top of the script:

   ```bash
   if [ "${use_gpu}" = "true" ]; then
       source brainana_gpu_slot.sh
   fi
   ```

   Do not export `CUDA_VISIBLE_DEVICES` yourself, and do not echo GPU ids or
   `params.gpu_count` in the script (rule 4).
2. **Workflow:** pass `Channel.value(paramResolver.resolveUseGpu(params))`. Use a narrower
   gate when the GPU is only used sometimes, as `registrationUsesGpu` does for FireANTs.
   A step that holds a slot while doing CPU work blocks other GPU steps.
3. **Python:** follow the next recipe.
4. **Tests:** add the process to `GPU_PROCESSES` in `tests/test_gpu_scheduling.py`.
5. **Docs:** add a row to the table above.

### Writing Python that uses a device

```python
from nhp_mri_prep.utils.gpu_device import resolve_device, run_with_cpu_fallback

device = resolve_device(config.get("general", {}).get("gpu_device", "auto"))
result = run_with_cpu_fallback(lambda dev: run_model(..., device=dev), device,
                               "my network", logger)
```

- **Always** go through `resolve_device()`, passing the step's `general.gpu_device`.
  **Never** call `torch.cuda.is_available()` / `device_count()` directly, build
  `"cuda:0"` strings, or pick "the least busy GPU". The first loads the driver in CPU
  tasks; the others ignore the scheduler's choice.
- Pass the resolved `torch.device` down. Do not re-resolve deeper in the stack: FireANTs
  once probed one device and ran on another.
- Pass the caller's config into helpers that start their own GPU work (e.g. a registration
  inside a segmentation step). Helpers that built their own config from package defaults
  ignored the user's `gpu_device` and `enable_fireants`.
- **Threads:** `get_num_threads()` for torch, and `OMP_NUM_THREADS` for subprocesses.
  Never hard-code a count, and never set thread env vars after `import torch`, because
  it has no effect by then.
- **Models:** load weights with `map_location="cpu"` and then `.to(device)` (loading
  straight onto MPS yields zeros). Run inference under `torch.no_grad()` and in `eval()` mode.
- **Freeing memory:** guard `torch.cuda.empty_cache()` with `torch.cuda.is_initialized()`,
  so CPU runs never touch the driver.
- **Metadata:** write step metadata with `save_metadata()`, which adds the device record
  automatically.

### Keeping `-resume` stable (rule 4)

- **Placeholder files:** a workflow that stands in a file for an absent input must create
  it only when missing:
  `file("${workDir}/x.dummy").tap { if (!it.exists()) it.toFile().text = "" }`.
  Rewriting it on every run changes its modification time, so every task that takes it
  re-runs. `tests/test_placeholder_files.py` enforces this.
- **Grouped runs:** `groupTuple` emits items in completion order, which varies between
  runs. When a grouped list reaches a task script (a JSON list of paths, "the first run's
  name"), order it by a stable key first, keeping parallel lists aligned. See
  `func_for_averaging_ch` and `tsnr_grouped` in `workflows/functional_workflow.nf`. Do not
  use `groupTuple(sort: true)` for this: it sorts each list independently. Sort a
  `toList()`, not a range: a range is immutable and `sort` fails.

## 4. Don'ts

| Don't | Because |
|---|---|
| Hand a GPU to a task through a process input or script text | It enters the task hash, so `-resume` re-runs whatever drew another GPU |
| Pass GPUs between processes as tokens (a queue) | A task that fails for good never returns its token, and the run hangs. `flock` slots are released by the kernel |
| Use `errorStrategy 'ignore'` casually on GPU steps | Safe with slots, but it hides a failed subject. Choose it deliberately |
| Add a per-step device config key | `general.gpu_device` is the one knob; the legacy `fastSurferCNN.gpu_device` and `skullstripping.gpu_device` remain only as fallbacks |
| Let `auto` resolve to MPS, or schedule MPS | Unsupported, unscheduled, and known to misbehave (zeros on direct load) |
| Assume `nvidia-smi` indices equal CUDA indices | They agree only with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, which the pipeline and image set |
| Read `nvidia-smi` for GPU choice inside a task | It lists every physical GPU, not the task's one visible GPU |
| Rely on Python `logger.info` reaching the task log | In most tasks it does not reach `.command.err`; put facts you need later in `metadata.json` |

## 5. Testing and verification

**Unit tests** (no GPU needed):

- `tests/test_gpu_device.py`: the resolver across spec × `CUDA_VISIBLE_DEVICES` ×
  `BRAINANA_DEVICE`. It fails if hidden CUDA is probed at all. It also covers the OOM
  fallback, the device report and config validation.
- `tests/test_gpu_scheduling.py`: the default-deny `beforeScript`, the GPU processes and
  their slot lines, that no process hashes a GPU id, and the slot helper under
  `bash -ue`: waiting, release, CPU mode.
- `tests/test_placeholder_files.py`.

**Pipeline check** (local, `--no_docker`; see `run_brainana.sh`):

- Dataset: `dataset_supereasy` (1 subject, T1w + 1 bold) with
  `template.output_space: "NMT2Sym:res-1"` and `anat.surface_reconstruction.enabled: false`.
- Run it in GPU mode, in CPU mode (`general.gpu_device: -1` with GPUs visible), and once
  more with `-resume`. Expect every task cached on the resume.
- For anything about concurrency across GPUs, use `dataset_devtest` (4 subjects).
  `dataset_supereasy` never runs two GPU steps at once.
- `run_brainana.sh` exits 0 even when Nextflow fails. Read
  `<output>/nextflow_reports/run_status.json`.
- To see which tasks really touched a GPU, sample
  `nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory` every few seconds and
  keep the processes whose `/proc/<pid>/cwd` is inside the run's work dir. In CPU mode that
  set must be empty.

**Hosts not testable on the dev server** (run on `dataset_supereasy` after each image build):

- [ ] Windows/WSL2, `--gpus all`, default config: GPU steps' `metadata.json` says `cuda:0`.
- [ ] Windows/WSL2, `--gpus all` + `general.gpu_device: -1`: completes (this used to abort
      with exit 134). Until this passes, the FAQ keeps advising to omit `--gpus all` for
      CPU runs.
- [ ] Windows/WSL2 and macOS without `--gpus`: completes.
- [ ] macOS: start-up prints the `NXF_MAX_*` clamp warning when the Docker Desktop VM is
      smaller than 8 CPUs / 20 GB.
- [ ] Any host, `docker run --cpus 2 --memory 8g`: warning printed, and FUNC_COMPUTE_REGISTRATION
      (3 CPUs) still runs.

## 6. Reference measurements

Dev server (2× RTX A6000, 48 GB each), res-1, surface reconstruction off.

Peak GPU memory per task (`dataset_devtest`), against the `perJobVramMiB = 4096` budget:

| Process | Peak MiB |
|---|---|
| ANAT_SKULLSTRIPPING | 812 |
| ANAT_REGISTRATION (FireANTs) | 712 |
| FUNC_COMPUTE_REGISTRATION (FireANTs), FUNC_COMPUTE_BRAIN_MASK | 482 |
| ANAT_CONFORM, FUNC_COMPUTE_CONFORM | 428 |

GPU vs CPU wall time (`dataset_supereasy`):

| Process | GPU | CPU |
|---|---|---|
| ANAT_CONFORM | 22 s | 64 s |
| ANAT_SKULLSTRIPPING | 27 s | 4m27s |
| ANAT_REGISTRATION | 15 s | 5m29s |
| FUNC_COMPUTE_BRAIN_MASK | 7 s | 58 s |

Whole run: about 4 min on the GPU and 13–14 min on the CPU.

## 7. Known gaps

- **Per-job VRAM budget.** `perJobVramMiB` stays at 4096 until res-05 (the default
  template, about 8× the voxels of res-1) is measured. The res-1 peaks above do not justify
  lowering it.
- **Waiting for a slot costs a reservation.** A GPU step waiting for a slot has already
  been launched and holds its CPU/memory reservation. The per-process `maxForks` caps keep
  this small.
- **Registration and conform carry `label 'cpu'`**, so `withLabel: 'gpu'` settings do not
  apply to them. The slots still bound their GPU use.
- **Surface reconstruction has its own thread setting** (`processing.threads`, default 1,
  cap 16), which overwrites `OMP_NUM_THREADS` inside the step. It matches its `cpus = 1`
  today; keep them in step if either changes.
- **FastSurfer's inference padding** has a "move to CPU" branch that moves the tensor back
  to the GPU. It has no effect on results, and the OOM fallback covers the failure it was
  meant to avoid.
- **GPU detection runs once**, when the config is parsed. GPUs added or freed later in a
  run are not seen.
