# Device management (CPU / GPU)

How brainana chooses between the CPU and the GPU, how it sizes CPU threads and
memory, what was wrong with it, and what is left to do. This replaces
`GPU_RESOURCE_REVIEW.md`; everything in it that still holds is folded in here and
its proposals are tracked in [§6](#6-status-of-the-earlier-review-gpu_resource_reviewmd).

Supported targets: the Docker image on Linux, macOS (Docker Desktop, Intel and
Apple silicon) and Windows (Docker Desktop with WSL2), and local
`run_brainana.sh --no_docker` runs on Linux. Native macOS/MPS, Apptainer/Singularity
and SLURM/HPC are **not** supported for now.

---

## 1. Rules

1. **Nothing can see a GPU unless it was given one.** The global `beforeScript` in
   `nextflow.config` exports `CUDA_VISIBLE_DEVICES=""` and `BRAINANA_DEVICE=cpu` for
   every task. A GPU step sources `bin/brainana_gpu_slot.sh`, which takes a GPU slot
   and re-exports `CUDA_VISIBLE_DEVICES=<gpu>` and `BRAINANA_DEVICE=cuda`. A CPU task
   therefore cannot load the CUDA driver, whatever its Python code probes.
2. **Nextflow decides; Python follows.** `general.gpu_device` is read in two places:
   - `nextflow.config`, for `--gpus all` and the slot table (`gpuEnabled`, `gpuIndex`,
     `gpuSlotIds`);
   - `workflows/param_resolver.groovy` (`parseGpuDevice`, `resolveUseGpu`,
     `resolveGpuIds`, `registrationUsesGpu`), for which steps run on a GPU and to stop
     the run at start-up on a GPU index that does not exist.

   In Python, `resolve_device()` follows `BRAINANA_DEVICE` whenever it is set: `cpu` means
   CPU; `cuda` means the slot's GPU as `cuda:0` (CPU with a warning if CUDA is unusable).
   An explicit `cpu`/`-1` is always honoured, which is what the out-of-memory retry uses.
3. **One device policy in Python.** It is implemented once, in
   `fastsurfer_nn/utils/gpu_utils.py`, because `nhp_mri_prep` imports `fastsurfer_nn`
   when the package loads and the reverse import would be circular.
   `nhp_mri_prep.utils.gpu_device` (the name the pipeline uses) re-exports it;
   `fastsurfer_nn.utils.common.find_device` and `nhp_skullstrip_nn.utils.gpu` delegate
   to it. Outside the pipeline (standalone, Lite) the spec decides:
   - `auto` (and `cuda`/`gpu`) → a usable CUDA GPU, else CPU. It never picks MPS.
   - `-1`/`cpu` → CPU.
   - An explicit index or `cuda:N` → that GPU:
     - with CUDA hidden, CPU plus a warning;
     - with one visible GPU, `cuda:0`;
     - otherwise it is validated against `device_count`;
     - `RuntimeError` if CUDA is unusable.
   - An unknown spec → `ValueError`, inside the pipeline too.
   - Every decision is logged as `[Device] ...` and recorded; `save_metadata()` writes the
     record into the step's `metadata.json` under `device`.
4. **A CUDA out-of-memory error costs speed, not the step.** `run_with_cpu_fallback()`
   reruns the segmentation network, the skull-strip network or FireANTs (when
   `fireants_allow_cpu`) once on the CPU, logs a warning, and records it in
   `metadata.json` (`device.cpu_fallbacks`). Any other error propagates as before.
5. **Threads = the task's CPUs.** `beforeScript` exports `OMP/MKL/NUMEXPR/OPENBLAS/ITK`
   thread counts equal to `task.cpus`. The ANTs CLI, torch (`get_num_threads()` in
   `fastsurfer_nn/utils/threads.py`, the one implementation, follows `OMP_NUM_THREADS`)
   and FireANTs on CPU all inherit them.
6. **Requests never exceed the limits.** `NXF_MAX_CPUS`/`NXF_MAX_MEMORY` are clamped by
   the entrypoint to what the container has. Per-process `cpus`/`memory` are clamped to
   those limits (`capCpus`, `capMemory`, `retryMemory` in `nextflow.config`).

## 2. How it fits together

```
nvidia-smi --list-gpus ──► gpuCount ─┐                  (nextflow.config, parse time)
nvidia-smi memory.free ──► maxJobsPerGpu (VRAM per job, 1..4)
user config general.gpu_device ──► gpuForcedCpu, gpuIndex ─┴► gpuEnabled ─► --gpus all, maxForks
                                                         └──► gpuSlotIds: each GPU id × maxJobsPerGpu
beforeScript (every task): CUDA_VISIBLE_DEVICES=""  BRAINANA_DEVICE=cpu
                           BRAINANA_GPU_SLOTS="<gpuSlotIds>"  BRAINANA_GPU_LOCK_DIR=<work>/.gpu_slots
                                                         │
workflow: GPU step? ─yes─► val use_gpu = true  ─► script: source brainana_gpu_slot.sh
                    │                                    flock a free slot (wait if none),
                    │                                    CUDA_VISIBLE_DEVICES=<gpu>, BRAINANA_DEVICE=cuda;
                    │                                    the kernel releases it when the task ends
                    └no──► val use_gpu = false ─► stays on the CPU
                                                         │
python: resolve_device(...) ─► [Device] cuda:0 | cpu ─► metadata.json "device"
```

Why a slot and not a token passed between processes (the design before Phase 2):

- **`-resume` depends only on the mode.** A token was a task input and was written
  into the script, so it was part of the task hash; on a multi-GPU host a task that drew
  a different GPU re-ran, with everything downstream. The slot's GPU id never reaches the
  script. Only `use_gpu` (true/false) does, so switching CPU ↔ GPU still re-runs, on
  purpose: the outputs differ.
- **Slots cannot leak.** A task that failed for good never returned its token, and the
  never-closed queue then hung the run; that is why ANAT_SURFACE_BASE_ATLAS could not use
  `errorStrategy 'ignore'`. `flock` is released by the kernel however the task ends.
- **Trade-off:** a task waiting for a slot has already been launched and holds its CPU
  and memory reservation. The per-process `maxForks` caps keep that small.

Which steps run on a GPU slot:

| Process | Uses GPU for | Slot when |
|---|---|---|
| ANAT_CONFORM, FUNC_COMPUTE_CONFORM | skull-strip network before the rigid conform registration | GPU scheduling on |
| ANAT_SKULLSTRIPPING | segmentation network (+ FireANTs in the V1 WM fix) | GPU scheduling on |
| FUNC_COMPUTE_BRAIN_MASK | skull-strip network | GPU scheduling on |
| ANAT_SURFACE_BASE_ATLAS | segmentation network on the longitudinal base | GPU scheduling on |
| ANAT_REGISTRATION | FireANTs SyN | on, `enable_fireants`, `anat2template_xfm_type: syn` |
| FUNC_COMPUTE_REGISTRATION | FireANTs SyN | on, `enable_fireants`, func2anat or func2template `syn` |
| FUNC_WITHIN_SES_COREG | nothing (rigid registration) | never |
| everything else | nothing | never |

The conform steps used to run their network on the GPU with no token (unscheduled). Rule
1 alone would have moved them to the CPU, which took ANAT_CONFORM from ~10–30 s to ~65 s,
so they run on a slot like the other network steps.

## 3. Operating system × mode

| Host | GPU mode | CPU mode | Notes |
|---|---|---|---|
| Linux, Docker | `--gpus all` (NVIDIA Container Toolkit) | omit `--gpus`, or keep it and set `gpu_device: -1` | Pipeline verified with `--no_docker` on this box (2× A6000); in the current `brainana:latest` image only the pieces were checked (FireANTs with/without `LD_PRELOAD`, the entrypoint clamp). Re-run end-to-end after the next image build |
| Windows, Docker Desktop + WSL2 | `--gpus all` (NVIDIA driver on Windows) | **omit `--gpus all`** (FAQ) | CPU mode with `--gpus all` aborted (exit 134). Fixed by rule 1 and by not preloading `libcudart` in CPU mode, but **not verified on a Windows host**. Keep the FAQ advice until it is |
| macOS Intel, Docker Desktop | n/a | default | Docker Desktop VM memory defaults to 50% of RAM; the entrypoint now clamps to it and warns |
| macOS Apple silicon, Docker Desktop | n/a | default, amd64 emulation | Slower; the image is `linux/amd64` only |
| Linux, `--no_docker` | automatic when `nvidia-smi` sees a GPU | `gpu_device: -1` | A `CUDA_VISIBLE_DEVICES` set in your own shell is replaced per task |

Checklist for hosts we cannot test here (run on the small `dataset_supereasy`):

- [ ] Windows/WSL2, `--gpus all`, default config: GPU steps log `[Device] cuda:0`.
- [ ] Windows/WSL2, `--gpus all` + `general.gpu_device: -1`: completes, no exit 134.
- [ ] Windows/WSL2, no `--gpus`: completes.
- [ ] macOS (either CPU), default: start-up prints the `NXF_MAX_*` clamp warning if the VM
      is smaller than 8 CPUs / 20 GB, and the run completes.
- [ ] `docker run --cpus 2 --memory 8g` (any host): warning printed, FUNC_COMPUTE_REGISTRATION
      (asks for 3 CPUs) still runs.

## 4. Review findings

Status: **fixed** = in the working tree with tests; **open** = Phase 2 (§5).

### 4.1 CPU mode could still use the GPU

| # | Finding | Status |
|---|---|---|
| B1 | ANAT_REGISTRATION, FUNC_COMPUTE_BRAIN_MASK, FUNC_COMPUTE_REGISTRATION and FUNC_WITHIN_SES_COREG set `CUDA_VISIBLE_DEVICES` only when they held a token; in CPU mode they left it unset. With `--gpus all` + `gpu_device: -1`, the FireANTs gate then initialised CUDA and ran a kernel — the crash class 42d7ca5 tried to remove. | fixed (rule 1) |
| B2 | The conform step's skull strip passed `config=None`, i.e. `gpu_device: auto`, in a CPU-labelled task with no token, so it used the GPU unscheduled and ignored `-1`. | fixed (config passed through; conform now takes a GPU token) |
| B3 | The V1 WM fix's template registration built its config from package defaults, so it ignored `general.gpu_device` and `registration.enable_fireants` (and logged a made-up "32 threads"). | fixed |
| B5 | Host-side Docker executor passed `--gpus all` whenever a GPU existed, even with `gpu_device: -1`. | fixed (`gpuEnabled`) |
| — | `neuroenv.sh` preloads `libcudart.so.12` (`LD_PRELOAD`) into every process. Tested in `brainana:latest`: FireANTs' CPU optimizer and its GPU fused ops both load without it (the CUDA runtime is on `LD_LIBRARY_PATH` too). | CPU-mode runs now `unset LD_PRELOAD`; GPU mode unchanged |
| — | The FireANTs gate probed the GPU whatever device the registration would use. | fixed (probes the resolved device) |

### 4.2 Wrong device or silent swap

| # | Finding | Status |
|---|---|---|
| B7 | Under a token the GPU is `cuda:0`, so `gpu_device: 1` asked for `cuda:1` (does not exist) and `0` used whichever GPU the token named. A quoted `"-1"` reached `torch.device("-1")`. `general.gpu_device` was not validated. | fixed: N pins the token pool to GPU N; validated at load and at start-up |
| — | Three resolvers disagreed: `resolve_device`, `gpu_utils.setup_device`, `common.find_device`. `"cuda"` was CPU / crash / least-busy depending on which one ran. | fixed for the spec; the duplicate code goes in Phase 2 |
| — | `nvidia-smi` numbers GPUs in PCI order, CUDA in fastest-first order; `CUDA_DEVICE_ORDER` was never set, so a token could name a different physical GPU on a mixed-GPU box. | fixed (`PCI_BUS_ID` in `beforeScript` and the entrypoint) |
| — | `gpu_utils.get_least_busy_gpu` ignored a multi-GPU `CUDA_VISIBLE_DEVICES` (could return an invalid index), and its fallback called `set_device()` on every GPU and compared this process's own allocations (always GPU 0). | fixed |
| — | `resolve_device("auto")` chose MPS on a Mac. Nothing schedules MPS, the skull-strip loader maps weights straight to MPS (FastSurfer's own comment says that returns zeros), and the FireANTs gate never probed it. | fixed: `auto` never picks MPS |
| — | `cuda_hidden()` did not recognise `NoDevFiles`. | fixed |
| — | `resolve_device` logged nothing, so a silent CPU fallback was invisible. | fixed (`[Device]` log) |

### 4.3 Threads, CPUs and memory

| # | Finding | Status |
|---|---|---|
| B4 | The segmentation network always ran `torch.set_num_threads(8)`; ANAT_SKULLSTRIPPING has 2 CPUs and 2 forks. | fixed (follows `OMP_NUM_THREADS`) |
| — | `NXF_MAX_CPUS`/`NXF_MAX_MEMORY` default to 8 / 20g whatever the container has. On Docker Desktop (VM = 50% of host RAM) Nextflow over-committed memory and tasks died with 137. | fixed (entrypoint clamp + warning) |
| — | Fixed `cpus`/`memory` requests were not clamped: `NXF_MAX_CPUS=2` left the 3-CPU processes unschedulable. | fixed (`capCpus`/`capMemory`) |
| — | `get_num_threads()` exists twice (`fastsurfer_nn`, `nhp_skullstrip_nn`), and `setup_pytorch_threads` is dead and sets env vars after torch is imported. | fixed (Phase 2: one `get_num_threads`, dead code removed) |
| — | Surface recon's threading helper caps at 16 and overwrites `OMP_NUM_THREADS` with `processing.threads` (default 1, matching its `cpus = 1`). | open: consistent today, but a second thread knob; low priority |

### 4.4 Scheduling and resume

| # | Finding | Status |
|---|---|---|
| — | FUNC_WITHIN_SES_COREG (always rigid) held a GPU token; registration held one even when not running FireANTs SyN. With one GPU and <8 GiB free, this queued most of the pipeline one task at a time. | fixed |
| B6 | `gpu_id` is a `val` input and is written into the script, so it is part of the task hash. On a multi-GPU host `-resume` re-runs a task that draws a different GPU, plus everything downstream. | fixed (Phase 2: run-time GPU slots) |
| — | Per-job VRAM is a fixed 4096 MiB guess; the pool is sized once at config parse time. | see §5 item 6 (measured) |
| — | Registration and conform processes carry `label 'cpu'`, so the `withLabel:'gpu'` `maxForks` does not apply to them; the GPU slots still bound them. | open (cosmetic) |

### 4.5 Robustness

| Finding | Status |
|---|---|
| No `torch.cuda.OutOfMemoryError` handling anywhere; `handle_cuda_memory_exception` had no callers. | fixed (Phase 2: CPU retry, rule 4; dead helper removed) |
| Skull-strip inference ran without `no_grad` (built autograd graphs), and its loader mapped weights straight onto the device (zeros on MPS). | fixed (Phase 2: `@torch.no_grad()`, load on CPU then move) |
| FastSurfer's `DataParallel` path was dead (`find_device` always adds an index). | fixed (removed) |
| The inference padding "move to CPU" branch moves the tensor back to the GPU. | open (no effect on results; the OOM retry covers the failure it was meant to avoid) |
| `known_flags.txt` still accepts `--use_gpu` silently. | fixed (Phase 2: `main.nf` warns it is ignored) |

## 5. Phase 2: unify — status

| # | Item | Status |
|---|---|---|
| 1 | One implementation | Done. `fastsurfer_nn/utils/gpu_utils.py` holds the policy (moved there to avoid an import cycle, see rule 3). `nhp_mri_prep.utils.gpu_device` re-exports it, and `common.find_device` delegates (keeping its `ValueError` and the view-aggregation `min_memory` check). One `get_num_threads` (`fastsurfer_nn/utils/threads.py`). Removed: `nhp_skullstrip_nn/utils/threads.py`, `setup_pytorch_threads`, `handle_cuda_memory_exception`, and the `DataParallel` path |
| 2 | `BRAINANA_DEVICE` + device in metadata | Done (rules 2–3). `metadata.json` gains `device: {device, resolutions, cpu_fallbacks?, BRAINANA_DEVICE, CUDA_VISIBLE_DEVICES}` for steps that resolved a device |
| 3 | Resume-stable GPU slots (B6) | Done: `bin/brainana_gpu_slot.sh`; token queue removed; slots interleaved across GPUs. Verified below |
| 4 | Out-of-memory policy | Done (rule 4). Unit-tested with a simulated OOM; not triggered for real (no step comes close to 48 GB at res-1) |
| 5 | Skull-strip inference | Done: `@torch.no_grad()` on `predict_volumes`, and weights loaded on the CPU and then moved. `no_grad` rather than `inference_mode`, so the returned tensors stay ordinary tensors. `model.eval()` was already called by `predict_volumes` |
| 6 | Measured VRAM | Measured at res-1 (§7): per-task peaks 0.4–0.8 GiB against the 4 GiB/job budget. The budget is **kept** until res-05 (the default template) is measured; lowering it from res-1 numbers could run default runs out of GPU memory |
| 7 | `--use_gpu` | Done: `main.nf` warns that it is ignored |

Still open:

- Measure VRAM at res-05, then set `perJobVramMiB`.
- Surface recon's second thread knob (§4.3).
- The padding branch (§4.5).
- An unrelated resume leak found during verification: the functional workflow rewrites
  its `*.dummy` placeholder files (`workflows/functional_workflow.nf`) in the work dir on
  every run. The new modification time makes every task that takes one miss the cache
  (func-only subjects, QC_TSNR). The fix is to write them only when missing.

## 6. Status of the earlier review (`GPU_RESOURCE_REVIEW.md`)

| Proposal | Status |
|---|---|
| 4.1 central `resolve_device()` honouring `CUDA_VISIBLE_DEVICES` | Done. Used by skull strip, FireANTs and segmentation; FastSurfer's own `find_device` still exists → Phase 2 item 1 |
| 4.2 FireANTs device from config, not `cuda:0` | Done (and now resolved once by the dispatcher) |
| 4.3 GPU token for FireANTs registration | Done, and now only when it actually runs FireANTs SyN |
| 4.4 one `general.gpu_device` | Done; the legacy per-model keys remain as fallbacks and are validated the same way |
| 4.5 ROCm / memory checks / `[Device]` logging / unit tests | Logging and tests done (`tests/test_gpu_device.py`, `tests/test_gpu_scheduling.py`); ROCm not planned; memory → Phase 2 item 6 |
| "Nextflow GPU detection only at parse time" | Unchanged by design: a run is sized once at start-up |

## 7. Verification

Unit tests: `pytest tests/` — 838 passed, 5 skipped, including the new files. `tests/test_gpu_device.py` covers the
resolver across spec × `CUDA_VISIBLE_DEVICES`, and fails if hidden CUDA is probed at all.
`tests/test_gpu_scheduling.py` checks the default-deny `beforeScript` and the token
processes.

Pipeline runs on `dataset_supereasy` (1 subject, T1w + 1 bold),
`template.output_space: NMT2Sym:res-1`, surface reconstruction off, `--no_docker` on
this box (2× RTX A6000):

| Run | Result | GPU use (nvidia-smi sampled every 2 s, filtered by the run's work dir) |
|---|---|---|
| GPU mode (`auto`), fresh | success, 33/33 tasks, 3m57s; `GPU scheduling: enabled`, pool GPUs 0,1 × 4 | exactly the 6 token processes: ANAT_CONFORM, ANAT_SKULLSTRIPPING, ANAT_REGISTRATION, FUNC_COMPUTE_CONFORM, FUNC_COMPUTE_BRAIN_MASK, FUNC_COMPUTE_REGISTRATION. Both registrations `engine: fireants` |
| same, `-resume` | success, 14.7s; everything cached except QC_TSNR (no GPU token; it re-runs on resume for an unrelated reason) | — |
| CPU mode (`gpu_device: -1`, both GPUs visible) | success, 33/33 tasks, 12m56s; `GPU scheduling: disabled (2 GPU(s) detected)`; no `[GPU Assignment]` in any task; `.command.run` has `CUDA_VISIBLE_DEVICES=""` and `unset LD_PRELOAD` | none in 366 samples. Both registrations still `engine: fireants` (CPU path) |

Timing, GPU vs CPU mode: ANAT_CONFORM 22.5 s vs 64 s, ANAT_SKULLSTRIPPING 27 s vs 4m27s,
ANAT_REGISTRATION 15 s vs 5m29s, FUNC_COMPUTE_BRAIN_MASK 7 s vs 58 s.

The `-resume` run cannot show B6: with one subject every GPU task drew GPU 0 both times.
It needs several subjects running GPU steps concurrently.

Container checks on `brainana:latest` (built before this change, with the new
`entrypoint.sh` mounted):

- FireANTs' CPU optimizer and its GPU fused ops both load and run with and without
  `LD_PRELOAD`.
- `docker run --cpus 2 --memory 8g` clamps 8 CPUs / 20g to 2 / 7372MB, with warnings.
  This caught a bug: the image's `mawk` prints `MemTotal` in scientific notation.

Python `logger.info` output from most tasks does not reach `.command.err` (it is empty),
so `[Device]` lines show only in standalone and Lite runs. Phase 2 item 2 (`device_used`
in metadata) closes that gap.

### Phase 2 runs

Same settings (res-1, surface reconstruction off, `--no_docker`, this box), with the
`nvidia-smi` sampler now also recording each process's GPU memory:

| Run | Result | GPU use |
|---|---|---|
| `dataset_supereasy`, GPU mode | success, 33/33, 4m2s | only the 6 GPU steps, spread over GPU 0 (5 steps) and GPU 1 (1); 15 `metadata.json` files record `"device": "cuda:0"`; both registrations `engine: fireants` |
| same, `-resume` | success, 15.1s: 32 cached, only QC_TSNR re-ran (the `*.dummy` placeholder issue, §5) | — |
| `dataset_supereasy`, CPU mode (`gpu_device: -1`, both GPUs visible) | success, 33/33, 13m50s; no step took a slot | none in 394 samples; 14 `metadata.json` files record `"device": "cpu"`; FireANTs ran on the CPU |
| `dataset_devtest` (4 subjects), GPU mode | success, 230/230, 12m44s; up to 4 GPU steps at once | all on GPU 0 (slots 0–3): the slot table then listed GPU 0's slots first. Now interleaved |
| same, `-resume` after interleaving the slot table (a config change between the runs) | success: 198 cached, 32 re-ran | no GPU step re-ran. The 32 are the functional chain of the func-only subject `sub-032183` plus QC, which take `*.dummy` placeholders (§5), identical script and input contents |

Peak GPU memory per task at res-1 (`dataset_devtest`), against the 4096 MiB/job budget:

| Process | Tasks | Peak MiB |
|---|---|---|
| ANAT_SKULLSTRIPPING | 3 | 812 |
| ANAT_REGISTRATION (FireANTs) | 3 | 712 |
| FUNC_COMPUTE_REGISTRATION (FireANTs) | 6 | 482 |
| FUNC_COMPUTE_BRAIN_MASK | 6 | 482 |
| ANAT_CONFORM / FUNC_COMPUTE_CONFORM | 3 / 6 | 428 |

The first Phase 2 CPU-mode and resume runs failed because `defaults.yaml` had been deleted
by hand from the working tree during the run (unrelated to the pipeline). It was restored
and both runs above were repeated.
