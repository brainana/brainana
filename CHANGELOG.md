# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


## [3.1.0] - 2026-09-30

### Upgrading from 3.0.0

No config key or output file was renamed or removed, and a 3.0.0 config runs unchanged. Default outputs do change, so re-run rather than mixing 3.0.0 and newer anatomical or surface outputs in one analysis:

- **Surfaces start from the NMT2Sym template.** Every subject now shares the template's ico6 mesh and vertex numbering; V1's white surface is held at the template. `anat.surface_reconstruction.template_surface.enabled: false` restores the 3.0.0 tessellation.
- **The V1 white-matter fill is off unless tessellated surfaces are built** (`fix_V1_WM: "auto"`): off with template surfaces, and off without surface reconstruction (Brainana Lite, volume-only runs). Set `true` to keep it.
- **Segmentation can run a second pass**, intensity rescaling is capped relative to brain, and `part-phase`/`part-real`/`part-imag` anatomicals are skipped. Masks change only for images these affect.
- **GPU steps really run on the GPU.** In 3.0.0 they ran on the CPU on every machine; results differ within numerical noise.
- **An explicit GPU request the machine cannot serve now stops the run** (`general.gpu_device: cuda` or an index) instead of falling back to the CPU.
- **A config without `anat.synthesis_level` now gets `subject`**, as documented; 3.0.0 used `session` for it.
- **The work directory must support file locks.** GPU steps take a GPU slot with `flock` under `<work_dir>/.gpu_slots`; on Lustre or NFS without lock support the first GPU step fails with a message. Use a work directory on local disk.

### Added

- **Surfaces start from the NMT2Sym template, with V1 held at the template** (`anat.surface_reconstruction.template_surface.enabled`, default on). Instead of tessellating the white-matter segmentation and correcting its topology, surface reconstruction carries the NMT2Sym template's white surface (`template_zoo/fastsurfer/sub-NMT2Sym`, ico6) into the subject with the subject-to-NMT2Sym registration and fits it to the image. V1's white surface stays at the template; the rest of the white surface and the whole pial surface are fitted as before. On T1w data the image-fitted V1 white surface sat about 0.4 mm too far out (median V1 thickness 1.3–1.4 mm, against 1.7–2.0 mm expected; 1.7–1.9 mm with V1 held), and on scans with strong bias or a surface coil V1 lost its fold shape. In blind hemisphere-by-hemisphere comparisons the template surface was chosen for every strong-bias subject of a hard set, and on 24 normal-quality PRIME-DE subjects from 13 sites it was better on 14, tied on 7 and worse on 3 (no site pattern). Freeing V1 within 1–2 mm of the template, or re-registering the occipital lobe first, did not help. Every subject now shares the template's mesh and vertex numbering; `label/?h.template.V1.label` and `scripts/?h.template_init.json` record what was held and which transform was used; the latter (also copied into `scripts/surface_qc.json`) gives the share of the warped template inside the brain mask, and a share under 95% is logged as a suspect registration (99.95–99.98% on devtest). The anatomical registration is reused when `template.output_space` is NMT2Sym; for other output spaces surface reconstruction registers to NMT2Sym itself, so the prior never depends on the output space. If that transform cannot be computed, or puts less than 80% of the template white surface inside the brain mask, the subject is tessellated instead (`template_surface_fallback` in the step metadata). Longitudinal timepoints inherit the base's mesh and V1 label. `template_surface.enabled: false` restores the v3.0.0 tessellation

- **Second segmentation pass on a bias-corrected input when the first one misses brain** (`anat.skullstripping_segmentation.fastSurferCNN.pre_inference_n4`, `.template_prior`). Nothing corrected the bias field before the segmentation network, and strong intensity non-uniformity (a surface receive coil, or sites such as PRIME-DE rockefeller) made it leave out brain that looked too dark — the occipital pole far from a coil, or whole lobes — with a mask that could still look plausible by volume. After the first pass the NMT2Sym template is registered to the subject and the brain it shows outside the mask is measured. When that exceeds 0.5 cm³ (or 0.25 cm³ with a mask under 80% of the template brain), N4 is fitted over the brain and its surroundings and the network runs again on the corrected image; the second result is kept only if it agrees with the template at least as well as the first. The template only decides; the segmentation always comes from the network. On 17 hard cases the brain mask of a surface-coil subject went from 86 to 96 cm³ and of three rockefeller subjects from 42–72 to 82–115 cm³; devtest, the example dataset and 11 other correctly segmented subjects are unchanged. `pre_inference_n4.enabled: true` always runs the second pass (kept by the same rule); `template_prior.enabled: false` skips the registration (about 30 s on a GPU) and falls back to the mask volume. The corrected image is only network input; the bias-correction step still starts from the uncorrected image. If the N4, the template registration or the second pass itself fails, the first pass is kept and the error recorded in the sidecar (`SegmentationCheckFailed`, `PreInferenceN4.Error`); the step does not fail
- **Segmentation provenance in the brain-mask and segmentation sidecars** — `MaskVolumeCm3`, `TemplateBrainVolumeCm3`, `MaskToTemplateBrainRatio`, `MaskUndersized`, `IntensityCapApplied`, `IntensityCapRatio`, `SegmentationPasses`, `TemplatePrior` (missed brain volume and agreement with the registered template), and for a second pass `Pass1` and `PreInferenceN4` (trigger, whether it was kept). A mask under 80% of the template brain is also logged as a warning; a failed segmentation used to pass silently and surface as a failure much later in surface reconstruction
- **Optional relabelling of small label fragments** (`fastSurferCNN.label_island_min_volume_mm3`, default 0 = off). Detached pieces of a label smaller than the threshold take the label surrounding them; the largest piece of every label is always kept. The threshold is a volume, so it means the same thing at every resolution. Across 473 PRIME-DE reconstructions such fragments (median 52 per subject) tracked the number of topology defects
- **`all_subjects_report.html` to browse every subject's QC from one page**, written next to the per-subject `sub-XXX.html` reports when a run has two or more subjects. The subject chip switches subjects (dropdown with filter, ‹ ›, ← →), and a "By step" view stacks one QC figure — e.g. skullstripping — for every subject, naming any subject that has no such figure. Each per-subject report's `brainana` badge links back to it
- **`scripts/surface_qc.json` records the topology-fix path and cortical thickness** — per hemisphere, whether `mris_fix_topology` ran (tessellated surfaces only, i.e. `template_surface.enabled: false`) with `-ga`, fell back without it, or ran without it, whether the mesh was repaired or flipped, and the median thickness with the fraction of vertices under 0.5 mm. A median under 1.0 mm is flagged `collapsed` and logged: a pial surface that never left the white surface used to produce no signal at all. The lowest median across 912 PRIME-DE hemispheres (0.27–1.0 mm voxels) is 1.42 mm

### Changed

- **`fastSurferCNN.fix_V1_WM` defaults to `"auto"`**: the V1 white-matter fill runs only when tessellated surfaces are built (surface reconstruction on, template surfaces off). With tessellated surfaces it helped (in blind comparisons the version without it was never preferred), but with template surfaces V1's white surface no longer comes from the segmentation and the fill made no visible difference. Without surface reconstruction (Brainana Lite, volume-only runs) it only relabelled ~200 voxels of the segmentation, and moving the image by 0.05° changed half of them, so it is off there too. `true`/`false` behave as before

- **The V1 white-matter fix runs once, on the segmentation that is kept**, instead of inside every network pass: with a second pass, the first pass's template registration was thrown away
- **Intensity rescaling is capped relative to brain intensity.** The 0–255 rescale before segmentation (and for the 8-bit volume surface reconstruction starts from) mapped the brightest 0.1% of *all* voxels to 255, so non-brain tissue far brighter than brain — fat and muscle beside a surface coil — squeezed the brain into a few grey levels and the network returned a 7 cm³ "brain". The upper limit is now at most 2.5× the 99th percentile of the image's central box, which after conform is ~90% brain. On the uncorrected conformed T1w of 98 PRIME-DE images from 17 sites the cap binds on 1; the other 97 are rescaled byte-identically. Whether it applied is recorded in the sidecars
- **With tessellated surfaces (`template_surface.enabled: false`), surface reconstruction maps the uncorrected surface with FreeSurfer's quasi-homeomorphic sphere** (`mris_sphere -q`, as `recon-all -qsphere`) instead of a spectral projection. `mris_fix_topology` treats faces that overlap on this map as defects; the spectral map folded more, so defects grew and merged and the topology fix cut away cortex — most visibly the occipital pole, where V1 white matter is thin. On a 7-subject test set, judged hemisphere by hemisphere, the FreeSurfer map was better on 9 of 14 hemispheres and indistinguishable on the rest. It adds 5–35 s per hemisphere. `processing.use_fs_qsphere: false` in the surface-reconstruction package defaults (`src/fastsurfer_surfrecon/config/default.yaml`; not a brainana config key) restores the spectral projection
- **Discovery skips non-magnitude anatomicals.** T1w/T2w files labelled `part-phase`, `part-real` or `part-imag` are listed as "will not be processed" and no longer count as extra runs for multi-run averaging — a phase image used to be averaged into the subject's T1w

### Fixed

- **Output spaces other than NMT2Sym: the NMT2Sym registrations started 25 mm off and failed.** The template prior, the second segmentation pass's N4 region, `fix_V1_WM` and the template surface register NMT2Sym to the subject with FireANTs, which refines a near alignment but does not recover a large shift. Conform aligns the subject to the *output* template. MEBRAINS, D99 and Yerkes19 use an origin about 25 mm from NMT2's (D99 is also tilted about 10°), and a custom template or `anat.conform.enabled: false` can be anywhere. On a devtest subject the template prior then reached a Dice of 0.45–0.57 instead of 0.96, reported 2–7 cm³ of "missed" brain (enough to trigger a second pass), the N4 region lost up to 12 cm³ of brain, and the template surface landed largely outside the brain. `fix_V1_WM` changed 0–15 voxels, none of them correct, instead of 184; this affects 3.0.0 runs with those output spaces too, including Brainana Lite. The image is now first placed in NMT2Sym coordinates by its header alone. NMT2Sym and NMT2Asym need nothing. MEBRAINS, D99 and Yerkes19 use a bundled rigid (`template_zoo/template/xfm/`, made by `scripts/dev/nhp_mriprep/make_template_nmt2sym_xfm.py`). Custom templates and conform off use a rigid registration of the brain to NMT2Sym with the conform method (FLIRT, or SimpleITK in Brainana Lite; 5–10 s). Every frame then matches the NMT2Sym results (prior Dice 0.96, surface 99.9% inside the mask). If the rigid fails, the template checks and `fix_V1_WM` are skipped and the first pass is kept (`Nmt2SymFrame` in the sidecars). Default NMT2Sym runs take the same path as before
- **The undersized-mask rule compares with the NMT2Sym brain (92.5 cm³) whatever the output space.** It used the output template's brain, which moved the cut from 74 cm³ to 82–93 cm³ for `NMT2Sym:res-1`, MEBRAINS, Yerkes19 and D99 and flagged correct ~90 cm³ masks; custom templates had no check at all
- **`T1w` is no longer offered as an output space** in the docs and the config generator. It was listed but cannot run: no template resolves for it

- **GPU steps ran on CPU even when a GPU was detected.** `nextflow.config` declared `use_gpu = false`, and a param set in the config cannot be reassigned by the script, so the runtime switch in `main.nf` was silently ignored. Skull stripping, anatomical and functional registration, functional coregistration, the functional brain mask and the longitudinal base segmentation all ran with `CUDA_VISIBLE_DEVICES=""` on every machine (the task log still read `of N available`). GPU use is now derived from the detected GPU count and `general.gpu_device`, and `main.nf` logs `GPU scheduling: enabled/disabled` at startup
- **CPU-mode tasks no longer load the CUDA driver.** A task with every GPU hidden (`CUDA_VISIBLE_DEVICES=""`) still asked PyTorch whether CUDA was available, which initializes the driver. On a Windows (WSL2) machine with an NVIDIA GPU, skull stripping in CPU mode aborted (exit 134) while the same step passed on Linux and macOS and on the GPU; the driver probe is the likely trigger but was not confirmed on that machine. The segmentation network's device checks now skip CUDA when it is hidden; outputs are unchanged
- **`general.gpu_device: 0` was read as `auto` by the anatomical segmentation** — the setting was looked up with `or`, and 0 is false
- **Every task now starts with no visible GPU; only a task holding a GPU token gets one.** Anatomical and functional registration, the functional brain mask and within-session coregistration exposed a GPU when they held a token but never hid it otherwise. The conform step's skull strip ignored `general.gpu_device` and used the GPU without a token; it now takes one. With `--gpus all` and `general.gpu_device: -1`, these could still initialise CUDA and run on the GPU. CPU-mode runs also no longer preload the CUDA runtime (`LD_PRELOAD`) into every process, and host-side Docker runs no longer pass `--gpus all` when the config forces CPU. When Nextflow starts one container per task, these settings (and the thread limits) are now passed into the container (`docker.envWhitelist`); they used to stay on the host
- **GPU tokens only for steps that use the GPU.** Within-session coregistration (always rigid) no longer holds one, and registration holds one only for FireANTs SyN. On a single GPU with little free memory, CPU-only registrations used to queue behind skull stripping
- **`general.gpu_device: N` now means physical GPU N for the whole run.** Each GPU task sees its GPU as `cuda:0`, so `1` used to request a device that did not exist, and `0` used whichever GPU the task drew. The token pool is now limited to GPU N, and the run stops at start-up if it does not exist. `"-1"` in quotes no longer crashes the segmentation, and `general.gpu_device` is validated when the config is loaded
- **GPU numbering is consistent.** Tasks export `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so a GPU token (an `nvidia-smi` index) names the same physical GPU inside CUDA on machines with mixed GPU models
- **One device policy for every step.** `gpu_device: cuda` silently ran skull stripping and FireANTs on the CPU. An explicit GPU request the machine cannot serve now fails with a message instead of falling back or crashing later. `auto` no longer picks Apple MPS, which the pipeline does not schedule. Least-busy GPU selection no longer returns an index that is not visible to the task. The chosen device is logged as `[Device] ...`
- **FireANTs checks the device it will run on.** Its availability probe ran a CUDA kernel even when the registration was going to run on the CPU. It now probes the resolved device only, and the V1 white-matter fix's template registration follows `general.gpu_device` and `registration.enable_fireants`
- **The segmentation network uses the task's CPU allocation** (`OMP_NUM_THREADS`) instead of a fixed 8 threads. Two concurrent skull-strip tasks (2 CPUs each) could run 16 threads
- **`-resume` no longer re-runs GPU steps that drew a different GPU.** The GPU a task used was a task input, so on a machine with several GPUs a resumed run re-ran any GPU step that happened to draw another GPU, plus everything downstream of it. GPU steps now take a GPU slot when they start (`bin/brainana_gpu_slot.sh`, an `flock` under `<work_dir>/.gpu_slots`), so the GPU is no longer part of the task. Switching between CPU and GPU mode still re-runs these steps, because the results differ. A GPU step that fails can no longer hold on to its GPU and hang the run. A work directory whose file system cannot lock (Lustre, NFS without lockd) fails the step with a message instead of waiting forever. No GPU step runs more tasks at once than there are GPU slots, because a task waits for its slot inside its own time limit
- **`-resume` no longer re-runs functional steps that did not change.** Two causes. First, the workflows rewrote their empty placeholder files (`<work_dir>/*.dummy`, stand-ins for absent inputs) on every run; the new modification time changed every consuming task's cache key, so a subject without anatomy re-ran its whole functional chain, and the tSNR QC re-ran, on each resume. Placeholders are now written only when missing. Second, runs grouped per session (tmean averaging, session tSNR) were passed in completion order, which varies between runs and changed the task script; they are now ordered by run id
- **A CUDA out-of-memory error no longer fails the step.** The segmentation network, the skull-strip network and FireANTs (with `fireants_allow_cpu`) are rerun once on the CPU with a warning. Each step's `metadata.json` now records the device it ran on and any such fallback under `device`
- **`--use_gpu` prints a warning that it is ignored**; GPU use follows `general.gpu_device`
- **The container limits `NXF_MAX_CPUS`/`NXF_MAX_MEMORY` to what it has** (Docker Desktop VM size, `docker run --cpus/--memory`), with a warning, and no step asks for more CPUs or memory than those limits. A 3-CPU step could never be scheduled with `NXF_MAX_CPUS=2`
- **A native crash now leaves a Python stack trace** in the task's `.command.err` (`PYTHONFAULTHANDLER=1`). An abort used to leave only bash echoing the task's script, with no sign of where it failed
- **A partial `--config` could silently switch anatomical synthesis to per-session.** BIDS discovery read the user's YAML without the package defaults — `validate_config()` merged them but its result was discarded — and fell back to its own `synthesis_level: "session"` where `defaults.yaml` says `"subject"`. A config that omitted the key produced one T1w per session for multi-session subjects, while `nextflow_reports/config.yaml` recorded `subject`. Affects 2.1.0 and 3.0.0 runs whose config did not set `anat.synthesis_level`. Discovery now runs on the merged config, and an empty section (`bids_filtering:`) no longer crashes it
- **`--anat_only` on the command line now reaches BIDS discovery**, so the start-up summary and `functional_jobs.json` agree with what runs. The functional workflow was already skipped correctly
- **`mris_fix_topology -ga` crashing no longer aborts the hemisphere** (this and the next two: tessellated surfaces only, `template_surface.enabled: false`). FreeSurfer 7.4.1's genetic-algorithm search can die outright on large defects ("stack smashing detected"); stage 12 now retries with the default search, whose result faces the same pre-orig gate. `processing.topology_fix_ga: false` in the surface-reconstruction package defaults (`src/fastsurfer_surfrecon/config/default.yaml`) skips the GA attempt
- **`-resume` keeps the rescued mesh.** A resumed reconstruction re-ran the `-ga` topology fix and discarded a mesh that the default-search rescue had won; it now takes the recorded result. A sphere that has to be redone is re-inflated first
- **A `-ga` result that is a failed search no longer silently costs cortex.** `mris_fix_topology -ga` can finish without error yet return an open or non-genus-0 mesh; pymeshfix then closed it by deleting the region around the defect (on one subject, the occipital pole). Stage 12 now also runs the default search in that case, and when pymeshfix would remove more than 2% of the vertices, and keeps whichever repaired mesh retains more surface. Both attempts are recorded in `scripts/<hemi>.topology_fix.json` (mode `no_ga_rescue` when the default search wins)
- **An inside-out premesh no longer fails the pre-orig gate.** A mesh that is closed, consistently wound and genus 0 but inverted skipped the pymeshfix repair (which only looked at closed/oriented/Euler) and was then rejected by the gate, which requires outward normals. It is now flipped before the gate
- **`mris_place_surface` options the wrapper does not know are rejected** instead of being dropped without a word
- **The report generator's fallback config is read over the package defaults**, like every other reader
- **Surface reconstruction normalised intensities on cortex instead of white matter.** Stage 2 takes white matter to be aseg labels 2/41 but was given the atlas volume, whose labels are raw atlas IDs — in ARM2, 2 is right anterior cingulate cortex. `norm.mgz` therefore put that cortex at 105 and true white matter at ~125–175 instead of ~105. It now normalises on `aseg.auto_noCCseg.mgz`


## [3.0.0] - 2026-09-19

### Upgrading from 2.1.0

Output filenames changed and input validation is stricter. Re-run rather than mixing 2.1.0 and 3.0.0 output in one derivatives tree.

**Runs that used to succeed may now abort.** BIDS validation is always on and has no skip flag. It errors only when a directory and a filename disagree about subject or session (`BIDS101`–`BIDS104`), or a subject mixes session and datatype directories (`BIDS105`). Both meant the earlier run was already wrong — mislabelled derivatives in the first case, silently dropped input in the second — so 2.1.0 output for those datasets should not be trusted. Every error names the affected files and the rename or move that resolves it.

**Renamed outputs.** Anything keying on an exact 2.1.0 filename — scripts, viewer configs, bookmarks — needs updating:

| 2.1.0 | 3.0.0 |
| --- | --- |
| `…_space-<tpl>_desc-preproc_desc-func2target_bold.png` | `…_space-<tpl>_desc-func2target_bold.png` |
| `sub-01_ses-001ref` (from a `_boldref` input) | `sub-01_ses-001` |
| `…_desc-a_desc-b_…` | `…_desc-b_…` |
| `…_space-scanner_space-T1w_desc-preproc_…` | `…_space-T1w_desc-preproc_…` |
| custom entities emitted alphabetically | emitted in source order |

The repeated-key names could not round-trip: `parse_bids_entities` keeps only the last value of a key. Custom-entity order differs only for datasets carrying two or more non-standard entities.

**Removed:** two Brainana Lite config overrides that were already no-ops — `registration.fireants_allow_cpu` (already the default) and `general.anat_only` (read only by BIDS discovery).

### Added

- **`anat.synthesis_level: "session_longitudinal"` — within-subject longitudinal surface reconstruction.** A strict superset of `"session"`: identical anatomical selection, plus an unbiased within-subject base template per subject and a base-seeded reconstruction per session, so vertex *i* is the same anatomical point at every timepoint and per-vertex differencing is valid with no surface registration. The base is built with `mri_robust_template` alone — no atlas priors, no species assumptions — where `recon-all -base`/`-long` would run a human GCA volume stream and clobber the CNN segmentation the surfaces are built on. Requires `anat.surface_reconstruction.enabled` and at least two sessions with anatomy; single-session subjects are skipped. Tunable under `anat.surface_reconstruction.longitudinal`: `iscale`, `subsample`, `max_cbv_dist`, `pial_blend_weight`, `time_column`
  - Publishes `fastsurfer/sub-<id>_base/` and `fastsurfer/sub-<id>_ses-<id>_long/`, base-space derivatives and atlases under `sub-<id>/anat/` marked `space-base`, and within-subject rates of change: `surf/?h.long.<measure>-{rate,avg,spc}.mgh`, `stats/?h.long.roi-rates.csv`, `stats/long.change-stats.json`, and `scripts/long.qdec.table.dat` for FreeSurfer's own longitudinal tools
  - Rates are fitted against real elapsed time when `<bids_dir>/sub-XX/sub-XX_sessions.tsv` carries an `age` or `acq_time` column, or one named by `longitudinal.time_column`; otherwise they fall back to digits in the session label, then to scan order. Resolution is all-or-nothing per subject rather than mixing ages and scan indices in one regression, and `long.change-stats.json` records `time_source`
  - The functional and fsnative-atlas streams deliberately keep seeing the cross-sectional trees: a `_long` tree's `orig.mgz` is in base space, and `mri_vol2surf --regheader` would misregister by exactly the timepoint-to-base transform
  - `scripts/base_segmentation_agreement.json` records per-label Dice between the base's own segmentation and each timepoint's, so whether the CNN's domain shift on a robust average matters is answerable from the run's own outputs

- **A declared output-naming contract, and a validator for it.** `src/nhp_mri_prep/utils/bids.py` declares `ENTITY_ROLES` — every entity is *identity* (inherited from the raw data; brainana copies and never sets it), *frame*, *variant* or *product* — and `derive_output_name()` is the one constructor, raising rather than resolving the two ways a name loses information: overwriting an inherited identity entity, or emitting a key twice. `validate_output_name()` checks a published name; `scripts/check_output_naming.py` checks a whole output tree and exits non-zero on anything undeclared
  - `BIDS_ENTITY_ORDER` gains the entities brainana emits but never listed — `atlas`, `from`/`to`/`mode`, `res`, `den`, `hemi`, `stat` — at the positions already published. They had fallen into the alphabetised `others` slot, which is emitted *before* `space` and `desc`
  - Also found by the audit: nine QC figures carried two `desc-` entities; `FUNC_APPLY_TRANSFORMS` declared one glob for two output slots, so the consumer asking for the boldref was handed the 4-D BOLD; the publish-time `desc-preproc` rewrite injected one `space-T1w` per `desc-` token; `isT1wFile` matched any path with `T1w` in a directory name; `dataset_description.json` omitted `BIDSVersion`; and `QC_BIAS_CORRECTION_FUNC` was dead
  - The `_brain` suffix tail and the subject-last atlas names are **declared deviations**, recorded against the consumer that depends on each (brainana-viewer's data contract) rather than left looking accidental
  - **The longitudinal stream is marked `space-base`, not `acq-base`/`acq-long`.** `acq` belongs to the raw data, so a stream marker there destroyed a timepoint's own `acq-` label and could collapse two acquisitions onto one filename — which `publishDir overwrite:false` makes a missing file rather than an error. Affects `session_longitudinal` only, which is unreleased; no published tree changes

- **Uncropped conformed anatomical (`desc-conformFullFOV`).** The conform field of view is sized from the template, so a recording chamber, head-post, coil markers or the neck fall outside it and are cropped from every derivative with no way to recover them. Conform now also writes the same conform on a grid enlarged just enough to contain every voxel of the scanner-space input — same transform, orientation and resolution, differing by a whole number of voxels, so cropping at that offset recovers the processed image. A leaf output: nothing downstream reads it. Anatomical only, both rigid backends (`flirt`, `sitk`). If the grid cannot be sized or would exceed a voxel cap, the run warns and falls back to the target field of view rather than failing
  - Conform QC gains a paired figure on the uncropped grid, with the processing field of view drawn as a lavender box, so the crop is legible rather than invisible. The pair appears only when the grid was actually enlarged; otherwise the two would be pixel-for-pixel identical and only the existing figure is rendered

- **BIDS input layout is validated before processing starts.** Discovery cross-checks every input file's `sub-`/`ses-` entities against the directories it sits in and aborts with a report naming each affected file. pybids resolves such a conflict in favour of the *directory* and drops the filename entity, but brainana names output directories from the directory and output filenames from the filename stem — so `sub-monkey1/anat/sub-monkey_ses-1_run-1_T1w.nii.gz` used to process without a warning and publish `sub-monkey1/` full of `sub-monkey_*` files. Scoped to the subjects and sessions the run will actually process, so one malformed subject cannot block the rest of a dataset
  - Errors: `BIDS101` subject label mismatch · `BIDS102` no `sub-` entity · `BIDS103` session label mismatch · `BIDS104` no `ses-` entity where the subject has several sessions · `BIDS105` subject mixes `ses-*/` with subject-level datatype directories
  - Warnings, which never stop a run: `BIDS201` `ses-` entity with no `ses-*/` level · `BIDS202` NIfTI that will not be processed · `BIDS203` no `dataset_description.json` · `BIDS204` no `ses-` entity, single session

- **`func.confounds.fd_radius_mm` — the FD rotation radius is now configurable.** It was pinned at the macaque 27 mm with no way to reach it from configuration, so any other primate silently got FD computed for a macaque head. Validated as a positive number and recorded per column in the JSON sidecar, since FD values are only comparable across runs computed at the same radius

- **Confounds and motion QC figures now show which frames were flagged.** Vertical bands behind the traces — gray for non-steady-state volumes, red for motion outliers — plus dashed threshold reference lines on the FD and DVARS panels. Contiguous frames merge into one band, and bands are clamped to the shared frame axis, so pixel alignment between the stacked figures is unchanged. Indicators only; no volumes are removed from the BOLD data. The motion figure reads the confounds TSV as an *optional* input, so it is produced exactly as before when confounds are disabled or failed

- **Brainana Lite publishes the outputs it was already computing** — the uncropped `desc-conformFullFOV` conform, the paired conform QC figure, the hemisphere mask, and ingest-normalization provenance (`Input4DCollapsed`, `OrientationRecovered`, `QformSformReconciled`, `InputHeaderWarnings`) in the scanner-space sidecar. Lite was already paying for the enlarged-grid resample and discarding it; the normalization repairs likewise happened but went unrecorded, which mattered most for `OrientationRecovered` and its left/right mirror caveat. Lite now also writes a `dataset_description.json` and the effective merged config under `lite_reports/`, making its output a valid BIDS-derivatives dataset, and `tests/test_lite_notebook_api.py` brings the notebook under CI

### Changed

- **`_desc-conform_` intermediates are matched by exact entity token, not substring**, so the new `_desc-conformFullFOV_` image publishes normally instead of being silently dropped and appended to the downstream channel
- **Lite fetches atlases the way the runtime selects them** — the sparse checkout required an exact `_space-{template}_res-05` match while `discover_atlases` matches by space and then picks the nearest resolution, so any atlas stored at another resolution was silently dropped from backprojection. Worst for `TEMPLATE="D99"`, which discarded the D99 parcellation itself. Atlas-segmentation QC also now uses the pre-bias brain as its underlay
- **Lite QC figures are published at `sub-XX/figures`**, never under a `ses-` level, matching the pipeline, and are named from the BIDS stem rather than a hand-assembled prefix, so `run-` and `acq-` entities survive
- **Lite config overrides go through a checked setter** — `load_config()` deliberately does not validate, so a mistyped key created an entry nothing read and the run continued on the default the user believed they had overridden
- **MacBNA lookup table gained an `ID_nohemi` column** — MacBNA splits 152 regions into 304 hemisphere-specific IDs (1–152 left, 153–304 right), so nothing paired a region with its contralateral homologue. `ID_nohemi` is the hemisphere-independent index: `ID` 1 and `ID` 153 are both `FP.d` and now share `ID_nohemi` 1. Inserted as the second column, so any consumer reading the table by column position must be updated. The file also moved from CRLF to LF, and `*.tsv` is now covered by `.gitattributes`

### Fixed

- **Confounds JSON sidecar reported the wrong rotation radius** — `RotationRadiusMM` was written from the module constant rather than the radius actually used, so an overridden radius disagreed with the FD values in the very TSV it describes. Unchanged for the default 27 mm
- **`func.confounds` thresholds are validated** — only `enabled` was checked, so `fd_outlier_threshold_mm: loose` or a negative radius passed validation and failed much later inside a Nextflow process, far from the cause
- **Motion QC step read the wrong result key** — `qc_motion_correction` read `snapshot_file`, but `create_motion_correction_qc` returns `motion_plot`, so the recorded `qc_files` entry was never the value the snapshot function returned
- **ARM6 atlas: area 32 (ID 1004) carried the hemisphere label `kg`** — a typo, now `lh`, matching ARM4/ARM5 and the ID's left-hemisphere range
- **ARM4 atlas lookup table normalized to the shape every other ARM atlas uses** — dropped the `key_L1`–`name_L3` hierarchy columns (referenced by nothing) and the repeated abbreviations in `name_full` (`claustrum (Cl)` → `claustrum`), and uncoloured its lone coloured subcortical row so viewers assign a procedural colour. IDs, labels, regions, names, hemispheres and cortical colours are unchanged
- **Brainana Lite** — fixed a Colab hang at "Ensuring SuiteSparse headers" (an unbounded `apt-get install` for a build dependency Lite never installs; removed rather than hardened); re-cloning and reinstalling on every run (the reuse check compared `rev-parse --abbrev-ref HEAD` against the ref, which returns the literal `HEAD` for a tag, so it could never succeed for the pinned default); `RUN_DEMO=True` aborting on a second run over the file the first had downloaded; a `BRAINANA_URL` still naming the pre-migration `xingyu-liu/brainana` while the Colab badge had moved; unbounded network calls with no stdin in the environment cell, which hung the cell forever on a credential prompt; a single failed QC figure aborting an otherwise complete run, where the pipeline gives every QC process `errorStrategy 'ignore'`; and dead code from a patch whose guard has existed since before 2.1.0


## [2.1.0] - 2026-07-26

### Added

- **"Data findings" section in the QC report** — the per-subject HTML report now shows what ingest repaired ("Repaired automatically") and what it could not ("Not repaired — no safe automatic fix"), with the left/right caveat spelled out on the orientation entry. It renders only when there is something to report, so a well-formed dataset produces an unchanged report. Deliberately separate from the run-status badge: that tier answers "did the pipeline execute", and demoting a green run because a header had odd units would train people to ignore it
- **Surface topology QC record** — every run writes `scripts/surface_qc.json` with the measured state of each key surface (vertices, faces, closed, oriented, Euler, signed volume), so surface defects are checkable after the fact rather than only inferable from run status
- **Surface reconstruction unit tests** — previously none of it was covered. The tests synthesise meshes in numpy, so they need neither FreeSurfer binaries nor subject data and run in under a second

### Fixed

#### Anatomical ingest and input validation

- **4D anatomical inputs no longer crash the pipeline** — some scanners and DICOM converters emit T1w/T2w with a trailing singleton frame axis (e.g. `(144, 144, 60, 1)`, `dim[0] = 4`). These are geometrically 3D but broke `ANAT_CONFORM`, which failed during skullstripping with `ValueError: This function can only deal with 3D images`. Anatomicals are now normalized to 3D once at ingest (`ANAT_SYNTHESIS`, before any other step): a trailing singleton is dropped losslessly, a genuine multi-volume anatomical is averaged over its last axis. Recorded as `Input4DCollapsed` in the JSON sidecar. Already-3D inputs pass through byte for byte; BOLD timeseries and ANTs displacement fields keep their non-spatial dimensions
- **Anatomicals with no stored orientation are made explicit** — a NIfTI with `qform_code = 0` *and* `sform_code = 0` declares no spatial orientation, and readers do not agree on what to assume: nibabel and FSL fall back to the header's base affine (LAS, origin at the centre of the voxel grid), while ITK — and therefore ANTs — uses an identity direction in LPS with the origin at the *corner*. One grid silently meant two different geometries within a single run, so files genuinely on the same grid came out disagreeing by an axis flip and a half-FOV translation. Ingest now writes the nibabel/FSL fallback into both qform and sform with code 2. **Header only — voxel data is not resampled**, and existing numeric behaviour is unchanged; what changes is that ANTs reads the same geometry as everything else. Recorded as `OrientationRecovered`
  - **Caveat:** this recovers a *convention*, not ground truth. The assumed affine puts +x at the subject's left; if the acquisition ran the other way the result is a left/right mirror that no rigid or affine registration can undo and that is invisible on inspection. If a sidecar carries `OrientationRecovered`, confirm handedness against an external record before trusting hemisphere-wise results
  - **Scope:** anatomicals only. A BOLD run with `qform_code = 0` and `sform_code = 0` is still subject to the reader-dependent fallback
- **Disagreeing qform and sform are reconciled** — a NIfTI can store its geometry twice and nothing enforces that the two agree; which one wins is the *reader's* policy. nibabel, FSL and every other brainana step read the sform, while FastSurfer's `check_affine_in_nifti` resolves toward the qform — so an unreconciled header meant brainana registered against one grid and segmented against another, with only a warning buried in the FastSurfer log. Ingest now writes the sform into both forms, preserving the sform's own code. Header only. Recorded as `QformSformReconciled`
- **Uncompressed `.nii` anatomicals are converted to `.nii.gz` at ingest** — previously they flowed through uncompressed and were gzipped only at publish time. Converting once, up front, means no intermediate step handles a raw `.nii`. This also removes a latent crash class: nibabel mmaps an uncompressed `.nii`, so any path that rewrote such a file onto its own path died with SIGBUS — exit 135, no traceback, nothing Nextflow could report
- **Header defects with no safe automatic repair are reported instead of guessed at** — rescaling or rewriting them could just as easily turn a recoverable dataset into confidently wrong output, so two cases are detected at ingest and surfaced: `xyzt_units` declaring something other than mm (nibabel returns raw `pixdim` regardless, so a metre-unit header is read 1000× too small while ITK/ANTs converts it correctly), and `pixdim` disagreeing with the affine's voxel scale (a self-inconsistent header; FastSurfer aborts on this deep inside segmentation). Written to the sidecar as `InputHeaderWarnings` and shown in the QC report
- **`ANAT_SYNTHESIS` now publishes its JSON sidecar** — the sidecar for the scanner-space anatomical was written into the task directory and then silently dropped, so `Sources`, `SkullStripped` and `Synthesized` never reached the output directory for that file. `publishDir` only publishes *declared outputs*, and the process declared `path "metadata.json"` where every other process in the module declares `path "*.json"`. Pre-existing since sidecars were introduced in 1.3.0; it also suppressed the new ingest-normalization keys
- **Skullstripping and segmentation hardened against 4D input** — `nhp_skullstrip_nn` collapses a frame axis before its anisotropic-voxel resampling step, so the standalone CLI works on such files too; `fastsurfer_nn` no longer passes a 4D `out_shape` when resampling a segmentation back to native space (`RuntimeError: affine matrix has wrong number of columns`), which was reachable with `anat.conform.enabled: false`

#### Surface reconstruction

- **Surface topology repair silently stopped working, and could produce quietly wrong surfaces** — `pyvista` was dropped from the dependency set on the strength of `grep "import pyvista" src/`, which cannot see that it is a *call-time* requirement of `pymeshfix` rather than an import of ours: `pymeshfix.MeshFix.__init__` probes `find_spec("pyvista.core")`, which *raises* when pyvista is absent. A broad `except Exception` logged the failure as a warning, so `mris_fix_topology` output that needed repair was passed through unrepaired. Repair now uses pymeshfix's `PyTMesh` API — the same call sequence `MeshFix.repair()` performs, with no pyvista involved — and the floor is raised to `pymeshfix>=0.18.1`
  - **Who is affected.** Any environment created or refreshed after the dependency was removed. When the defective premesh was *open*, the run crashed later in spherical projection (`ValueError: Can only project closed meshes`) and nothing bad was published. When it was closed but not genus 0, projection succeeded and the run **completed normally with an unrepaired surface**. Affected subjects are therefore not identifiable from run status alone — re-run surface reconstruction for any subject processed by such an environment
- **Inside-out surfaces are detected and corrected** — repairing a non-oriented mesh can return one that is *consistently* wound but entirely inverted, which passes every topology check there is (closed, oriented, Euler 2); only the sign of the enclosed volume distinguishes it, and nothing was checking that. Because `mris_autodet_gwstats` estimates the gray/white intensity thresholds by sampling *along surface normals*, an inverted surface made it read inside for outside: on an affected subject the white and gray means came out swapped (110/91 became 91/110), inverting every threshold used to place the white and pial surfaces, with no error logged anywhere. Repair now normalises the winding sign, `fix_surface_orientation` flips an inverted surface rather than declaring it fine, and `orig` is gated on outward-facing normals
- **A broken mesh now fails at the stage that produced it** — topology is validated in-process as closed *and* consistently oriented *and* Euler 2, rather than by parsing `mris_euler_number` output. Euler alone is insufficient (one backwards-wound triangle is still closed with Euler 2), and the old check failed *open*: a missing binary, a timeout or unparsed output all returned `None`, which skipped the entire validate-and-repair block silently. A defective mesh can no longer be promoted to `orig`, and since `mris_place_surface` preserves connectivity, gating `orig` transitively protects `white` and `pial`
  - **Behaviour change — runs that previously completed may now abort.** When pymeshfix cannot reach a closed, oriented, genus-0 mesh within its 5 iterations, stage 12 now raises instead of promoting the best-effort result and continuing. Such a subject used to finish and publish surfaces built on a defective `orig`; it now fails at the stage that produced the defect. This gate is deliberately *not* covered by `processing.strict_surface_checks` — that flag governs the warn-only checks at stages 8, 9, 11 and 15, whereas nothing downstream of a broken `orig` is meaningful. A subject that starts failing here was already producing unreliable surfaces; inspect `scripts/surface_qc.json` for what the meshes actually look like
- **Stages can no longer report success without producing their outputs** — `Completed {stage}` previously fired whenever the stage body returned without raising, and FreeSurfer wrappers returned their output path without checking anything was written. Stages now declare their outputs, which are verified before the stage is recorded complete, and commands that exit 0 without writing raise instead
- **Resuming into a half-finished stage no longer skips the rest of it** — stage 12 wrote `orig` at step 2 of 8 but used `orig` alone as its "already complete" signal, so a run that died at step 8 skipped the stage entirely on the next invocation. Skip checks now require the stage's full output set, including a file it writes last, and stage 12 regenerates `smoothwm`/`inflated` when `orig` changes rather than reusing artifacts built from a superseded mesh
- **A failure in one hemisphere no longer discards the other's completed work** — the parallel hemisphere runner raised on the first failure, but `ThreadPoolExecutor` waits for the other worker regardless, so its result was computed and then thrown away and a second failure was never reported. Both outcomes are now collected and logged before raising
- **`fix_surface_orientation` no longer claims success it did not verify** — it logged "Fixed and saved" after calling `orient_()` without re-checking, and `orient_()` cannot orient a mesh with boundary edges. It now refuses such meshes up front and re-reads from disk to confirm

#### Runtime

- **Local runs pin the Nextflow version** — `run_brainana.sh` exports `NXF_VER=25.10.2` (matching the Dockerfile) unless already set. A freshly installed launcher otherwise self-downloads the newest release, and Nextflow 26.x defaults to the strict config parser, which rejects the Groovy in `nextflow.config` and aborts before any work starts. Docker runs are unaffected

### Changed

- **Dependency changes are now checked by CI** — the only existing workflow installs with `pip install --no-deps` and therefore cannot detect a missing dependency by construction. A new `Dependencies` workflow installs for real: it verifies the lockfile is in sync, imports every shipped module under a **core-only** install (where a module needing an extra actually shows up), and runs the test suite on the full set. Because the pyvista class of bug is invisible to any import check, the surface tests perform a real mesh repair — that is the layer that catches it. `CONTRIBUTING.md` documents why grep is not sufficient evidence for removing a dependency
- **`psutil` is now declared in the `train` extra** as well as `full` — the training data-prep scripts import it unguarded, and `full` deliberately excludes `train`, so `train` was not self-sufficient
- **Development scripts no longer live under `src/`** — 17 notebook-style scratch drivers sat inside the installable package tree across all four packages, every one of them referenced by nothing and every one carrying hardcoded absolute paths. Most ran real work at *import* time — a batch atlas backprojection over a whole dataset root, a GPU registration, a torch model load, NIfTI resampling and writing — so merely importing one ran it. Because `[tool.setuptools.packages.find]` defaults to `namespaces = true`, they shipped in the wheel and the Docker image despite having no `__init__.py`. They now live under `scripts/dev/`, grouped by the package they drive (`fastsurfer_seg/`, `fastsurfer_recon/`, `nhp_mriprep/`, `nhp_skullstrip/`), which is already excluded from the image. What remains under `src/` is library code, the `nhp_skullstrip_nn` prediction CLI, the `nextflow_scripts/` pipeline plumbing and the two training drivers. Separately, a genuine pytest suite that had been misfiled under `src/` moved to `tests/`, where it runs for the first time
- **BIDS discovery summary distinguishes inputs from jobs** — the anatomical section previously printed one "Total jobs" count, which under multi-run synthesis reported N input files as a single job and read as if data had gone missing. It now prints `BIDS inputs → processing jobs` per modality, with the cross-session / within-session / no-synthesis breakdown underneath. Under `general.anat_only` the functional section prints an explicit "skipped" notice instead of a bare `0`
- **QC report titles include the session** — the report heading is derived from the report filename rather than the subject ID alone, so per-session reports are distinguishable

### Removed

- **`nextflow_scripts/read_yaml_config.py`** — dead since it was added; no `.nf` file, shell script or Python module ever called it, unlike its three siblings which `main.nf` and `run_brainana.sh` do invoke
- **`ANAT_REORIENT` and `FUNC_REORIENT` processes** and the AFNI-backed helpers behind them (`operations.reorient`, `utils.reorient_image_to_target`, `utils.reorient_image_to_orientation`, `utils.get_image_orientation`). Both processes were already unreachable — `main.nf` had not referenced them since orientation handling moved into `ANAT_CONFORM` — but the helpers were exported from `nhp_mri_prep.utils` and `nhp_mri_prep.operations`, so anything importing them directly must be updated. Reorientation to the reference grid is performed by the conform step; the ingest normalization above covers the missing-orientation case


## [2.0.0] - 2026-07-20

### Added

- **Brainana Viewer companion docs** — a new [Brainana Viewer](https://brainana.readthedocs.io/en/stable/viewer.html) page introduces the cross-platform NiiVue desktop viewer for exploring per-subject pipeline output (anatomical volumes, cortical surfaces, atlas overlays, and functional maps), with a link to the [viewer's repository](https://github.com/brainana/brainana-viewer)
- **Bundled demo dataset + "Try a demo" guide** — a small, ready-to-run BIDS dataset (`examples/dataset_example/`, one macaque subject: two T1w, one T2w, two resting-state runs) exercises the full pipeline end to end; the [Try a demo](https://brainana.readthedocs.io/en/stable/demo.html) page documents the run command, expected run time, and expected output
- **New bundled atlases** — **D99**, **MacBNA** (Macaque Brainnetome Atlas), and **FuncNetwork** (functional network parcellation) added to `template_zoo/atlas/`, with accompanying `.tsv`/`.md`/`.bib` metadata sidecars

### Changed

- **ARM atlas metadata** — ARM1–ARM6 TSVs now include a `color` column (hex RGB lookup) for consistent region coloring; rows reformatted to match


## [1.3.0] - 2026-07-13

### Added

- **Atlas surface projection (fsnative)** — when surface reconstruction is enabled, T1w-space atlases are projected into FastSurfer space and onto the cortical surface (nearest-neighbour throughout), published under `anat/atlas_space-fsnative/` as resampled label volumes `atlas-<name>_space-fsnative_<prefix>.nii.gz` and per-hemisphere maps `atlas-<name>_space-fsnative_hemi-<L|R>_<prefix>.func.gii`
- **Custom template files** — `--output_space` (`template.output_space`) accepts an absolute path to a `.nii/.nii.gz` file; outputs use the fixed BIDS space label `template`. Strict validation (wrong extension or missing file aborts at run start, no silent fallback); the resolved path is recorded in each sidecar and in `dataset_description.json` (`TemplateSource.Custom`). Custom templates have no bundled atlases, so atlas outputs are skipped for that space
- **JSON sidecars on derivatives** — anatomical *and* functional publish processes emit BIDS JSON sidecars (`write_derivative_sidecar`) recording `TemplateSource`, `SkullStripped`/`Type`/`Sources`, and BOLD timeseries fields; a `dataset_description.json` is written at run start
- **Engine-aware transform sidecars** — `*_xfm.json` `GeneratedBy` names the actual registration engine after any runtime fallback (FireANTs / ANTs / ANTsPy / FLIRT / SimpleITK)
- **Atlas metadata sidecars** — `atlas-{name}.tsv`, `.md`, and `.bib` are copied into every output space (T1w, scanner, fsnative)
- **Centralized CLI flag handling** — `flags.sh` + `known_flags.txt` as the single source of truth; `-h/--help` prints `USAGE.txt` before any heavy setup; unknown args are rejected fast; underscore is canonical with hyphenated aliases (`--work-dir` → `--work_dir`)

### Changed

- QC reports read the merged Nextflow config, so they reflect CLI overrides such as custom `output_space` templates
- `stc_enabled` default aligned to `defaults.yaml` (`false` → `true`)

### Fixed

- **Anisotropic skull strip** — resample a NIfTI-path input to isotropic (0.5 mm) before 2.5D U-Net inference, then map back to the native grid; previously an anisotropic T1w distorted the aspect ratio and collapsed the mask, breaking anatomical conform (native grid/affine/header unchanged)
- Atlas projected counter increments only when a hemisphere is actually projected
- Dropped the `template_dir` override; the bundled template manager is always used
- Clarified the `main.nf` hyphen-normalization error message


## [1.2.0] - 2026-07-03

### Added

- **Functional confound regressors** — fMRIPrep-compatible nuisance regressors written per run as `*_desc-confounds_timeseries.tsv` (+ JSON sidecar), compatible with `nilearn...load_confounds`: 24-parameter motion, framewise displacement (macaque 27 mm radius) and RMSD, DVARS/std-DVARS, global-signal and (when a T1w segmentation is available) CSF/WM tissue regressors, plus non-steady-state and motion-outlier indicators. Regressors only — the BOLD image is never scrubbed ([docs](https://brainana.readthedocs.io/en/stable/processing.html))
- **Confounds QC** — fMRIPrep-style confounds panel (global signal, CSF, WM, DVARS, FD) in the HTML report
- **QC run-status badge** — reports are now always generated on completion and carry a status badge: **Pass**, **Pass with warnings** (an optional step failed; partial outputs), or **Fail** (early abort)
- `func.confounds.enabled` toggle and support for runs without motion correction
- Config-validation hardening with defaults/config-generator consistency tests

### Changed

- Confound computation is gated behind `func.confounds.enabled`

### Fixed

- Functional pipeline emits a dummy sentinel for skipped tSNR runs
- QC report: About/Methods headings wrapped in prose measure; confounds pipeline fixes

### Docs

- **SEO metadata** — canonical URLs, Open Graph / Twitter cards, per-page meta descriptions, `SoftwareApplication` JSON-LD, and Google Search Console verification
- Restructured functional processing docs: added confound-regressors, tSNR, and despike method sections and renumbered the functional steps
- Template/atlas zoo note that more atlases are bundled; documented the QC status badge


## [1.1.0] - 2026-06-03

### Added

- **Brainana Lite** — lightweight, notebook-driven T1w volumetric preprocessing for Jupyter and Google Colab (no Docker required): BIDS organization → synthesis → conform → skull strip/segment → bias correction → template registration → atlas backprojection, with inline QC figures ([`examples/BrainanaLite.ipynb`](examples/BrainanaLite.ipynb), [docs](https://brainana.readthedocs.io/en/stable/brainana_lite.html))
- Colab vs local Jupyter auto-detection; Google Drive mounts, T1w input validation/sync checks, demo mode (`RUN_DEMO`), and isolated env under `WORKING_DIR/brainana_lite_env/`
- Example T1w (`examples/exam_T1w_ple.nii.gz`) and local launcher (`examples/test_BrainanaLite_local_instruction.sh`)
- **SimpleITK rigid registration** — FSL-free alternative to FLIRT for anatomical conform; selectable via `anat.conform.rigid_method` (`flirt` | `sitk`)
- **ANTsPy fallback** (`antspyx_ops.py`) when ANTs CLI is absent; Lite uses `brainana[lite]` and `set_ants_backend("antspyx")`
- **FireANTs CPU syn** — `registration.fireants_allow_cpu` (default `true`) runs affine+greedy on CPU when no GPU (FireANTs 1.5.0)
- Optional dependency extras: `[lite]`, `[surf]`, `[func]`, `[train]`, `[full]` (see `pyproject.toml`)
- `tests/test_create_output_link.py`

### Changed

- Core Python dependencies slimmed; full Docker/runtime should install `brainana[full]` (surfaces, func/BIDS, psutil)
- Lite defaults: SimpleITK rigid (`rigid_method=sitk`), ANTsPy backend, FireANTs syn on CPU when no GPU
- **Logging** — `quiet_external_output()` suppresses third-party print/tqdm when not verbose; sibling package loggers deduplicated for Jupyter; `fix_roi_wm` uses logger instead of `print`
- Dev/scratch scripts moved from `tests/` to `scripts/scratch/` (Docker and local test scripts, surf recon helpers; `test_brainana_local.sh` adds `ENABLE_GPU` toggle)
- `.dockerignore` expanded (excludes `examples/`, `scripts/`, `docs_temp/`, notebooks, local venvs) for slimmer image builds
- Docs CI uses docs-only install in the Sphinx workflow

### Fixed

- **GPU scheduling** — skull stripping and functional brain-mask no longer claim GPU when `use_gpu=false` / CPU mode
- SimpleITK conform output matrix aligned with FLIRT-style convention; center-of-gravity direction and resume-directory bugs
- Colab FUSE / file-existence checks in the Lite notebook
- Sphinx `-W` build: RST warnings corrected; `brainana_lite` page added; README and installation updated for Lite vs full pipeline

### Removed

- Obsolete `scripts/brainana_lite.ipynb`, `scripts/bak_*`, and `tests/test_anat_conformation.py`


## [1.0.0] - 2026-05-28

First public release of **Brainana**, a unified preprocessing framework for macaque MRI: BIDS in, anatomical and functional preprocessing, optional cortical surface reconstruction, and HTML QC reports.

- **Run via Docker** — `docker pull liuxingyu987/brainana:1.0.0` (see [Installation](https://brainana.readthedocs.io/en/stable/installation.html))
- **Nextflow pipeline** — parallel processing across subjects/sessions/runs with resume on failure
- **Anatomical** — synthesis, conform, skull strip/segmentation, bias correction, template registration, optional T2w coregistration and surface reconstruction
- **Functional** — slice timing (when metadata allow), motion correction, registration to anatomy/template, tSNR
- **QC** — per-step snapshots and a combined HTML report
- **Docs** — [Read the Docs](https://brainana.readthedocs.io/en/stable/) (usage, outputs, templates/atlases, FAQ)

Research software, beta stage — see README for license and citation.
