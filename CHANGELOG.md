# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **DataLad / git-annex datasets**: discovery found no subjects, because every NIfTI is a symlink into `.git/annex`. Symlinked files are now found where they sit in the dataset.
- **Scans that include the neck or body**: the anatomical conform could align the brain upside down or tilted, and every later step failed while the run still reported success. Conform now registers only the region around the brain found by its skull strip; outputs keep the full field of view. Cropping scans by hand is no longer needed.
- **Conform could align background noise instead of the brain**: when the skull-strip model also marked a brain-sized patch of background, conform could keep that patch, and every later step then ran on noise. The model's most confident region is now kept, not the largest; the same applies to the functional brain mask.
- **A skull strip that finds no brain now stops that subject**: a brain mask under 10% of the template brain fails `ANAT_SKULLSTRIPPING` for that subject only, and the rest of the run continues. Previously later steps ran on an empty mask and the first error came from surface reconstruction. Check the subject's conform QC figure first. The QC report now lists failed tasks by name.
- **`desc-conformFullFOV` kept only the standard field of view** for scans that include the neck or body, and for high-resolution scans (about 0.3 mm or finer). It now always holds the whole input, up to 512 voxels per axis; a longer axis keeps its central 512 voxels. The sidecar's `FullFOVPadding.status` reads `clipped` when that happens.
- **Atlas surface maps had unlabelled cortex** (`anat/atlas_space-fsnative/*.func.gii`): up to about 9% of cortex vertices had no label, and labels spilled past the cortex. Vertices are now sampled across cortical depth, maps are limited to the cortex, and unlabelled cortex in label atlases takes its neighbours' label. Each map moves by about 1–2% of vertices at area borders. Continuous maps (retinotopy, somatotopy, cortical hierarchy) are not filled.

## [3.1.0] - 2026-09-30

### Upgrading from 3.0.0

A 3.0.0 config runs unchanged; no config key or output file was renamed or removed. Default anatomical and surface outputs change, so do not mix 3.0.0 and 3.1.0 results in one analysis.

- Surfaces start from the NMT2Sym template (shared mesh and vertex numbering, V1 held at the template). `anat.surface_reconstruction.template_surface.enabled: false` restores the 3.0.0 tessellation.
- `fix_V1_WM: "auto"` fills V1 white matter only for tessellated surfaces; set `true` to keep it.
- GPU steps now run on the GPU. When GPUs are visible, a `general.gpu_device` index that does not exist stops the run at start-up.
- A config without `anat.synthesis_level` gets `subject`, as documented (3.0.0 used `session`).
- The work directory must support file locks (GPU slots under `<work_dir>/.gpu_slots`).

### Added

- **Template-initialised surfaces** (`anat.surface_reconstruction.template_surface.enabled`, default on): the NMT2Sym white surface is warped into the subject and fitted to the image, except V1, which stays at the template. Records: `label/?h.template.V1.label`, `scripts/?h.template_init.json`. Falls back to tessellation when the warped surface does not fit the brain mask. Longitudinal timepoints inherit the base's mesh and V1 label.
- **Second segmentation pass when the first one misses brain** (`fastSurferCNN.pre_inference_n4`, `.template_prior`): the NMT2Sym template is registered to the subject; when it shows brain outside the mask, the image is N4-corrected and segmented again, and the second result is kept only if it agrees with the template at least as well. If any of this fails, the first pass is kept.
- **Segmentation provenance in the mask and segmentation sidecars**: mask volume and ratio to the template brain (`MaskUndersized`), intensity cap, segmentation passes, template agreement (`TemplatePrior`).
- **Optional relabelling of small label fragments** (`fastSurferCNN.label_island_min_volume_mm3`, default off).
- **`all_subjects_report.html`**: every subject's QC on one page, by subject or by step.
- **`scripts/surface_qc.json` records cortical thickness** (flagged `collapsed` when implausibly thin) and, for tessellated surfaces, the topology-fix path.

### Changed

- **`fix_V1_WM` defaults to `"auto"`** (on only for tessellated surfaces) and runs once, on the segmentation that is kept.
- **Intensity rescaling before segmentation is capped relative to brain intensity.**
- **Tessellated surfaces use FreeSurfer's quasi-homeomorphic sphere** (`mris_sphere -q`) for the topology fix.
- **Anatomicals labelled `part-phase`, `part-real` or `part-imag` are skipped.**

### Fixed

- **Output spaces other than NMT2Sym** (MEBRAINS, D99, Yerkes19, custom templates, conform off): the template prior, the second pass, `fix_V1_WM` and the template surface misregistered. The image is now placed in NMT2Sym coordinates first (bundled rigid transforms in `template_zoo/template/xfm/`, or a rigid registration for custom templates and conform off), recorded as `Nmt2SymFrame`. `fix_V1_WM` was affected in 3.0.0 too.
- **The undersized-mask rule always compares with the NMT2Sym brain.**
- **`T1w` is no longer offered as an output space**; it could not run.
- **GPU use**: GPU steps ran on the CPU; every task now starts with no visible GPU and only GPU steps take one; CPU tasks no longer initialise CUDA (CPU mode aborted on Windows/WSL2); `general.gpu_device: N` means physical GPU N; one device policy for every step; a CUDA out-of-memory error reruns the step on the CPU; `--use_gpu` is ignored with a warning.
- **`-resume`** no longer re-runs steps whose GPU, placeholder files or input order changed.
- **Resources and diagnostics**: the segmentation network uses the task's CPU allocation; `NXF_MAX_CPUS`/`NXF_MAX_MEMORY` are clamped to the container; an unlockable work directory fails with a message; per-task containers get the GPU and thread settings; native crashes leave a Python stack trace.
- **A partial `--config` could switch anatomical synthesis to per-session** (2.1.0 and 3.0.0).
- **`--anat_only` on the command line reaches BIDS discovery.**
- **A single numeric value for `--subjects`, `--sessions`, `--tasks` or `--runs` no longer aborts the run.** Lists are comma-separated.
- **Surface reconstruction normalised intensities on cortex instead of white matter.**
- **Topology fix (tessellated surfaces)**: a failed `mris_fix_topology -ga` search is retried with the default search, also on `-resume`; an inside-out premesh is flipped; unknown `mris_place_surface` options are an error.
- The report generator's fallback config is read over the package defaults.

## [3.0.0] - 2026-09-19

### Upgrading from 2.1.0

Output filenames changed and input validation is stricter; re-run rather than mixing 2.1.0 and 3.0.0 output in one derivatives tree.

- **Runs may abort on a malformed BIDS layout** (always on): a directory and a filename that disagree about subject or session (`BIDS101`–`BIDS104`), or a subject mixing session and datatype directories (`BIDS105`). Each error names the files and the fix.
- **Renamed outputs**:

| 2.1.0 | 3.0.0 |
| --- | --- |
| `…_space-<tpl>_desc-preproc_desc-func2target_bold.png` | `…_space-<tpl>_desc-func2target_bold.png` |
| `sub-01_ses-001ref` (from a `_boldref` input) | `sub-01_ses-001` |
| `…_desc-a_desc-b_…` | `…_desc-b_…` |
| `…_space-scanner_space-T1w_desc-preproc_…` | `…_space-T1w_desc-preproc_…` |
| custom entities emitted alphabetically | emitted in source order |

- **Removed** two Brainana Lite overrides that were already no-ops: `registration.fireants_allow_cpu` and `general.anat_only`.

### Added

- **`anat.synthesis_level: "session_longitudinal"`**: within-subject longitudinal surfaces. Each subject gets a base template and a base-seeded reconstruction per session, with per-vertex and per-ROI rates of change. Needs surface reconstruction and two or more anatomical sessions; settings under `anat.surface_reconstruction.longitudinal`.
- **Output-naming contract** and validator (`scripts/check_output_naming.py`).
- **Uncropped conformed anatomical (`desc-conformFullFOV`)**, with a paired QC figure.
- **BIDS layout validation** before processing (errors `BIDS101`–`BIDS105`, warnings `BIDS201`–`BIDS204`).
- **`func.confounds.fd_radius_mm`**: configurable FD head radius.
- **Confounds and motion QC figures mark flagged frames** and thresholds.
- **Brainana Lite publishes** the uncropped conform, the hemisphere mask, ingest provenance, a `dataset_description.json` and its effective config.

### Changed

- **MacBNA lookup table gains an `ID_nohemi` column** (second column); consumers reading by column position must update.
- **Brainana Lite** fetches atlases at the resolution the runtime picks, publishes QC figures at `sub-XX/figures`, and validates config overrides.

### Fixed

- The confounds sidecar reports the FD radius actually used; `func.confounds` thresholds are validated.
- Motion QC recorded the wrong figure path.
- ARM6 area 32 had the wrong hemisphere label; the ARM4 lookup table now matches the other ARM tables.
- Brainana Lite: several hangs, unneeded re-clones and reruns failing, and one failed QC figure aborting a complete run.

## [2.1.0] - 2026-07-26

### Added

- **"Data findings" in the QC report**: what ingest repaired and what it could not.
- **`scripts/surface_qc.json`**: the measured topology of each key surface.

### Fixed

#### Anatomical ingest

- **4D anatomicals** no longer crash conform; they are made 3D at ingest (`Input4DCollapsed`).
- **Anatomicals with no stored orientation** get the FSL convention written into the header (`OrientationRecovered`). **Caveat:** this is a convention, not ground truth; confirm left/right handedness before trusting hemisphere-wise results.
- **Disagreeing qform and sform** are reconciled to the sform (`QformSformReconciled`).
- Uncompressed `.nii` anatomicals are compressed at ingest.
- Header defects with no safe repair are reported as `InputHeaderWarnings`.
- The scanner-space anatomical's JSON sidecar was never published.

#### Surface reconstruction

- **Topology repair had silently stopped working**, so some runs completed with unrepaired surfaces. Re-run surface reconstruction for subjects processed without `pyvista` installed; run status does not identify them.
- **Inside-out surfaces** are detected and flipped.
- **A defective mesh fails at the stage that made it** instead of producing surfaces built on it.
- Stages verify their outputs before reporting success; resuming into a half-finished stage reruns it; a failure in one hemisphere keeps the other's work.

#### Runtime

- Local runs pin Nextflow 25.10.2 (`NXF_VER`).

### Changed

- CI checks dependencies with a real install; `psutil` is declared in the `train` extra; development scripts moved to `scripts/dev/`.
- The BIDS discovery summary separates input files from processing jobs; QC report titles include the session.

### Removed

- The unreachable `ANAT_REORIENT`/`FUNC_REORIENT` processes and their AFNI helpers (update direct imports), and the unused `read_yaml_config.py`.

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
