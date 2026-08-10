# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **BIDS input layout is validated before processing starts** — discovery now cross-checks every input file's `sub-`/`ses-` entities against the directories it sits in, and aborts the run with a report naming each affected file if they disagree. This closes a silent-corruption path: pybids resolves such a conflict in favour of the *directory* and discards the filename entity, so nothing downstream ever saw a conflict — but brainana names output directories from the (directory-derived) subject ID and output *filenames* from the original filename stem. A dataset with `sub-monkey1/anat/sub-monkey_ses-1_run-1_T1w.nii.gz` therefore processed to completion without a single warning and published `sub-monkey1/` filled with `sub-monkey_*` files: not valid BIDS derivatives, and broken for any tool keying on subject ID. Errors are `BIDS101` (subject label mismatch), `BIDS102` (no `sub-` entity), `BIDS103` (session label mismatch) and `BIDS104` (no `ses-` entity where the subject has several sessions). Validation is scoped to the subjects and sessions the run will actually process — whether you set them with `--subjects`/`--sessions` or in `bids_filtering` — so one malformed subject or session cannot block data you did not ask to process
- **Subjects that mix session and data directories are rejected** — a subject holding both `sub-01/ses-001/` and `sub-01/anat/` is not valid BIDS (a subject directory contains *either* session directories *or* datatype directories) and discovery, which reads one session at a time, never looked inside the subject-level `anat/`. Those files were dropped from the run without appearing anywhere in the logs, so the pipeline reported success having quietly ignored part of your input. This is now a `BIDS105` error naming every orphaned file
- **Files that are silently skipped are now reported** — a NIfTI under a subject directory that discovery would not select (outside `anat/`/`func/`, or carrying a suffix brainana does not read) is reported as a `BIDS202` warning saying it will not be processed, instead of vanishing without comment. A filename carrying a `ses-` entity with no matching `ses-*/` directory level is `BIDS201`, a filename missing a `ses-` entity where only one session is being processed is `BIDS204`, and a missing `dataset_description.json` is `BIDS203`. All are warnings and never stop a run

### Changed

- **Behaviour change — datasets that previously processed may now abort.** There is deliberately no flag to skip validation. An error fires in exactly two situations: the input directory and the input filename disagree about which subject or session a file belongs to (`BIDS101`–`BIDS104`), or a subject mixes session directories with subject-level data directories (`BIDS105`). Both mean the previous run was already wrong — the first published mislabelled derivatives, the second silently processed only part of your input — so output from before this release should not be trusted in either case. Errors are raised only for files brainana actually reads, so extra volumes kept alongside your anatomicals (CT, histology, PET) are reported as warnings and do not block a run. Every error names the affected files and the rename or move that resolves it


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

- **Brainana Viewer companion docs** — a new [Brainana Viewer](https://brainana.readthedocs.io/en/stable/viewer.html) page introduces the cross-platform NiiVue desktop viewer for exploring per-subject pipeline output (anatomical volumes, cortical surfaces, atlas overlays, and functional maps), with a link to the [viewer's repository](https://github.com/arcaro-lab/brainana_tools)
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
