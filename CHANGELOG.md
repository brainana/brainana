# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

- **A declared output-naming contract, and a validator for it.** `src/nhp_mri_prep/utils/bids.py` declares `ENTITY_ROLES` — every entity is *identity* (inherited from the raw data; brainana copies and never sets it), *frame*, *variant* or *product* — and `derive_output_name()` is the one constructor, raising rather than resolving the two ways a name loses information: overwriting an inherited identity entity, or emitting a key twice. `validate_output_name()` checks a published name; `scripts/check_output_naming.py` checks a whole output tree and exits non-zero on anything undeclared. See `docs/output_naming.rst`
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
