# Surface reconstruction: template-surface start with V1 held (developer notes)

How surface reconstruction builds its starting mesh from the NMT2Sym template instead of
tessellating the white-matter segmentation, why V1's white surface is held at the template,
and what was tried on the way. For maintainers; the user-facing summary is in
`docs/processing.rst` (section 3) and `docs/outputs.rst`.

Code: `src/fastsurfer_surfrecon/processing/template_init.py`,
`src/fastsurfer_surfrecon/stages/s12b_template_init.py` (`TemplateInit`, `frozen_label`),
`stages/s13_white_preaparc.py`, `stages/s15_surface_placement.py`, `stages/s00_long_init.py`,
`src/fastsurfer_surfrecon/config.py` (`template_*` fields, `validate_template_init`),
`src/nhp_mri_prep/steps/anatomical.py` (`surface_prior_transform`, `anat_surface_reconstruction`),
`src/nhp_mri_prep/operations/preprocessing.py` (`resolve_fix_v1_wm`),
`workflows/surfrecon_workflow.nf`, `workflows/anatomical_workflow.nf`, `modules/anatomical.nf`
(`ANAT_SURFACE_RECONSTRUCTION`). Template: `template_zoo/fastsurfer/sub-NMT2Sym/` (its README
records provenance).
Tests: `tests/surfrecon/test_template_init.py`, `tests/surfrecon/test_long_init_seed_set.py`,
`tests/test_template_surface.py`.

---

## 1. The problem

The occipital surfaces, V1 in particular, were often wrong and lost V1's characteristic
fold shape. Two things combine:

| Cause | Example | What goes wrong |
|---|---|---|
| Strong bias field / surface coil | PRIME-DE rockefeller 032116/121, sub-monkeyB, uwmadison 1224, bordeaux m03 | the white-matter segmentation at the pole is fragmented; the tessellated mesh inherits it before any placement runs |
| V1's own contrast | every subject, including normal-quality ones | V1 is thin and heavily myelinated, so its grey/white contrast on T1w is weak; the image-fitted white surface stops ~0.4 mm too far out |

Measurements that shaped the design:

- **The registration is good enough to be a prior.** The NMT2Sym white surface carried into
  each subject by the pipeline's own T1w→NMT2Sym warp sits a median 0.25–0.5 mm from the
  subject's white surface, in V1 and elsewhere alike (24 hard + 5 normal subjects).
- **Broken V1s are far from the template.** Share of subject V1 vertices > 1.5 mm from the
  warped template: normal data 0.1–1.2%; worst hard hemispheres 5–10% (monkeyB-lh 10.3,
  032116-rh 8.2, 032121-rh 8.2, 1224-rh 7.7, m03-rh 6.2), matching the QC figures.
- **Most of the damage precedes placement.** For monkeyB, 032121, m03 and 1224 the deviation
  is already there on `orig.nofix` (the tessellated mesh); placement adds more on 032116-rh
  (3% → 8%) and 1224-rh. So fixing placement alone cannot work.
- **V1 thickness is biased low by image fitting.** Median V1 thickness (per hemisphere, then
  median over hemispheres): v3.0.0 1.42 mm on the hard set, 1.50 mm on the normal set; the
  template's own V1 is 1.87 mm (rest of cortex 2.12). Macaque V1 is ~1.7–2.0 mm.

## 2. Flow

★ = added by this design. With `template_surface.enabled: false` the pipeline runs the v3.0.0
path (s08–s12 on the left) exactly as before.

```
 ANAT_REGISTRATION (earlier step)
   skull-stripped T1w ─► NMT2Sym (FireANTs SyN by default)
   forward transform from-T1w_to-<output space> ─────────────────┐
        │                                                         │
        ▼                                                         │
 ANAT_SURFACE_RECONSTRUCTION ─ anat_surface_reconstruction()       │
 ┌─────────────────────────────────────────────────────────────┐  │
 │ ★ surface_prior_transform()                                  │◄─┘
 │    name has "_to-NMT2Sym_" and is non-empty ─► reuse it      │
 │    otherwise (other output space, registration off,          │
 │    longitudinal base, missing) ─► ants_register(skull-       │
 │    stripped T1w → tpl-NMT2Sym_res-05_T1w_brain), same        │
 │    registration settings                                     │
 │ postprocess_for_freesurfer ─► orig.mgz (LIA, T1w world)       │
 │ volume stages s00–s07b (unchanged)                            │
 └─────────────────────────────────────────────────────────────┘
        │  per hemisphere
        ▼
   template_surface on?
     no ─► s08 tessellate ─► s09 smooth ─► s10 inflate ─► s11 qsphere ─► s12 topology fix
     yes (s08–s12 disable themselves)
        ▼
 ★ s12b TemplateInit
     template surf/?h.white (surface RAS) ─► scanner RAS (template world)
       ─► antsApplyTransformsToPoints (forward transform; LPS)
       ─► subject scanner RAS ─► orig.mgz surface RAS           = orig.nofix
     mris_remove_intersection ─► orig ; orientation check
     mris_smooth ─► smoothwm (= smoothwm.nofix) ; mris_inflate ─► inflated, sulc
     template surf/?h.sphere ─► sphere, qsphere, qsphere.nofix (copied: shared mesh)
     template label/?h.V1.label ─► label/?h.template.V1.label
        │
        ▼
   s13 white.preaparc   mris_place_surface --white --nsmooth 3 --max-cbv-dist 5
                        ★ --rip-label template.movable   (= all vertices − V1)
   s14 parcellation     (unchanged: subject cortex.label, subject ARM2 annot)
   s15 white            ★ --rip-label template.cortex_movable (= cortex.label − V1)
       pial             (unchanged: fitted everywhere, V1 included)
   s16 morphometry, s17 registration, s18–s22 (unchanged)
```

The subject ends up with the template's mesh (ico6, 40962 vertices per hemisphere) and
vertex numbering. V1's white surface is the warped template plus light smoothing; everything
else, and the whole pial surface, is fitted to the subject's image as before.

## 3. The stages

### 3.1 The transform (`surface_prior_transform`)

The prior is always NMT2Sym, whatever `template.output_space` is: surface reconstruction runs
in the subject's space, and only the mesh and its held label come from the template.

- **Reuse** when the anatomical registration's forward transform targets NMT2Sym (the name
  carries `_to-NMT2Sym_`; any resolution). This is the transform the study used.
- **Register** otherwise: the skull-stripped T1w (T1w × brain mask) to
  `tpl-NMT2Sym_res-05_T1w_brain`, with `ants_register` and the run's `registration` settings
  (`anat2template_xfm_type`, `enable_fireants`, `fireants_allow_cpu`). This covers other output
  spaces, `registration.enabled: false` (the passthrough writes `from-T1w_to-T1w`), and the
  longitudinal base, which is built by `reconstruct_base()` without a transform. On CPU it
  took about a minute (032116) and placed V1 within a median 0.10–0.17 mm of the reused one.
- **Rejected:** reusing the segmentation template-prior registration
  (`segmentation_prior.register_template_prior`), which is computed for every subject anyway.
  It registers the head, on the unstripped image, earlier in the pipeline; its accuracy at the
  occipital pole was never measured, and the whole approach depends on it.
- An affine-only transform (`.mat`) is reused but logged as a warning: V1 is placed less
  accurately without SyN.

Nextflow passes the transform as an optional input (`anat_reg_transforms` → `[sub, ses, fwd]`,
joined with `remainder: true`, empty placeholder `dummy_template_xfm.dummy` when absent;
transform-only remainder rows are filtered out).

### 3.2 Coordinates (`template_init.py`)

FreeSurfer surfaces store surface RAS (tkr) of the volume they were made from. **Surface RAS
is always framed LIA, whatever the volume's orientation**, so the conversion goes through
voxels: `scanner = vox2ras · inv(vox2ras_tkr) · surface`, both matrices rebuilt from the
volume-geometry block each surface carries (`volume`, `voxelsize`, `x/y/zras`, `cras`). Only
for LIA volumes — conformed `orig.mgz` and the template's own volume — does this reduce to
`scanner = surface + c_ras`. The first implementation used that shortcut; the oblique case in
`test_surface_scanner_conversion_matches_freesurfer` caught it.

Points go template → subject through the subject-to-template *image* transform: an ANTs image
transform maps fixed-space (template) points to moving-space (subject) points.
`antsApplyTransformsToPoints` reads `.nii.gz` fields, `.h5` composites and `.mat` affines
alike, in LPS. The subject's `orig.mgz` is conformed from the same T1w and keeps its world
space, which is what makes the T1w transform valid for it.

Check against the study (which sampled the displacement field in numpy): 032116 and 1224,
both hemispheres, max difference 0.0001 mm. The volume-geometry block written by
`volume_info_from_mgz` matches a pipeline-written surface field for field.

### 3.3 s12b outputs

`orig`, `orig.nofix`, `smoothwm`, `smoothwm.nofix`, `inflated` (+ `sulc`), `sphere`, `qsphere`,
`qsphere.nofix`, `label/?h.template.V1.label`, `scripts/?h.template_init.json`.

- `mris_remove_intersection` because a smooth warp keeps the topology but can push opposite
  banks of a tight sulcus through each other. The study logs show 16–57 intersections found
  per hemisphere; 5/48 hemispheres kept 3–10 afterwards and placement ran normally on all.
- The `*.nofix` files and the copied sphere are what s00 seeds a longitudinal timepoint from
  and what the s08–s11 skip rules look for; with them a template-initialised base seeds
  exactly like a tessellated one.
- `verify_outputs` asserts `orig` is closed, oriented, Euler 2 (strict only with
  `processing.strict_surface_checks`).

### 3.4 Holding V1 (s13, s15)

`mris_place_surface --rip-label L` holds every vertex **outside** `L`. So the label passed is
the movable set: all vertices minus V1 in s13, `cortex.label` minus V1 in s15. The pial pass is
unchanged. `frozen_label()` applies the hold only when the mesh is the template's
(`template_init` or `longitudinal`) and the label file exists, so a stale label in a reused
tree cannot freeze a tessellated mesh whose numbering it does not match.

**The hold is not absolute.** s13 runs with `--nsmooth 3`, and `mris_place_surface` smooths the
whole surface *before* ripping (log: "Smoothing surface before ripping"). V1 therefore moves by
smoothing only — at most ~1.2 mm — and never by the image.
This is exactly the arm that won the comparisons (measured on CN: 31–52% of V1 vertices per
subject moved > 0.1 mm); kept as is.

V1 label: CHARM level 2 "V1" ∩ the template's `cortex.label` — 4789 (lh) / 4714 (rh) vertices.

Differences from the study arm (CN): the study used the template's `cortex.label` and ARM2
annotation for placement; the pipeline uses the subject's own (built in s14), the natural
choice. V1 is unaffected (identical, §6.3); elsewhere the white surface moves a median
0.12–0.18 mm, below the 0.6 mm (p90) that placement varies between otherwise identical runs.

### 3.5 fix_V1_WM: "auto"

The CNN-stage V1 white-matter fill (`fix_roi_wm`) pastes the template's V1 white matter into
the segmentation with no intensity check. Audit on the hard set: 1–8.7k voxels added per
subject (63k on the 0.27 mm 1224); in 14/24 subjects their T1w intensity was grey-matter-like
(median index ≤ 0.2 on a local GM = 0 / WM = 1 scale), though mostly at plausible depth
(2.3–3.5 mm below the brain edge; too shallow in 032197 55%, 032194 41%, 032300 28%,
m03 22%, 032190 16%). On 032116/032117 the tessellated white surface reached the pole tip
through these voxels.

Blind result: with tessellation the fill helps (AN — tessellation without it — was never
picked over A); with the template surface it barely matters (all noted ties were C vs CN,
CN picked 24 vs C 10). `"auto"` = on only for anatomy whose surfaces are not template-built
(surface reconstruction off, or `template_surface` off). Note it changes the segmentation
itself, not only the surfaces.

### 3.6 Longitudinal

The base is reconstructed with the template (option-2 registration, since `reconstruct_base()`
passes no transform). s00 seeds each timepoint from the base's surfaces and now also copies
`label/?h.template.V1.label` when present, so timepoints hold the same vertices. Timepoints
never warp anything, so `validate_template_init` does not require a transform for them.

## 4. Provenance

| File | Meaning |
|---|---|
| `scripts/?h.template_init.json` | template dir, transform path, held label, `warped_fraction_in_mask`, `registration_suspect` |
| `label/?h.template.V1.label` | held vertices (template indices) |
| `label/?h.template.movable.label`, `label/?h.template.cortex_movable.label` | the rip labels s13 / s15 passed |
| step `metadata.json` | `template_surface` (bool), `template_surface_transform` (`reused` / `registered`) |
| `surf/?h.orig.noorient` | only if the orientation check had to flip `orig` |

Work files of a registered transform: `work/surface_prior_registration/`.

## 5. Configuration

```yaml
anat:
  skullstripping_segmentation:
    fastSurferCNN:
      fix_V1_WM: "auto"        # true | false | "auto" (on only when template surfaces are off)
  surface_reconstruction:
    template_surface:
      enabled: true            # false = v3.0.0 tessellation
```

Library (`fastsurfer_surfrecon/config/default.yaml`, set by the step, not by users):
`template_init` (false in the library; the step turns it on), `template_subject_dir`
(`template_zoo/fastsurfer/sub-NMT2Sym`), `template_xfm`, `template_freeze_label` (`V1`; `null`
fits the whole white surface — the "B" arm).

Constants, deliberately not exposed: the template (`SURFACE_PRIOR_TEMPLATE = "NMT2Sym"`,
`SURFACE_PRIOR_TEMPLATE_SPEC = "NMT2Sym:res-05"` in `steps/anatomical.py`), s13's
`--nsmooth 3 --max-cbv-dist 5` (unchanged from v3.0.0).

Cost: s12b replaces s08–s12 (no tessellation, sphere or topology fix to compute); not timed
against them in isolation. The whole surface task took 8 min (032116) and 18 min (032125)
with three tasks sharing 4 CPUs; the v3.0.0 surface tasks of the hard run took 9–12 min each.
Option-2 registration added ~1 min on CPU.

## 6. Validation

All comparisons were blind: per subject the versions were shuffled and labelled Version 1..n,
picked per hemisphere, arms revealed afterwards; figures were an occipital close-up (white
surface + curvature, posterior/ventral/lateral), occipital axial and sagittal slices with white
and pial outlines, and the pipeline's own two QC figures. One rater (the pipeline author).
Both hemispheres of a subject always got the same pick.

### 6.1 Rounds (hard set, `dataset_surf` hard half, 24 subjects then)

| Round | Arms | Picks (hemispheres) |
|---|---|---|
| 1 | A v3.0.0 · B template start, all fitted · C template start, V1 held | C 30, B 10, A 8 |
| 2 | A · AN (A, fix off) · C · CN (C, fix off) | CN 24, C 10, A 14, AN 0 |
| 3 | CN · BN · SN1/SN2 soft V1 prior (032116, 032121) | CN 4/4 |
| 4 | CN · RN/RWN local occipital re-registration (032116, 032121, 1224) | CN 4, tie 2 |

The template-V1 family won 17/24 subjects in round 2 and every strong-bias subject in every
round. A won 032155, 032165, 032172, 032194, 032197, 032206, 032212 in round 2.

### 6.2 Normal-quality data (`dataset_surf` normal half)

24 PRIME-DE subjects without a `_badQC` tag, 13 sites (seeded random pick per site; list in
`dataset_surf/normal_subjects.tsv`). v3.0.0 run CPU-only (`dataset_surf_normal_v3.0.0_cpu`),
CN built from it.

| | A (v3.0.0) better | tie | CN better |
|---|---|---|---|
| subjects | 3 (032288 ecnuChen, 032119 rockefeller, 032308 sbri) | 7 | 14 |

No site pattern in A's three wins (each of those sites also has a CN win or tie). 032177 and
032288 were marked "both bad" — an upstream problem, not this design.

V1 median thickness (median over hemispheres):

| set | A | AN | C | CN | template |
|---|---|---|---|---|---|
| hard (48 hemis) | 1.42 | 1.57 | 1.77 | 1.77 | 1.87 |
| normal (48 hemis) | 1.50 | — | — | 1.81 | 1.87 |

CN's V1 thickness is partly by construction — its white surface is the template's.

### 6.3 Implementation vs the study arm

The pipeline step, on each subject's own surface-task inputs with the pre-fix segmentation
(`preproc/tplinit_impl_check/`, `scripts/impl_check.py`, `impl_compare.py`):

| case | V1 white vs CN | rest of white, median / p90 | V1 pial median | V1 thickness new / CN |
|---|---|---|---|---|
| 032116 (hard), reused transform | 0.000 mm | 0.15–0.18 / 0.45–0.54 | 0.11–0.12 | 1.75/1.78, 1.71/1.72 |
| 032125 (normal), reused transform | 0.000 mm | 0.12–0.14 / 0.35–0.39 | 0.09–0.18 | 1.61/1.73, 1.65/1.69 |
| 032116, registered (option 2) | 0.10–0.17 (max 0.8) | 0.25–0.27 / 0.67–0.69 | 0.22–0.29 | 1.76/1.78, 1.71/1.72 |

Study material: `preproc/tplinit_study/` (hard set: arms `A B C AN BN CN SN1 SN2 RN RWN`,
blind pages `index.html`, `index_nofix.html`, `index_soft.html`, `index_reg.html`, picks
`tplinit_study_*picks.json`, `approaches_summary.html`), `preproc/tplinit_normal/` (normal set,
`index_normal.html`, picks). Scripts: `preproc/tplinit_study/scripts/` (`study.py` builds the
template arms from a run, `run_nofix.py` re-runs a run's surface task on the pre-fix
segmentation, `soft.py`, `reg_local.py`, `render_study.py` + `build_page.py` +
`page_template.html` make the blind pages, `e2e.js` checks every revealed arm name against the
image it labels).

## 7. Known limits

- **Seam at the V1 border.** Where this animal's V1 white matter differs in shape from the
  template's — typically a finger reaching into the pole tip — the held template sheet and the
  image-fitted surface around it meet across grey matter (a false white-matter connection at
  axial z ≈ −7/−8 in 032116, 032118, 032121; part of the sheet in grey matter on 1224).
  Neither a soft hold nor a better registration fixes it (§8): a smooth warp cannot create a
  fold the template does not have. Not yet tried: holding a larger region (V1+V2, or the pole)
  so the seam falls where the image contrast is better.
- **Individual V1 variation is suppressed.** V1's white surface is the template's, so V1
  thickness and shape vary across subjects mainly through the pial surface and the
  registration. Analyses of V1 thickness differences between animals should know this.
- **Mesh resolution.** ico6, 40962 vertices per hemisphere; tessellated meshes had 51–66k at
  0.5 mm. Enough for the placement, coarser for high-resolution (e.g. 0.27 mm) data.
- **Registration quality is inherited.** A failed or affine-only registration misplaces V1
  directly. s12b measures the share of the warped template's vertices inside `mask.mgz`
  (`fraction_in_mask`; a fitted white surface is at 1.0 on every devtest reconstruction) and
  below `MIN_FRACTION_IN_MASK = 0.95` logs a warning and records `registration_suspect` in
  `scripts/?h.template_init.json`. It does not fall back to tessellation; whether it should,
  and at what threshold, needs the share measured on known misregistrations first.
- **Surface registration (s17) with the shipped template** would need
  `atlas/?h.folding.atlas.tif`, which is not shipped. s17 is off by default; before this
  change `registration_template` pointed to a directory that did not exist at all.
- **Not yet run end-to-end through Nextflow**: the step was validated by calling it directly on
  real task inputs (§6.3). The Nextflow wiring is lint-clean (no errors beyond the 93
  pre-existing strict-syntax ones) but untested in a run.

## 8. What was tried and removed

- **Template start, everything fitted (B/BN).** Fixes the tessellation-stage outliers
  (round-1 arm B, monkeyB-lh: 10.3% → 1.7% of V1 > 1.5 mm from the template) but not V1: thickness stays at
  v3.0.0's level and the fold shape still follows a poor image. Lost to C in round 1 (10 vs
  30). Available as `template_freeze_label: null` in the library.
- **Capping V1 travel with `--max-cbv-dist 1`.** It limits the per-iteration intensity search,
  not total travel: V1 moved up to 2.6 mm. Dropped before any comparison.
- **Soft V1 prior (SN1, SN2).** Free fit, then each V1 vertex pulled back to ≤ 1 or 2 mm of
  the template (tapering to 4 mm over 2 mm at the V1 border), then a seam pass. The free fit
  already sits within ~1 mm, so SN1 clamped 2–7% of V1 and SN2 almost nothing; V1 thickness
  stayed at BN's 1.27–1.39 mm (CN 1.70–1.83): the image-fitted white surface is systematically
  ~0.4 mm out, inside any cap. Lost 0/4. The author's verdict: on these images a soft prior
  cannot help.
- **Local occipital re-registration (RN intensity, RWN white-matter masks).** ANTs SyN on a
  box around V1 before warping; RWN raised the white-matter agreement at the pole (0.94–0.98
  → 0.98–1.00) but did not change the seam. Lost (CN 4, tie 2, RN/RWN 0).
- **Tessellation without the V1 fill (AN).** Never picked.
