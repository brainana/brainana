# Anatomical CNN segmentation: two-pass design (developer notes)

How the anatomical brain segmentation decides whether to run the network twice,
and why each piece is the way it is. For maintainers; the user-facing summary is
two paragraphs in `docs/processing.rst` (section 2.3).

Code: `src/nhp_mri_prep/operations/preprocessing.py` (`apply_segmentation`,
`_anat_segmentation_checks`, `_pre_inference_n4`),
`src/nhp_mri_prep/operations/segmentation_prior.py`,
`src/fastsurfer_nn/inference/segmentation.py` (`run_segmentation`, `apply_roi_wm_fix`).
Tests: `tests/test_seg_robustness.py`.

---

## 1. The problem

The segmentation network returns a confident but incomplete brain when part of the
brain is darker than it expects. Field reports and PRIME-DE show three versions of it:

| Cause | Example | What the CNN does |
|---|---|---|
| Surface receive coil | sub-monkeyB (UC Berkeley) | drops the medial occipital pole, far from the coil; mask still 98% of template volume |
| Strong bias field | PRIME-DE rockefeller 032116/117/121 | drops whole dark lobes (032121: 57 cm³ of a 115 cm³ brain) |
| MP2RAGE-like (UNI) contrast | PRIME-DE uwo 032190/194/197 | ragged mask, misses the frontal pole |

Two facts shape the design:

- **The misses are confident.** On monkeyB, counting every voxel with background
  probability < 0.99 as brain added 0.7 cm³ of the missing pole. A lower threshold,
  other view weights or test-time tricks cannot recover it; the input has to change.
- **Volume is a weak failure signal.** monkeyB's failed mask is 98% of the template
  brain; healthy small animals (bordeaux m03, sbri 032300) are 63–68%. An 80% volume
  rule both misses real failures and fires on healthy brains.

## 2. Flow

★ = added by this design (v3.0.0 ran pass 1 and the V1 fix only).

```
 ANAT_CONFORM (earlier step)
   raw T1w ─► rigidly aligned to NMT2Sym, resampled onto the template grid (+6 mm pad)
   = "conformed T1w" ── the only image this step reads
        │
        ▼
 ANAT_SKULLSTRIPPING ─ apply_segmentation()
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ PASS 1 ─ run_segmentation(conformed T1w)                                  │
 │  1. network space: reorient to LIA, resample to the checkpoint voxel size│
 │  2. rescale 0–255: top = brightest 0.1%,                                 │
 │     ★ capped at 2.5 × centre-box p99                                     │
 │  3. three 2-D networks (coronal/axial/sagittal, weights 0.4/0.4/0.2),    │
 │     scores summed                                                        │
 │  4. highest score ─► ARM2 labels; stray white-matter islands fixed       │
 │  5. labels back to the conformed grid                                    │
 │  6. brain mask = labels ≠ 0, 3 rounds of                                 │
 │       largest piece → fill holes → dilate 2 mm / erode                   │
 │  7. labels outside the mask → 0; hemisphere mask                         │
 │  ★ (no V1 fix here for anatomy; see step "V1 fix")                       │
 └──────────────────────────────────────────────────────────────────────────┘
        │
        ▼
 ★ record: intensity cap applied? mask volume / template brain (80% rule)
        │
   template_prior off (or no atlas)  AND  not forced  AND  not undersized?
        │ yes ─────────────────────────────────────────────────────┐
        │ no                                                       │
        ▼                                                          │
 ★ N4 region = [template brain placed by conform alone, + 15 mm]   │
               ∩ [head mask: > 10% of brain brightness, filled]    │
 ★ N4 inside that region (shrink 4, 60 mm spline) ─► "N4 image"    │
     used only as CNN and registration input                       │
        │                                                          │
        ▼                                                          │
 ★ Template prior (measures only, never edits):                    │
     register NMT2Sym head T1w onto the N4 image                   │
     (affine + SyN); warp the NMT2Sym brain mask and ARM2 atlas    │
     missed = brain-bright (p2–p99.5), atlas-labelled voxels       │
              > 1 mm inside the template brain, outside the CNN    │
              mask, ≥ 1 mm thick, connected to it                  │
     Dice   = CNN mask vs template brain                           │
     (registration fails ─► prior = none, logged)                  │
        │                                                          │
        ▼                                                          │
 ★ Second pass?                                                    │
     forced (pre_inference_n4: true)                               │
     OR missed > 0.5 cm³                                           │
     OR (mask < 80% of template AND missed > 0.25 cm³)             │
     OR (no prior AND mask < 80% of template)                      │
        │ no ──────────────────────────────────────────────────────┤
        │ yes                                                      │
        ▼                                                          │
   PASS 2 ─ steps 1–7 on the N4 image (pass-1 files → pass1/)      │
        │                                                          │
 ★ keep pass 2 only if Dice(pass 2) ≥ Dice(pass 1) − 0.005          │
     otherwise: pass 2 → pass2_rejected/, pass 1 restored           │
     (also when pass 2 raises; the forced mode uses the same rule)  │
        │◄─────────────────────────────────────────────────────────┘
        ▼
 ★ V1 white-matter fix, once, on the kept pass
     (crop V1, register the template's V1, add its thin white matter;
      uses the image that pass segmented)
        │
        ▼
   sidecars ─► work/ (mask, labels, hemisphere mask, lookup table)
   conformed T1w × mask ─► skull-stripped T1w
        │
        ▼
 DOWNSTREAM (unchanged)
   ANAT_BIAS_CORRECTION  N4 on the *uncorrected* conformed T1w, final mask
   ANAT_REGISTRATION → QC → atlas backprojection → surface reconstruction
```

The output is always one CNN pass, whole. The template never adds or removes voxels;
it only decides whether a second pass runs and whether it is kept. Nothing after this
step sees the N4 image.

## 3. The stages

### 3.1 Intensity cap (both passes)

The 0–255 rescale mapped the brightest 0.1% of all voxels to 255. Beside a surface coil
those are muscle and fat, several times brighter than brain, and the brain was squeezed
into a few grey levels (monkeyB's brain median sat at ~60/255; normal data at 125–170).
The top is capped at 2.5 × the 99th percentile of the central half-box, which after
conform is ~90% brain. On 98 PRIME-DE images the cap binds on 1. See the comment block
on `CENTER_CAP` in `fastsurfer_nn/data_loader/conform.py`.

### 3.2 N4 region

N4 is fitted inside **the template brain placed by world coordinates (after 3.7), widened
15 mm, intersected with the head mask** (`segmentation_prior.search_region`).

- *Not the first-pass mask (the original design, +6 mm):* a lobe the first pass dropped
  is exactly what needs correcting, and outside the mask N4 only extrapolates.
- *Not the head mask alone:* in MP2RAGE UNI images the background is brighter than brain
  (uwo 032194: brain median 1603, air 2029), so every threshold keeps the whole field of
  view, and the fit and the registration are pulled into noise. An Otsu threshold was
  worse (it locked onto fat: 1.8% of the image on 032197).
- *Not the placed template alone:* at +10–30 mm it reaches past the thin occipital scalp
  into air. Fitting the field on near-zero voxels darkens the occipital pole and the CNN
  drops it: monkeyB/032121 lost 0.5–4 cm³ of occipital volume against the head mask, and
  widening made it worse.
- *Why placing the template without registration is enough:* conform is rigid, so the
  placed template brain is off by the subject's scale and shape (Dice 0.80–0.95 with the
  final brain), but as a container it holds 99.7–100% of every brain at 6 mm and 100% at
  10 mm, across 65–136 cm³ brains. It is never used as an outline.
- The intersection covers ≥ 99.66% of every final brain. The missing voxels are dark
  surface voxels open to the outside, which hole filling cannot reach; widening the head
  mask by 2 mm got only to 99.89% and added ~33 cm³ of air back. The fitted field is
  smooth and applied to the whole image, so they are still corrected.

Spline distance 60 mm (ANTs `-b [60]`, a distance in mm). The pipeline's 150 mm is one
span across a ~70 mm head. Pass-2 volume by spline on the three test subjects:

| spline | monkeyB | 032117 | 032121 |
|---|---|---|---|
| 150 mm | 93.8 | 80.1 | 109.3 |
| 60 mm | **95.7** | **81.8** | **114.6** |
| 30 mm | 95.2 | 81.0 | 103.9 |

A brain-referenced clip of the N4 image (clip at the brain's p99 before the CNN rescale)
was tried and dropped: after head-wide N4 it shrank masks slightly on every subject.

### 3.3 Template prior (`segmentation_prior.py`)

The bundled NMT2Sym head T1w is registered onto the N4 image (`ants_register`, SyN;
FireANTs on GPU, ~30 s), and the NMT2Sym brain mask (linear) and the ARM2 atlas (nearest
neighbour) are warped into the subject grid. Always NMT2Sym + ARM2, whatever
`template.output_space`: that is what the CNN was trained on, and the CNN's labels are
ARM2 IDs exactly.

`missed_brain()` counts voxels that are all of:

| condition | why |
|---|---|
| > 1 mm inside the warped template brain | a registration error at the brain edge is not a miss |
| labelled by the warped ARM2 atlas | the template brain mask also covers pituitary and olfactory tissue; the atlas does not |
| outside the CNN mask | |
| within p2–p99.5 of the CNN brain's intensity (N4 image) | CSF and bright non-brain are not brain |
| survives a 1 mm opening | removes the thin ventral slivers a registration error leaves |
| connected to the CNN mask | |

It also reports Dice(CNN mask, template brain) and `Reliable` (Dice ≥ 0.92). Correct
segmentations register at 0.94–0.98.

Measured pass-1 `MissedCm3` on the 28 test subjects: clear failures 0.87–40; 032117
(a mild failure, +5.5 cm³ from pass 2) 0.46; every correct segmentation ≤ 0.37.

### 3.4 Second-pass decision

| Rule | Catches | Evidence |
|---|---|---|
| `pre_inference_n4.enabled: true` | forced | |
| missed > `min_missed_cm3` (0.5) | failures whatever their volume | monkeyB (98% of template volume) 1.15; uwo 11–31; rockefeller 26–40 |
| mask < 80% of template AND missed > 0.25 cm³ | mild failures, two weak signals agreeing | 032117: 0.46 missed, 75% volume |
| no prior AND mask < 80% of template | fallback when `template_prior: false` or its registration failed | the pre-prior rule |

A small mask the template agrees with does not trigger: 032206, 032215 (small animals,
missed ~0) stay single-pass. Under the old volume-only rule they got a second pass.

### 3.5 Keeping pass 2

Pass 2 is kept unless its Dice with the template drops below pass 1's by more than
0.005. A rejected pass 2 is moved to `fastsurfercnn_output/pass2_rejected/` and pass 1
restored; the sidecar says `Pass2Kept: false`. On the 28 test subjects every pass 2 so
far agreed at least as well as pass 1; the rule is a safety net, e.g. for a false trigger
on a registration that went wrong.

### 3.6 V1 fix, once

The ROI white-matter fix (a cropped template registration that adds the thin V1 white
matter the CNN misses) used to run inside every CNN pass, so with two passes the first
one's registration was thrown away. For anatomy `apply_segmentation` now runs the passes
with `fix_roi_wm=False` and calls `apply_roi_wm_fix` once on the kept pass, with that
pass's input image. Functional segmentation is unchanged. The fix's registration is between
V1 crops, so it needs the subject in NMT2Sym world too: `fix_roi_wm(world_to_nmt2sym=...)`
writes the subject crop with the placed header (3.7).

### 3.7 Placing the image in NMT2Sym world (`nmt2sym_frame.py`)

The N4 region, the template prior and the V1 fix all place or register **NMT2Sym**
material on the subject, whatever `template.output_space` is. FireANTs (the default SyN
engine) refines a near alignment but does not recover a large shift or rotation; the main
registration gets its coarse alignment from conform. Conform aligns to the *output*
template, so these steps only start close when that template shares NMT2Sym's world.

| Output frame | Offset from NMT2Sym (brain centroid) | How the image is placed |
|---|---|---|
| NMT2Sym (any res), NMT2Asym | 0 (NMT2Asym brain-mask Dice 0.991 untransformed) | identity: nothing changes |
| MEBRAINS / D99 / Yerkes19 | 24.8 / 23.3 / 24.6 mm; D99 pitched ~10° | bundled rigid, `template_zoo/template/xfm/from-X_to-NMT2Sym_mode-image_desc-rigid_xfm.txt` |
| custom template, conform off | arbitrary (devtest scanner frame: 20 mm) | runtime rigid of the brain (pass-1 mask × image) to `tpl-NMT2Sym_res-05_T1w_brain`, `anat.conform.rigid_method` (FLIRT 5 s; SimpleITK 10 s, input padded 20 voxels) |

The placement is header-only (`reframe`: affine' = W · affine, same voxels), so the CNN
keeps the image's own header and every array warped onto the copy is on the original grid.
The region and prior are computed on the copy; the N4 itself runs on the original. If the
rigid fails, the checks and the V1 fix are skipped, pass 1 is kept, and
`Nmt2SymFrame: {Source: failed}` is recorded. A non-identity frame is
recorded as `Nmt2SymFrame` {`Source`, `Detail`, `TranslationMm`}.

**Measured** (devtest sub-032309, 2026-09-30; FireANTs on GPU):

| Frame | Prior Dice / MissedCm3 | Region coverage | Template surface in mask | V1 fix relabelled |
|---|---|---|---|---|
| NMT2Sym conform (baseline) | 0.962 / 0.0 | 0.998 | 99.95% | 184 |
| MEBRAINS/D99/Yerkes19 as before | 0.45–0.57 / 2–7 | 0.87–0.94 | — (surface Dice 0.41–0.63) | 0 / 15 / 0 (none correct) |
| MEBRAINS/D99/Yerkes19, bundled rigid | 0.963 / 0.0 | 0.998 | 99.95% | 143–153 (Dice 0.46–0.65 with the baseline's changes) |
| real FLIRT conform to MEBRAINS/D99/Yerkes19, bundled rigid | 0.960–0.964 / ≤0.11 | 1.0 | 99.89–99.94% | — |
| scanner frame, FLIRT / SimpleITK rigid | 0.958–0.959 / ≤0.05 | 1.0 | 99.94% | — |
| NMT2Sym baseline nudged 0.05° / 0.02 mm (identity frame) | 0.963 / 0.0 | 0.998 | 99.95% | 147 (Dice 0.47) |

The V1 fix is deterministic (a repeat of the baseline gives the same 184 voxels) but very
sensitive: moving the image by 0.05° / 0.02 mm, with no frame involved, changes it as much as
the bundled-rigid frames do. Its exact voxels are therefore not a measure of placement; the
prior and surface numbers are.

The bundled rigids: brain-mask Dice with NMT2Sym 0.94 (MEBRAINS), 0.91 (D99), 0.89 (Yerkes19)
from the rigid alone, against 0.51–0.55 untransformed and 0.95–0.96 for the full affine.
Only the rigid is stored: the templates differ in size by 2–8%, which FireANTs absorbs, and a
template-to-template warp would bias the prior toward that correspondence.

The undersized rule compares with NMT2Sym's brain (92.5 cm³) in every frame: the output
template's brain moved the cut to 82–93 cm³.

**Not used, and why.** FireANTs' own initialisation (`init_rigid`, moments) was ruled out:
the coarse step is exactly what it is unreliable at. A full ANTs CLI SyN converges from 25 mm
but took 25 min on 2 threads. ANTs translation→rigid→affine before FireANTs works (Dice 0.963)
but needs affine+warp composition at every site.

## 4. Provenance (mask and segmentation sidecars)

| Field | Meaning |
|---|---|
| `MaskVolumeCm3`, `TemplateBrainVolumeCm3`, `MaskToTemplateBrainRatio`, `MaskUndersized` | final mask vs the output template's brain (80% rule) |
| `IntensityCapApplied`, `IntensityCapRatio` | whether the 0–255 cap bound, for the image that produced the final segmentation |
| `SegmentationPasses` | 1 or 2 |
| `Pass1` | pass-1 volume and cap fields, when pass 2 was kept |
| `PreInferenceN4` | `Trigger`, `Region`, `ShrinkFactor`, `BSplineFitting`, `Pass2Kept`; `Error` when pass 2 itself failed (pass 1 kept) |
| `SegmentationCheckFailed` | the N4 or the prior failed before any decision; pass 1 kept, single pass |
| `TemplatePrior` | `Pass1` / `Pass2` {`TemplateDice`, `MissedCm3`, `Reliable`}, final `TemplateDice`, `MissedCm3`, `Reliable`, `MinMissedCm3`, `Engine`; or `RegistrationFailed` |

Work files: `pre_inference_n4/{search_region,pre_inference_n4}.nii.gz`,
`template_prior/template_{brain,atlas}_in_subject.nii.gz`,
`fastsurfercnn_output/{pass1,pass2_rejected}/`.

## 5. Configuration

```yaml
anat:
  skullstripping_segmentation:
    fastSurferCNN:
      pre_inference_n4:
        enabled: "auto"      # auto | true | false
        shrink_factor: 4
        bspline_fitting: 60  # spline distance, mm
      template_prior:
        enabled: true        # false skips the registration; the volume rule takes over
        min_missed_cm3: 0.5
```

Constants, deliberately not exposed: `SEARCH_REGION_MARGIN_MM = 15`,
`HEAD_THRESHOLD_FRACTION = 0.1`, `MISSED_CORE_MM = 1`, `MISSED_OPENING_MM = 1`,
`MISSED_INTENSITY_PERCENTILES = (2, 99.5)`, `MIN_TEMPLATE_DICE = 0.92`
(`segmentation_prior.py`); `UNDERSIZED_MIN_MISSED_CM3 = 0.25`,
`PASS2_DICE_TOLERANCE = 0.005`, `MASK_UNDERSIZED_RATIO = 0.8` (`preprocessing.py`).

Cost on a healthy subject with defaults: one N4 (shrink 4) and one template registration,
~30 s on GPU. A second pass adds one CNN run. CPU cost not yet measured.

## 6. Validation

Test sets (conformed T1w reused so only the segmentation step varies; config
`preproc/config_res-1_surf-off_anatonly.yaml`):

- `dataset_seg_hard` (17): monkeyB, PRIME-DE rockefeller ×4, sbri 032300, ecnuChen ×3,
  oxford ×5, uwo ×3. Its README lists each subject's failure type.
- Regression (11): `dataset_devtest` ×4, `dataset_example`, and PRIME-DE 032306, 032215,
  1224, 032212, bordeaux m03, mcgill 032206.

Mask volume (cm³), v3.0.0 → final design, subjects that differ:

| subject | v3.0.0 | final | |
|---|---|---|---|
| monkeyB | 86.1 | 95.9 | pass 2 (template) |
| rockefeller 032116 | 71.8 | 113.0 | pass 2 (template) |
| rockefeller 032117 | 41.7 | 81.8 | cap + pass 2 (undersized + template) |
| rockefeller 032121 | 56.7 | 114.9 | pass 2 (template) |
| rockefeller 032118 | 91.1 | 97.1 | cap only |
| mcgill 032206 | 64.7 | 65.2 | cap only |
| uwo 032190 / 032194 / 032197 | 87.1 / 83.5 / 60.3 | 97.0 / 86.6 / 67.5 | pass 2 (template) |
| ecnuChen 032279 | 97.2 | 102.5 | pass 2 (template) |
| bordeaux m03 | 69.6 | 70.5 | pass 2 (template) |
| sbri 032300 | 64.1 | 65.2 or 64.1 | see §7, GPU jitter |

All other subjects (devtest, example, oxford ×5, ecnuChen 032284/032287, 032306, 032215,
1224, 032212) are unchanged. "Cap only" rows are single-pass subjects changed by the
intensity cap (§3.1), not by the two-pass logic. Visual review: before/after pages in
`preproc/seg_repair_qc/` (`seg_fourway.html`: v3.0.0 / previous design / first
streamlined / final); scripts to regenerate them in `preproc/seg_repair_qc/scripts/`
(`seg_run.py` runs one code tree's segmentation on a case list; `make_html3.py` builds
the page).

## 7. Known limits

- **Registration jitter.** FireANTs on GPU is not bit-reproducible; `MissedCm3` moves by
  ~0.3 cm³ between identical runs (032300: 0.33, then 0.67). A subject near a threshold
  can flip between one and two passes across reruns. The keep rule bounds the effect
  (032300: 64.1 vs 65.2 cm³).
- **UNI (MP2RAGE) registration.** The template registers poorly on uwo (Dice 0.74–0.87):
  the whole-head registration is pulled toward the bright background. It still triggers
  pass 2 correctly (the pass-2 masks are much cleaner), but `MissedCm3` there is not a
  measurement of anatomy, and the CNN still misses the uwo frontal pole in both passes.
  Tried and rejected: cutting both images to the placed template (+8 mm) and affine-first
  then cut — both fix UNI and small brains but anchor large brains at template size
  (032121: Dice 0.967 → 0.85–0.92); cutting only the subject image trades the same way
  with its margin (10 mm good for UNI, 25 mm needed for 032121). The real fix is UNI
  contrast in the CNN's training data.
- **Template overshoot on small brains.** On 032300 and m03 the warped template sits a
  few mm outside the frontal pole. Harmless now that the template only measures.

## 8. What was tried and removed

- **Template repair** (the previous design): added missed voxels from the registered
  template (atlas-labelled, brightness-gated) and trimmed CNN voxels > 3 mm outside it,
  gated on Dice ≥ 0.92. On the 28 subjects it changed the mask by at most 0.09 cm³ after
  pass 2 — the second pass does the recovery — while being the most complex part
  (write-back, `*_prerepair` files, hemisphere mask rebuild). Without its gates it grew
  masks into temporalis muscle and orbit on m03, where the template did not shrink onto a
  small brain. Removed; the prior now only measures.
- **Head-mask-only N4 region**, **template-only region (+10 mm)**, **brain clip**,
  **30/150 mm splines**: see §3.2.
- **Probability threshold instead of argmax**: see §1.
