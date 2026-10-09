# Skull strips that find no brain

Field case (PRIME-DE v3.1.0 batch, 2026-10-08): `site-carmenlyon/sub-12/ses-01`. The run
reported "pass with warnings, 1 failed task" — `ANAT_SURFACE_RECONSTRUCTION (12_01)`,
`mri_normalize: could not find enough control points`. The real failure was four steps
earlier, at conform.

## Cascade

| Step | What happened | Flagged |
|---|---|---|
| Conform skull strip (`nhp_skullstrip_nn`, T1w model) | kept a 72 cm³ blob in the air behind/above the head | no |
| Conform FLIRT | registered that blob; matrix swaps x↔y and flips z (site-mates: clean 30–65° pitch) | no |
| Conform crop | box on background; `desc-conform` is pure noise | no |
| Main skull strip (fastSurferCNN) | `QC: CNN brain mask is 0.0 cm^3` | warning only |
| Template prior | `missed_brain` crashed on the empty mask (`index -1 is out of bounds`), pass 1 kept | warning only |
| Bias, registration, atlases | ran on noise, "completed" | no |
| Surface recon | empty aseg → `mri_normalize` fails | the one failed (ignored) task |

The raw T1w is fine (normal header, same pose and contrast as sub-10/11/13).

## Cause: largest-component selection on a near-tie

`predict_volumes` thresholds the averaged probability at 0.5 and kept the largest connected
component. For sub-12 (0.5 mm inference grid):

| Component | Voxels | Mean p | Share of voxels with p > 0.9 | Mean intensity | Inside head |
|---|---|---|---|---|---|
| Air blob (kept) | 577,929 | 0.61 | 0.00 | 25 | 0.00 |
| Brain | 551,613 | 0.95 | 0.88 | 405 | 0.99 |

The decision was a coin flip. Zeroing 12 slices of pure-noise background (x < 12) flipped
it (blob 544,410 vs brain 551,884). Clipping at the 99.5th percentile did not.
The `brain_for_reg` crop (`conform_registration_crop.md`) would not have helped: it crops
around whatever mask it is given.

## Fix 1: keep the confident component (`select_confident_component`)

`nhp_skullstrip_nn/utils/morphology.py`. Among the 0.5-level components, keep the one with
the most voxels above `CONFIDENT_CORE_THRESHOLD = 0.9`; with no voxel above 0.9, keep the
largest (old behaviour). The kept component's voxels are unchanged — only which one is
kept. Used in both the binary and multiclass branches of `predict_volumes`, so it covers
anat conform, func conform and the func brain mask.

A runner-up ≥ `NEAR_TIE_RATIO = 0.5` of the kept size logs a `QC:` warning (with "largest
component was dropped" when the choice differs from the old one). It is not in the step
metadata: `predict_volumes` has no metadata channel.

Rejected:
- **Probability mass (Σp)**: a large faint blob can still win.
- **Mean intensity / overlap with an Otsu head mask**: works for sub-12, but MP2RAGE UNI has
  bright noisy background and EPI has dropout; the network's own confidence is
  modality-agnostic.
- **Distance to the image centre**: fails with neck/body in the FOV.

## Fix 2: an empty main mask stops the subject

`_require_nonempty_mask` (`operations/preprocessing.py`), called in `apply_segmentation`
for anat after the final mask is validated (outside the try that rewraps errors). Mask
< `MASK_EMPTY_RATIO = 0.1` × NMT2Sym template brain (92.5 cm³ → 9.25 cm³) raises
`EmptyBrainMaskError`. Separate from `MASK_UNDERSIZED_RATIO = 0.8`: 41–73 cm³ masks are real
misses the second pass can recover and stay warnings.

- `ANAT_SKULLSTRIPPING` catches it and exits `EMPTY_BRAIN_MASK_EXIT_CODE = 3`;
  `nextflow.config` maps 3 → `ignore` (OOM → retry, else terminate). The subject's later
  steps never start; the rest of the run continues.
- Missing-subject safety: no `groupKey` with a size anywhere; func sessions fall back to
  another session's anat or the template (`performFuncAnatomicalSelection`).
- The longitudinal base calls `apply_segmentation` too; there the error exits 1, and the
  base processes already ignore failures.
- Report banner (`run_status.py`): "optional steps failed" → lists FAILED task names from the
  trace (`failed_task_names`; a FAILED row whose name later COMPLETED/CACHED — an OOM retry —
  is not listed).

## Tests

`tests/test_empty_brain_mask.py`: selection (confident beats larger faint, both confident →
largest, no core → largest, rim voxels kept, empty), empty vs 60 cm³ mask, config/module
exit-code wiring, trace parsing, banner.

## Regression (2026-10-08)

Old (largest) vs new (confident core) selection on the same network pass, 58 images:
29 `dataset_surf` subjects, all 20 carmenlyon subjects, 3 MP2RAGE UNI (2 uwo + 1 other
site), 6 func tmeans (first 30 volumes; uwo, newcastle, ion, oxford, ucdavis,
mountsinaiP; EPI model).

- 57/58 masks voxel-identical. The one change is carmenlyon sub-12: kept 551,612 voxels
  (brain) instead of 577,926 (air blob).
- Near-tie warning (runner-up ≥ 50% of kept) fired only on sub-12. Everywhere else the
  runner-up was small.
- carmenlyon re-run with v3.1.0 + the two `nhp_skullstrip_nn` files overlaid and sub-12's
  cached conform moved aside, so only sub-12 recomputes (log:
  `PRIME-DE_brainana_v3.1.0/_logs/site-carmenlyon_sub12fix.log`).
