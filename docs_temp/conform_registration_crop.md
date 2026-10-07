# Conform: register the brain, not the scan

Field report (v3.1.0 Docker, 2026-10): a T1w with the neck and shoulders in view
(`dataset_CM`, 190 × 240 × 240 mm, 1 mm) conformed upside down; skull stripping and
everything after it failed, and the run reported success. The user had to crop by hand.

## Cause

Conform (`conform_to_template`, `operations/preprocessing.py`) skull-strips the input with
the conform NN, then runs a 6-DOF FLIRT with a ±180° search on all three axes (corratio)
from the stripped brain to the padded template.

- The NN mask was **fine** on the full FOV (103 ml vs 106 ml on a hand-cropped copy).
- FLIRT returned 116° / −177° (Euler xyz) on the full grid and −31° (correct; the head is
  pitched) on the cropped one.
- The cost function was not at fault: on the full grid the correct pose scores 0.74 and the
  wrong one 0.94 (lower is better). The search never reaches the right basin.
- The grid size alone decides it. The same masked brain, cropped to its bounding box
  ±5 / 20 / 40 voxels, gives −31° every time; the brain from the hand-cropped input, zero-padded
  back out to 230 × 240 × 240, fails again (60° / 88°). The brain is ~1% of the 240 mm grid.
- The SimpleITK backend found −31.8° on the full grid.

## Fix

For `modal == "anat"` only, between the skull strip and the rigid registration:

1. `crop_to_nonzero_bbox` (`utils/mri.py`) crops `brain_for_conform` to its nonzero bounding
   box → `brain_for_reg.nii.gz`. Margin per axis is `max(25% of the brain's extent, 16 mm)`
   (`DEFAULT_CONFORM_REG_CROP_MARGIN_FRACTION`, `DEFAULT_CONFORM_REG_CROP_MIN_MM`), clipped to
   the image.
   - Why a fraction and not voxels: voxel sizes span 0.25–1.5 mm, so voxels are the wrong
     unit; a fraction of the brain scales across species.
   - Why the 16 mm floor: it is two of FLIRT's coarsest (8 mm) voxels, which matters for
     small brains.
   - Margins from ~8% to ~60% of this brain all worked, so the exact value is not critical.
2. Both backends register the cropped brain.
3. FSL matrices are tied to the moving grid, so the forward matrix is re-expressed for the
   uncropped grid with `fsl_mat_for_new_moving` (`sitk_rigid_registration.py`, the moving-side
   mirror of `fsl_mat_for_new_reference`): `mat_full = mat_crop @ C(full → crop)`. The inverse
   is rewritten as `inv(mat_full)`. This applies to the sitk backend's FSL `.mat` files too; its
   transform object and `.world.mat` are world-space and unchanged.
4. Apply (Step 5) and full-FOV (Step 5b) still read the uncropped input, so every output keeps
   the same grid and FOV as before.

The crop copies the input header and moves only the stored origins (`qoffset_*`, `srow_*[3]`).
The first version used nibabel's `slicer`, which rebuilds the header from the affine and drops
the qform. On an oblique scan (site-ohsu: qfac −1, float32 sform agreeing with the qform only to
~1e-7) SimpleITK then read the direction from the sform for the crop but from the qform for the
original, and `fsl_mat_for_new_moving` correctly refused the pair (direction differs > 1e-9).

Func conform (tmean) is unchanged.

## The full-FOV growth cap is not a correctness test

`reference_padding_to_cover` refuses to enlarge the full-FOV grid past 3× the template box per
axis. Its message used to say "the rigid transform is probably wrong". On `dataset_CM`,
the correct pose also exceeds the cap (4.34× vs 4.58× for the wrong pose), so the number
measures how far the scan reaches beyond the head. The message now says so. A cross-check
against a second backend was considered and dropped: with the crop, FLIRT alone gets this case
right, and the check would cost 1–2 min per run.

## Tests

- `tests/test_conform_full_fov.py`: `fsl_mat_for_new_moving` reproduces the full-grid matrix
  exactly for both x-flip branches and an asymmetric crop; rejects a different direction;
  `crop_to_nonzero_bbox` margin fraction, mm floor, clipping and no-op cases.
- `tests/test_conform_registration_crop.py`: the NMT2Sym res-1 brain, pitched 30°, embedded
  high in a 190 × 240 × 240 mm zero grid, conforms to within 1° with both backends. Without
  the crop FLIRT misses by 2.1° here (with it, 0.1°); SimpleITK is within 0.2° either way.

## Regression (PRIME-DE, 2026-10-06)

`conform_to_template`, default config, v3.1.0 vs this fix, one T1w per site (26 sites with a
T1w; `site-ds003989_annex` skipped as a duplicate; utrecht's rhesus subject) plus `sub-CM`.
Outputs, `summary.csv` and `contact_sheet.png` in
`brainana_test/preproc/conform_crop_regression/` (final new-code pass in `*/new2`).

- 26/26 PRIME-DE sites: same pose, at most 1.92° / 0.94 mm (site-sbri); median ~0.2° / 0.1 mm.
- `sub-CM`: 174° change (the fix); final brain mask 47.6 ml → 108.0 ml in a full pipeline run.
- Conform is faster on 24 of 27 samples (smaller FLIRT input; the other 3 unchanged), e.g. uwmadison 194 → 149 s.
