# Atlas projection onto fsnative surfaces

Step: `ANAT_PROJECT_ATLASES_TO_SURFACE` → `anat_project_atlases_to_surface`
(`src/nhp_mri_prep/steps/anatomical.py`). Output: `anat/atlas_space-fsnative/`.

## Problem (v3.1.0)

The step sampled each atlas once, at mid-thickness (`mri_vol2surf --projfrac 0.5 --interp
nearest`), with no cortex mask and no hole filling. The atlas grey matter (warped from the
template) and the subject's ribbon never line up exactly, so the mid-thickness point often
lands on a 0 voxel or a white-matter voxel.

PRIME-DE v3.1.0, `site-bordeaux/sub-m05` (40962 vertices per hemisphere; lh cortex 38617):

| atlas | empty cortex vertices lh / rh | labelled vertices outside cortex lh / rh |
|---|---|---|
| ARM2 | 1653 / 1574 | 2804 / 2946 |
| D99 | 2410 / 1802 | 1268 / 1370 |
| MacBNA | 3393 / 2827 | 1241 / 1340 |
| FuncNetwork (lh-only) | 4349 / – | 76 / – |

## Method (after the fix)

1. `mri_vol2vol` to `mri/T1.mgz` (unchanged).
2. `mri_vol2surf --interp nearest` at projfrac 0.0, 0.1, …, 1.0 (11 calls per hemi).
3. `_combine_depth_samples`, per vertex:
   - **label atlas** (has `atlas-{name}.tsv`): most common **positive** id across depths, ties to
     the depth nearest 0.5. Negative ids are ARM tissue codes (-1 cortical WM, -2 lateral
     ventricle, -501 cerebellar WM, -100x for lh) and would win near the white surface.
   - **continuous map** (no tsv: retinotopy, somatotopy, CortHierarchy): the nonzero sample
     nearest mid-thickness. No median or mean, so polar angle (±180°) never wraps and values
     are never invented. Frame 0 picks the depth for all frames, so PA/ECC stay paired with
     their F statistics.
4. Mask to `label/{hemi}.cortex.label`.
5. Label atlases only: `_fill_label_holes_in_cortex` repeats the 1-ring `mode_filter`
   (`fastsurfer_surfrecon.processing.parcellation`) on label-0 cortex vertices until nothing
   changes. Labelled vertices never change; out-of-cortex vertices are never donors.
   - **Coverage guard**: skip the fill when less than 50% of cortex is labelled after step 3.
     lh-only atlases (FuncNetwork, CortHierarchy) catch a few midline voxels on rh (30 / 15 on
     sub-m05); an unguarded fill would flood rh from them.
   - Continuous maps are not filled: CortHierarchy has real gaps (lh 65% coverage on sub-m05).

Why not fill the ribbon in the volume first: a nearest-neighbour fill in 3D crosses sulcal
banks and the midline (close in voxels, far on the cortex). Filling on the mesh follows the
cortex.

Status JSON gains `depths`, `surface_fill` and `cortex_coverage` (fraction of cortex labelled
before filling, per `{atlas}_hemi-{L|R}`).

## Result on sub-m05

| atlas | empty cortex lh / rh | outside cortex | labels changed where both labelled |
|---|---|---|---|
| ARM2 | 0 / 0 | 0 | 0.7–0.8% |
| ARM6 | 0 / 0 | 0 | 2.1–2.3% |
| D99 | 0 / 0 | 0 | 2.3–2.4% |
| MacBNA | 0 / 0 | 0 | 2.1–2.3% |
| FuncNetwork lh | 0 | 0 | 1.5% |
| CortHierarchy lh | 13066 (was 15801; not filled) | 0 | 0% |
| retinotopy / somatotopy | 0 / 0 | 0 (was ~3000) | 0% |

No output id is missing from the fsnative volume. The changed vertices are at area borders,
where several depths now vote. Runtime: about 80 s per subject for 12 atlases (264
`mri_vol2surf` calls, ~0.35 s each).

## Not handled

- ARM subcortical ids (≥ 500 after removing the lh +1000 offset) can still win at a vertex if
  most depths hit them; none were seen on sub-m05.
- Longitudinal base projection (`ANAT_PROJECT_ATLASES_TO_SURFACE_BASE`) uses the same
  function, so it gets the same method.

## Backfill

PRIME-DE v3.1.0 outputs were re-projected with
`scripts/scratch/surf_recon/backfill_atlas_surface_projection.py`; see
`atlas/dataset_information/PRIME-DE/NOTE.txt`.

## Zoo atlases in every template space (2026-10-09)

Before: most atlases existed in only one or two zoo spaces (see the old list in
`docs/template_atlas_zoo.rst`). Missing atlas × space volumes were warped with
`/home/star/github/macaque/propogate_atlases_across_space/` (`1_warp_atlases.py`):
NMT2Sym zoo volume → `from-NMT2Sym_to-<T>_desc-syn_xfm` (`atlas/macaque/xfm`), nearest
neighbour, onto `tpl-<T>_res-05_T1w_brain`; FuncConnGrad from Yerkes19 through NMT2Sym in one
resampling. Volumes already in the zoo were kept (D99 res-025 in D99, D99 res-04 and
retinotopy res-1 in MEBRAINS, MacBNA in NMT2Asym). xfm direction checked by warping the
NMT2Sym T1w: r ≈ 0.95 as named vs ≈ 0 reversed. On the template surfaces, label area shares
match NMT2Sym (r ≥ 0.985 for D99, MacBNA, FuncNetwork). `discover_atlases` picks the new
files up with no code change.
