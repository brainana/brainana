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

## 112RM-SL template + UWPaxinos atlas (2026-11-20)

- **Template `112RMSL`** (display name 112RM-SL; hyphen dropped for a BIDS-valid label):
  McLaren et al. 2009, NeuroDebian `mclaren-rhesus-macaque-atlas` 0.20081212 (GPL-3). It sits in
  Saleem & Logothetis space but is a 112-monkey average, not the D99 specimen (brainana's D99 is
  D99 v2.0, voxel-identical to the AFNI `D99_v2.0_dist`). The distributed T1 is already skull-stripped;
  `brainmask` = GM+WM+CSF priors > 0.5 (114 cm³). Reoriented LAS→RAS and cropped to brain + 6 vox,
  no resampling. Built by `macaque/propogate_atlases_across_space/make_112rmsl_template_and_atlas.py`;
  originals + NOTE in `atlas/macaque/template/112RMSL/`.
- **Rigid frame** `template/xfm/from-112RMSL_to-NMT2Sym_*`: brain-mask Dice with NMT2Sym 0.876
  identity / 0.89 rigid / 0.962 affine. 112RM-SL is already near the NMT2Sym frame (1.7 mm), so
  `test_table_places_the_template_brain_on_nmt2sym` checks rigid > identity > inverse for it instead
  of the "inverse is far off" check used for D99/MEBRAINS/Yerkes19.
- **SyN to/from NMT2Sym** via brainana's `ants_register` (`make_syn_xfm.py`): FireANTs (GPU) used;
  stored as `.nii.gz` warps in `atlas/macaque/xfm/`. Direction check: NMT2Sym→112RMSL r=0.912
  (reversed 0.265), 112RMSL→NMT2Sym r=0.800 (reversed 0.256). ANTs SyN run for comparison gave
  0.884 / 0.807, so FireANTs kept; the lower forward r is the image pair (112RM-SL contrast,
  CSF-inclusive mask), not the engine.
- **UWPaxinos atlas**: Wisconsin Rhesus Brain Parcellation (NITRC `rhesusROIs_20180830`, public
  domain), indexed `Paxinos_to_112RM-SL_20121014` on the original (non-symmetrized) template.
  290 codes = 286 Paxinos elementary + 992 Cb, 997 VTA, 998 DR, 999 SN (997–999 identified by voxel
  overlap with `ROIs/In_house/*`; names from `Rhesus_ROIs.xls`). The `.img` header origin differs
  from the T1's on the same voxel grid; the T1 affine is used (99.8% of labelled voxels in the brainmask).
  Bilateral labels. Volume only (not projected onto template surfaces).
- **Propagation**: `1_warp_atlases.py` warped every zoo atlas (incl. ARM1–6, added to `ATLASES` with
  `surface=False`) into 112RMSL, and UWPaxinos into the other five spaces; all 290 labels survive in
  every space. 112RMSL has no `sub-112RMSL` surfaces, so steps 2–3 default to `SURFACE_TEMPLATES` /
  `SURFACE_ATLASES` and skip it.
- Docs: the D99 row in `template_atlas_zoo.rst` used to cite McLaren 2009 (the 112RM-SL paper);
  now Saleem et al. 2021. The zoo overview figure (`template_atlas_zoo.png`) was not regenerated.
