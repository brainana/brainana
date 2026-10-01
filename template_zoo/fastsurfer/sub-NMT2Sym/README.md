# sub-NMT2Sym (FreeSurfer-format template surfaces)

Template surfaces used by surface reconstruction when `anat.surface_reconstruction.template_surface.enabled`
is true, and by surface registration (`registration_template` in the surface-reconstruction config).

The mesh is icosahedral (ico6, 40962 vertices per hemisphere). Every subject reconstructed with the
template prior shares this vertex numbering, so labels defined here index the same place in every subject.

| File | Use |
|---|---|
| `surf/{lh,rh}.white` | Starting white surface. It is carried into each subject by the subject-to-NMT2Sym registration |
| `surf/{lh,rh}.sphere` | Spherical map. It is copied to each subject, since the mesh is shared |
| `label/{lh,rh}.V1.label` | V1 vertices. Their white surface is held at the template (not moved by white-surface placement) |
| `label/{lh,rh}.aparc.ARM2atlas.mapped.annot` | ARM2 parcellation on the template surface (surface registration) |

## Provenance

Copied from the brainana atlas repository, `public/macaque_template_surfaces/sub-NMT2Sym`. Those surfaces
were derived from the white and pial surfaces distributed with the NMT v2 symmetric 0.5 mm template, and
processed with brainana (sphere, labels, parcellations).

`{lh,rh}.V1.label` was made once from that repository's CHARM level-2 parcellation
(`atlas/CHARM2_{lh,rh}.annot`, region "V1"), restricted to `label/{lh,rh}.cortex.label`:
4789 (lh) and 4714 (rh) vertices.

## Why V1 is held at the template

On T1w data the white surface fitted to the image in V1 sat about 0.4 mm too far out, which gave a V1
thickness of 1.3–1.4 mm, against the expected 1.7–2.0 mm. On images with strong bias it also lost V1's
characteristic fold shape. Holding V1's white surface at the template, and fitting everything else (and the
whole pial surface) to the image, was preferred in blind visual comparisons on both hard and normal-quality
data (2026-09).
