# Output Naming — Reference & Update Guide

A single place to understand how brainana names the files it publishes, and
exactly what to touch when you add an output, a `desc` value or a space. Read
Part 1 once; use the checklists in Part 4 every time you add something.

> **Golden rule:** an output name is *the raw file's entities, copied* + *one
> entity per thing brainana has to say* + *one BIDS suffix*. Identity entities
> belong to the raw data — brainana copies them and never sets one. Build names
> with `derive_output_name()`; it refuses the two ways a name loses information.

---

## Part 1 — The full picture

### The four roles

`ENTITY_ROLES` in `src/nhp_mri_prep/utils/bids.py` is the source of truth.

| Role | Entities | Who writes them |
|---|---|---|
| **identity** | `sub` `ses` `task` `acq` `ce` `dir` `rec` `run` `echo` `flip` `inv` `mt` `part` `recording`, plus any entity brainana does not recognise | **The raw data only.** brainana copies them and never sets or changes one. |
| **frame** | `space` `from` `to` `mode` `res` `den` `hemi` | brainana — which reference frame the data are in. |
| **variant** | `desc` `split` | brainana — which processing variant. At most one `desc` per name. |
| **product** | `atlas` `stat` | brainana — what kind of product. |

Each concern has exactly one slot. Writing a pipeline concern into an *identity*
entity is the mistake this table exists to prevent: the entity already means
something, so the two meanings become indistinguishable, and what the scanner
recorded is destroyed.

That is not hypothetical. The longitudinal stream shipped marking its products
`acq-base` / `acq-long`, in a dataset that also carries a real `acq-test`. Three
consequences, none of which announced itself:

- a timepoint's own `acq-mprage` was overwritten, so two acquisitions in one
  session resolved to one filename — see layer (2) above;
- no consumer could tell a stream marker from an acquisition label;
- the QC report groups anatomical figures by their identity entities, so
  `ses-001` and `ses-001 + acq-long` read as two different sessions. A
  three-session subject listed six entries under **Structural** where the
  `session` synthesis level listed three.

It is now `space-base`, because that is what the distinction *is*: a base-seeded
reconstruction lives in the subject's base space. Which is already the documented
reason the functional and fsnative-atlas streams keep using the cross-sectional
trees — `mri_vol2surf --regheader` assumes header agreement with the session's
own volumes. The report excludes `space` from its group key, so the fix also
collapsed the nav back to one entry per session with no report change at all.

### The six surfaces that must stay in sync

1. **Roles** — `ENTITY_ROLES` in `src/nhp_mri_prep/utils/bids.py`.
2. **Order** — `BIDS_ENTITY_ORDER`, same file.
3. **The constructor** — `derive_output_name()` / `create_bids_output_filename()`.
4. **The validator** — `validate_output_name()`, plus
   `scripts/check_output_naming.py` for a whole tree.
5. **The published docs** — `docs/outputs.rst` (the actual filenames). The rule
   itself is not published; it lives in `ENTITY_ROLES` and this guideline.
6. **`brainana-viewer`'s data contract** — `docs/data-contract.md` in that repo.
   It names specific output shapes; two of the deviations in Part 3 exist for it.

Plus the drift guards: `tests/test_output_naming_contract.py` and the two
pattern checks in `tests/test_nextflow_script_bodies.py`.

### `BIDS_ENTITY_ORDER`: positions are a contract

Entity positions are **the ones brainana already publishes, not the abstract BIDS
order**, because published names are consumed:

- `hemi` after `space` — `atlas-<n>_space-fsnative_hemi-L_<prefix>.func.gii`,
  which the Viewer discovers by that exact shape.
- `stat` after `desc` — `<prefix>_space-T1w_desc-preproc_stat-tsnr_boldmap`.
- `atlas` first — see Part 3.

An entity **missing** from this list falls into the `others` slot, which is
emitted *before* `space` and `desc`. `stat` and `hemi` were missing, so a tSNR
map rebuilt from its own entities came back as `..._stat-tsnr_desc-preproc_...`
— a different file. If you emit a new entity, list it, and list it where the
pipeline actually writes it.

### How a name is built at runtime

```
raw BIDS input                sub-01/ses-003/anat/sub-01_ses-003_run-1_T1w.nii.gz
      │
      ▼  ANAT_SYNTHESIS (anat) / channel map (func)
naming template               travels through channels as val(bids_name), and is
"bids_name"                   NOT rewritten by any step except the longitudinal
      │                       one (longitudinal_bids_name → space-base)
      ├──────────────► derive_output_name(bids_name, suffix=…, set_entities=…)
      │                       copies identity entities, sets brainana's own,
      │                       emits in BIDS_ENTITY_ORDER, raises on a conflict
      │
      └──────────────► create_bids_output_filename(bids_name, suffix=…, modality=…)
                              stem-preserving: keeps an unknown raw entity in its
                              original position, and strips whatever the suffix
                              re-sets
      │
      ▼  publishDir (mode: 'copy', overwrite: false)
<output_dir>/sub-01/ses-003/anat/…    or    <output_dir>/sub-01/figures/…
```

**The rule in one line: name from the template, never from a staged file.** The
template comes from the raw data and carries no derivative entity. A staged file
is a previous step's output and carries them all — which is how nine published
QC figures ended up as `..._space-NMT2Sym_desc-preproc_desc-func2target_bold.png`.
`tests/test_nextflow_script_bodies.py` enforces this: `get_filename_stem()` may
only be called on a naming template, or on an argument listed in `_NOT_A_NAME`
with its reason.

### Which helper to use

| Situation | Use |
|---|---|
| Naming a product from the run's template | `derive_output_name(bids_name, suffix=…, set_entities={…})` |
| Same, where the suffix is a legacy multi-entity string | `create_bids_output_filename(bids_name, suffix='space-X_desc-y', modality='T1w')` |
| Replacing only the frame on an existing name | `replace_bids_space(stem, new_space)` — strips every existing `space-` first |
| A session-level product (drop task/run/acq) | `get_bids_prefix(bids_name)` — keeps `sub`/`ses` only |
| The base template or a longitudinal timepoint | `longitudinal_bids_name(bids_name, 'base' \| 'long')` |
| A `.json` beside an image | `create_bids_sidecar_filename(image_path)` |

`get_bids_prefix()`'s session-level branch **drops `space`**. A product whose
frame is not the session's own must take the frame separately — see
`_backprojection_space_label()` in `src/nhp_mri_prep/steps/anatomical.py`, which is why a
base atlas is `atlas-ARM2_space-base_sub-01.nii.gz` and not `space-T1w`.

---

## Part 2 — The vocabulary in use

Reuse a value before inventing one. A new value that duplicates an old meaning is
much harder to undo than to avoid.

### `desc` — variant and processing stage

*Published derivative variants* (what the image **is**):
`preproc`, `brain`

*Processing-stage markers* (mostly QC figures — which **step** made it):
`conform`, `conformFullFOV`, `biascorrect`, `skullstrip`, `sliceTiming`,
`motion`, `despike`, `coreg`, `sescoreg`, `T1wT2wCombined`, `T2w2T1w`,
`T2w2template`, `anat2template`, `func2anat`, `func2target`,
`surfReconTissueSeg`, `corticalSurfAndMeasures`, `atlasSegmentation`,
`atlasSurfaceProjection`, `tSNR`, `confounds`

Both families share one entity, which is why stacking a stage marker onto
`desc-preproc` was possible at all. The constructor now *replaces* rather than
appends, so the published name carries the stage — the `preproc` is implied,
every QC source being a preprocessed image.

### `space` — reference frames

`scanner`, `T2wScanner`, `T1w`, `bold`, `fsnative`, `base`, and the output-space
label (`NMT2Sym`, or literally `template` for a custom template file — see
`space_label_for()` in `utils/templates.py`).

`space-T1w` is **relative**: it means *this subject-session's own conformed T1w
grid*. That is why base-space products say `space-base` rather than `space-T1w` —
the latter would name two different grids identically.

### Other entities brainana writes

`from` / `to` / `mode` on transforms (`<prefix>_from-<src>_to-<dst>_mode-image_xfm.<ext>`
— `docs/spaces_and_transforms.rst` has the table of legal pairs); `atlas` on
parcellations; `hemi` on per-hemisphere surface maps; `stat` on statistical maps
(`stat-tsnr`); `res` on template resolutions.

---

## Part 3 — Declared deviations

Three things are not what the specification would give, and are kept **on
purpose**. Each is in `DECLARED_DEVIATIONS` with its reason;
`validate_output_name()` reports them as declared, never as failures.

**Do not "fix" one without changing its consumer first.**

### `_brain` suffix tail

`<prefix>_space-T1w_desc-preproc_T1w_brain.nii.gz` — a second token after the
BIDS suffix, so no BIDS parser can read it back, including brainana's own input
validator (`discover_bids_for_nextflow.py`, which takes the last `_`-token as the
suffix). Kept because `brainana-viewer`'s data contract names it in the
base-volume fallback chain, and `docs/outputs.rst` publishes it.

### Atlas names lead with `atlas-`, subject last

`atlas-ARM1_space-T1w_sub-01_ses-001.nii.gz` — 590 files in a dev-test run. Atlas
*discovery* matches on the leading `atlas-` token, in three places: the Viewer,
`utils/templates.py` (`discover_atlases_in_space`) and `steps/anatomical.py`
(`_atlas_name_from_filename`). Canonical BIDS order would break all three, across
two repos, and invalidate every existing output tree.

### Unknown raw entities pass through

Real dev-test data carries `test-xxx`. It is preserved verbatim, in its original
position, rather than dropped — dropping it would collapse two runs that differ
only by it onto one filename, which by layer (2) of Part 1 is a missing file.

Note the asymmetry this creates and live with it knowingly: the stem-preserving
helper keeps such an entity *in place*, while the parse-and-rebuild helper emits
it at the `others` slot. Both are stable and neither drops it.

---

## Part 4 — Update checklists

### ➕ Add a new published output

1. Pick the **slot** from the roles table. If the thing you want to say does not
   fit a slot, that is the finding — do not borrow an identity entity.
2. Reuse an existing **value** from Part 2 if one fits.
3. Build the name with `derive_output_name(bids_name, …)`. Do not concatenate,
   and do not name from a staged file.
4. Add the name shape to `docs/outputs.rst` (the round-trip test reads it).
5. `pytest tests/test_output_naming_contract.py tests/test_nextflow_script_bodies.py`
6. Run the pipeline, then
   `python scripts/check_output_naming.py <output_dir>`.

### ➕ Add a new `desc` value

1. Check Part 2 — a near-synonym of an existing value is worse than reusing it.
2. If the figure is a QC snapshot, add the mapping in
   `SNAPSHOT_MAPPINGS` and a caption in `FIGURE_DESCRIPTIONS`
   (`src/nhp_mri_prep/quality_control/reports.py`), or the report will not
   recognise it.
3. Pass it as `set_entities={'desc': …}`, never inside `suffix`.

### ➕ Add a new space

1. Add the label to `docs/spaces_and_transforms.rst`, including its `from-`/`to-`
   pairs if a transform is written.
2. If anything reads the frame back off a filename, make sure it splits on the
   `_space-` token rather than on a hard-coded value.

### ➕ Emit a new entity key

1. Add it to `ENTITY_ROLES` with its role.
2. Add it to `BIDS_ENTITY_ORDER` **at the position you actually publish it** —
   see Part 1.
3. `tests/test_output_naming_contract.py::test_every_entity_brainana_orders_has_a_role`
   fails if you do one and not the other.

### ✏️ Change an existing published name

1. Grep the whole repo *and* `brainana-viewer` for the old shape. Glob patterns
   in `path("…")` output declarations and `publishDir pattern:` are consumers: a
   glob that matches nothing makes a task fail, and a `publishDir` pattern that
   matches nothing silently publishes nothing.
2. Update `docs/outputs.rst`, and `brainana-viewer/docs/data-contract.md` if it
   names the shape.
3. Say so in `CHANGELOG.md` — an output rename invalidates existing trees.

---

## Part 5 — Automated guards

| Guard | Catches |
|---|---|
| `derive_output_name()` raising `NamingConflictError` | Setting an identity entity that the raw data already set differently; any duplicated key |
| `tests/test_output_naming_contract.py` | One case per defect class the audit found; every name in `docs/outputs.rst` validating **and** round-tripping; the roles and order tables agreeing |
| `tests/test_nextflow_script_bodies.py` (two pattern checks) | Splicing a `desc` into a stem by string replacement; taking a stem from anything but a naming template |
| `scripts/check_output_naming.py` | A real output tree, grouped by violation class, non-zero exit on anything undeclared |
| `modules/anatomical.nf` `validate_preproc_modality()` | The publish step's rewritten name not ending in a recognised suffix |

CI note: `.github/workflows/deps.yml` is the only workflow that runs pytest, and
its PR trigger is path-filtered. `modules/**`, `workflows/**` and `main.nf` are in
that filter — keep them there. Most output naming lives in the `.nf` process
bodies, and before they were added a change confined to them ran no tests at all.

---

## Part 6 — Gotchas

### `parse_bids_entities` is deliberately permissive

Its regex is `([a-zA-Z]+)-([a-zA-Z0-9-]+)`, unanchored, with no allow-list. It
therefore also harvests `from-`, `to-`, `mode-`, `desc-`, `space-` — anything
shaped like an entity. Consequences worth knowing: a value containing `_` or `.`
is truncated, and a **repeated key keeps the last value**.

### `get_bids_prefix()` session-level destroys almost everything

It keeps `sub`/`ses` and nothing else — not `task`, `run`, `acq`, `space`, or a
custom entity. That is correct for a genuinely session-level product and wrong
for anything else. Two consequences: pass a non-empty `run_identifier` for
per-run products, and take the frame separately for a base-space product.

### `space-T1w` on a base-space file was a real name

Base derivatives used to be `sub-X_acq-base_space-T1w_desc-brain_mask.nii.gz` —
`space-T1w` naming a frame the file is not in. If you find yourself writing a
frame that is only true "relative to something", that is the signal the frame
needs its own label.

### A `_boldref` input asked for modality `bold`

`create_bids_output_filename` used to `str.replace('_bold', '')` unbounded, so
`sub-01_ses-001_boldref` came back as `sub-01_ses-001ref`. It now strips a
recognised trailing suffix token, longest-first. The general lesson: an unbounded
`str.replace` on a filename will eventually eat a substring of something else.

### Two output slots, one glob

`FUNC_APPLY_TRANSFORMS` declared `path("*space-*desc-*.nii.gz")` twice, for two
different outputs. Nextflow gives **both** slots the entire sorted match list, and
`..._bold.nii.gz` sorts before `..._boldref.nii.gz` — so the consumer that asked
for the boldref received the 4-D BOLD. If two output slots can match the same
files, their globs need to differ.

### `-resume` will not re-run a task whose Python changed

Nextflow's cache key is the process *script* — the text of the `.nf` body — plus
its inputs. It does not hash the Python modules that body imports. So a naming
change made inside `src/` leaves every cached task looking valid, and `-resume`
reuses the old output with the old name. A naming change has to be verified by a
**clean run into a fresh output directory**.

Two corollaries. `publishDir` copies and never cleans, so a stale name from an
earlier run sits in the output tree next to the new one — an output directory is
only trustworthy if it was built from empty. And the check that actually catches a
position change is a **before/after diff of two trees**, not the validator alone:
`scripts/check_output_naming.py` reports a name's shape, and it happily passed
`atlas-ARM1_sub-01_space-scanner.nii.gz` until the atlas shape check was added.

### `replace_bids_space` appends when it finds no anchor

It strips every `_space-*` then re-inserts at a BIDS anchor: before `_desc-`, else
before a trailing modality suffix, else **appended**. An atlas name has neither
anchor, so routing the scanner-space rename through it turned
`atlas-ARM1_space-T1w_sub-01.nii.gz` into `atlas-ARM1_sub-01_space-scanner.nii.gz`
— which no longer matches `atlas-<name>_space-*`, the shape every atlas consumer
discovers by. Use it only on names that have one of its anchors;
`_rename_to_scanner_space()` in `steps/anatomical.py` replaces in place instead.

Two general lessons. A helper that re-inserts at a canonical position will *move*
a token in any name whose convention is not canonical — and this repo has such a
convention on purpose (Part 3). And a regex value class that admits `.`
(for `res-0.5`) will swallow `.nii.gz` when applied to a full filename rather than
a stem; `_FRAME_TOKEN` excludes it for that reason.

### Groovy has its own parsers

`workflows/channel_helpers.groovy` reimplements entity parsing, extension
stripping and `space-` extraction, and `param_resolver.groovy` + `main.nf`
duplicate custom-template detection. They agree with the Python ones on current
inputs and share no definition. If you change a parsing rule, grep `*.groovy` too.
