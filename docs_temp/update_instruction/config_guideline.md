# Configuration System — Reference & Update Guide

A single place to understand how config works in brainana and exactly what to
touch when you add, change, or remove a parameter. Read Part 1 once; use the
checklists in Part 3 every time you make a change.

> **Golden rule:** `src/nhp_mri_prep/config/defaults.yaml` is the source of truth.
> Every other surface (validator, generator UI, docs) must agree with it, and
> `tests/test_config_consistency.py` fails the build if they drift.

---

## Part 1 — The full picture

### Two config layers

| Layer | Source of truth | Holds | Read by |
|---|---|---|---|
| **Runner / Nextflow** | `nextflow.config` `params { }` block | I/O paths, CLI-overridable options, GPU & executor tuning | `main.nf`, `workflows/*.nf`, `run_brainana.sh`, `entrypoint.sh` |
| **Scientific / pipeline** | `src/nhp_mri_prep/config/defaults.yaml` | every processing parameter (anat/func/registration/…) | the Python ops + the Groovy resolver |

### The five surfaces that must stay in sync

1. **Definition** — `src/nhp_mri_prep/config/defaults.yaml` (value + inline `# type: …` comment).
2. **Validation (Python)** — `src/nhp_mri_prep/config/config_validation.py` (allowed values / types / ranges).
3. **Resolution (Groovy)** — `workflows/param_resolver.groovy`, `workflows/config_helpers.groovy` (priority merge + effective-config generation).
4. **Generation UI** — `docs/_static/config_generator.html` (interactive form users download a YAML from).
5. **Docs** — `docs/usage_notes.rst` (CLI args), `docs/processing.rst` (prose), and this `docs_temp/` reference.

Plus the **drift guard**: `tests/test_config_consistency.py`.

> There is intentionally **no `nextflow_schema.json` / nf-schema plugin** today.
> The "schema" role is the four surfaces above kept in sync by hand + the test.
> (Adopting nf-schema is the long-term way to collapse surfaces 1–4 into one —
> see "Future direction" at the end.)

### How a parameter flows at runtime

```
CLI  --output_space X          nextflow.config params{}  (declares CLI params, default null)
user --config_file my.yaml            │
defaults.yaml  ────────────────┐      │
                               ▼      ▼
        workflows/param_resolver.groovy
          generateEffectiveConfig()  =  defaults.yaml  ◀ user YAML ◀ CLI --param
                               │   (priority: CLI > user YAML > defaults.yaml)
                               ▼
        <output_dir>/nextflow_reports/config.yaml   ← the single "effective config"
                               │
   ┌───────────────────────────┼───────────────────────────┐
   ▼                           ▼                            ▼
discover_bids_for_nextflow.py  each Nextflow process        (reports)
  validate_config(effective)   load_config('${config_file}')
  → HARD FAIL on bad values     → passes config= into the op functions
```

Key facts that follow from this diagram:

- **Validation runs once, early**, in `discover_bids_for_nextflow.py` (right after
  the config is loaded), and hard-fails with a friendly message. This is the only
  place bad user values are caught before compute.
- **Every process passes `config=` explicitly** into the operation functions, so
  user overrides always take effect. `get_config()` returns **defaults only** and
  is just a fallback for direct Python API use — never rely on it inside the
  pipeline; pass the loaded config.
- The **effective config** written to `nextflow_reports/config.yaml` is the exact,
  fully-resolved settings a run used (good for debugging / reproducibility).

### CLI-overridable vs YAML-only parameters

Most parameters are **YAML-only** (set them in a config file). A small set are also
**CLI-overridable**, wired through `PARAM_MAPPING` in `param_resolver.groovy`:

| CLI flag | YAML key |
|---|---|
| `--output_space` | `template.output_space` |
| `--anat_only` | `general.anat_only` |
| `--subjects` | `bids_filtering.subjects` |
| `--sessions` | `bids_filtering.sessions` |
| `--tasks` | `bids_filtering.tasks` |
| `--runs` | `bids_filtering.runs` |

Workflow-only params (not in YAML) live only in `nextflow.config` `params{}`:
`bids_dir`, `output_dir`, `config_file`, `work_dir`,
`gpu_count`, `max_jobs_per_gpu`, `python_exe`.
(Executor limits are **env vars**, not params: `NXF_MAX_CPUS`, `NXF_MAX_MEMORY`.)

---

## Part 2 — Key files at a glance

| File | Role |
|---|---|
| `src/nhp_mri_prep/config/defaults.yaml` | **Source of truth.** Every YAML param + inline doc. |
| `src/nhp_mri_prep/config/config_validation.py` | Allowed-value/type/range checks. `validate_config()` dispatches to per-section validators. |
| `src/nhp_mri_prep/config/config_io.py` | Load/merge/save YAML+JSON; `get_default_config()`, `load_config()`, `_deep_merge()`. |
| `src/nhp_mri_prep/config/config.py` | `Config` class; `get_config()` (defaults-only global), `set_config()` (seed user config for direct API use). |
| `src/nhp_mri_prep/utils/nextflow.py` | `load_config()` used inside every process (raw YAML load). |
| `workflows/param_resolver.groovy` | Priority merge, type coercion, `generateEffectiveConfig()`, `PARAM_MAPPING`. |
| `nextflow.config` | `params{}`, executor/profiles, process resources. |
| `src/nhp_mri_prep/nextflow_scripts/discover_bids_for_nextflow.py` | Pre-flight: BIDS structure check + `validate_config()` + job discovery. |
| `docs/_static/config_generator.html` | Interactive config builder (mirrors `defaults.yaml`). |
| `docs/usage_notes.rst` / `docs/processing.rst` | User-facing CLI + processing docs. |
| `tests/test_config_consistency.py` | Drift guard: defaults validate + generator covers every default key. |

---

## Part 3 — Update checklists

### ➕ Add a new YAML parameter (`section.subsection.key`)

1. **`defaults.yaml`** — add the key with a value and an inline comment stating
   type/range, e.g. `dof: 6  # int: degrees of freedom (6, 9, or 12)`.
2. **`config_validation.py`** — add a check in the matching `validate_*` function
   (create one and register it in `validate_config()` if the section is new).
   Enum-like or typed params **must** be validated (that's what the audit's F5 was).
3. **`config_generator.html`** — update **all three** sync points (or deliberately
   skip; see "internal parameters" below):
   - the **form field** (an `<input>`/`<select>` with `id="section_key"`),
   - the **embedded default object** (the JS `defaultConfig` literal),
   - the **JS field-collection** block (`config.section.key = document.getElementById('…').…`).
4. **Consumer code** — read it defensively: `config.get("section.key", <default>)`
   (Python `Config`) or `reg_config.get("key", <default>)` (plain dict in a process).
5. **Docs** — if user-facing, mention it in `docs/processing.rst` (prose) and, if
   CLI-overridable, in `docs/usage_notes.rst`.
6. **Run the guard:** `pytest tests/test_config_consistency.py`.

### ➕ Also make it CLI-overridable (`--my_param`)

Do everything above, **plus**:

7. **`nextflow.config`** — declare it in `params { }` (default `null`).
8. **`param_resolver.groovy`** — add `'my_param': 'section.key'` to `PARAM_MAPPING`.
9. **Workflow `.nf`** — read it via the resolver: `getParam(params,'my_param')`,
   `getParamBool(...)`, `getParamList(...)`, etc.
10. **`run_brainana.sh` / `entrypoint.sh`** — only if you need an alias
    (e.g. `--work-dir` ↔ `--work_dir`) or entrypoint-level handling.
11. **`docs/usage_notes.rst`** — add it to the Command-line arguments section.

### ✏️ Change an existing parameter

- **New default value:** edit `defaults.yaml` **and** the generator's embedded
  default + the field's shown value. (Consumer `.get(key, X)` fallbacks should
  match the new default too.)
- **New allowed values / range:** edit the enum/range in `config_validation.py`
  **and** the generator control (`<select>` options, `min`/`max`). Confirm
  `defaults.yaml`'s value still validates (`test_defaults_validate_clean`).
- **Rename a key:** treat as remove-old + add-new across all surfaces, and grep
  consumers: `grep -rn "old_key" src/ modules/ workflows/ docs*/`.

### ➖ Remove a parameter

Delete it from: `defaults.yaml`, its validator block, the generator's three sync
points, the docs, and every consumer (`grep -rn "the_key"`). Run the guard test.

### 🔒 Internal parameter (advanced, **not** user-exposed)

Some knobs should exist for developers but not be advertised (e.g.
`registration.fireants_allow_cpu`). Convention:

- **Do NOT** put it in `defaults.yaml`, the generator, or user docs.
- Give it a hardcoded default at the read site: `reg_config.get("key", <default>)`.
- Optionally keep a defensive check in `config_validation.py` so a power user who
  sets it in raw YAML still gets validated.

This keeps `test_config_consistency.py` green (it only requires *defaults.yaml*
keys to appear in the generator) while leaving an escape hatch.

---

## Part 4 — The validation model

- `validate_config(config)` (in `config_validation.py`) merges the user config
  over defaults, then dispatches to per-section validators
  (`validate_func_config`, `validate_anat_config`, `validate_registration_config`,
  `validate_motion_correction_config`, …). Each raises `ValueError`/`TypeError`
  with the house-style message ending in *"Please fix this in your configuration file."*
- It is invoked **hard-fail** in `discover_bids_for_nextflow.py` (pre-flight) and
  also inside the `Config` class (`__init__`/`update`/`validate`) for direct API use.
- Unknown/optional sections (`quality_control`, `orientation_mismatch_correction`)
  are only validated **if present** — validators never reject unknown keys, so
  adding a new section to defaults is safe.
- The Groovy resolver (`getYamlBool/Int/Float/List`) validates **soft** (warns +
  falls back to default). That's fine because Python validation already hard-failed
  on the same effective config upstream. Don't rely on Groovy for hard validation.

**To add a new allowed-value check:** find/creat the section's `validate_*`
function, raise with the standard message, and add a rejection case to the
`test_validators_reject_bad_values` parametrize list in the consistency test.

---

## Part 5 — The drift guard (`tests/test_config_consistency.py`)

Run it after any config change:

```bash
source .venv/bin/activate            # env with torchio etc.
pytest tests/test_config_consistency.py -q
```

It asserts:

- **`test_defaults_validate_clean`** — `defaults.yaml` passes its own validator.
- **`test_generator_covers_every_default_key`** — every leaf key in `defaults.yaml`
  appears somewhere in `config_generator.html`. (This is what would have caught the
  missing `func.confounds` and `fireants_allow_cpu` fields.)
- **`test_validators_reject_bad_values`** — enum/typed params actually get rejected.

**When it fails:**
- *"generator is missing … X"* → either add X to the generator (all 3 sync points),
  or, if X is intentionally internal, remove it from `defaults.yaml` and give it a
  code default (see Part 3, "internal parameter").
- *"defaults … validate"* → your new validator is stricter than the shipped default,
  or the default value is wrong. Fix one of them.

---

## Part 6 — Gotchas & conventions

- **`build/` and `docs/_build/` are gitignored** — they're regenerated artifacts.
  Never edit config there; a stale copy under `build/lib/.../defaults.yaml` is just
  a local build output, not repo state.
- **`gpu_device` has two scopes.** `general.gpu_device` is the **global** policy
  (`main.nf` reads it to decide if GPU scheduling is on at all; set `-1`/`"cpu"` to
  force CPU everywhere). Per-model knobs like
  `anat.skullstripping_segmentation.fastSurferCNN.gpu_device` pick a device *within*
  that policy.
- **`output_space` is validated twice, differently:** Groovy `validateOutputSpace`
  checks **format** (regex), while `utils/templates.py` checks **existence** against
  the actual template zoo at resolve time. A value can pass the format check and
  still fail later if that template isn't installed.
- **No unknown-param detection.** A misspelled `--anat_ony` is silently ignored by
  Nextflow. Double-check flag spelling; there's no guard for this yet.
- **Effective config** is at `<output_dir>/nextflow_reports/config.yaml` — inspect
  it to see exactly what a run resolved to.

---

## Future direction (optional, larger change)

To eliminate hand-syncing entirely, promote `defaults.yaml` to a real schema (typed
comments or a sibling `nextflow_schema.json`) and **generate** the other surfaces
from it: the generator fields, a published "Parameters" doc page, and the validator's
allowed-values. Adopting **nf-schema** (`validateParameters()`) would additionally
give typed validation and unknown-parameter detection at the Nextflow boundary. Until
then, the checklists above + the drift test are the contract.
