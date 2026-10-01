# brainana Parameters — Wiki & Reference

**Canonical, single-source reference for every input brainana accepts** — Docker
entrypoint flags, local/CLI flags, BIDS-discovery arguments, Nextflow `params.`*, and
environment variables — plus how they are resolved, the connector convention, known
footguns, and the roadmap for making the parameter surface more standard and robust.

> This document supersedes the old split between `ARGUMENT_REFERENCE.md` (what you can
> pass) and `PARAMETER_MANAGEMENT.md` (how it is resolved). `ARGUMENT_REFERENCE.md` now
> points here. Keep this file in sync with the code — see
> [§11 Where arguments are defined in code](#11-where-arguments-are-defined-in-code).

---

## 1. Architecture: the three invocation layers

A brainana run passes through up to three layers. Which layer *consumes* a flag matters,
because it determines the flag's naming rules and whether a typo is caught.

```
 docker run … image [input] [output] [flags]
        │
        ▼
 ┌─────────────────┐   entrypoint-only flags (‑w/‑‑work_dir, ‑‑config, ‑‑no_resume,
 │  entrypoint.sh  │   ‑‑freesurfer_license) consumed here; sets up user/GPU/FS_LICENSE,
 │  (Docker only)  │   rebuilds a canonical call to run_brainana.sh.
 └────────┬────────┘
          │  ./run_brainana.sh run main.nf --bids_dir … --output_dir … --config_file … [extra]
          ▼
 ┌─────────────────┐   normalizes flag connectors (hyphen→underscore), resolves config,
 │ run_brainana.sh │   runs BIDS discovery, then launches Nextflow. Local runs enter HERE
 │ (Docker+local)  │   directly (no positionals; everything named).
 └────────┬────────┘
          │            ┌────────────────────────┐
          ├───────────▶│  BIDS discovery (py)   │  a SUBPROCESS spawned by run_brainana.sh
          │            │  discover_bids_*.py    │  (NOT a layer). reads bids_dir, output_dir,
          │            └────────────────────────┘  config_file, subjects/sessions/tasks/runs
          ▼
 ┌─────────────────┐   receives params.* ; param_resolver.groovy merges CLI > YAML > defaults
 │    Nextflow     │   into an effective config used by every process.
 │   (main.nf)     │
 └─────────────────┘
```

The three invocation layers are `**entrypoint.sh` → `run_brainana.sh` → Nextflow**. BIDS
discovery is **not** a fourth layer — it is a helper *subprocess* that `run_brainana.sh`
spawns before launching Nextflow, and it only *reads* a subset of the already-parsed args.

**What Docker accepts.** The Docker form is
`docker run … image [input_dir] [output_dir] [flags]` (the two positionals default to `/input`
`/output`). `entrypoint.sh` consumes **only** the four flags shown above and forwards
*everything else unchanged*, so **every §3.1 param also works from Docker** — e.g.
`--output_space /tpl.nii.gz` — as do Nextflow natives like `-resume` / `-profile`. There is
nothing Docker-specific about those flags; they pass straight through to Nextflow.

**Core constraint.** Any flag that survives into Nextflow as `params.X` must use
`snake_case` (underscore). Nextflow maps `--foo_bar` → `params.foo_bar` (clean Groovy
dotted access), whereas `--foo-bar` → `params.'foo-bar'` (awkward quoted access). This is
why the pipeline's canonical connector is the underscore — see [§4](#4-connector-convention).

---

## 2. Parameter resolution priority

Parameters are resolved highest-to-lowest by `workflows/param_resolver.groovy`:

1. **Command-line arguments** (`--param value`) — highest priority.
2. **User YAML config** (`--config` / `--config_file`) — medium.
3. `**defaults.yaml`** (`src/nhp_mri_prep/config/defaults.yaml`) — lowest.

`nextflow.config` no longer carries default *values* for anything settable via YAML; the
defaults live in `defaults.yaml`. `param_resolver` generates one **effective config** at run
start (written to `<output_dir>/nextflow_reports/config.yaml`) that every process reads.

---

## 3. Complete flag & parameter inventory

### 3.1 Nextflow pipeline params (canonical: underscore)

These reach Nextflow as `params.X`. All are `snake_case`. Column **"Read by"** = which reader
consumes the value — `D` = the BIDS-discovery subprocess, `N` = Nextflow. This is *which reader
uses the value*, not an invocation layer (discovery is a subprocess of `run_brainana.sh`, see
[§1](#1-architecture-the-three-invocation-layers)). Every flag here is also accepted from Docker.


| Flag / param     | Default                                         | Read by | YAML key (if mapped)      | Description                                                                                            |
| ---------------- | ----------------------------------------------- | ------- | ------------------------- | ------------------------------------------------------------------------------------------------------ |
| `--bids_dir`     | (required)                                      | D, N    | —                         | BIDS dataset root. In Docker, set from positional `input_dir`.                                         |
| `--output_dir`   | (required)                                      | D, N    | —                         | Output directory. In Docker, set from positional `output_dir`.                                         |
| `--config_file`  | `defaults.yaml`                                 | D, N    | —                         | YAML config path. `--config` is an alias (see §3.2).                                                   |
| `--work_dir`     | script dir (local) / `<output_dir>_wd` (Docker) | N       | —                         | Nextflow launch/work dir (`.nextflow/`, resume cache).                                                 |
| `--output_space` | (from YAML)                                     | N       | `template.output_space`   | Output template space, e.g. `NMT2Sym:res-1`, `T1w`, **or a path to a custom template** `.nii/.nii.gz`. |
| `--anat_only`    | (from YAML)                                     | N       | `general.anat_only`       | Run only the anatomical pipeline. Boolean-like.                                                        |
| `--subjects`     | (all)                                           | D, N    | `bids_filtering.subjects` | Restrict to subject ID(s).                                                                             |
| `--sessions`     | (all)                                           | D, N    | `bids_filtering.sessions` | Restrict to session ID(s).                                                                             |
| `--tasks`        | (all)                                           | D, N    | `bids_filtering.tasks`    | Restrict to task name(s) (functional).                                                                 |
| `--runs`         | (all)                                           | D, N    | `bids_filtering.runs`     | Restrict to run number(s).                                                                             |


> **GPU control.** The only user-facing GPU switch is the YAML key `**general.gpu_device*`*:
> `auto` (use a GPU if one is detected) or `cpu` / `-1` (force CPU-only). Everything else
> GPU-related is derived automatically — see the internals note below.

> **Auto-detected internals (not user-set).** These `params.*` exist and are tolerated by the
> unknown-flag guard, but you do **not** pass them; they are computed at config-parse or
> recomputed each run:
>
> - `gpu_count` — detected via `nvidia-smi`.
> - `max_jobs_per_gpu` — derived from free VRAM (~4 GiB/job, bounded 1–4).
> - `use_gpu` — not a real param (kept in `known_flags.txt` only so an old `--use_gpu` does not
> abort). GPU scheduling is derived per workflow by `paramResolver.resolveUseGpu(params)` from
> `gpu_count` + `general.gpu_device`; **a value you pass is ignored**. Never declare it in
> `nextflow.config` `params{}` — config params are read-only to the script, which pinned every
> GPU process to CPU.
> - `python_exe` — the active venv/conda `python3` (else bare `python3`), for config-parse
> helper calls only.

### 3.2 Bash-wrapper flags (consumed before Nextflow)

These are intercepted by `entrypoint.sh` and/or `run_brainana.sh`. Canonical form is
underscore; the hyphenated spellings are accepted aliases.


| Flag (canonical)       | Aliases                | Consumed by                                      | Forwarded to Nextflow?                                      | Description                                                                                |
| ---------------------- | ---------------------- | ------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `--config`             | `--config_file`        | entrypoint + run_brainana.sh                     | as `--config_file`                                          | YAML config path. Collapsed to one `--config_file` before Nextflow.                        |
| `--work_dir`           | `-w`, `--work-dir`     | entrypoint (`-w`) + run_brainana.sh              | yes (`--work_dir`)                                          | Nextflow work/launch dir.                                                                  |
| `--no_resume`          | `--no-resume`          | entrypoint                                       | no (converts to omitting `-resume`)                         | Disable resume; run from scratch. Docker adds `-resume` by default.                        |
| `--no_docker`          | `--no-docker`          | run_brainana.sh                                  | no (sets `NXF_NO_DOCKER`)                                   | Local dev: don't run Nextflow tasks under Docker.                                          |
| `--freesurfer_license` | `--freesurfer-license` | entrypoint + run_brainana.sh (sets `FS_LICENSE`) | no (consumed as `FS_LICENSE` env, stripped before Nextflow) | FreeSurfer license path; required when surface reconstruction is on.                       |
| `-h`, `--help`         | —                      | entrypoint + run_brainana.sh                     | no (prints usage, exits 0)                                  | Print the full argument listing (from `USAGE.txt`) and exit before any GPU/Nextflow setup. |


Special Docker first-argument: `bash` / `sh` / `-bash` / `-sh` → the entrypoint execs an
interactive shell instead of the pipeline.

**Help text (`--help`) source.** The runtime usage listing lives in a single file,
`**USAGE.txt`** at the repo root (`/opt/brainana/USAGE.txt` in the image). Both wrappers
`cat` it on `-h`/`--help`, and the unknown-flag guard in `main.nf` embeds it in its error
(see [§5](#5-unknown-flag-behavior)). The machine allowlist is `known_flags.txt` (loaded by
`flags.sh` and by `main.nf`); keep `USAGE.txt`, `known_flags.txt`, the `params{}` block in
`nextflow.config`, and `docs/usage_notes.rst` in sync when adding or renaming a flag.

### 3.3 Nextflow native options (single dash — pass through)

Not brainana params; handled by Nextflow itself: `-resume` (Docker adds by default unless
`--no_resume`), `-profile <name>` (e.g. `minimal`, `recommended`; see `nextflow.config`
profiles), `-log`, `-C` (both set internally by the wrappers). Single-dash options are
never touched by the connector normalizer.

### 3.4 CLI → YAML mapping

From `workflows/param_resolver.groovy` (`PARAM_MAPPING`). Only params whose YAML path
differs from the flag name need an entry:


| CLI param      | YAML key                  |
| -------------- | ------------------------- |
| `output_space` | `template.output_space`   |
| `anat_only`    | `general.anat_only`       |
| `subjects`     | `bids_filtering.subjects` |
| `sessions`     | `bids_filtering.sessions` |
| `tasks`        | `bids_filtering.tasks`    |
| `runs`         | `bids_filtering.runs`     |


### 3.5 Environment variables


| Variable               | Set by                                 | Description                                           |
| ---------------------- | -------------------------------------- | ----------------------------------------------------- |
| `NXF_WORK`             | entrypoint / run_brainana.sh           | Nextflow work dir (process scratch).                  |
| `NXF_HOME`             | entrypoint / run_brainana.sh           | Nextflow home (cache, history).                       |
| `NXF_LAUNCH_DIR`       | entrypoint / run_brainana.sh           | Dir Nextflow launches from (`.nextflow/` lives here). |
| `NXF_MAX_CPUS`         | Docker env (default `8`)               | Executor CPU cap. **Not a CLI param.**                |
| `NXF_MAX_MEMORY`       | Docker env (default `20g`)             | Executor memory cap. **Not a CLI param.**             |
| `NXF_NO_DOCKER`        | `--no_docker` sets it                  | Disable Docker for Nextflow tasks.                    |
| `NXF_ANSI_LOG`         | optional (`true` default)              | Set `false` to disable colored logs.                  |
| `FS_LICENSE`           | entrypoint from `--freesurfer_license` | FreeSurfer license path.                              |
| `CUDA_VISIBLE_DEVICES` | entrypoint (emptied when no GPU)       | Forces CPU fallback when no GPU is accessible.        |
| `DISPLAY`              | entrypoint (`:99` via Xvfb)            | Virtual display for QC snapshots.                     |


> **Executor limits are env-only.** Older docs listed `--max_cpus` / `--max_memory` as CLI
> params — that was incorrect. Set `NXF_MAX_CPUS` / `NXF_MAX_MEMORY` (e.g.
> `docker run -e NXF_MAX_CPUS=8 …`).

---

## 4. Connector convention

**Canonical connector: underscore (`snake_case`).** Every user-facing flag has an
underscore canonical spelling. For backward compatibility the historical hyphenated
spellings are accepted as **aliases**.

- Flags that reach Nextflow *must* be underscore (Groovy constraint, [§1](#1-architecture-the-three-invocation-layers)).
- Bash-only flags historically used GNU hyphens; they now accept both.

**How aliasing works.** `flags.sh` (sourced by `run_brainana.sh` and `entrypoint.sh`)
provides an **allowlist-driven normalizer** (`normalize_flag` + `KNOWN_FLAGS`, loaded from
`known_flags.txt`). Before args reach Nextflow it rewrites `--foo-bar` →
`--foo_bar`, but **only when `foo_bar` is a known flag name**. This is the single choke
point both the Docker and local paths pass through. `entrypoint.sh` additionally recognizes
the underscore spellings of the four flags it consumes for its own logic.

Guarantees:

- Unknown tokens (e.g. a typo `--anat-onlyy`) pass through **verbatim** — they are *not*
silently "corrected", so they remain detectable (see [§5](#5-unknown-flag-behavior)).
- Only the flag **name** (the part before `=`) is rewritten. Values such as
`--output_space NMT2Sym:res-1` or paths like `/a-b/c.nii.gz` are never altered.
- Both `--flag value` and `--flag=value` forms are handled.
- Single-dash options (`-w`, `-resume`, `-profile`) are never rewritten.

Result: `--work-dir /w --anat-only true --output-space X` and
`--work_dir /w --anat_only true --output_space X` produce a byte-identical Nextflow command.

---

## 5. Unknown-flag behavior

**Why this needs a guard:** Nextflow accepts *any* `--foo bar` and turns it into an unused
`params.foo` — it has no built-in rejection of unknown params. So a wrong flag **name**
would never error; it would be silently ignored.

Real example (the motivating bug): `--custom-template /tpl.nii.gz` is not a real flag (the
correct one is `--output_space /tpl.nii.gz`). It silently became an unused param (Nextflow
maps `--custom-template` to `params.customTemplate`), was never read, and the run completed
"successfully" using the *default* template instead — an 11-minute run that did the wrong
thing without any warning.

**Now — validation at every layer, from one allowlist.** Unrecognized arguments are
rejected at **three** points, all reading the **single** allowlist file
`**known_flags.txt`** (repo root; `/opt/brainana/known_flags.txt` in the image):

1. `**entrypoint.sh` (Docker, fail-fast):** after arg parsing and before the FS-license
  probe / GPU / Xvfb, `validate_flags` checks the forwarded flags. A typo is rejected in
   **~1 second**, before any heavy setup.
2. `**run_brainana.sh` (local + Docker):** after connector normalization and **before BIDS
  discovery / Nextflow**, `validate_flags` runs again. Rejecting here means **no discovery
   runs and no partial QC report is written**.
3. `**main.nf` (backstop):** the Nextflow-layer guard still runs, covering direct
  `nextflow run main.nf` invocations that bypass the wrappers.

The error is a standard, argparse-style message: the offending flag(s), the **full argument
listing** (the same `USAGE.txt` text as `--help`), and a hint about the common
`--output_space`/`--custom-template` mistake. In `main.nf` it also renders readably in
`run_status.json` and the QC "Run status" HTML `<pre>`.

Layers 1–2 use `flags.sh` (`KNOWN_FLAGS` + `normalize_flag` + `validate_flags` +
`print_usage`); layer 3 reads `known_flags.txt` directly in Groovy. All three **fail-open**:
if the allowlist file is missing they skip validation (warn) rather than break valid runs.
The connector normalizer ([§4](#4-connector-convention)) does not mask typos — it only
rewrites *known* names, so an unknown token like `--custom-template` reaches the guard
verbatim. Run `--help` any time to see the listing without triggering an error.

> **Single source of truth.** `known_flags.txt` is the machine allowlist. When adding or
> renaming a flag, update `**known_flags.txt`** together with the `params{}` block in
> `nextflow.config` (the actual param declaration) and `**USAGE.txt**` (the human `--help`).

Contrast: an unknown **value** for a *known* flag is also caught — e.g. a malformed
`--output_space` or a nonexistent custom-template path fails fast in `main.nf`.

---

## 6. Using parameters in Nextflow (resolver API)

### Initialization

```groovy
// main.nf
def paramResolver = evaluate(new File("${projectDir}/workflows/param_resolver.groovy").text)

workflow {
    paramResolver.initialize(params, projectDir)
    def output_space = paramResolver.getParamOutputSpace(params, 'output_space')
    def anat_only    = paramResolver.getParamBool(params, 'anat_only', false)
}
```

### CLI params (CLI → YAML → default)

```groovy
def output_space = paramResolver.getParamOutputSpace(params, 'output_space')   // with format/path validation
def anat_only    = paramResolver.getParamBool(params, 'anat_only', false)
def subjects     = paramResolver.getParamList(params, 'subjects', null)
def shrink       = paramResolver.getParamInt(params, 'some_int', 2, 1, 10)      // value, default, min, max
```

### YAML-only params (not exposed on the CLI)

```groovy
def reorient   = paramResolver.getYamlBool("func.reorient.enabled", true)
def xfm_type   = paramResolver.getYamlString("registration.anat2template_xfm_type", "syn")
def shrink_fac = paramResolver.getYamlInt("anat.bias_correction.shrink_factor", 2, 1, null)
```

Available getters: `getParam`, `getParamBool`, `getParamInt`, `getParamFloat`,
`getParamList`, `getParamOutputSpace`; `getYamlParam`, `getYamlBool`, `getYamlString`,
`getYamlInt`, `getYamlFloat`, `getYamlList`; plus `generateEffectiveConfig`.

### Validation

The resolver validates as it reads: `output_space` (format `T1w` / `NAME` / `NAME:DESC`, or
a `.nii/.nii.gz` path), booleans (`true/false`, `1/0`, `yes/no`, `on/off`), ints/floats
(numeric + optional range), and lists. `main.nf` additionally fail-fasts on a custom-template
`output_space` whose extension is wrong or whose file is missing. Invalid values abort the
run with a descriptive message.

---

## 7. Improvement roadmap

Tracks work to make the parameter surface more standard and robust. Status as of this
revision.

1. **Connector unification — DONE.** Underscore canonical + hyphen aliases via the
  allowlist normalizer in `run_brainana.sh` and alias recognition in `entrypoint.sh`.
   User docs (`docs/usage_notes.rst`, `docs/faq.rst`) updated. See [§4](#4-connector-convention).
2. **Unknown-flag guard — DONE.** `main.nf` (after `paramResolver.initialize`) errors on any
  `params.X` not in a canonical `KNOWN_PARAMS` set, listing the known params and hinting at
   the `--output_space` mistake. Catches `--custom-template`. Composes with item 1 because
   the normalizer leaves unknown tokens verbatim. See [§5](#5-unknown-flag-behavior).
3. `**--custom-template` typo fix — DONE.** `scripts/scratch/test_brainana_docker.sh` now
  uses `--output_space /custom_template.nii.gz`.
4. `**freesurfer_license` forwarding — DONE.** No longer forwarded to Nextflow: `entrypoint.sh`
  consumes it into `FS_LICENSE` without adding it to the forwarded args, and `run_brainana.sh`
   strips it (setting `FS_LICENSE` for local runs) before launching Nextflow. It remains in the
   guard allowlist as a defensive tolerance.
5. **Executor limits — DOC-ONLY.** `max_cpus`/`max_memory` are env-only
  (`NXF_MAX_CPUS`/`NXF_MAX_MEMORY`); documented in [§3.5](#35-environment-variables).
   Could be exposed as flags later if there's demand.

---

## 8. Examples

### CLI overrides YAML

```bash
# config.yaml → template.output_space: "NMT2Sym:res-1"
./run_brainana.sh run main.nf --output_space "NMT2Sym:res-05" --config_file config.yaml
# Result: uses "NMT2Sym:res-05" (CLI wins)
```

### YAML overrides defaults

```bash
# config.yaml → general.anat_only: true ; func.motion_correction.enabled: false
./run_brainana.sh run main.nf --config_file config.yaml
```

### Custom template (correct usage)

```bash
docker run --rm --gpus all \
  -v /data/bids:/input -v /data/out:/output \
  -v /fs_license.txt:/fs_license.txt \
  -v /data/tpl.nii.gz:/custom_template.nii.gz \
  liuxingyu987/brainana:<version> /input /output \
  --work_dir /output/wd \
  --freesurfer_license /fs_license.txt \
  --output_space /custom_template.nii.gz          # NOT --custom-template
```

### Connector aliases are interchangeable

```bash
# These two are equivalent:
./run_brainana.sh run main.nf ... --work-dir /w --anat-only true --output-space X
./run_brainana.sh run main.nf ... --work_dir /w --anat_only true --output_space X
```

---

## 9. Troubleshooting


| Symptom                                            | Likely cause                                                            | Fix                                                                                                                            |
| -------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| A flag "does nothing" / wrong default used         | Unknown flag **name** silently ignored ([§5](#5-unknown-flag-behavior)) | Check spelling against [§3](#3-complete-flag--parameter-inventory); e.g. use `--output_space <file>`, not `--custom-template`. |
| Custom template ignored                            | Passed via a non-existent flag                                          | Pass the file path as `--output_space /path.nii.gz`.                                                                           |
| "Invalid output_space" / "template file not found" | Bad `output_space` value or path                                        | Use `T1w` / `NAME[:DESC]`, or a valid `.nii/.nii.gz` path.                                                                     |
| Executor ignores `--max_cpus`/`--max_memory`       | Those are env-only                                                      | Set `NXF_MAX_CPUS` / `NXF_MAX_MEMORY`.                                                                                         |
| License error though `--freesurfer_license` passed | File not mounted / path mismatch                                        | Mount the license and match the in-container path.                                                                             |
| Resume not working                                 | No persistent work dir                                                  | Pass `--work_dir` (Docker: mount it) so `.nextflow/` persists.                                                                 |


---

## 10. Dry-run verification

`run_brainana.sh` supports a dry-run that prints the exact assembled Nextflow command and
exits without running discovery or Nextflow — useful to confirm two spellings resolve
identically:

```bash
BRAINANA_DRY_RUN=1 ./run_brainana.sh run main.nf --bids_dir /b --output_dir /o --work-dir /w --output-space X
BRAINANA_DRY_RUN=1 ./run_brainana.sh run main.nf --bids_dir /b --output_dir /o --work_dir /w --output_space X
# diff the two outputs → must be byte-identical
```

---

## 11. Where arguments are defined in code


| Layer                          | File(s)                                                                                                                                                                                                  |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Docker entrypoint              | `entrypoint.sh` — positionals; consumes `--config`/`--config_file`, `-w`/`--work_dir`/`--work-dir`, `--no_resume`/`--no-resume`, `--freesurfer_license`/`--freesurfer-license`; rebuilds canonical call. |
| Shared flag helpers | `flags.sh` (repo root) — `KNOWN_FLAGS` (from `known_flags.txt`), `normalize_flag`, `validate_flags`, `print_usage`; sourced by `entrypoint.sh` + `run_brainana.sh`. |
| Wrapper + connector normalizer | `run_brainana.sh` — sources `flags.sh`; config resolution; `--no_docker`→`NXF_NO_DOCKER`; `NXF_LAUNCH_DIR`; `BRAINANA_DRY_RUN`; forwards to Nextflow.                                        |
| BIDS discovery                 | `src/nhp_mri_prep/nextflow_scripts/discover_bids_for_nextflow.py` — argparse: bids_dir, output_dir, config_file, subjects, sessions, tasks, runs.                                                        |
| Nextflow params                | `nextflow.config` (`params { … }`), `main.nf` (reads `params.*`, custom-template validation), `workflows/param_resolver.groovy` (priority resolution, CLI↔YAML mapping, validation).                     |


User-facing docs mirroring this reference: `docs/usage_notes.rst` (command-line arguments
section) and `docs/faq.rst`.