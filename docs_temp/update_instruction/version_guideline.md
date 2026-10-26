# Brainana — Version Management

How version numbers are defined, resolved at runtime, stamped into outputs, and kept in sync across releases.

For the full release workflow (PR path, Docker publish, RTD, Colab), see [general_guideline.md](general_guideline.md). This document covers **version organization** and **what to update when bumping a release**.

---

## 1. Single source of truth

**`pyproject.toml`** is the canonical version:

```toml
version = "1.2.0"
```

Everything else derives from this — directly (Sphinx, `get_version()` in source-tree runs) or indirectly (dist-info after `uv pip install -e .`, Docker build-arg).

**Convention:**

| Context | Format | Example |
|---------|--------|---------|
| Python package / sidecars | no `v` prefix | `1.2.0` |
| Git tag | `v` prefix | `v1.2.0` |
| Docker Hub tag | no `v` prefix | `1.2.0` |
| Brainana Lite notebook `BRAINANA_REF` | `v` prefix (git ref) | `v1.2.0` |

---

## 2. Version resolution chain

```
pyproject.toml  (source of truth)
       │
       ├─► get_version()  in  src/nhp_mri_prep/version.py
       │   │
       │   │  Priority 1: BRAINANA_IMAGE_TAG env var (Docker)
       │   │  Priority 2: pyproject.toml at repo root (source-tree dev)
       │   │  Priority 3: importlib.metadata.version("brainana")
       │   │  Priority 4: "0.0.0" fallback
       │   │
       │   └──► _generated_by() in sidecar.py → all derivative JSON sidecars
       │
       ├─► pip / uv install -e .  ──►  dist-info in site-packages
       │         (importlib.metadata — Priority 3; checked by pytest)
       │
       ├─► Dockerfile  --build-arg BRAINANA_VERSION=<tag>
       │         └── ENV BRAINANA_IMAGE_TAG=${BRAINANA_VERSION}
       │
       ├─► docs/conf.py  (reads pyproject.toml directly via tomllib)
       │         → Sphinx release / version variables
       │
       └─► examples/BrainanaLite.ipynb
                 BRAINANA_REF = "v1.2.0"  (git ref for clone, not get_version())
```

### Summary diagram

```
                 pyproject.toml (version = "X.Y.Z")
                        │
          ┌─────────────┼──────────────────┐
          │             │                  │
          ▼             ▼                  ▼
   uv pip install   docs/conf.py       BrainanaLite.ipynb
   (-e .)           (tomllib, direct)  (BRAINANA_REF, manual)
          │
          ▼
   dist-info in site-packages        Dockerfile
   (frozen at install time)    --build-arg BRAINANA_VERSION
          │                               │
          │                               ▼
          │                     BRAINANA_IMAGE_TAG env var
          │                               │
          └──────────────► get_version()  ◄──────────────┘
                           Priority 1: env var
                           Priority 2: pyproject.toml (source tree)
                           Priority 3: dist-info
                           Priority 4: "0.0.0"
                                    │
                                    ▼
                         _generated_by() in sidecar.py
                                    │
                         ┌──────────┴───────────────┐
                         ▼                          ▼
               dataset_description.json    all *_xfm.json,
                                           preproc sidecars, etc.
```

---

## 3. Where version lives — file by file

### 3.1 `pyproject.toml`

Canonical version string. Update this first on release day.

**Downstream consumers:**
- `get_version()` Priority 2 — read directly when running from the source tree
- `uv pip install -e .` → writes to `dist-info/METADATA`
- `docs/conf.py` reads it at Sphinx build time
- Docker build must pass matching `--build-arg BRAINANA_VERSION=<tag>`

### 3.2 `src/nhp_mri_prep/version.py` — runtime resolver

All internal code should call `get_version()`, not `importlib.metadata` directly.

```python
def get_version() -> str:
    tag = os.environ.get("BRAINANA_IMAGE_TAG", "").strip()
    if tag:
        return tag
    pyproject_version = _pyproject_version()   # reads repo-root pyproject.toml
    if pyproject_version:
        return pyproject_version
    try:
        return version("brainana")
    except PackageNotFoundError:
        return "0.0.0"
```

| Priority | Source | When active |
|----------|--------|-------------|
| 1 | `BRAINANA_IMAGE_TAG` env var | Docker runs (baked in at build time) |
| 2 | `pyproject.toml` at repo root | Local `--no-docker` / editable dev from source tree |
| 3 | `importlib.metadata.version("brainana")` | Non-editable pip install; fallback when pyproject not found |
| 4 | `"0.0.0"` | Package not installed |

**Callers:**
- `src/nhp_mri_prep/__init__.py` → `__version__`
- `src/nhp_mri_prep/utils/sidecar.py` → `_generated_by()` → stamps every derivative JSON sidecar and `dataset_description.json`

### 3.3 Output sidecars — `GeneratedBy.Version`

Every derivative JSON sidecar includes:

```json
"GeneratedBy": [{"Name": "brainana", "Version": "1.2.0"}]
```

Written by `write_derivative_sidecar()` (anatomical/functional Nextflow processes) and `write_dataset_description()` (run start). If you see a wrong version in `*_xfm.json`, trace it to `get_version()` — not to Nextflow resume or the scratch script `version=` variable.

### 3.4 `Dockerfile` — Docker image version

```dockerfile
ARG BRAINANA_VERSION=unknown
ENV BRAINANA_IMAGE_TAG=${BRAINANA_VERSION}
```

Build with an explicit tag (not read from `pyproject.toml` automatically):

```bash
export VERSION=$(grep '^version' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
docker build --build-arg BRAINANA_VERSION=${VERSION} \
  -t liuxingyu987/brainana:${VERSION} .
```

If the build-arg is omitted, sidecars inside the container will show `"unknown"`.

### 3.5 `docs/conf.py` — Sphinx / RTD

Reads `pyproject.toml` directly via `tomllib`. No install step needed. RTD **stable** tracks the highest semver git tag after push.

### 3.6 `examples/BrainanaLite.ipynb` — Lite git ref

```python
BRAINANA_REF = "v1.2.0"
```

Manual string for `git clone --branch`. Not wired to `get_version()`. Colab smoke with `v${VERSION}` only works **after** the tag is pushed to GitHub.

### 3.7 Installed `dist-info` metadata

Editable installs (`.venv`, conda) update code on the fly but **freeze the version field at install time**. `pytest tests/test_version.py::test_dist_info_matches_pyproject` fails if dist-info is stale — run `uv pip install -e .` to fix.

---

## 4. What to update on release

Use one `VERSION` (e.g. `1.2.0`) and update these **together on `main` before pre-tag gates** (leave uncommitted until tests pass).

**Run the script — do not walk the table by hand:**

```bash
python scripts/bump_version.py ${VERSION}             # apply
python scripts/bump_version.py ${VERSION} --dry-run   # preview only
```

It applies every **bold** row below, prints a per-file `old -> new` report, then re-runs
the quick grep and lists anything still mentioning the old version. The table remains the
reference for *what* it touches and why; the two unbold rows it cannot do
(`uv pip install -e .` and the Docker build-arg) are printed as reminders when it finishes.

The v2.1.0 bump was done by hand and missed four of these files, which is why the script
exists. `scripts/scratch/test_brainana_*.sh` is matched by glob, so a new scratch runner is
picked up automatically; runners with no `version=` line are skipped silently.

| File / area | What to update | Notes |
| ----------- | -------------- | ----- |
| **pyproject.toml** | `version = "${VERSION}"` | Single source of truth |
| **uv.lock** | the `brainana` package's `version` line | CI `uv lock --check` fails if it lags (3.3.0 PR). One line edited in place — do not relock with another uv version: CI pins 0.6.16, which rewrites the whole file into an older format |
| **`.venv` editable install** | `uv pip install -e .` | Refreshes dist-info; sidecar stamping reads pyproject directly, but pytest checks dist-info alignment |
| **CHANGELOG.md** | `[Unreleased]` → `[${VERSION}] - YYYY-MM-DD`; add empty `[Unreleased]` | Copy into `gh release create --notes-file` |
| **README.md** | Release highlights, links, Lite pointers | No pinned version required |
| **docs/** (RTD) | Old example versions, new/changed behavior | `rg -n "${VERSION}\|1\\.0\\.0\|<version>" docs/` |
| **examples/BrainanaLite.ipynb** | `BRAINANA_REF = "v${VERSION}"` | Git tag uses `v`; Docker tag does not |
| **`scripts/scratch/test_brainana_*.sh`** | `version=${VERSION}` | Output-dir naming only |
| **RTD build (local)** | `sphinx-build` or `bash docs/build_rtd_local.sh` | After pyproject bump |
| **Docker image** | `--build-arg BRAINANA_VERSION=${VERSION}` | Must match pyproject |

**Dev PRs (before release):** only edit `CHANGELOG.md` under `[Unreleased]`. Do **not** bump `pyproject.toml` version until release day on `main` — except in a release PR (below).

**Release PR: bumping on the branch.** The last PR before a release may carry the bump
itself, which saves one CI run (`deps.yml` runs on the PR and again on every push to
`main`, so bumping on `main` afterwards costs a third). Allowed when all of these hold:

1. **The branch sits directly on `main`'s tip** (`git merge-base --is-ancestor origin/main HEAD`),
   so the tree that passed the gates is the tree that lands. Rebase first if `main` moved.
2. **Every pre-tag gate passes on the branch before merging** — pytest, ruff, Sphinx,
   Docker smoke, Lite local (section 5). `main` reads `X.Y.Z` from the merge on; tag only
   once it is good, and fix forward on `main` if not.
3. **Merge without squashing** (fast-forward or merge commit), then tag the commit that
   lands on `main` — never the branch commit.
4. **The `CHANGELOG.md` date is the tag day**; re-date it if the merge slips.

The bonus: CI installs the bumped package fresh, so `tests/test_version.py` checks the new
version. The alternative — the bump on `main` as usual with `[skip ci]` in the release
commit message — saves the same run but never tests the bump.

**Quick grep before build:**

`bump_version.py` runs this sweep itself and prints what is left, so this is the manual
equivalent — useful when verifying a bump someone else made, or on a tag.

```bash
export VERSION=1.2.0   # set to your release
grep '^version' pyproject.toml
rg -n "${VERSION}|1\\.0\\.0|<version>" pyproject.toml CHANGELOG.md README.md docs/
```

Expect one legitimate hit after a bump: the previous release's own `## [X.Y.Z]` heading in
`CHANGELOG.md`. `docs/_build/` and the bundled example QC report also match — the former is
regenerated, the latter carries a minified JS blob that matches almost any digit sequence;
the script filters both out.

---

## 5. Pre-tag gates (version-specific)

All must pass **before** `git commit` and **before** `git tag`. Full gate list is in [general_guideline.md](general_guideline.md#pre-tag-test-gates).

**Version steps:**

0. After bumping `pyproject.toml`, refresh editable install:
   ```bash
   uv pip install -e .
   ```
   Sidecar stamping reads pyproject directly (Priority 2), but `pytest tests/test_version.py` also asserts dist-info alignment.

1. Run version tests:
   ```bash
   pytest tests/test_version.py -v
   ```

2. Run full test suite (includes sidecar `GeneratedBy.Version` checks):
   ```bash
   pytest
   ```

3. Build Docker with matching tag and smoke-test:
   ```bash
   docker build --build-arg BRAINANA_VERSION=${VERSION} \
     -t liuxingyu987/brainana:${VERSION} .
   # scripts/scratch/test_brainana_docker.sh or _cpu.sh
   ```

4. Optional: inspect a local run sidecar:
   ```bash
   # After a conform step, check:
   cat sub-*_from-scanner_to-T1w_mode-image_xfm.json | grep Version
   ```

---

## 6. Release-day checklist

| Step | Command / action |
|------|------------------|
| Bump every version site | `python scripts/bump_version.py X.Y.Z` |
| ↳ covers | `pyproject.toml`, `uv.lock`, `CHANGELOG.md` heading, `BrainanaLite.ipynb` ref, `docs/*.rst` example tags, `test_brainana_*.sh` |
| Write the CHANGELOG body | move `[Unreleased]` entries under the new heading |
| Refresh `.venv` dist-info | `uv pip install -e .` (script cannot do this) |
| Grep for stale references | script prints this; see quick grep above to re-check |
| Run version tests | `pytest tests/test_version.py -v` |
| Build Docker | `--build-arg BRAINANA_VERSION=X.Y.Z` |
| Verify sidecar version | inspect `GeneratedBy.Version` in a test run |
| Verify Sphinx | built HTML shows new `release` |
| Commit, tag, push | `git tag vX.Y.Z && git push origin vX.Y.Z` |
| Post-tag | Colab Lite, `gh release create`, `docker push`, RTD stable |

---

## 7. Automated guards

| Test | File | What it checks |
|------|------|----------------|
| `test_get_version_reads_pyproject` | `tests/test_version.py` | `get_version()` matches `pyproject.toml` |
| `test_dist_info_matches_pyproject` | `tests/test_version.py` | dist-info matches pyproject (fails → run `uv pip install -e .`) |
| `test_notebook_ref_matches_pyproject` | `tests/test_version.py` | `BRAINANA_REF` matches pyproject **when pinned to a `vX.Y.Z` tag**; skipped for `main` / SHA / `feat/<topic>` (see section 8) |
| Sidecar `GeneratedBy.Version` | `tests/test_templates_and_sidecar.py` | Sidecars stamp `get_version()` |

Only `BRAINANA_REF` is guarded among the section 4 files, because it is the only one where
staleness is *breaking* rather than cosmetic: Colab clones that ref, so a stale pin silently
installs the wrong code. A stale `docs/` example tag or scratch `version=` is harmless — the
old Docker tag is still published, and the scratch variable only names output directories.
Enforcing those would block releases on cosmetics and train people to bypass the guard.

---

## 8. Gotchas and pitfalls

### Stale dist-info after a bump

**Symptom:** `pytest tests/test_version.py::test_dist_info_matches_pyproject` fails with `dist-info stale: '1.1.0' != '1.2.0'`.

**Fix:** `uv pip install -e .` from repo root (in `.venv`).

**Note:** Sidecar stamping is correct immediately after a pyproject bump (Priority 2), even before reinstall. Reinstall is still required for pytest and dist-info consistency.

### Wrong version in output sidecars (historical)

Before Priority 2 was added, local `--no-docker` runs read stale dist-info and stamped old versions into all `*_xfm.json` and `dataset_description.json`. This is fixed for new runs from the source tree.

### Docker build-arg not auto-synced

The Dockerfile does not read `pyproject.toml`. Always pass `--build-arg BRAINANA_VERSION=${VERSION}` explicitly, or derive `VERSION` from pyproject (see section 3.4).

### `BRAINANA_IMAGE_TAG=unknown`

If Docker is built without the build-arg, sidecars inside the container show `"unknown"`. Always pass the build-arg for release images.

### Scratch script `version=` variable

The `version=` variable in `scripts/scratch/test_brainana_*.sh` only names output directories. It does **not** control what version is stamped in sidecars.

### Brainana Lite `BRAINANA_REF`

- Pre-tag: use `main`, commit SHA, or local editable install — not `v${VERSION}` until the tag exists on GitHub.
- Post-tag: Colab smoke requires `BRAINANA_REF = "v${VERSION}"` after push.
- `test_notebook_ref_matches_pyproject` (section 7) only fires on the pinned `vX.Y.Z` form,
  precisely so it cannot fail during this pre-tag window. `python scripts/bump_version.py
  X.Y.Z --pre-tag` sets the ref to `main` instead of the tag for the same reason.

---

## 9. Verify release artifacts

After publishing, confirm:

| Artifact | Where to check | Expected |
| -------- | -------------- | -------- |
| Version in repo | `grep '^version' pyproject.toml` on tag `v${VERSION}` | `${VERSION}` |
| Brainana Lite notebook | `examples/BrainanaLite.ipynb` on tag | `BRAINANA_REF = "v${VERSION}"` |
| Git tag | `git ls-remote --tags origin v${VERSION}` | one ref |
| GitHub Release | [releases](https://github.com/xingyu-liu/brainana/releases) | `v${VERSION}` with CHANGELOG notes |
| Docker Hub | [tags](https://hub.docker.com/r/liuxingyu987/brainana/tags) | `${VERSION}` (no `v`) |
| RTD stable | [brainana.readthedocs.io/en/stable/](https://brainana.readthedocs.io/en/stable/) | `${VERSION}` in built docs |
| Output sidecars | any derivative `*.json` | `GeneratedBy[0].Version == ${VERSION}` |
