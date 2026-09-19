#!/usr/bin/env python3
"""Apply a version bump across every file the release ritual touches.

This is the executable form of section 4 ("What to update on release") in
``docs_temp/update_instruction/version_guideline.md``. That table is the source
of truth; this script exists because the doc's own verification step is a manual
grep, which has the same forget-to-run failure mode as the bump itself.

Usage::

    python scripts/bump_version.py 2.2.0            # apply
    python scripts/bump_version.py 2.2.0 --dry-run  # show what would change
    python scripts/bump_version.py 2.2.0 --pre-tag  # set notebook ref to "main"

Deliberately stdlib-only and importing nothing from ``src/``: it must run before
``uv pip install -e .`` has caught up with the new version.

What it does NOT do, because it cannot:

- ``uv pip install -e .`` to refresh dist-info (section 4). ``pytest`` fails until you do.
- The Docker ``--build-arg BRAINANA_VERSION`` (section 8, "Docker build-arg not
  auto-synced"). The Dockerfile does not read pyproject.toml.

Both are printed as reminders on success.
"""

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")

# Files whose version strings are rewritten, and how to find them. Each entry is
# (path, description, builder) where builder(old, new) returns the (pattern,
# replacement) pair applied to the file text.
#
# Every pattern anchors on the *current* version rather than matching any
# version-shaped string, so unrelated numbers are untouched: pymeshfix>=0.18.1,
# the aparc_vinn_axial_v2.0.0.pkl checkpoint name, the semver.org/spec/v2.0.0
# link, historical CHANGELOG headings, and src/fastsurfer_nn/__init__.py (which
# per section 3 does not track the release).
TEXT_TARGETS = [
    (
        "pyproject.toml",
        "project version (source of truth)",
        lambda old, new: (rf'^version = "{re.escape(old)}"$', f'version = "{new}"'),
    ),
    (
        "docs/demo.rst",
        "example Docker Hub tag",
        lambda old, new: (rf"\(e\.g\. ``{re.escape(old)}``\)", f"(e.g. ``{new}``)"),
    ),
    (
        "docs/installation.rst",
        "example Docker Hub tag",
        lambda old, new: (
            rf"for example ``{re.escape(old)}``",
            f"for example ``{new}``",
        ),
    ),
    (
        "docs/usage_notes.rst",
        "example Docker Hub tag",
        lambda old, new: (
            rf"for example ``{re.escape(old)}``",
            f"for example ``{new}``",
        ),
    ),
]

# Section 4 names these by glob (`scripts/scratch/test_brainana_*.sh`), so they are
# discovered rather than listed — a new scratch runner is picked up automatically.
# Discovered late (module import time is too early for a missing scripts/scratch).
SCRATCH_GLOB = "scripts/scratch/test_brainana_*.sh"


def scratch_targets() -> list:
    """Section 4's `scripts/scratch/test_brainana_*.sh` row, expanded.

    Only the runners that actually declare a `version=` participate. Several
    scratch scripts have no such line (they hardcode an image tag or do not run
    Docker at all), and reporting those as unmatched every release would be noise
    that trains you to ignore the real "check by hand" warnings.
    """
    targets = []
    for path in sorted(REPO.glob(SCRATCH_GLOB)):
        if not re.search(r"^version=", path.read_text(encoding="utf-8"), re.MULTILINE):
            continue
        targets.append(
            (
                str(path.relative_to(REPO)),
                "output-dir naming only (section 8)",
                lambda old, new: (rf"^version={re.escape(old)}$", f"version={new}"),
            )
        )
    return targets


NOTEBOOK = "examples/BrainanaLite.ipynb"
CHANGELOG = "CHANGELOG.md"

# Files the section 4 "quick grep" sweeps for leftovers.
GREP_SCOPE = [
    "pyproject.toml",
    CHANGELOG,
    "README.md",
    "docs",
    "examples",
    "scripts/scratch",
]


class Change:
    """One pending edit, rendered as a report row and applied on demand."""

    def __init__(self, path: Path, what: str, before: str, after: str, text: str):
        self.path = path
        self.what = what
        self.before = before
        self.after = after
        self.text = text

    def apply(self) -> None:
        self.path.write_text(self.text, encoding="utf-8")


def current_version() -> str:
    """Read the version from pyproject.toml without a TOML parse of the whole file."""
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    if not match:
        sys.exit('error: could not find `version = "..."` in pyproject.toml')
    return match.group(1)


def plan_text_targets(old: str, new: str) -> list:
    """Build the pending edits for every plain-text target."""
    changes = []
    for rel, what, builder in TEXT_TARGETS + scratch_targets():
        path = REPO / rel
        if not path.is_file():
            print(f"  ! {rel}: missing, skipped")
            continue
        pattern, replacement = builder(old, new)
        text = path.read_text(encoding="utf-8")
        updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
        if count == 0:
            print(f"  ! {rel}: no {old} occurrence matched ({what}) — check by hand")
            continue
        changes.append(Change(path, what, old, new, updated))
    return changes


def plan_notebook(old: str, new_ref: str) -> list:
    """Build the pending edit for the Lite notebook's BRAINANA_REF.

    Rewritten through json rather than as text so the notebook stays valid and
    the diff stays to the single line: Jupyter writes indent=1, which
    json.dumps reproduces exactly.
    """
    path = REPO / NOTEBOOK
    if not path.is_file():
        print(f"  ! {NOTEBOOK}: missing, skipped")
        return []

    notebook = json.loads(path.read_text(encoding="utf-8"))
    assign = re.compile(r'^(BRAINANA_REF\s*=\s*)"([^"]+)"$')
    found = []
    for cell in notebook["cells"]:
        for i, line in enumerate(cell["source"]):
            match = assign.match(line.strip())
            if not match:
                continue
            found.append(match.group(2))
            cell["source"][i] = line.replace(f'"{match.group(2)}"', f'"{new_ref}"')

    if len(found) != 1:
        print(f"  ! {NOTEBOOK}: expected 1 BRAINANA_REF assignment, found {found}")
        return []
    if found[0] == new_ref:
        return []

    text = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
    return [Change(path, "Colab git ref (breaking if stale)", found[0], new_ref, text)]


def plan_changelog(new: str, today: str) -> list:
    """Insert a dated release heading directly below ``## [Unreleased]``."""
    path = REPO / CHANGELOG
    text = path.read_text(encoding="utf-8")

    if re.search(rf"^## \[{re.escape(new)}\]", text, re.MULTILINE):
        print(f"  = {CHANGELOG}: [{new}] section already present")
        return []
    if "## [Unreleased]" not in text:
        print(
            f"  ! {CHANGELOG}: no '## [Unreleased]' heading — add the section by hand"
        )
        return []

    updated = text.replace(
        "## [Unreleased]\n",
        f"## [Unreleased]\n\n\n## [{new}] - {today}\n",
        1,
    )
    return [
        Change(
            path, "new release heading", "[Unreleased]", f"[{new}] - {today}", updated
        )
    ]


def leftovers(old: str) -> list:
    """Section 4's quick grep: any remaining mention of the old version in scope.

    Restricted to hand-edited source formats. Sweeping everything drowns the real
    hits in generated noise: docs/_build/ is a rebuildable Sphinx tree, and the
    bundled example QC report carries a minified JS blob that contains almost any
    digit sequence you care to search for.
    """
    suffixes = {".toml", ".md", ".rst", ".sh", ".ipynb", ".py", ".yml", ".yaml", ".cfg"}
    skip_dirs = {"_build", ".ipynb_checkpoints", "__pycache__"}

    hits = []
    for rel in GREP_SCOPE:
        target = REPO / rel
        files = sorted(target.rglob("*")) if target.is_dir() else [target]
        for path in files:
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if skip_dirs & set(path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for n, line in enumerate(text.splitlines(), 1):
                if old in line:
                    hits.append((path.relative_to(REPO), n, line.strip()))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="new version, e.g. 2.2.0 (no 'v' prefix)")
    parser.add_argument(
        "--dry-run", action="store_true", help="report intended edits, change nothing"
    )
    parser.add_argument(
        "--pre-tag",
        action="store_true",
        help='set the notebook ref to "main" instead of the version tag (section 8: '
        "the tag does not exist until it is pushed)",
    )
    parser.add_argument(
        "--date", help="release date for the CHANGELOG heading (default: today)"
    )
    args = parser.parse_args()

    new = args.version.lstrip("v")
    if not SEMVER.match(new):
        sys.exit(f"error: {args.version!r} is not a X.Y.Z version")

    old = current_version()
    if old == new and not args.pre_tag:
        print(f"Already at {new}; nothing to do.")
        return 0

    today = args.date or datetime.date.today().isoformat()
    new_ref = "main" if args.pre_tag else f"v{new}"

    print(f"Bumping {old} -> {new}\n")
    changes = plan_text_targets(old, new)
    changes += plan_notebook(old, new_ref)
    changes += plan_changelog(new, today)

    if not changes:
        print("\nNo edits to apply.")
        return 0

    width = max(len(str(c.path.relative_to(REPO))) for c in changes)
    for change in changes:
        rel = str(change.path.relative_to(REPO))
        print(f"  {rel:<{width}}  {change.before} -> {change.after}   ({change.what})")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    for change in changes:
        change.apply()

    print(f"\nWrote {len(changes)} file(s).")

    remaining = leftovers(old)
    if remaining:
        print(f"\nStill mentioning {old} (review — some are legitimately historical):")
        for rel, n, line in remaining:
            print(f"  {rel}:{n}: {line[:100]}")
    else:
        print(f"\nNo remaining references to {old} in the release file set.")

    print(
        "\nNot handled here (see version_guideline.md):\n"
        "  uv pip install -e .                          refresh dist-info, or pytest fails\n"
        f"  docker build --build-arg BRAINANA_VERSION={new} ...   Dockerfile does not read pyproject"
    )
    if args.pre_tag:
        print(
            f'\n  --pre-tag: BRAINANA_REF left at "main". Set it to "v{new}" '
            "in the release commit."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
