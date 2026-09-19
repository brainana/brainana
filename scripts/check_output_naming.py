#!/usr/bin/env python3
"""Check every published name in an output tree against the naming contract.

The executable form of ``docs_temp/update_instruction/name_guideline.md``. The
unit tests pin what the naming API builds; this checks what a real run actually
wrote, which is the only place a name assembled by string concatenation inside a
Nextflow process body ever appears.

It exists because the pipeline's failure mode here is silence. All of a subject's
QC figures land in one flat ``sub-<id>/figures``, every QC process carries
``errorStrategy 'ignore'``, and ``publishDir`` defaults to ``overwrite: false`` --
so two names that collapse into one are a missing file, not an error. A name with
a duplicated entity is worse: it is present, and ``parse_bids_entities`` silently
drops one of the two values, so every consumer sees a file that claims less than
it is.

Usage::

    python scripts/check_output_naming.py <output_dir>
    python scripts/check_output_naming.py <output_dir> --verbose   # list declared too

Exits non-zero when an undeclared violation is found. The three deliberate
deviations in ``DECLARED_DEVIATIONS`` are reported as declared, never as
failures -- brainana-viewer's data contract depends on two of them.

Skipped by design:

- ``fastsurfer/`` -- FreeSurfer-native naming (``lh.pial``, ``recon-all.log``,
  ``sub-X_ses-Y_to_sub-X_base.lta``). Not BIDS, not ours to name.
- ``nextflow_reports/`` -- run logs and the effective config.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from nhp_mri_prep.utils.bids import (  # noqa: E402
    DECLARED_DEVIATIONS,
    validate_output_name,
)

# Directories whose contents another tool names.
SKIP_DIRS = frozenset({"fastsurfer", "nextflow_reports"})

# Files that are not per-subject products and carry no entity chain.
SKIP_NAMES = frozenset({"dataset_description.json"})


def _published_files(root: Path):
    """Every file brainana named, in sorted order."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if SKIP_DIRS & set(path.relative_to(root).parts):
            continue
        if path.name in SKIP_NAMES:
            continue
        yield path


def check(root: Path, verbose: bool = False) -> int:
    """Report violations grouped by class. Returns a process exit code."""
    by_class: dict[str, list[str]] = defaultdict(list)
    checked = 0

    for path in _published_files(root):
        checked += 1
        for violation in validate_output_name(path.name):
            # The message's leading phrase is the class; the detail follows.
            klass = violation.split("(")[0].split(";")[0].strip()
            by_class[klass].append(f"{path.relative_to(root)}: {violation}")

    print(f"Checked {checked} published names under {root}")

    if verbose:
        print(f"\nDeclared deviations (not failures, {len(DECLARED_DEVIATIONS)}):")
        for name, reason in DECLARED_DEVIATIONS.items():
            print(f"  - {name}: {reason}")

    if not by_class:
        print("\nNo undeclared naming violations.")
        return 0

    total = sum(len(v) for v in by_class.values())
    print(f"\n{total} undeclared violation(s), in {len(by_class)} class(es):\n")
    for klass, entries in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
        print(f"  {klass}  ({len(entries)} file(s))")
        for entry in entries[:10]:
            print(f"      {entry}")
        if len(entries) > 10:
            print(f"      ... and {len(entries) - 10} more")
        print()
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir", type=Path, help="A brainana output directory")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also list the declared deviations and their reasons",
    )
    args = parser.parse_args()

    if not args.output_dir.is_dir():
        parser.error(f"not a directory: {args.output_dir}")
    return check(args.output_dir, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
