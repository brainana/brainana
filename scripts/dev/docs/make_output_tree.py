"""Write the output-directory tree in docs/outputs.rst, and check it against a real run.

The tree is the TREE data below: (name, comment, children). It is rendered as a text tree
with the comments set off by ``#`` (the docs colour them; see the ``tree`` lexer in
docs/conf.py) and written into docs/outputs.rst between the ``output-tree-start`` and
``output-tree-end`` markers.

    python scripts/dev/docs/make_output_tree.py                  # rewrite the block
    python scripts/dev/docs/make_output_tree.py --check <out>    # every file pattern exists?

``--check`` turns each file name into a glob (``<...>`` -> ``*``, ``{a|b}`` -> each
alternative, ``[x]`` -> with and without) and looks for a match anywhere under ``<out>``,
the ``preprocessed`` folder of a pipeline run. Point it at a run with surface
reconstruction and functional data (e.g. the devtest dataset).
"""

import argparse
import itertools
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RST = REPO / "docs" / "outputs.rst"
START, END = ".. output-tree-start", ".. output-tree-end"

# (name, comment, children). A name ending in "/" is a folder.
TREE = [
    ("dataset_description.json", "", []),
    ("sub-<id>.html", "QC report, one per subject", []),
    ("all_subjects_report.html", "QC for all subjects (2+ subjects)", []),
    ("sub-<id>/", "", [
        ("anat/", "or in ses-<id>/", [
            ("<anat>_space-scanner_T1w.nii.gz", "", []),
            ("<anat>_space-T1w_desc-preproc_{T1w|T2w}[_brain].nii.gz", "", []),
            ("<anat>_space-T1w_desc-brain_{mask|hemimask|atlasARM2}.nii.gz", "", []),
            ("<anat>_space-<template>_desc-preproc_{T1w|T2w}.nii.gz", "", []),
            ("<anat>_from-<A>_to-<B>_mode-image_xfm.<ext>", "transforms", []),
            ("atlas_space-{T1w|scanner|fsnative}/", "backprojected atlases", []),
        ]),
        ("ses-<id>/", "", [
            ("func/", "", [
                ("<ses>_space-bold_{boldref|desc-brain_mask}.nii.gz", "", []),
                ("<ses>_from-<A>_to-<B>_mode-image_xfm.<ext>", "", []),
                ("<run>_space-{bold|T1w|<template>}_desc-preproc_{bold|boldref}.nii.gz", "", []),
                ("<run>_space-T1w_desc-preproc_stat-tsnr_boldmap.nii.gz", "", []),
                ("<run>_desc-confounds_timeseries.tsv", "", []),
            ]),
        ]),
        ("figures/", "QC snapshots", []),
    ]),
    ("fastsurfer/", "", [
        ("sub-<id>[_ses-<id>]/", "label/ mri/ surf/ stats/ scripts/", []),
    ]),
    ("nextflow_reports/", "run reports, job lists, config", []),
]


def render():
    rows = [("<output_dir>/", "")]

    def walk(items, prefix):
        for i, (name, comment, children) in enumerate(items):
            last = i == len(items) - 1
            rows.append((f"{prefix}{'└── ' if last else '├── '}{name}", comment))
            walk(children, prefix + ("    " if last else "│   "))

    walk(TREE, "")
    # One comment column, just past the longest line that carries a comment.
    column = max(len(line) for line, comment in rows if comment) + 2
    return [f"{line.ljust(column)}# {comment}" if comment else line for line, comment in rows]


def write():
    text = RST.read_text()
    a, b = text.index(START), text.index(END)
    block = ".. code-block:: tree\n\n" + "\n".join("   " + line for line in render()) + "\n\n"
    RST.write_text(text[: a + len(START)] + "\n\n" + block + text[b:])
    print(f"wrote the tree into {RST}")


def globs(name):
    """File-name pattern -> globs: <...> -> *, {a|b} -> each, [x] -> with/without."""
    name = name.rstrip("/")
    parts = re.split(r"(\{[^}]*\}|\[[^\]]*\])", name)
    choices = []
    for p in parts:
        if p.startswith("{"):
            choices.append(p[1:-1].split("|"))
        elif p.startswith("["):
            choices.append(["", p[1:-1]])
        else:
            choices.append([p])
    for combo in itertools.product(*choices):
        yield re.sub(r"<[^>]*>", "*", "".join(combo))


def check(out_dir):
    out_dir = Path(out_dir)
    missing = []

    def walk(items):
        for name, _, children in items:
            for g in globs(name):
                if not any(out_dir.rglob(g)):
                    missing.append(g)
            walk(children)

    walk(TREE)
    for g in missing:
        print(f"missing: {g}")
    print(f"{'FAIL' if missing else 'OK'}: checked the tree against {out_dir}")
    return 1 if missing else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", metavar="OUTPUT_DIR", help="check the tree against a real run")
    args = parser.parse_args()
    sys.exit(check(args.check) if args.check else write())
