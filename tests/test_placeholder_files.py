"""Workflow placeholder files must be written only when missing.

Nextflow's standard cache hashes an input file by path, size and mtime. The
workflows stand in `${workDir}/*.dummy` files for absent inputs; rewriting one on
every run gave it a new mtime, so every task consuming it (the functional chain of
a subject without anatomy, QC_TSNR) missed the cache on each -resume.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = sorted((REPO / "workflows").glob("*.nf"))


def test_placeholders_are_not_rewritten():
    for path in WORKFLOWS:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r'toFile\(\)\.text\s*=\s*""', line):
                assert ".exists()" in line, f"{path.name}:{lineno}: {line.strip()}"
