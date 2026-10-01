"""BIDS filter options (--subjects/--sessions/--tasks/--runs) through the real resolver.

Nextflow parses a single numeric CLI value (``--subjects 01``, ``--runs 1``) into a
number. The workflows' list check used to reject it and abort the run, although
discovery (which does the actual filtering) had already received the right text.
Runs workflows/param_resolver.groovy inside Nextflow, so it needs the pinned
Nextflow launcher and Java; skipped where they are missing.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
LAUNCHER = REPO / "nextflow"

SCRIPT = """
def paramResolver = evaluate(new File("%s/workflows/param_resolver.groovy").text)
paramResolver.initialize(params, "%s")
workflow {
    for (p in ['subjects', 'sessions', 'tasks', 'runs']) {
        if (params[p] != null) {
            println "${p}=${paramResolver.getParamList(params, p, null)}"
        }
    }
}
"""

pytestmark = pytest.mark.skipif(
    not LAUNCHER.exists() or shutil.which("java") is None,
    reason="needs the nextflow launcher and Java",
)


def _resolve(tmp_path, *args):
    (tmp_path / "main.nf").write_text(SCRIPT % (REPO, REPO))
    # The version run_brainana.sh pins; 26.x rejects the repo's Groovy config.
    env = dict(os.environ, NXF_VER=os.environ.get("NXF_VER", "25.10.2"))
    proc = subprocess.run(
        [str(LAUNCHER), "-q", "run", "main.nf", *args],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)


def test_single_numeric_values_are_accepted(tmp_path):
    out = _resolve(tmp_path, "--subjects", "01", "--runs", "1")
    assert out == {"subjects": "[1]", "runs": "[1]"}


def test_comma_separated_lists(tmp_path):
    out = _resolve(tmp_path, "--subjects", "01,02", "--sessions", "001", "--tasks", "rest,movie")
    assert out == {"subjects": "[01, 02]", "sessions": "[1]", "tasks": "[rest, movie]"}
