"""Version consistency guards for brainana.

Ensures ``get_version()`` reflects ``pyproject.toml``, that installed dist-info
metadata stays aligned after editable installs, and that the Brainana Lite
notebook does not ship pinned to a stale release tag.

See ``docs_temp/update_instruction/version_guideline.md`` for the full list of
files a release touches; ``scripts/bump_version.py`` applies them all at once.
"""

import json
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
import tomllib

from nhp_mri_prep.version import get_version

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "examples" / "BrainanaLite.ipynb"

# A ref pinned to a release tag, e.g. "v2.1.0".
_PINNED_TAG = re.compile(r"^v\d+\.\d+\.\d+$")
_REF_ASSIGN = re.compile(r'^BRAINANA_REF\s*=\s*"([^"]+)"$')


def _pyproject_version() -> str:
    with open(REPO / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def _notebook_ref() -> str:
    """Return the single ``BRAINANA_REF`` value assigned in the Lite notebook."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    refs = [
        match.group(1)
        for cell in notebook["cells"]
        for line in cell["source"]
        if (match := _REF_ASSIGN.match(line.strip()))
    ]
    detail = (
        f"expected exactly one BRAINANA_REF assignment in {NOTEBOOK.name}, got {refs}"
    )
    assert len(refs) == 1, detail
    return refs[0]


def test_get_version_reads_pyproject(monkeypatch):
    """get_version() must match pyproject.toml when running from the source tree."""
    # BRAINANA_IMAGE_TAG takes priority in get_version(); clear it so this test checks
    # the pyproject path rather than failing under the Docker/release env that sets it.
    monkeypatch.delenv("BRAINANA_IMAGE_TAG", raising=False)
    assert get_version() == _pyproject_version()


def test_dist_info_matches_pyproject():
    """Editable dist-info must match pyproject.toml (run ``uv pip install -e .`` if stale)."""
    expected = _pyproject_version()
    try:
        installed = version("brainana")
    except PackageNotFoundError:
        pytest.skip("brainana not installed")
    assert (
        installed == expected
    ), f"dist-info stale: {installed!r} != {expected!r}; run: uv pip install -e ."


def test_notebook_ref_matches_pyproject():
    """A version-pinned Lite notebook must point at *this* release.

    Colab clones whatever ``BRAINANA_REF`` names, so a stale pin silently
    installs the wrong code — the one version site where drift is breaking
    rather than cosmetic.

    Only the pinned form is checked. ``version_guideline.md`` section 8 states that
    pre-tag the ref should be ``main``, a commit SHA or ``feat/<topic>``, because
    ``v${VERSION}`` does not exist until the tag is pushed; failing on those would
    make the guard fire during the workflow it is meant to protect.
    """
    ref = _notebook_ref()
    if not _PINNED_TAG.match(ref):
        pytest.skip(
            f"BRAINANA_REF is {ref!r} (unpinned pre-tag state), not a release tag"
        )

    expected = f"v{_pyproject_version()}"
    detail = (
        f"{NOTEBOOK.name} pins BRAINANA_REF={ref!r} but pyproject.toml says "
        f"{_pyproject_version()!r}; run: python scripts/bump_version.py "
        f"{_pyproject_version()}"
    )
    assert ref == expected, detail
