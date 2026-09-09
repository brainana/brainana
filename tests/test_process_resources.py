"""Heavy Nextflow processes must actually receive their resource configuration.

Nextflow matches a `withName:` selector against the **full** process name, so a
selector written for `FOO` does not apply to `FOO_LONG` or to any other member of
the same family. A process that misses its selector silently falls back to the
1 CPU / 2 GB process defaults -- and when it also carries
`errorStrategy 'ignore'`, it is OOM-killed without failing the run, so the
pipeline reports success and produces nothing.

That is exactly what happened when the longitudinal processes were added:
`ANAT_SURFACE_RECONSTRUCTION_LONG` runs the same s01-s22 pipeline as
`ANAT_SURFACE_RECONSTRUCTION` (configured for 8 GB, 8 h, OOM retry) but matched no
selector. These tests make the whole class of mistake fail in CI instead.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "nextflow.config"
MODULES = sorted((REPO / "modules").glob("*.nf"))
WORKFLOWS = sorted((REPO / "workflows").glob("*.nf"))

# Processes that must never run on the light defaults. Matched as substrings of the
# process name, so a new sibling (e.g. a further _LONG or _BASE variant) is picked
# up automatically and has to be given a selector.
#
# Scope note: deliberately limited to the surface-reconstruction family, which is
# what this guard was written for. Widening it to SKULLSTRIPPING / REGISTRATION /
# APPLY_TRANSFORMS surfaces pre-existing processes that match no selector either --
# ANAT_REGISTRATION_T2W, ANAT_REGISTRATION_PASSTHROUGH(_T2W),
# ANAT_SKULLSTRIPPING_T2W, FUNC_APPLY_TRANSFORMS_MASK, QC_REGISTRATION(_FUNC,
# _FUNC_INTERMEDIATE), QC_SKULLSTRIPPING(_FUNC), QC_T2W_TO_T1W_REGISTRATION. Some of
# those are genuinely light and some may be under-resourced for the same reason
# documented above, but auditing and re-sizing them is a separate change. Widen
# HEAVY_MARKERS once that has been done.
HEAVY_MARKERS = (
    "SURFACE_RECONSTRUCTION",
    "SURFACE_BASE",
    "SURFACE_LONG",
)

# Names carrying a heavy marker that are deliberately light. Keep this short and
# justified -- it is the escape hatch, so every entry needs a reason.
HEAVY_EXCEPTIONS: set[str] = set()


def _declared_processes(paths):
    """Process names declared in the given .nf files."""
    names = set()
    for path in paths:
        names |= set(
            re.findall(r"^process\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", path.read_text(), re.M)
        )
    return names


def _aliased_processes(paths):
    """Names introduced by `include { X as Y }`, which are separate for selector purposes."""
    names = set()
    for path in paths:
        text = path.read_text()
        names |= set(
            re.findall(r"include\s*\{\s*[A-Za-z_][A-Za-z0-9_]*\s+as\s+([A-Za-z_][A-Za-z0-9_]*)\s*\}", text)
        )
    return names


def _selectors():
    """The `withName:` selector strings in nextflow.config."""
    return re.findall(r"withName:\s*'([^']+)'", CONFIG.read_text())


def _matches(name, selector):
    """Nextflow's rule: the selector is a regex matched against the WHOLE name.

    Mirrors nextflow.script.dsl.ProcessConfigBuilder.matchesSelector, verified
    against the 25.10.2 jar: 'ANAT_SURFACE_RECONSTRUCTION' does not match
    'ANAT_SURFACE_RECONSTRUCTION_LONG', while '...(_LONG)?' and '....*' do.
    """
    return re.fullmatch(selector, name) is not None


ALL_NAMES = sorted(_declared_processes(MODULES) | _aliased_processes(WORKFLOWS))
HEAVY_NAMES = sorted(
    n
    for n in ALL_NAMES
    if any(m in n for m in HEAVY_MARKERS) and n not in HEAVY_EXCEPTIONS
)


def test_there_are_heavy_processes_to_check():
    """Guard against the discovery regexes silently matching nothing."""
    assert HEAVY_NAMES, "no heavy processes discovered -- the parser is broken"


@pytest.mark.parametrize("name", HEAVY_NAMES)
def test_heavy_process_matches_a_selector(name):
    selectors = _selectors()
    matching = [s for s in selectors if _matches(name, s)]
    assert matching, (
        f"{name} matches no withName: selector in nextflow.config, so it will run "
        f"on the 1 CPU / 2 GB process defaults. Add a selector, widen an existing "
        f"one (e.g. 'FOO(_LONG)?'), or add {name} to HEAVY_EXCEPTIONS with a "
        f"reason. Remember withName matches the FULL name."
    )


def test_surface_reconstruction_family_shares_one_selector():
    """Every process running the s01-s22 pipeline needs the same generous limits.

    They do identical work, so drifting resource blocks between them would be a
    latent OOM waiting for whichever one was forgotten.
    """
    family = [
        "ANAT_SURFACE_RECONSTRUCTION",
        "ANAT_SURFACE_RECONSTRUCTION_LONG",
        "ANAT_SURFACE_BASE_RECON",
    ]
    selectors = _selectors()
    per_member = {
        name: {s for s in selectors if _matches(name, s)} for name in family
    }
    missing = [n for n, sels in per_member.items() if not sels]
    assert not missing, f"no selector for {missing}"
    shared = set.intersection(*per_member.values())
    assert shared, (
        "the surface-reconstruction family is covered by different selectors "
        f"({per_member}); they must share one so their limits cannot drift"
    )


def test_qc_aliases_share_their_originals_selector():
    """The _LONG QC aliases do identical work on identical trees."""
    selectors = _selectors()
    for original in ("QC_SURF_RECON_TISSUE_SEG", "QC_CORTICAL_SURF_AND_MEASURES"):
        alias = f"{original}_LONG"
        shared = {
            s
            for s in selectors
            if _matches(original, s) and _matches(alias, s)
        }
        assert shared, (
            f"{alias} is not covered by the same selector as {original}; it would "
            f"fall through to the 1 CPU / 2 GB default"
        )


def test_full_match_semantics_are_what_we_assume():
    """Pin the assumption the rest of this file rests on.

    If a future Nextflow made selectors substring-matched, these tests would be
    checking nothing, so assert the semantics directly.
    """
    assert not _matches("FOO_LONG", "FOO")
    assert _matches("FOO_LONG", "FOO(_LONG)?")
    assert _matches("FOO", "FOO(_LONG)?")
    assert _matches("FOO_LONG", "FOO.*")
    assert not _matches("FOO_LONG", "BAR")
