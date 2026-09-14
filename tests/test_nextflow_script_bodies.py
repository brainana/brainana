"""Nextflow process scripts are Python, but nothing imports them.

Each process body is a Python heredoc embedded in a Groovy string. Nothing type
checks it, no linter reads it, and pytest never imports it -- so a renamed helper
or a dropped keyword argument is found only when the task runs. For the
longitudinal family that is the worst case in the pipeline: the base and its
timepoints are gathered behind every cross-sectional reconstruction, so a typo
there surfaces after hours of compute, and the processes carry
`errorStrategy 'ignore'`, which turns it into a subject that silently produced
nothing.

These tests compile each body and resolve the names it calls. Scoped to the
processes this branch added, matching how test_process_resources.py scopes its
own guard; widen deliberately.
"""

import ast
import importlib
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MODULES = REPO / "modules" / "anatomical.nf"
NF_FILES = sorted((REPO / "workflows").glob("*.nf")) + sorted((REPO / "modules").glob("*.nf"))

LONGITUDINAL_PROCESSES = [
    "ANAT_SURFACE_RECONSTRUCTION",
    "ANAT_SURFACE_BASE_TEMPLATE",
    "ANAT_SURFACE_BASE_ATLAS",
    "ANAT_SURFACE_BASE_RECON",
    "ANAT_SURFACE_RECONSTRUCTION_LONG",
    "ANAT_SURFACE_LONG_CHANGE_STATS",
]

# Groovy interpolates ${...} into the heredoc before the shell sees it. Every
# occurrence sits where a Python string literal or expression goes, so a bare
# placeholder keeps the body parseable without pretending to know the value.
_INTERPOLATION = re.compile(r"(?<!\\)\$\{[^{}]*\}")


def _process_bodies():
    text = MODULES.read_text()
    bodies = {}
    for name in LONGITUDINAL_PROCESSES:
        start = text.index(f"process {name} {{")
        # The next `process ` at column 0, or end of file.
        nxt = text.find("\nprocess ", start + 1)
        block = text[start : nxt if nxt != -1 else len(text)]
        match = re.search(r"<<EOF\n(.*?)\nEOF", block, re.S)
        assert match, f"no python heredoc found in {name}"
        bodies[name] = _INTERPOLATION.sub("PLACEHOLDER", match.group(1))
    return bodies


BODIES = _process_bodies()


def test_every_process_body_was_found():
    """Guard against the extraction regex silently matching nothing."""
    assert set(BODIES) == set(LONGITUDINAL_PROCESSES)


@pytest.mark.parametrize("name", LONGITUDINAL_PROCESSES)
def test_body_is_valid_python(name):
    ast.parse(BODIES[name], filename=f"{name} script body")


@pytest.mark.parametrize("name", LONGITUDINAL_PROCESSES)
def test_every_import_resolves(name):
    """A renamed helper must fail here, not inside a task hours later."""
    tree = ast.parse(BODIES[name])
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if not node.module.startswith(("nhp_mri_prep", "fastsurfer")):
            continue
        module = importlib.import_module(node.module)
        for alias in node.names:
            assert hasattr(module, alias.name), (
                f"{name} imports {alias.name} from {node.module}, which does not "
                f"define it"
            )


@pytest.mark.parametrize("name", LONGITUDINAL_PROCESSES)
def test_keyword_arguments_match_the_functions_they_call(name):
    """The failure mode this file exists for: a kwarg the step no longer takes."""
    import inspect

    tree = ast.parse(BODIES[name])
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
            ("nhp_mri_prep", "fastsurfer")
        ):
            module = importlib.import_module(node.module)
            for alias in node.names:
                obj = getattr(module, alias.name, None)
                if inspect.isfunction(obj):
                    imported[alias.asname or alias.name] = obj

    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        func = imported.get(node.func.id)
        if func is None:
            continue
        params = inspect.signature(func).parameters
        accepts_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        for kw in node.keywords:
            if kw.arg is None or accepts_kwargs:
                continue
            assert kw.arg in params, (
                f"{name} calls {node.func.id}({kw.arg}=...), but "
                f"{func.__module__}.{func.__qualname__} has no such parameter"
            )
            checked += 1

    if imported:
        assert checked, f"{name}: no keyword arguments checked -- parser broken?"


# -------------------------------------------------------------------------------------------------
# Groovy footguns in the workflow files
# -------------------------------------------------------------------------------------------------

# `(0..<n)` is an IntRange -- an immutable AbstractList. Groovy's sort(Closure)
# mutates in place, so sorting one throws UnsupportedOperationException and takes
# the whole run down with it. `.toList().sort {}` (or `.sort(false) {}`) is the fix.
#
# This shipped in the longitudinal gather closures, where it fires only once
# groupTuple emits -- after every cross-sectional reconstruction has finished. So
# it cost hours per discovery and no test could see it, which is exactly the kind
# of thing a cheap static check should own.
_RANGE_LITERAL = re.compile(r"^\s*\d+\s*\.\.")


def _sorts_a_range_in_place(line):
    """True when `line` sorts a numeric range literal in place.

    A regex cannot do this: `(0..<xs.size()).sort {` and
    `(0..<xs.size()).toList().sort {` differ only in which `(` the `)` before
    `.sort` belongs to, which needs paren matching. So find each `.sort`, walk
    back over the balanced parens immediately preceding it, and ask whether what
    they enclose is a range literal.
    """
    for match in re.finditer(r"\.sort\s*[({]", line):
        head = line[: match.start()].rstrip()
        if not head.endswith(")"):
            continue
        depth, opener = 0, None
        for i in range(len(head) - 1, -1, -1):
            if head[i] == ")":
                depth += 1
            elif head[i] == "(":
                depth -= 1
                if depth == 0:
                    opener = i
                    break
        if opener is not None and _RANGE_LITERAL.match(head[opener + 1 : -1]):
            return True
    return False


@pytest.mark.parametrize("path", NF_FILES, ids=lambda p: p.name)
def test_no_in_place_sort_on_a_range(path):
    offenders = [
        f"{path.name}:{i}: {line.strip()}"
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if _sorts_a_range_in_place(line)
    ]
    assert not offenders, (
        "Sorting a Groovy range in place throws UnsupportedOperationException at "
        "runtime, aborting the pipeline. Use `.toList().sort { ... }`:\n  "
        + "\n  ".join(offenders)
    )


def test_the_range_sort_pattern_actually_matches():
    """Guard against the regex silently matching nothing forever."""
    assert _sorts_a_range_in_place("def order = (0..<ses_list.size()).sort { a, b -> 0 }")
    assert _sorts_a_range_in_place("(0..n).sort {")
    assert _sorts_a_range_in_place("x = (0 ..< n).sort(c)")
    # The fix must not read as the bug -- this is the pair a regex confuses.
    assert not _sorts_a_range_in_place(
        "def order = (0..<ses_list.size()).toList().sort { a, b -> 0 }"
    )
    assert not _sorts_a_range_in_place("things.sort { it }")
    assert not _sorts_a_range_in_place("(names).sort { it }")


# -------------------------------------------------------------------------------------------------
# Every process must declare a publishDir path
# -------------------------------------------------------------------------------------------------


def _processes_without_publish_dir(path):
    """Process names in `path` whose directive block declares no publishDir."""
    text = path.read_text()
    missing = []
    for match in re.finditer(r"^process\s+(\w+)\s*\{", text, re.M):
        nxt = text.find("\nprocess ", match.start() + 1)
        block = text[match.start() : nxt if nxt != -1 else len(text)]
        # Directives only -- everything before the script body.
        head = re.split(r"^\s*(?:script|shell|exec)\s*:", block, maxsplit=1, flags=re.M)[0]
        if "publishDir" not in head:
            missing.append(match.group(1))
    return missing


@pytest.mark.parametrize(
    "path", sorted((REPO / "modules").glob("*.nf")), ids=lambda p: p.name
)
def test_every_process_declares_a_publish_dir(path):
    """A process with no publishDir inherits a pathless default and dies.

    nextflow.config sets `publishDir = [mode: 'copy', overwrite: false]` for every
    process -- with no `path`. A process that declares its own supplies the path
    and is fine; one that declares none inherits that dict verbatim and is killed
    while being finalized, with "Target path for directive publishDir cannot be
    null" -- after its work is done, which for the base template is a multi-hour
    robust average.

    A process that should not publish still needs a path: give it one and add
    `enabled: false`, as ANAT_BIAS_CORRECTION and ANAT_SURFACE_BASE_TEMPLATE do.
    """
    missing = _processes_without_publish_dir(path)
    assert not missing, (
        f"{path.name}: {', '.join(missing)} declare no publishDir, so they "
        f"inherit the pathless default from nextflow.config and will fail at task "
        f"finalization. Add a publishDir with a real path (plus `enabled: false` "
        f"if it should not actually publish)."
    )


def test_the_config_default_publish_dir_still_has_no_path():
    """Pin the assumption the test above rests on.

    If nextflow.config ever gains a default `path`, that test is checking nothing
    and should be reconsidered rather than left as decoration.
    """
    config = (REPO / "nextflow.config").read_text()
    block = re.search(r"publishDir\s*=\s*\[(.*?)\]", config, re.S)
    assert block, "no default publishDir found in nextflow.config"
    assert "path" not in block.group(1), (
        "nextflow.config's default publishDir now sets a path; "
        "test_every_process_declares_a_publish_dir may no longer be needed"
    )
