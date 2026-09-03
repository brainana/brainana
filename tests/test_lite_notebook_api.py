"""Guards that the Brainana Lite notebook still matches the code it drives.

``examples/BrainanaLite.ipynb`` does not go through Nextflow: it imports brainana's step
functions and calls them directly. Nothing else in the test suite executes it, so a rename or
a signature change in ``src/`` can leave the notebook broken with no failing test --
``tests/test_version.py`` only checks the *ref string* it pins, not the calls it makes.

These checks are static and import-only: no subject data, no GPU, no ANTs, no network. They are
deliberately name- and signature-level rather than behavioural, to stay low-maintenance while
still catching whole-symbol drift.
"""

import ast
import inspect
import json
import re
import subprocess
from importlib import import_module
from pathlib import Path

import pytest

from nhp_mri_prep.config.config_io import load_yaml_config

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "examples" / "BrainanaLite.ipynb"
DEFAULTS = REPO / "src" / "nhp_mri_prep" / "config" / "defaults.yaml"
TEMPLATE_ZOO = REPO / "template_zoo"

# The notebook drives the anatomical path only; these are the packages it may import from.
BRAINANA_PACKAGES = ("nhp_mri_prep", "fastsurfer_nn", "fastsurfer_surfrecon", "nhp_skullstrip_nn")

# Templates offered in the notebook's USER SETTINGS cell.
OFFERED_TEMPLATES = ("NMT2Sym", "NMT2Asym", "MEBRAINS", "Yerkes19", "D99")


def _code_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _notebook_ast() -> list[ast.Module]:
    return [ast.parse(src) for src in _code_cells()]


def _brainana_imports() -> dict[str, str]:
    """Map each name the notebook imports from brainana to its source module."""
    imports: dict[str, str] = {}
    for tree in _notebook_ast():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in BRAINANA_PACKAGES:
                    for alias in node.names:
                        imports[alias.asname or alias.name] = node.module
    return imports


def _literal(name: str):
    """Value of a module-level ``name = <literal>`` assignment in any code cell."""
    for tree in _notebook_ast():
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets
            ):
                try:
                    return ast.literal_eval(node.value)
                except ValueError:
                    continue
    raise AssertionError(f"no literal assignment to {name} found in the notebook")


# -------------------------------------------------------------------------------------------------
# 1. Every imported symbol still exists
# -------------------------------------------------------------------------------------------------


def test_notebook_imports_resolve():
    """Every name the notebook imports from brainana must still be importable from that module."""
    missing = []
    for name, module in sorted(_brainana_imports().items()):
        try:
            mod = import_module(module)
        except ImportError as exc:  # pragma: no cover - a real packaging break
            missing.append(f"{module} (import failed: {exc})")
            continue
        if not hasattr(mod, name):
            missing.append(f"{module}.{name}")
    assert not missing, "notebook imports names that no longer exist: " + ", ".join(missing)


# -------------------------------------------------------------------------------------------------
# 2. Every call site still binds against the current signature
# -------------------------------------------------------------------------------------------------


def _resolve(name: str):
    imports = _brainana_imports()
    return getattr(import_module(imports[name]), name)


def _keywords(call: ast.Call) -> tuple[list[str], bool]:
    """Keyword names in a call, plus whether it also splats ``**kwargs``."""
    names = [kw.arg for kw in call.keywords if kw.arg is not None]
    splat = any(kw.arg is None for kw in call.keywords)
    return names, splat


def _iter_calls():
    """Yield (callee_name, positional_count, keyword_names, has_splat) for brainana calls.

    Handles the notebook's ``run_qc(qc_fn, save_f, title, **kwargs)`` indirection: the QC
    snapshot function is passed as an argument rather than called, so without this the QC
    signatures would silently stop being covered.
    """
    imported = _brainana_imports()
    for tree in _notebook_ast():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            fname = node.func.id
            if fname == "run_qc" and node.args and isinstance(node.args[0], ast.Name):
                qc_name = node.args[0].id
                if qc_name in imported:
                    names, splat = _keywords(node)
                    # run_qc forwards save_f itself, plus every keyword it was given.
                    yield qc_name, 0, ["save_f", *names], splat
                continue
            if fname in imported:
                names, splat = _keywords(node)
                yield fname, len(node.args), names, splat


def test_notebook_calls_bind_to_current_signatures():
    """Arguments the notebook passes must still be accepted by the functions it calls."""
    failures = []
    for name, n_pos, kwnames, splat in _iter_calls():
        target = _resolve(name)
        if not callable(target):
            continue
        try:
            sig = inspect.signature(target)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            continue
        params = sig.parameters
        # Every keyword must be an EXPLICIT parameter, even when the callee also takes
        # **kwargs. Without this, the QC snapshot functions (which all end in **kwargs) would
        # silently swallow a renamed argument -- e.g. full_fov_save_f -- and the guard would
        # pass while the figure quietly stopped being produced.
        unknown = [k for k in kwnames if k not in params]
        if unknown:
            failures.append(f"{name}{sig}: keyword(s) not in signature: {', '.join(unknown)}")
            continue
        if any(p.kind is p.VAR_KEYWORD for p in params.values()):
            splat = False  # the callee accepts **kwargs, so a splat cannot break it
        if splat:
            continue  # cannot know the splatted names statically
        try:
            sig.bind_partial(*[object()] * n_pos, **{k: object() for k in kwnames})
        except TypeError as exc:
            failures.append(f"{name}{sig}: {exc}")
    assert not failures, "notebook call sites no longer match src/ signatures:\n  " + "\n  ".join(
        failures
    )


# -------------------------------------------------------------------------------------------------
# 3. Config keys the notebook overrides must exist
# -------------------------------------------------------------------------------------------------


def _set_config_paths() -> list[str]:
    paths = []
    for tree in _notebook_ast():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "set_config"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                paths.append(node.args[0].value)
    return paths


def _raw_config_assignments() -> list[str]:
    """Dotted paths *assigned* through raw ``config[...]`` subscripting rather than set_config().

    Only assignments count: reading ``config["template"]["output_space"]`` is fine, since a
    mistyped read raises KeyError immediately.
    """
    found = []
    for tree in _notebook_ast():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                keys, cur = [], target
                while isinstance(cur, ast.Subscript):
                    if isinstance(cur.slice, ast.Constant):
                        keys.append(str(cur.slice.value))
                    cur = cur.value
                if isinstance(cur, ast.Name) and cur.id == "config" and keys:
                    found.append(".".join(reversed(keys)))
    return found


def test_notebook_uses_set_config_not_raw_subscript():
    """Config overrides must go through set_config(), which rejects keys nothing reads.

    load_config() does not validate, so a raw ``config["a"]["b"] = ...`` with a typo silently
    creates a key no code reads and the run proceeds on the unchanged default.
    """
    assert _set_config_paths(), "no set_config() calls found - did the config cell change shape?"
    raw = _raw_config_assignments()
    assert not raw, (
        "notebook assigns config keys by raw subscript ("
        + ", ".join(sorted(raw))
        + "); use set_config() so a mistyped key is an error instead of a silent no-op"
    )


def test_notebook_config_overrides_exist():
    """Every key the notebook overrides must exist in defaults.yaml (or be a declared exception)."""
    defaults = load_yaml_config(DEFAULTS)
    allowed = set(_literal("_UNDECLARED_OK"))
    missing = []
    for path in _set_config_paths():
        if path in allowed:
            continue
        node = defaults
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                missing.append(path)
                break
            node = node[key]
    assert not missing, (
        "notebook overrides config keys absent from defaults.yaml: "
        + ", ".join(sorted(missing))
        + " (add them to defaults.yaml, or to the notebook's _UNDECLARED_OK with a reason)"
    )


# -------------------------------------------------------------------------------------------------
# 4. The notebook clones the repo it ships in
# -------------------------------------------------------------------------------------------------


def _owner_repo(url: str) -> str | None:
    match = re.search(r"[:/]([^/:]+/[^/]+?)(?:\.git)?/?$", url.strip())
    return match.group(1).lower() if match else None


def test_notebook_clone_url_matches_origin():
    """BRAINANA_URL must point at this repository.

    The Colab badge opens the notebook from one repo; if BRAINANA_URL names another, users run
    this notebook against a different repository's code.
    """
    try:
        origin = subprocess.run(
            ["git", "-C", str(REPO), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("no git origin available")
    expected = _owner_repo(origin)
    actual = _owner_repo(_literal("BRAINANA_URL"))
    assert actual == expected, f"notebook clones {actual!r} but origin is {expected!r}"


# -------------------------------------------------------------------------------------------------
# 5. The sparse checkout fetches exactly the atlases the runtime will select
# -------------------------------------------------------------------------------------------------


def _notebook_atlas_helpers() -> dict:
    """Exec just the atlas-selection helpers out of the notebook."""
    wanted = {"_lite_res_value", "_lite_atlas_res", "_lite_atlas_paths"}
    ns: dict = {}
    for src in _code_cells():
        tree = ast.parse(src)
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
        if funcs:
            exec(compile(ast.Module(body=funcs, type_ignores=[]), "notebook", "exec"), ns)
    assert wanted <= ns.keys(), f"notebook is missing atlas helpers: {sorted(wanted - ns.keys())}"
    return ns


@pytest.mark.parametrize("template", OFFERED_TEMPLATES)
def test_sparse_checkout_matches_runtime_atlas_discovery(template):
    """The notebook must download the atlas files TemplateManager will later choose.

    The two used to disagree: the checkout filter required an exact ``res-05`` match while
    discover_atlases matches by space and picks the nearest resolution, so atlases stored at
    another resolution were never fetched and silently dropped from backprojection --
    TEMPLATE="D99" lost atlas-D99 itself.
    """
    from nhp_mri_prep.utils.templates import TemplateManager

    listing = [
        str(p.relative_to(REPO)) for p in TEMPLATE_ZOO.rglob("*") if p.is_file()
    ]
    picked = sorted(_notebook_atlas_helpers()["_lite_atlas_paths"](listing, template, "res-05"))

    manager = TemplateManager(template_dir=str(TEMPLATE_ZOO))
    runtime = sorted(str(p.relative_to(REPO)) for _, p in manager.discover_atlases(template, "res-05"))

    assert picked == runtime, (
        f"{template}: sparse checkout and runtime atlas discovery disagree.\n"
        f"  only in checkout: {sorted(set(picked) - set(runtime))}\n"
        f"  only in runtime:  {sorted(set(runtime) - set(picked))}"
    )


@pytest.mark.parametrize("template", OFFERED_TEMPLATES)
def test_every_offered_template_resolves(template):
    """Each TEMPLATE the notebook offers must have a res-05 brain volume in template_zoo."""
    from nhp_mri_prep.utils.templates import resolve_template

    assert Path(resolve_template(f"{template}:res-05")).is_file()
