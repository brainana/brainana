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
        # walk, not tree.body: several of these live inside the env-setup if/else.
        for node in ast.walk(tree):
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


def _run_qc_own_params() -> set[str]:
    """Parameter names run_qc consumes itself rather than forwarding to the QC function.

    Read from the notebook so adding a keyword to run_qc does not make this guard report it
    as an argument the snapshot function fails to accept.
    """
    for src in _code_cells():
        for node in ast.parse(src).body:
            if isinstance(node, ast.FunctionDef) and node.name == "run_qc":
                args = node.args
                return {
                    a.arg
                    for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]
                }
    return set()


def _iter_calls():
    """Yield (callee_name, positional_count, keyword_names, has_splat) for brainana calls.

    Handles the notebook's ``run_qc(qc_fn, save_f, title, **kwargs)`` indirection: the QC
    snapshot function is passed as an argument rather than called, so without this the QC
    signatures would silently stop being covered.
    """
    imported = _brainana_imports()
    run_qc_own = _run_qc_own_params()
    for tree in _notebook_ast():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            fname = node.func.id
            if fname == "run_qc" and node.args and isinstance(node.args[0], ast.Name):
                qc_name = node.args[0].id
                if qc_name in imported:
                    names, splat = _keywords(node)
                    # run_qc forwards save_f itself, plus every keyword it does not consume.
                    forwarded = [n for n in names if n not in run_qc_own]
                    yield qc_name, 0, ["save_f", *forwarded], splat
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


def _notebook_names(names: set[str]) -> dict:
    """Exec the named notebook functions and `_LITE_*` constants into a namespace."""
    ns: dict = {}
    for src in _code_cells():
        tree = ast.parse(src)
        picked = [
            n
            for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (
                isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "").startswith("_LITE_")
            )
        ]
        if any(isinstance(n, ast.FunctionDef) for n in picked):
            exec(compile(ast.Module(body=picked, type_ignores=[]), "notebook", "exec"), ns)
    missing = names - ns.keys()
    assert not missing, f"notebook is missing helpers: {sorted(missing)}"
    return ns


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return [p for p in out if p]


def test_sparse_checkout_keeps_the_editable_install_inputs():
    """The allowlist must retain everything the editable install and runtime read.

    It is an allowlist, so anything newly required by the build or by the packages Lite
    imports is silently dropped rather than erroring -- the failure would surface only as a
    broken Colab run.
    """
    ns = _notebook_names({"_lite_code_paths"})
    ns.setdefault("DEMO_NIFTI", "exam_T1w_ple.nii.gz")
    kept = set(ns["_lite_code_paths"](_tracked_files()))

    import tomllib

    with open(REPO / "pyproject.toml", "rb") as fh:
        pyproject = tomllib.load(fh)
    readme = pyproject["project"]["readme"]
    readme_file = readme["file"] if isinstance(readme, dict) else readme

    required = {"pyproject.toml", readme_file}
    assert required <= kept, f"allowlist drops build inputs: {sorted(required - kept)}"

    # Every module of the packages the notebook actually imports must survive.
    for package in ("nhp_mri_prep", "fastsurfer_nn", "fastsurfer_surfrecon"):
        modules = {p for p in _tracked_files() if p.startswith(f"src/{package}/") and p.endswith(".py")}
        assert modules, f"no modules found for {package}"
        assert modules <= kept, f"allowlist drops {package} modules: {sorted(modules - kept)[:5]}"

    # fastSurferCNN inference weights: the default skullstripping method needs these.
    weights = {p for p in _tracked_files() if p.startswith("src/fastsurfer_nn/pretrained_model/")}
    assert weights and weights <= kept, "allowlist drops the fastSurferCNN weights"


def test_sparse_checkout_keeps_the_anat_skullstrip_weights():
    """anat_conform runs its OWN skullstripping, so the anat brainmask weights are required.

    Excluding them -- on the reasonable-sounding assumption that lite segments with
    fastSurferCNN and so never touches nhp_skullstrip_nn -- made the notebook die at Step 2
    with a FileNotFoundError raised deep inside conform. Nothing static caught it; only a real
    run did. The filename is read from prediction.py's own mapping, so a rename there fails
    here rather than on someone's Colab runtime.
    """
    source = (REPO / "src/nhp_skullstrip_nn/inference/prediction.py").read_text()
    match = re.search(r"model_mapping\s*=\s*(\{[^}]*\})", source)
    assert match, "could not find model_mapping in prediction.py"
    weights = ast.literal_eval(match.group(1))["anat"]

    ns = _notebook_names({"_lite_code_paths"})
    ns.setdefault("DEMO_NIFTI", "exam_T1w_ple.nii.gz")
    kept = set(ns["_lite_code_paths"](_tracked_files()))
    expected = f"src/nhp_skullstrip_nn/pretrained_model/{weights}"
    assert expected in kept, f"sparse checkout drops the anat skullstrip weights ({expected})"


def test_sparse_checkout_stays_lean():
    """The checkout must not drift back toward cloning the whole repository.

    It previously excluded only template_zoo/ and then checked out ~354 MB to avoid 122 MB.
    The ceiling is deliberately loose; it exists to catch a filter that stops filtering.
    """
    ns = _notebook_names({"_lite_code_paths"})
    ns.setdefault("DEMO_NIFTI", "exam_T1w_ple.nii.gz")
    kept = ns["_lite_code_paths"](_tracked_files())
    total = sum((REPO / p).stat().st_size for p in kept if (REPO / p).is_file())
    megabytes = total / 1048576
    assert megabytes < 150, (
        f"lite code checkout grew to {megabytes:.0f} MB (expected well under 150). "
        "Check whether a large directory now matches _LITE_CODE_DIRS."
    )


def test_lite_pins_an_antspyx_floor():
    """The notebook must floor antspyx, or uv silently picks a version with no wheel.

    antspyx 0.6.x caps scipy<1.16 and numpy<2.4. Because brainana floors those unbounded,
    uv prefers the newest scipy and backtracks antspyx to 0.5.3 -- the last release without
    that cap, and the last without a cp313 wheel. On Python 3.13 (Colab) that silently turns
    a wheel download into an ITK source build: tens of minutes, frequently an OOM. The floor
    is applied at the notebook's install call rather than in the [lite] extra, so uv.lock,
    the Docker image and CI keep their own resolution.

    0.6.0 is the first release with cp313 wheels, so that is the minimum acceptable floor.
    """
    floor = _literal("LITE_ANTSPYX_FLOOR")
    match = re.fullmatch(r"antspyx>=(\d+)\.(\d+)\.(\d+)", floor.strip())
    assert match, f"expected an antspyx>=X.Y.Z floor, got {floor!r}"
    assert tuple(int(g) for g in match.groups()) >= (0, 6, 0), (
        f"antspyx floor {floor!r} is below 0.6.0, the first release with cp313 wheels"
    )


def test_lite_install_and_preflight_use_the_same_arguments():
    """The dry-run preflight must describe the install that actually runs.

    They are separate subprocess calls; if they drift, the plan reported to the user is not
    the plan executed -- which is worse than reporting nothing.
    """
    installs = []
    for src in _code_cells():
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Call):
                continue
            fname = getattr(node.func, "attr", None)
            if fname not in {"run", "Popen"} or not node.args:
                continue
            flat = ast.dump(node.args[0])
            # Must be a uv call: the `pip install -q uv` bootstrap also says pip/install.
            if all(tok in flat for tok in ("'_uv'", "'pip'", "'install'")):
                installs.append(flat)
    assert len(installs) >= 2, f"expected a preflight and a real install, found {len(installs)}"
    assert all("_install_args" in f for f in installs), (
        "the uv preflight and the real install must share _install_args so they cannot diverge"
    )


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
