"""Consistency guards for the configuration system.

The config "contract" is described across several surfaces that are kept in sync
by hand:

- ``src/nhp_mri_prep/config/defaults.yaml``      — the definition (source of truth)
- ``src/nhp_mri_prep/config/config_validation.py`` — the validator
- ``docs/_static/config_generator.html``         — the generation UI

These tests fail when those surfaces drift apart — e.g. a parameter added to
``defaults.yaml`` but never surfaced in the generator, or defaults that no longer
pass their own validator. They are deliberately name-level (not structural) to
stay low-maintenance while still catching whole-parameter drift.
"""

from pathlib import Path

import pytest

from nhp_mri_prep.config.config_io import load_yaml_config
from nhp_mri_prep.config.config_validation import (
    validate_config,
    validate_confounds_config,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULTS = REPO / "src" / "nhp_mri_prep" / "config" / "defaults.yaml"
GENERATOR = REPO / "docs" / "_static" / "config_generator.html"

# Keys injected at load time, not user-facing parameters.
META_KEYS = {"_version", "_description"}


def _leaf_keys(node, prefix=""):
    """Yield (dotted_path, leaf_name) for every leaf in a nested dict."""
    for key, value in node.items():
        if key in META_KEYS:
            continue
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _leaf_keys(value, prefix=f"{path}.")
        else:
            yield path, key


def test_defaults_validate_clean():
    """defaults.yaml must always pass its own validator.

    Guards against adding a stricter validator that rejects the shipped default,
    or a default value that violates an existing rule.
    """
    validate_config(load_yaml_config(DEFAULTS))


def test_generator_covers_every_default_key():
    """Every parameter in defaults.yaml must be surfaced by the config generator.

    This is what would have caught the ``func.confounds`` drift: those keys lived
    in defaults.yaml but had no field in config_generator.html.
    """
    html = GENERATOR.read_text()
    missing = sorted(
        path
        for path, name in _leaf_keys(load_yaml_config(DEFAULTS))
        if name not in html
    )
    assert not missing, (
        "config_generator.html is missing parameters present in defaults.yaml: "
        + ", ".join(missing)
    )


@pytest.mark.parametrize(
    "bad_config",
    [
        {"anat": {"synthesis_level": "bogus"}},
        {"registration": {"anat2template_xfm_type": "nonlinear"}},
        {"registration": {"func2anat_xfm_type": "spline"}},
        {"registration": {"func2template_xfm_type": ""}},
        {"registration": {"keep_func_resolution": "yes"}},
    ],
)
def test_validators_reject_bad_values(bad_config):
    """Parameters that are enum-like or typed in defaults.yaml must be validated."""
    with pytest.raises((ValueError, TypeError)):
        validate_config(bad_config)


# -------------------------------------------------------------------------------------------------
# func.confounds numeric validation
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key", ["fd_outlier_threshold_mm", "std_dvars_outlier_threshold", "fd_radius_mm"]
)
def test_confounds_numeric_keys_accept_valid_values(key):
    validate_confounds_config({key: 0.5})
    validate_confounds_config({key: 3})  # ints are fine


@pytest.mark.parametrize(
    "key", ["fd_outlier_threshold_mm", "std_dvars_outlier_threshold", "fd_radius_mm"]
)
@pytest.mark.parametrize("bad", ["loose", None, [0.25], -1, 0])
def test_confounds_numeric_keys_reject_bad_values(key, bad):
    with pytest.raises(ValueError, match=key):
        validate_confounds_config({key: bad})


@pytest.mark.parametrize(
    "key", ["fd_outlier_threshold_mm", "std_dvars_outlier_threshold", "fd_radius_mm"]
)
def test_confounds_numeric_keys_reject_bool(key):
    """``bool`` is a subclass of ``int``: without an explicit check ``true`` would read as 1.0."""
    with pytest.raises(ValueError, match="must be a number"):
        validate_confounds_config({key: True})


def test_confounds_validation_ignores_absent_keys():
    validate_confounds_config({})
    validate_confounds_config({"enabled": True})
