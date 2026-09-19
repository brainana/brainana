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
    VALID_SYNTHESIS_LEVELS,
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


# anat.skullstripping_segmentation.fastSurferCNN.gpu_device has no field of its
# own on purpose: the generator mirrors general.gpu_device into it, so exposing a
# second control for the same device would let the two disagree.
GENERATOR_FIELDLESS_KEYS = {
    "anat.skullstripping_segmentation.fastSurferCNN.gpu_device",
}


def test_generator_field_ids_mirror_config_paths():
    """A generator field's id must be its config path with dots as underscores.

    test_generator_covers_every_default_key is satisfied by the leaf *name*
    appearing anywhere in the file, so ids drifted from the paths they stand for:
    ``bids_subjects`` for ``bids_filtering.subjects``, ``anat_fastSurferCNN_*``
    for keys two levels deeper than that. Nothing was broken by it, but reading
    or editing the collection code meant guessing which convention a given field
    followed -- the same guesswork that shipped ``session_longitudinal`` with no
    ``<option>``. Pinning the convention keeps the next field honest.
    """
    html = GENERATOR.read_text()
    missing = sorted(
        path
        for path, _ in _leaf_keys(load_yaml_config(DEFAULTS))
        if path not in GENERATOR_FIELDLESS_KEYS
        and f'id="{path.replace(".", "_")}"' not in html
    )
    assert not missing, (
        "config_generator.html has no field whose id is the config path for: "
        + ", ".join(missing)
        + ". Name each input id after its full path (dots -> underscores), or add "
        "it to GENERATOR_FIELDLESS_KEYS with the reason it has no control."
    )


@pytest.mark.parametrize("synthesis_level", VALID_SYNTHESIS_LEVELS)
def test_every_synthesis_level_validates(synthesis_level):
    """Every documented value must pass, so the enum cannot silently narrow."""
    validate_config({"anat": {"synthesis_level": synthesis_level}})


@pytest.mark.parametrize("synthesis_level", VALID_SYNTHESIS_LEVELS)
def test_generator_offers_every_synthesis_level(synthesis_level):
    """The generator must offer every value the validator accepts.

    test_generator_covers_every_default_key is deliberately name-level: it checks
    that each leaf *key* appears in the HTML, never that an enum's *values* do.
    So "session_longitudinal" shipped with four new longitudinal.* fields in the
    generator and no <option> to reach them -- the feature could not be turned on
    through the documented UI at all. This is the check that would have caught it.
    """
    html = GENERATOR.read_text()
    assert f'<option value="{synthesis_level}"' in html, (
        f"config_generator.html has no <option> for anat.synthesis_level "
        f"{synthesis_level!r}, so the UI cannot produce a config that selects it. "
        f"Add it to the #anat_synthesis_level <select>."
    )


def test_longitudinal_requires_surface_reconstruction():
    """The longitudinal stream lives inside surf recon.

    Selecting it with surf recon disabled would run an ordinary "session" pass
    and produce none of the longitudinal outputs, which is worth catching at
    config time rather than an hour into a run.
    """
    with pytest.raises(ValueError, match="surface_reconstruction"):
        validate_config(
            {
                "anat": {
                    "synthesis_level": "session_longitudinal",
                    "surface_reconstruction": {"enabled": False},
                }
            }
        )


def _long(**kwargs):
    return {"anat": {"surface_reconstruction": {"longitudinal": kwargs}}}


@pytest.mark.parametrize(
    "bad_longitudinal",
    [
        {"iscale": "yes"},
        {"subsample": 0},
        {"subsample": 1.5},
        {"max_cbv_dist": 0},
        {"max_cbv_dist": -1},
        {"max_cbv_dist": "far"},
        {"pial_blend_weight": 1.5},
        {"pial_blend_weight": -0.1},
        {"time_column": 3},
    ],
)
def test_longitudinal_subsection_rejects_bad_values(bad_longitudinal):
    """These knobs are read hours into a run, inside errorStrategy 'ignore' tasks.

    anat.surface_reconstruction.longitudinal had no validator at all, so a bad
    value did not produce an error message -- it produced a subject that quietly
    reconstructed nothing. Catch it at config load, naming the key.
    """
    key = next(iter(bad_longitudinal))
    with pytest.raises(ValueError, match=key):
        validate_config(_long(**bad_longitudinal))


def test_longitudinal_subsection_accepts_an_empty_key():
    """`longitudinal:` written with no value parses as None, not {}.

    Every consumer then did `.get('longitudinal', {}).get(...)` and hit
    AttributeError inside a Nextflow task. Null means "unset", not "invalid".
    """
    validate_config({"anat": {"surface_reconstruction": {"longitudinal": None}}})


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
