"""Null-safe config reads for values consumed inside Nextflow tasks.

YAML has two ways to say "I did not set this", and only one of them is what
`dict.get(key, default)` handles. A key written with an empty value --

    longitudinal:
    max_cbv_dist:

-- parses as ``None``, not as a missing key and not as ``{}``. So
``cfg.get("longitudinal", {}).get("iscale")`` raises AttributeError and
``float(cfg.get("max_cbv_dist", 3.5))`` raises TypeError, both from inside a
process that carries ``errorStrategy 'ignore'`` -- where a traceback is
indistinguishable from the subject simply producing nothing.
"""

import pytest

from nhp_mri_prep.utils.nextflow import config_section, config_value

NULL_SECTION = {"anat": {"surface_reconstruction": {"longitudinal": None}}}
REAL_SECTION = {
    "anat": {
        "surface_reconstruction": {
            "longitudinal": {"iscale": True, "max_cbv_dist": None, "subsample": 200}
        }
    }
}
DOTTED = "anat.surface_reconstruction.longitudinal"


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"anat": None},
        {"anat": {}},
        NULL_SECTION,
        {"anat": {"surface_reconstruction": 7}},
    ],
)
def test_missing_or_null_sections_read_as_empty(config):
    assert config_section(config, DOTTED) == {}


def test_a_real_section_is_returned_unchanged():
    assert config_section(REAL_SECTION, DOTTED)["iscale"] is True


def test_a_null_value_falls_back_to_the_default():
    """The TypeError case: `max_cbv_dist:` with nothing after it."""
    assert config_value(REAL_SECTION, f"{DOTTED}.max_cbv_dist", 3.5, float) == 3.5


def test_a_null_section_falls_back_to_the_default():
    """The AttributeError case: `longitudinal:` with nothing after it."""
    assert config_value(NULL_SECTION, f"{DOTTED}.max_cbv_dist", 3.5, float) == 3.5


def test_a_present_value_is_cast():
    value = config_value(REAL_SECTION, f"{DOTTED}.subsample", None, int)
    assert value == 200 and isinstance(value, int)


def test_the_default_is_never_cast():
    """cast=float must not turn a None default into a TypeError of its own."""
    assert config_value({}, f"{DOTTED}.subsample", None, int) is None


def test_a_top_level_key_needs_no_dots():
    assert config_value({"threads": 4}, "threads", 1) == 4
