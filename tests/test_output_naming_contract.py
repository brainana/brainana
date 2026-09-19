"""The output-naming contract: one declared slot per concern, and nothing silent.

brainana names its outputs by copying the raw file's entities and adding its own.
It had no single rule for that, so "which pipeline product is this" was written
into whichever entity was nearest to hand, and three of those collided:

* ``desc`` carried both the derivative variant (``preproc``, ``brain``) and the
  processing stage (``conform``, ``func2target``, 20-odd values). Stacking the
  second onto the first gave nine published figures with two ``desc-`` tokens --
  and ``parse_bids_entities`` keeps only the last, so the round trip dropped the
  first without a word.
* ``acq`` was borrowed to mark the longitudinal stream (``acq-base`` /
  ``acq-long``) in a dataset that also uses ``acq`` for a real acquisition label.
  A timepoint's own ``acq-mprage`` was destroyed, and the QC report -- which
  groups figures by identity entities -- read one session as two.
* the two naming helpers disagreed on where an unknown raw entity goes: one kept
  its position, the other alphabetised it.

These tests pin the rule instead of the symptoms. The failure mode they guard
against is quiet: QC figures all land in one flat ``sub-<id>/figures``, every QC
process runs ``errorStrategy 'ignore'``, and ``publishDir`` defaults to
``overwrite: false`` -- so two names that collapse to one are a missing file, not
an error.
"""

import re
from pathlib import Path

import pytest

from nhp_mri_prep.steps.anatomical import _rename_to_scanner_space
from nhp_mri_prep.utils.bids import (
    BIDS_ENTITY_ORDER,
    DECLARED_DEVIATIONS,
    ENTITY_ROLES,
    NamingConflictError,
    create_bids_filename,
    create_bids_output_filename,
    derive_output_name,
    parse_bids_entities,
    validate_output_name,
)

RAW = "sub-01_ses-a_task-rest_run-1_bold.nii.gz"


# --------------------------------------------------------------------------- #
# derive_output_name -- the single constructor
# --------------------------------------------------------------------------- #


def test_it_copies_identity_entities_and_adds_its_own():
    assert (
        derive_output_name(
            RAW, suffix="bold", set_entities={"space": "NMT2Sym", "desc": "preproc"}
        )
        == "sub-01_ses-a_task-rest_run-1_space-NMT2Sym_desc-preproc_bold.nii.gz"
    )


def test_setting_an_identity_entity_is_refused():
    """The bug this table exists for.

    acq records how the scan was acquired. Only the raw data may set it, so a step
    that wants to mark its own product has to use a slot brainana owns.
    """
    with pytest.raises(NamingConflictError, match="identity entity"):
        derive_output_name(
            "sub-01_ses-a_acq-mprage_T1w.nii.gz",
            suffix="T1w",
            set_entities={"acq": "long"},
        )


def test_an_identity_entity_may_be_reaffirmed_with_the_same_value():
    """Only a *change* is a conflict; restating what is already true is not."""
    assert (
        derive_output_name(
            "sub-01_ses-a_acq-mprage_T1w.nii.gz",
            suffix="T1w",
            set_entities={"acq": "mprage", "desc": "preproc"},
        )
        == "sub-01_ses-a_acq-mprage_desc-preproc_T1w.nii.gz"
    )


def test_an_entity_brainana_owns_is_replaced_not_stacked():
    """space and desc are brainana's, so re-setting them is the normal case."""
    name = derive_output_name(
        "sub-01_ses-a_space-scanner_desc-conform_T1w.nii.gz",
        suffix="T1w",
        set_entities={"space": "T1w", "desc": "preproc"},
    )
    assert name == "sub-01_ses-a_space-T1w_desc-preproc_T1w.nii.gz"
    assert name.count("space-") == 1 and name.count("desc-") == 1


def test_an_entity_that_stopped_being_true_must_be_dropped_explicitly():
    assert (
        derive_output_name(
            RAW,
            suffix="boldref",
            set_entities={"desc": "coreg"},
            drop_entities=("run",),
        )
        == "sub-01_ses-a_task-rest_desc-coreg_boldref.nii.gz"
    )


def test_an_unknown_raw_entity_is_preserved_in_place():
    """Dropping it would collide two runs that differ only by it.

    Real dev-test data carries ``test-xxx``. It is not a BIDS entity and brainana
    has nothing to say about it, so it travels through untouched.
    """
    name = derive_output_name(
        "sub-01_ses-002_task-rest_run-3_test-xxx_bold.nii.gz",
        suffix="bold",
        set_entities={"desc": "conform"},
    )
    assert name == "sub-01_ses-002_task-rest_run-3_test-xxx_desc-conform_bold.nii.gz"


def test_an_entity_in_the_suffix_is_refused():
    """Entities in the suffix are how a second desc used to get in."""
    with pytest.raises(ValueError, match="set_entities"):
        derive_output_name(RAW, suffix="desc-preproc_bold")


def test_a_missing_suffix_is_refused():
    with pytest.raises(ValueError, match="suffix"):
        derive_output_name(RAW, suffix="")


# --------------------------------------------------------------------------- #
# create_bids_output_filename -- the stem-preserving path, now de-duplicating
# --------------------------------------------------------------------------- #


def test_the_stem_preserving_path_no_longer_stacks_a_second_desc():
    """The nine published figures. It de-duplicated space only."""
    name = create_bids_output_filename(
        "sub-01_ses-a_space-NMT2Sym_desc-preproc_bold.nii.gz",
        "desc-func2target",
        "bold",
    )
    assert name.count("desc-") == 1
    assert name == "sub-01_ses-a_space-NMT2Sym_desc-func2target_bold.nii.gz"


def test_a_boldref_input_loses_its_whole_suffix_token():
    """'_bold' occurs inside '_boldref', and the replace was unbounded.

    'sub-01_ses-001_boldref' came back as 'sub-01_ses-001ref'.
    """
    assert (
        create_bids_output_filename(
            "sub-01_ses-001_boldref.nii.gz", "desc-preproc", "bold"
        )
        == "sub-01_ses-001_desc-preproc_bold.nii.gz"
    )


# --------------------------------------------------------------------------- #
# Round-trip: build -> parse -> build must be a fixed point
# --------------------------------------------------------------------------- #

# Every published name shape, mirroring docs/outputs.rst. Placeholders resolved
# the way a real run does.
_PLACEHOLDERS = {
    "<ses_prefix>": "sub-01_ses-001",
    "<run_prefix>": "sub-01_ses-001_task-rest_run-1",
    "<template>": "NMT2Sym",
    "<name>": "ARM2",
    "<ext>": ".h5",
    "<id>": "01",
}


def _documented_names():
    """Name patterns literal in docs/outputs.rst, with placeholders resolved."""
    text = (Path(__file__).resolve().parents[1] / "docs" / "outputs.rst").read_text()
    patterns = set(re.findall(r"``(<[a-z_]+>_[^`]+|atlas-<name>_[^`]+)``", text))
    resolved = set()
    for pattern in patterns:
        name = pattern
        for placeholder, value in _PLACEHOLDERS.items():
            name = name.replace(placeholder, value)
        if "<" not in name:
            resolved.add(name)
    return sorted(resolved)


def test_the_docs_list_names_we_can_check():
    """A guard on the extractor: a silent zero would make the two tests below vacuous."""
    assert len(_documented_names()) > 20


@pytest.mark.parametrize("name", _documented_names())
def test_every_documented_name_is_clean(name):
    assert validate_output_name(name) == [], name


@pytest.mark.parametrize("name", _documented_names())
def test_every_documented_name_survives_a_round_trip(name):
    """parse -> build must return the same name.

    This is what the entity-order table buys. ``stat`` and ``hemi`` were missing
    from it, so they fell into the alphabetised 'others' slot, which is emitted
    *before* space and desc -- and a tSNR map rebuilt from its own entities came
    back as '..._stat-tsnr_desc-preproc_...', a different file.

    The two declared deviations cannot round-trip by construction and are exempt:
    a '_brain' tail is invisible to a parser, and an atlas name has no suffix at all.
    """
    stem = name
    for ext in (".nii.gz", ".tsv", ".json", ".h5", ".mat", ".func.gii"):
        if stem.endswith(ext):
            stem, extension = stem[: -len(ext)], ext
            break
    else:
        pytest.skip(f"unhandled extension: {name}")

    tokens = stem.split("_")
    suffix_tokens = [t for t in tokens if "-" not in t]
    if stem.startswith("atlas-") or len(suffix_tokens) != 1:
        pytest.skip(f"declared deviation, cannot round-trip: {name}")

    rebuilt = create_bids_filename(
        parse_bids_entities(name), suffix=suffix_tokens[0], extension=extension
    )
    assert rebuilt == name


# --------------------------------------------------------------------------- #
# validate_output_name -- one case per defect class the audit found
# --------------------------------------------------------------------------- #


def test_a_duplicate_entity_is_reported():
    (violation,) = validate_output_name(
        "sub-01_ses-a_space-NMT2Sym_desc-preproc_desc-func2target_bold.png"
    )
    assert "duplicate entity 'desc'" in violation


def test_entities_out_of_order_are_reported():
    (violation,) = validate_output_name("sub-01_desc-preproc_space-T1w_T1w.nii.gz")
    assert "canonical order" in violation


def test_a_multi_token_suffix_is_reported():
    (violation,) = validate_output_name("sub-01_ses-a_desc-preproc_bold_cleaned.nii.gz")
    assert "multi-token suffix" in violation


@pytest.mark.parametrize(
    "name, deviation",
    [
        (
            "sub-01_ses-a_space-T1w_desc-preproc_T1w_brain.nii.gz",
            "brain_suffix_tail",
        ),
        ("atlas-ARM1_space-T1w_sub-01_ses-001.nii.gz", "atlas_subject_last"),
        (
            "sub-01_ses-002_task-rest_run-3_test-xxx_desc-conform_bold.png",
            "unknown_raw_entity",
        ),
    ],
)
def test_a_declared_deviation_is_not_a_violation(name, deviation):
    """Published on purpose, with a consumer that depends on it.

    brainana-viewer's data contract names the '_brain' tail in its base-volume
    fallback chain and discovers atlases by the leading 'atlas-' token; an unknown
    raw entity is what keeps two otherwise-identical runs apart. Each is recorded
    in DECLARED_DEVIATIONS with its reason.
    """
    assert deviation in DECLARED_DEVIATIONS
    assert validate_output_name(name) == []


# --------------------------------------------------------------------------- #
# The tables themselves
# --------------------------------------------------------------------------- #


def test_every_entity_brainana_orders_has_a_role():
    """An entity in one table and not the other is a slot nobody owns."""
    ordered = {e for e in BIDS_ENTITY_ORDER if e != "others"}
    assert ordered == set(ENTITY_ROLES)


def test_the_longitudinal_marker_is_not_an_identity_entity():
    """The whole point. Restated as an assertion so it cannot regress quietly."""
    from nhp_mri_prep.utils.bids import longitudinal_bids_name

    marker_entity = "space"
    assert ENTITY_ROLES[marker_entity] != "identity"
    assert f"{marker_entity}-base" in longitudinal_bids_name(
        "sub-01_ses-a_T1w.nii.gz", "base"
    )


# --------------------------------------------------------------------------- #
# The atlas tree's token order, which is a declared deviation with consumers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            "atlas-ARM1_space-T1w_sub-01.nii.gz",
            "atlas-ARM1_space-scanner_sub-01.nii.gz",
        ),
        (
            "atlas-ARM1_space-T1w_sub-032309m_ses-001.nii.gz",
            "atlas-ARM1_space-scanner_sub-032309m_ses-001.nii.gz",
        ),
        (
            "atlas-retinotopy_space-T1w_sub-01.json",
            "atlas-retinotopy_space-scanner_sub-01.json",
        ),
    ],
)
def test_the_scanner_space_atlas_keeps_its_frame_token_in_place(source, expected):
    """The frame is replaced where it stands, not moved to the end.

    Caught by diffing a real output tree: routing this through
    replace_bids_space() looked like a tightening, but that helper re-inserts at a
    BIDS anchor and an atlas name has neither a '_desc-' nor a trailing modality
    suffix -- so it appended, giving 'atlas-ARM1_sub-01_space-scanner.nii.gz'.
    That no longer matches 'atlas-<name>_space-*', which is how every atlas
    consumer discovers these files, brainana-viewer's data contract included.
    """
    assert _rename_to_scanner_space(source) == expected


def test_a_second_frame_token_is_dropped_rather_than_kept():
    """Two space entities would not survive a parse; the extra goes."""
    name = _rename_to_scanner_space("atlas-X_space-A_sub-01_space-B.nii.gz")
    assert name.count("space-") == 1
    assert name == "atlas-X_space-scanner_sub-01.nii.gz"
    assert validate_output_name(name) == []


def test_an_atlas_with_no_frame_gains_exactly_one():
    assert (
        _rename_to_scanner_space("atlas-X_sub-01.nii.gz")
        == "atlas-X_sub-01_space-scanner.nii.gz"
    )


def test_the_atlas_convention_has_a_shape_of_its_own():
    """The exemption in test_a_declared_deviation_is_not_a_violation is not a hole.

    Atlas names are exempt from the general entity-order check because they
    publish sub- last on purpose. That exemption let a real regression through:
    moving space- to the end of an atlas name validated clean, and was caught only
    by diffing an output tree against the previous run. So the convention itself
    is checked -- the frame token must follow the atlas token, which is the
    'atlas-<name>_space-*' shape consumers discover by.
    """
    assert validate_output_name("atlas-ARM1_space-scanner_sub-01.nii.gz") == []
    (violation,) = validate_output_name("atlas-ARM1_sub-01_space-scanner.nii.gz")
    assert "atlas-<name>_space-<frame>" in violation
    # A frameless sidecar carries no space entity and is not subject to the shape.
    assert validate_output_name("atlas-ARM1.tsv") == []
