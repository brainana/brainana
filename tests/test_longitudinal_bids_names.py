"""Naming for the base template and the longitudinal timepoints.

All of a subject's QC figures land in one ``sub-<id>/figures`` directory, so the
filename is the only thing separating a session's cross-sectional figure from its
longitudinal one -- and ``publishDir`` runs with ``overwrite: false``, so a
collision is not an error, it is a missing figure.

The first attempt at this was a pair of `^`-anchored Groovy regexes applied to
``bids_name``. That value is an absolute path (``create_synthesized_bids_filename``
returns ``original_file.parent / basename`` for session-level jobs), so the
anchors never matched, the marker entity was never injected, and every
longitudinal figure was silently dropped. Hence: one Python rule, and these tests.

The marker is ``space-base``. It was ``acq-base`` / ``acq-long``, which was wrong
in a way these tests could not express: ``acq`` is an identity entity the raw data
owns, so the old rule *destroyed* a timepoint's real ``acq-mprage`` -- and the old
test asserted that destruction as the contract (``"mprage" not in name``). A
base-seeded reconstruction is in base space; ``space`` is brainana's slot for a
frame, so nothing inherited has to be overwritten to say so.
"""

import pytest

from nhp_mri_prep.utils.bids import (
    create_bids_output_filename,
    longitudinal_bids_name,
)

CROSS = "sub-01_ses-a_T1w.nii.gz"
ABSOLUTE = "/data/bids/sub-01/ses-a/anat/sub-01_ses-a_T1w.nii.gz"
WITH_ENTITIES = "sub-01_ses-a_acq-mprage_run-1_T1w.nii.gz"


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("base", "sub-01_space-base_T1w.nii.gz"),
        ("long", "sub-01_ses-a_space-base_T1w.nii.gz"),
    ],
)
def test_the_basic_shapes(kind, expected):
    assert longitudinal_bids_name(CROSS, kind) == expected


@pytest.mark.parametrize("kind", ["base", "long"])
def test_an_absolute_path_gives_the_same_answer_as_a_basename(kind):
    """The regression. Callers hand this an absolute path, not a bare filename."""
    assert longitudinal_bids_name(ABSOLUTE, kind) == longitudinal_bids_name(CROSS, kind)


def test_a_raw_acquisition_label_survives_a_timepoint():
    """The reason the marker moved out of acq.

    acq is an identity entity: it records how the scan was acquired, and only the
    raw data may set it. The old rule overwrote it, so a subject scanned with two
    protocols in one session lost the distinction and both timepoints resolved to
    one filename -- which publishDir, running overwrite:false, turns into a
    missing file rather than an error.
    """
    name = longitudinal_bids_name(WITH_ENTITIES, "long")
    assert "acq-mprage" in name
    assert name.count("acq-") == 1


def test_only_one_space_entity_is_emitted():
    """A timepoint whose input already declares a frame gets it replaced, not stacked.

    Two space tokens would not survive a round trip: parse_bids_entities keeps the
    last, so the first is silently dropped.
    """
    name = longitudinal_bids_name("sub-01_ses-a_space-scanner_T1w.nii.gz", "long")
    assert name.count("space-") == 1
    assert "space-base" in name


def test_session_level_entities_survive_on_a_timepoint():
    """run- distinguishes real files; dropping it could collide two timepoints."""
    assert longitudinal_bids_name(WITH_ENTITIES, "long") == (
        "sub-01_ses-a_acq-mprage_run-1_space-base_T1w.nii.gz"
    )


def test_entities_stay_in_canonical_bids_order():
    """Identity entities precede space in BIDS_ENTITY_ORDER; the rebuild re-sorts."""
    name = longitudinal_bids_name(WITH_ENTITIES, "long")
    assert name.index("_acq-") < name.index("_run-") < name.index("_space-")


def test_the_base_name_does_not_depend_on_which_session_it_saw():
    """The base spans every session, so a session-level entity would be arbitrary.

    base_qc_input used to build this from bids_names[order[0]], which made the
    base's published name depend on a sort order that means nothing.
    """
    a = longitudinal_bids_name("sub-01_ses-a_run-1_T1w.nii.gz", "base")
    b = longitudinal_bids_name("sub-01_ses-b_acq-mprage_T1w.nii.gz", "base")
    assert a == b == "sub-01_space-base_T1w.nii.gz"


def test_two_sessions_get_different_longitudinal_names():
    assert longitudinal_bids_name("sub-01_ses-a_T1w.nii.gz", "long") != (
        longitudinal_bids_name("sub-01_ses-b_T1w.nii.gz", "long")
    )


def test_the_qc_figure_no_longer_collides_with_the_cross_sectional_one():
    """The bug, stated as the thing it broke.

    Both figures are named through create_bids_output_filename and published into
    the same sub-<id>/figures directory.
    """
    suffix, modality = "desc-surfReconTissueSeg", "T1w"
    cross_fig = create_bids_output_filename(CROSS, suffix, modality)
    long_fig = create_bids_output_filename(
        longitudinal_bids_name(ABSOLUTE, "long"), suffix, modality
    )
    base_fig = create_bids_output_filename(
        longitudinal_bids_name(ABSOLUTE, "base"), suffix, modality
    )
    assert len({cross_fig, long_fig, base_fig}) == 3


def test_the_modality_is_preserved():
    assert longitudinal_bids_name("sub-01_ses-a_T2w.nii.gz", "long").endswith(
        "_T2w.nii.gz"
    )


def test_an_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="kind"):
        longitudinal_bids_name(CROSS, "sideways")


def test_a_name_without_a_subject_is_rejected():
    with pytest.raises(ValueError, match="sub-"):
        longitudinal_bids_name("ses-a_T1w.nii.gz", "long")
