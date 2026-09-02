"""Tests for QC report figure rendering (grouping, dividers, captions)."""

import re

import pytest

from nhp_mri_prep.quality_control.reports import (
    _FRAME_SHADING_CAPTION,
    _REPORT_CSS,
    FIGURE_DESCRIPTIONS,
    SAME_STEP_FIGURE_PAIRS,
    HtmlGenerator,
)


def _snapshot(snapshot_type: str, filename: str, figure_description: str = "") -> dict:
    return {
        "snapshot_type": snapshot_type,
        "filename": filename,
        "path": f"figures/{filename}",
        "description": snapshot_type,
        "figure_description": figure_description,
        "modality": "anatomical",
    }


def _fig_classes(html: str) -> list:
    """The class attribute of each rendered figure block, in document order."""
    return re.findall(r'<div class="(fig[^"]*)" id=', html)


def _render(snapshots: dict) -> str:
    return HtmlGenerator._render_snapshots({"snapshots": snapshots}, "anat")


def test_conform_pair_renders_without_a_divider():
    """The two conform views are one processing step, so no rule separates them."""
    html = _render(
        {
            "a": _snapshot("conform_fullfov_overlay", "a_desc-conformFullFOV_T1w.png"),
            "b": _snapshot("conform_overlay", "b_desc-conform_T1w.png"),
        }
    )
    assert _fig_classes(html) == ["fig", "fig fig-same-step"]
    # The rule is suppressed in CSS keyed on that class.
    assert ".fig.fig-same-step{border-top:none" in _REPORT_CSS


def test_unrelated_figures_keep_their_divider():
    html = _render(
        {
            "a": _snapshot("conform_overlay", "a_desc-conform_T1w.png"),
            "b": _snapshot("skullstrip_overlay", "b_desc-skullstrip_T1w.png"),
        }
    )
    assert _fig_classes(html) == ["fig", "fig"]


def test_conform_keeps_divider_when_the_fullfov_figure_is_absent():
    """A run without the full-FOV figure must not lose the separator before its conform figure."""
    html = _render(
        {
            "a": _snapshot("bias_correction_comparison", "a_desc-biascorrect_T1w.png"),
            "b": _snapshot("conform_overlay", "b_desc-conform_T1w.png"),
        }
    )
    assert _fig_classes(html) == ["fig", "fig"]


def test_same_step_pairs_are_declared_by_snapshot_type():
    assert ("conform_fullfov_overlay", "conform_overlay") in SAME_STEP_FIGURE_PAIRS


def test_fullfov_caption_keeps_its_line_break_unescaped():
    """The caption is injected as HTML; an escaped <br> would print as literal text."""
    html = _render(
        {
            "a": _snapshot(
                "conform_fullfov_overlay",
                "a_desc-conformFullFOV_T1w.png",
                FIGURE_DESCRIPTIONS["conformFullFOV"],
            )
        }
    )
    cap = re.search(r'<div class="cap">(.*?)</div>', html, re.S).group(1)
    assert "<br>" in cap
    assert "&lt;br&gt;" not in cap
    # Rendered captions are sentence-cased by the renderer, so the stored text stays lowercase.
    assert cap.startswith("Rigid registered T1w")
    assert "Lavender box marks the field of view" in cap


def test_conform_caption_points_back_at_the_fullfov_figure():
    html = _render(
        {
            "a": _snapshot(
                "conform_overlay",
                "a_desc-conform_T1w.png",
                FIGURE_DESCRIPTIONS["conform"]["anatomical"],
            )
        }
    )
    cap = re.search(r'<div class="cap">(.*?)</div>', html, re.S).group(1)
    assert cap.startswith(
        "Lavender box of the above full-field-of-view figure, enlarged"
    )


@pytest.mark.parametrize("desc", ["motion", "confounds"])
def test_timeseries_captions_break_before_the_shading_sentence(desc):
    """Both stacked timeseries figures put the band explanation on its own line."""
    html = _render(
        {
            "a": _snapshot(
                f"{desc}_x", f"a_desc-{desc}_bold.png", FIGURE_DESCRIPTIONS[desc]
            )
        }
    )
    cap = re.search(r'<div class="cap">(.*?)</div>', html, re.S).group(1)
    assert "&lt;br&gt;" not in cap
    head, _, tail = cap.partition("<br>")
    assert tail == _FRAME_SHADING_CAPTION
    assert "Gray bands" not in head


def test_stacked_figures_share_one_shading_sentence():
    """Motion and confounds shade the same frames; the wording must not drift apart."""
    assert (
        FIGURE_DESCRIPTIONS["motion"].split("<br>")[1]
        == FIGURE_DESCRIPTIONS["confounds"].split("<br>")[1]
    )


# ---------------------------------------------------------------------------
# Conform caption: the back-reference only holds when both figures are there
# ---------------------------------------------------------------------------


def _parse(tmp_path, filenames: list) -> dict:
    """Run snapshot discovery over a directory holding just these PNG names."""
    import logging

    from nhp_mri_prep.quality_control.reports import SnapshotProcessor

    for name in filenames:
        (tmp_path / name).write_bytes(b"")
    parsed = SnapshotProcessor.discover_and_parse(tmp_path, logging.getLogger(__name__))
    return parsed["snapshots"]


def _caption_for(snapshots: dict, snapshot_type: str) -> str:
    entry = next(v for v in snapshots.values() if v["snapshot_type"] == snapshot_type)
    return entry["figure_description"]


def test_conform_caption_is_standalone_without_the_full_fov_figure(tmp_path):
    """A run whose full-FOV figure was skipped must not cite a figure that is not there."""
    snapshots = _parse(tmp_path, ["sub-01_desc-conform_T1w.png"])
    caption = _caption_for(snapshots, "conform_overlay")
    assert "full-field-of-view" not in caption
    assert caption.startswith("rigid registered T1w")


def test_conform_caption_cites_the_full_fov_figure_when_it_is_present(tmp_path):
    snapshots = _parse(
        tmp_path,
        ["sub-01_desc-conform_T1w.png", "sub-01_desc-conformFullFOV_T1w.png"],
    )
    caption = _caption_for(snapshots, "conform_overlay")
    assert caption.startswith("lavender box of the above full-field-of-view figure")


def test_functional_conform_caption_is_unaffected(tmp_path):
    """The functional conform figure has never had a full-FOV counterpart."""
    snapshots = _parse(tmp_path, ["sub-01_task-rest_desc-conform_bold.png"])
    assert _caption_for(snapshots, "conform_overlay") == (
        "rigid registered BOLD (underlaid); target space (contour)"
    )
