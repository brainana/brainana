"""Tests for aligned frame-index x-axes between motion and confounds QC figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nhp_mri_prep.quality_control.mri_plotting import (
    CONFOUNDS_QC_MARGINS,
    MOTION_OUTLIER_PREFIX,
    NONSTEADY_OUTLIER_PREFIX,
    TIMESERIES_QC_MARGINS,
    _contiguous_frame_spans,
    create_confounds_plot,
    create_motion_plot,
    frame_xlim,
    frame_xticks,
    outlier_frame_indices,
    save_timeseries_qc_figure,
    shade_frame_spans,
)
from nhp_mri_prep.quality_control import snapshots
from nhp_mri_prep.quality_control.snapshots import create_motion_correction_qc


def _apply_timeseries_margins(fig: plt.Figure) -> None:
    fig.subplots_adjust(**TIMESERIES_QC_MARGINS)


def _apply_confounds_margins(fig: plt.Figure) -> None:
    fig.subplots_adjust(**CONFOUNDS_QC_MARGINS)


@pytest.mark.parametrize("n_frames", [8, 120])
def test_motion_and_confounds_share_bottom_axis_geometry(n_frames: int) -> None:
    rng = np.random.RandomState(0)
    motion_data = np.cumsum(rng.normal(0, 0.01, size=(n_frames, 6)), axis=0)
    motion_fig = create_motion_plot(motion_data, title="")
    _apply_timeseries_margins(motion_fig)

    confounds_df = pd.DataFrame(
        {
            "framewise_displacement": np.abs(rng.normal(0, 0.05, n_frames)),
            "std_dvars": np.abs(rng.normal(1, 0.1, n_frames)),
            "global_signal": rng.normal(1000, 50, n_frames),
        }
    )
    confounds_fig = create_confounds_plot(confounds_df)
    _apply_confounds_margins(confounds_fig)

    motion_ax = motion_fig.axes[1]
    confounds_ax = confounds_fig.axes[-1]

    motion_pos = motion_ax.get_position()
    confounds_pos = confounds_ax.get_position()
    assert motion_pos.x0 == pytest.approx(confounds_pos.x0, abs=1e-6)
    assert motion_pos.width == pytest.approx(confounds_pos.width, abs=1e-6)
    assert motion_ax.get_xlim() == pytest.approx(confounds_ax.get_xlim(), abs=1e-6)
    assert motion_ax.get_xlim() == pytest.approx(frame_xlim(n_frames), abs=1e-6)
    assert np.allclose(motion_ax.get_xticks(), confounds_ax.get_xticks())
    assert confounds_ax.get_xlabel() == ""
    assert motion_fig.axes[1].spines["left"].get_visible()
    assert not confounds_fig.axes[0].spines["left"].get_visible()

    motion_fig.set_dpi(200)
    confounds_fig.set_dpi(200)
    motion_fig.canvas.draw()
    confounds_fig.canvas.draw()
    motion_frame0_px = motion_fig.axes[1].transData.transform((0, 0))[0]
    confounds_frame0_px = confounds_fig.axes[0].transData.transform((0, 0))[0]
    motion_last_px = motion_fig.axes[1].transData.transform((n_frames - 1, 0))[0]
    confounds_last_px = confounds_fig.axes[0].transData.transform((n_frames - 1, 0))[0]
    assert motion_frame0_px == pytest.approx(confounds_frame0_px, abs=1e-6)
    assert motion_last_px == pytest.approx(confounds_last_px, abs=1e-6)

    plt.close(motion_fig)
    plt.close(confounds_fig)


def test_frame_xticks_rule() -> None:
    assert np.array_equal(frame_xticks(5), np.arange(5))
    assert len(frame_xticks(120)) == 10
    assert frame_xticks(120)[0] == 0
    assert frame_xticks(120)[-1] == 119
    assert frame_xlim(120) == (0.0, 119.5)
    assert frame_xlim(1) == (0.0, 0.0)


def test_save_timeseries_qc_figure_writes_png(tmp_path: Path) -> None:
    motion_data = np.zeros((20, 6))
    fig = create_motion_plot(motion_data, title="")
    out = tmp_path / "motion.png"
    save_timeseries_qc_figure(fig, out)
    plt.close(fig)
    assert out.is_file()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------------------------
# Flagged-frame shading (non-steady-state + motion outliers)
# ---------------------------------------------------------------------------------------------


def _confounds_with_outliers(n_frames: int, nss: list, motion: list) -> pd.DataFrame:
    """Confounds frame carrying one-hot outlier columns, as operations.confounds writes them."""
    rng = np.random.RandomState(1)
    data = {
        "global_signal": rng.normal(1000, 50, n_frames),
        "std_dvars": np.abs(rng.normal(1, 0.1, n_frames)),
        "framewise_displacement": np.abs(rng.normal(0, 0.05, n_frames)),
    }
    for i, frame in enumerate(nss):
        col = np.zeros(n_frames)
        col[frame] = 1.0
        data[f"non_steady_state_outlier{i:02d}"] = col
    for i, frame in enumerate(motion):
        col = np.zeros(n_frames)
        col[frame] = 1.0
        data[f"motion_outlier{i:02d}"] = col
    return pd.DataFrame(data)


def _shading_patches(ax) -> list:
    """Patches added by the shading helpers.

    ``ax.patches`` holds only artists added to the axes, not the axes background patch, and these
    figures add no other patches -- so every entry here is an axvspan band. (Its concrete class is
    matplotlib-version dependent: Polygon on <3.10, Rectangle after.)
    """
    return list(ax.patches)


def test_outlier_frame_indices_reads_one_hot_columns() -> None:
    df = _confounds_with_outliers(20, nss=[0, 1, 2], motion=[7, 15])
    assert np.array_equal(
        outlier_frame_indices(df, NONSTEADY_OUTLIER_PREFIX), np.array([0, 1, 2])
    )
    assert np.array_equal(
        outlier_frame_indices(df, MOTION_OUTLIER_PREFIX), np.array([7, 15])
    )


def test_outlier_frame_indices_empty_without_columns() -> None:
    df = _confounds_with_outliers(20, nss=[], motion=[])
    assert outlier_frame_indices(df, NONSTEADY_OUTLIER_PREFIX).size == 0
    assert outlier_frame_indices(df, MOTION_OUTLIER_PREFIX).size == 0


def test_outlier_frame_indices_ignores_similarly_named_columns() -> None:
    # Bare prefixes without the ## suffix are not one-hot indicator columns.
    df = pd.DataFrame(
        {"motion_outlier": [1.0, 0.0], "motion_outlier_total": [1.0, 1.0]}
    )
    assert outlier_frame_indices(df, MOTION_OUTLIER_PREFIX).size == 0


def test_contiguous_frame_spans_merges_runs() -> None:
    assert _contiguous_frame_spans([0, 1, 2, 7]) == [(0, 2), (7, 7)]
    assert _contiguous_frame_spans([]) == []
    assert _contiguous_frame_spans([4]) == [(4, 4)]
    # Unsorted / duplicated input is normalized.
    assert _contiguous_frame_spans([5, 3, 4, 3]) == [(3, 5)]


def test_confounds_plot_shades_flagged_frames() -> None:
    n_frames = 60
    df = _confounds_with_outliers(n_frames, nss=[0, 1, 2], motion=[30, 59])
    fig = create_confounds_plot(df)

    for ax in fig.axes:
        # One merged band for frames 0-2, plus one per isolated motion outlier.
        assert len(_shading_patches(ax)) == 3

    plt.close(fig)


def test_shade_frame_spans_clamps_to_frame_limits() -> None:
    n_frames = 60
    lo_limit, hi_limit = frame_xlim(n_frames)
    fig, ax = plt.subplots()

    # Frame 0 and the final frame would otherwise overhang the tight axes limits.
    spans = shade_frame_spans(ax, [0, 1, 2, 30, 59], n_frames, color="0.5", alpha=0.2)
    assert spans == [(lo_limit, 2.5), (29.5, 30.5), (58.5, hi_limit)]
    assert len(_shading_patches(ax)) == 3

    plt.close(fig)


def test_shade_frame_spans_noop_on_empty_input() -> None:
    fig, ax = plt.subplots()
    assert shade_frame_spans(ax, [], 40, color="0.5", alpha=0.2) == []
    assert _shading_patches(ax) == []
    plt.close(fig)


def test_confounds_plot_unshaded_without_outlier_columns() -> None:
    df = _confounds_with_outliers(30, nss=[], motion=[])
    fig = create_confounds_plot(df)
    for ax in fig.axes:
        assert _shading_patches(ax) == []
    plt.close(fig)


def test_motion_plot_shading_is_opt_in_and_preserves_xaxis() -> None:
    n_frames = 40
    rng = np.random.RandomState(2)
    motion_data = np.cumsum(rng.normal(0, 0.01, size=(n_frames, 6)), axis=0)

    plain = create_motion_plot(motion_data, title="")
    assert all(_shading_patches(ax) == [] for ax in plain.axes)

    shaded = create_motion_plot(
        motion_data,
        title="",
        nonsteady_frames=np.array([0, 1]),
        motion_outlier_frames=np.array([20]),
    )
    for ax in shaded.axes:
        assert len(_shading_patches(ax)) == 2
    # Shading must not perturb the shared frame axis used for motion/confounds alignment.
    assert shaded.axes[1].get_xlim() == pytest.approx(frame_xlim(n_frames), abs=1e-6)
    assert np.allclose(shaded.axes[1].get_xticks(), plain.axes[1].get_xticks())

    plt.close(plain)
    plt.close(shaded)


# ---------------------------------------------------------------------------------------------
# Motion QC snapshot: confounds-driven shading must never cost the figure
# ---------------------------------------------------------------------------------------------


def _motion_tsv(path: Path, n_frames: int) -> Path:
    rng = np.random.RandomState(3)
    motion = np.cumsum(rng.normal(0, 0.01, size=(n_frames, 6)), axis=0)
    pd.DataFrame(
        motion, columns=["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
    ).to_csv(path, sep="\t", index=False)
    return path


@pytest.mark.parametrize(
    "confounds_frames, expect_shading",
    [(40, True), (37, False)],  # matching vs mismatched frame count
)
def test_motion_qc_shading_requires_matching_frame_count(
    tmp_path: Path, confounds_frames: int, expect_shading: bool
) -> None:
    motion_file = _motion_tsv(tmp_path / "motion.tsv", 40)
    confounds_file = tmp_path / "confounds.tsv"
    _confounds_with_outliers(confounds_frames, nss=[0, 1], motion=[20]).to_csv(
        confounds_file, sep="\t", index=False
    )
    out = tmp_path / "motion.png"

    captured = {}
    real_create_motion_plot = snapshots.create_motion_plot

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return real_create_motion_plot(*args, **kwargs)

    snapshots.create_motion_plot = _spy
    try:
        result = create_motion_correction_qc(
            motion_params=str(motion_file),
            save_f=str(out),
            confounds_file=str(confounds_file),
        )
    finally:
        snapshots.create_motion_plot = real_create_motion_plot

    assert result["motion_plot"] == str(out)
    assert out.is_file()
    if expect_shading:
        assert np.array_equal(captured["nonsteady_frames"], np.array([0, 1]))
        assert np.array_equal(captured["motion_outlier_frames"], np.array([20]))
    else:
        assert captured["nonsteady_frames"] is None
        assert captured["motion_outlier_frames"] is None


def test_motion_qc_survives_unreadable_confounds(tmp_path: Path) -> None:
    motion_file = _motion_tsv(tmp_path / "motion.tsv", 25)
    garbage = tmp_path / "garbage.tsv"
    garbage.write_bytes(b"\x00\x01not a tsv\x00")
    out = tmp_path / "motion.png"

    result = create_motion_correction_qc(
        motion_params=str(motion_file),
        save_f=str(out),
        confounds_file=str(garbage),
    )
    assert result["motion_plot"] == str(out)
    assert out.is_file()


def test_motion_qc_without_confounds_is_unchanged(tmp_path: Path) -> None:
    motion_file = _motion_tsv(tmp_path / "motion.tsv", 25)
    out = tmp_path / "motion.png"

    result = create_motion_correction_qc(
        motion_params=str(motion_file), save_f=str(out)
    )
    assert result["motion_plot"] == str(out)
    assert out.is_file()


def test_outlier_frame_indices_skips_non_numeric_columns() -> None:
    # A non-numeric column under an indicator name must not raise: create_confounds_plot has no
    # inner guard, so an exception here would cost the whole figure.
    df = pd.DataFrame(
        {
            "global_signal": [1.0, 2.0, 3.0],
            "motion_outlier00": ["a", "b", "c"],
            "motion_outlier01": [0.0, 1.0, 0.0],
        }
    )
    assert np.array_equal(
        outlier_frame_indices(df, MOTION_OUTLIER_PREFIX), np.array([1])
    )
    fig = create_confounds_plot(df)
    assert len(_shading_patches(fig.axes[0])) == 1
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Outlier threshold reference lines
# ---------------------------------------------------------------------------------------------


def _threshold_lines(ax) -> list:
    """Dashed reference lines drawn by plot_confound_panel (the p95 line is solid)."""
    return [ln for ln in ax.lines if ln.get_linestyle() != "-" or ln.is_dashed()]


def test_threshold_line_drawn_when_in_range() -> None:
    n = 60
    rng = np.random.RandomState(4)
    fd = np.abs(rng.normal(0.05, 0.02, n))
    fd[10] = 0.9  # a frame well past the threshold
    df = pd.DataFrame({"framewise_displacement": fd})
    fig = create_confounds_plot(
        df,
        thresholds={
            "fd_outlier_threshold_mm": 0.25,
            "std_dvars_outlier_threshold": 1.5,
        },
    )
    lines = _threshold_lines(fig.axes[0])
    assert len(lines) == 1
    assert np.allclose(lines[0].get_ydata(), 0.25)
    plt.close(fig)


def test_threshold_line_omitted_when_out_of_range() -> None:
    # A clean run: nothing approaches 0.25, so no band needs explaining and forcing the line into
    # view would rescale the trace to a flat line.
    df = pd.DataFrame(
        {"framewise_displacement": np.full(60, 0.02) + np.arange(60) * 1e-4}
    )
    fig = create_confounds_plot(df, thresholds={"fd_outlier_threshold_mm": 0.25})
    assert _threshold_lines(fig.axes[0]) == []
    plt.close(fig)


def test_threshold_lines_are_per_criterion() -> None:
    # FD gets the FD threshold, DVARS gets the std-DVARS threshold, and the tissue/GS panels
    # get neither -- they are not outlier criteria.
    n = 40
    rng = np.random.RandomState(5)
    df = pd.DataFrame(
        {
            "global_signal": rng.normal(1000, 200, n),
            "std_dvars": np.abs(rng.normal(1.0, 0.9, n)),
            "framewise_displacement": np.abs(rng.normal(0.2, 0.3, n)),
        }
    )
    fig = create_confounds_plot(
        df,
        thresholds={
            "fd_outlier_threshold_mm": 0.25,
            "std_dvars_outlier_threshold": 1.5,
        },
    )
    gs_ax, dvars_ax, fd_ax = fig.axes
    assert _threshold_lines(gs_ax) == []
    assert np.allclose(_threshold_lines(dvars_ax)[0].get_ydata(), 1.5)
    assert np.allclose(_threshold_lines(fd_ax)[0].get_ydata(), 0.25)
    plt.close(fig)


def test_no_threshold_lines_without_thresholds() -> None:
    df = pd.DataFrame({"framewise_displacement": np.abs(np.linspace(0, 1, 40))})
    fig = create_confounds_plot(df)
    assert _threshold_lines(fig.axes[0]) == []
    plt.close(fig)


def test_qc_confounds_reads_thresholds_from_config(tmp_path: Path) -> None:
    from nhp_mri_prep.steps.qc import qc_confounds

    n = 40
    fd = np.abs(np.random.RandomState(6).normal(0.05, 0.02, n))
    fd[5] = 0.9
    tsv = tmp_path / "confounds.tsv"
    pd.DataFrame({"framewise_displacement": fd}).to_csv(tsv, sep="\t", index=False)

    out = tmp_path / "confounds.png"
    result = qc_confounds(
        tsv,
        out,
        config={
            "quality_control": {"enabled": True},
            "func": {"confounds": {"fd_outlier_threshold_mm": 0.4}},
        },
    )
    assert out.is_file()
    assert result.metadata["thresholds"]["fd_outlier_threshold_mm"] == 0.4
    # The unset key falls back to the operation default rather than vanishing.
    assert result.metadata["thresholds"]["std_dvars_outlier_threshold"] == 1.5
