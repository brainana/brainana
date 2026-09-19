"""Within-subject rate-of-change statistics.

The load-bearing claim of the whole longitudinal feature is that vertex *i* is
the same anatomical point in every timepoint, so per-vertex differencing needs
no surface registration. These tests pin that down: a known linear trend must
come back as the right slope, and a vertex-count mismatch -- which is exactly
what a failed seeding looks like -- must be a hard error rather than silently
comparing unrelated points.
"""

import json
import warnings
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import pytest

from nhp_mri_prep.steps.surface_longitudinal import (
    collect_change_stats,
    parse_aparc_stats,
    parse_session_time,
    read_session_times,
    write_qdec_table,
)

N_VERTS = 12


def _write_morph(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fsio.write_morph_data(str(path), np.asarray(values, dtype=np.float32))


def _stats_file(path, rois, thickness):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Table of FreeSurfer cortical parcellation anatomical statistics",
        "# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg",
    ]
    for roi in rois:
        lines.append(f"{roi} 100 200.0 300.0 {thickness:.4f}")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def long_tree(tmp_path):
    """A base plus three timepoints whose thickness rises 0.1/unit time."""
    base = tmp_path / "sub-01_base"
    (base / "surf").mkdir(parents=True)
    (base / "scripts").mkdir(parents=True)

    long_dirs = {}
    for time, label in ((1.0, "001"), (2.0, "002"), (4.0, "004")):
        long_id = f"sub-01_ses-{label}_long"
        d = tmp_path / long_id
        for hemi in ("lh", "rh"):
            # thickness = 2.0 + 0.1*t, uniform across vertices
            _write_morph(
                d / "surf" / f"{hemi}.thickness", np.full(N_VERTS, 2.0 + 0.1 * time)
            )
            _write_morph(d / "surf" / f"{hemi}.area", np.full(N_VERTS, 1.0))
            _write_morph(d / "surf" / f"{hemi}.curv", np.zeros(N_VERTS))
            _stats_file(
                d / "stats" / f"{hemi}.aparc.ARM2atlas.mapped.stats",
                ["V1", "V2"],
                2.0 + 0.1 * time,
            )
        long_dirs[long_id] = d
    return base, long_dirs


class TestSessionTimeParsing:
    @pytest.mark.parametrize(
        "label, expected",
        [
            ("ses-12months", 12.0),
            ("ses-004", 4.0),
            ("12months", 12.0),
            ("008", 8.0),
            ("ses-abc", None),
            (None, None),
        ],
    )
    def test_parses_the_shapes_that_occur(self, label, expected):
        assert parse_session_time(label) == expected


LONG_IDS = ["sub-01_ses-001_long", "sub-01_ses-002_long", "sub-01_ses-004_long"]


def _sessions_tsv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(header)] + ["\t".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


class TestSessionTsvTimes:
    """Real elapsed time, which is what makes a "rate" a rate.

    Without a sessions.tsv the fit is keyed to digits in the session label, or to
    scan order. For ses-01/02/03 covering 2, 9 and 24 months that slope is wrong
    by an amount nothing in the outputs reveals -- so these tests also pin the
    refusal to half-resolve, which is what would make the error invisible.
    """

    def test_reads_a_named_column(self, tmp_path):
        tsv = _sessions_tsv(
            tmp_path / "sub-01_sessions.tsv",
            ["session_id", "scan_day"],
            [["ses-001", "0"], ["ses-002", "30"], ["ses-004", "90"]],
        )
        times, source = read_session_times(tsv, LONG_IDS, time_column="scan_day")
        assert times == {
            "sub-01_ses-001_long": 0.0,
            "sub-01_ses-002_long": 30.0,
            "sub-01_ses-004_long": 90.0,
        }
        assert source == "sub-01_sessions.tsv:scan_day"

    def test_auto_detects_age_before_acq_time(self, tmp_path):
        tsv = _sessions_tsv(
            tmp_path / "sub-01_sessions.tsv",
            ["session_id", "acq_time", "age"],
            [
                ["ses-001", "2024-01-01T09:00:00", "4.0"],
                ["ses-002", "2024-02-01T09:00:00", "4.5"],
                ["ses-004", "2024-04-01T09:00:00", "5.5"],
            ],
        )
        times, source = read_session_times(tsv, LONG_IDS)
        assert times["sub-01_ses-004_long"] == 5.5
        assert source.endswith(":age")

    def test_acq_time_becomes_days_from_the_first_session(self, tmp_path):
        """BIDS date-shifting preserves intervals; absolute dates are meaningless."""
        tsv = _sessions_tsv(
            tmp_path / "sub-01_sessions.tsv",
            ["session_id", "acq_time"],
            [
                ["ses-001", "2024-01-01T00:00:00"],
                ["ses-002", "2024-01-31T00:00:00"],
                ["ses-004", "2024-03-31T00:00:00"],
            ],
        )
        times, source = read_session_times(tsv, LONG_IDS)
        assert times["sub-01_ses-001_long"] == 0.0
        assert times["sub-01_ses-002_long"] == 30.0
        assert times["sub-01_ses-004_long"] == 90.0
        assert "days from first session" in source

    def test_a_missing_session_yields_no_times_at_all(self, tmp_path):
        """Partial coverage is refused on purpose.

        Filling the gap from scan order would put real ages and scan indices in
        one regression, and the summary would still call the result a rate. The
        honest fallback it would displace is strictly better.
        """
        tsv = _sessions_tsv(
            tmp_path / "sub-01_sessions.tsv",
            ["session_id", "age"],
            [["ses-001", "4.0"], ["ses-002", "4.5"]],
        )
        assert read_session_times(tsv, LONG_IDS) == ({}, None)

    def test_na_counts_as_missing(self, tmp_path):
        tsv = _sessions_tsv(
            tmp_path / "sub-01_sessions.tsv",
            ["session_id", "age"],
            [["ses-001", "4.0"], ["ses-002", "n/a"], ["ses-004", "5.5"]],
        )
        assert read_session_times(tsv, LONG_IDS) == ({}, None)

    def test_a_named_column_that_is_absent_falls_back(self, tmp_path):
        tsv = _sessions_tsv(
            tmp_path / "sub-01_sessions.tsv",
            ["session_id", "age"],
            [["ses-001", "4.0"], ["ses-002", "4.5"], ["ses-004", "5.5"]],
        )
        assert read_session_times(tsv, LONG_IDS, time_column="weight") == ({}, None)

    def test_no_usable_column_falls_back(self, tmp_path):
        tsv = _sessions_tsv(
            tmp_path / "sub-01_sessions.tsv",
            ["session_id", "handedness"],
            [["ses-001", "R"], ["ses-002", "R"], ["ses-004", "R"]],
        )
        assert read_session_times(tsv, LONG_IDS) == ({}, None)

    def test_one_constant_time_cannot_fit_a_rate(self, tmp_path):
        tsv = _sessions_tsv(
            tmp_path / "sub-01_sessions.tsv",
            ["session_id", "age"],
            [["ses-001", "4.0"], ["ses-002", "4.0"], ["ses-004", "4.0"]],
        )
        assert read_session_times(tsv, LONG_IDS) == ({}, None)

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert read_session_times(tmp_path / "absent.tsv", LONG_IDS) == ({}, None)

    def test_session_ids_without_the_ses_prefix_still_match(self, tmp_path):
        tsv = _sessions_tsv(
            tmp_path / "sub-01_sessions.tsv",
            ["session_id", "age"],
            [["001", "4.0"], ["002", "4.5"], ["004", "5.5"]],
        )
        times, _ = read_session_times(tsv, LONG_IDS)
        assert times["sub-01_ses-002_long"] == 4.5


class TestQdecTable:
    def test_writes_fsid_and_fsid_base_columns(self, tmp_path):
        path = write_qdec_table(
            tmp_path / "long.qdec.table.dat",
            "sub-01_base",
            ["sub-01_ses-001_long", "sub-01_ses-002_long"],
            [1.0, 2.0],
        )
        lines = path.read_text().strip().split("\n")
        assert lines[0].split() == ["fsid", "fsid-base", "time"]
        assert lines[1].split() == ["sub-01_ses-001_long", "sub-01_base", "1"]

    def test_mismatched_lengths_are_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="needs exactly one time"):
            write_qdec_table(tmp_path / "t.dat", "sub-01_base", ["a", "b"], [1.0])


class TestAparcStatsParsing:
    def test_uses_the_files_own_column_headers(self, tmp_path):
        """Columns vary with what mris_anatomical_stats produced (no eTIV here)."""
        path = tmp_path / "lh.aparc.ARM2atlas.mapped.stats"
        _stats_file(path, ["V1"], 2.5)
        parsed = parse_aparc_stats(path)
        assert parsed["V1"]["ThickAvg"] == pytest.approx(2.5)
        assert parsed["V1"]["NumVert"] == pytest.approx(100)


class TestCollectChangeStats:
    def test_recovers_a_known_linear_rate(self, long_tree):
        base, long_dirs = long_tree
        summary = collect_change_stats(base, long_dirs)
        rate_path = Path(summary["vertex_outputs"]["lh.thickness-rate"])
        rate = np.asarray(nib.load(str(rate_path)).dataobj).ravel()
        assert rate.shape == (N_VERTS,)
        np.testing.assert_allclose(rate, 0.1, atol=1e-5)

    def test_writes_avg_and_spc_maps(self, long_tree):
        base, long_dirs = long_tree
        summary = collect_change_stats(base, long_dirs)
        avg = np.asarray(
            nib.load(summary["vertex_outputs"]["lh.thickness-avg"]).dataobj
        ).ravel()
        spc = np.asarray(
            nib.load(summary["vertex_outputs"]["lh.thickness-spc"]).dataobj
        ).ravel()
        # mean of 2.1, 2.2, 2.4
        np.testing.assert_allclose(avg, np.mean([2.1, 2.2, 2.4]), atol=1e-5)
        np.testing.assert_allclose(spc, 100.0 * 0.1 / avg, atol=1e-4)

    def test_vertex_count_mismatch_is_a_hard_error(self, long_tree):
        """A mismatch means seeding failed; comparing anyway would be nonsense."""
        base, long_dirs = long_tree
        broken = sorted(long_dirs)[1]
        _write_morph(
            long_dirs[broken] / "surf" / "lh.thickness", np.full(N_VERTS + 3, 2.2)
        )
        with pytest.raises(RuntimeError, match="vertex counts differ"):
            collect_change_stats(base, long_dirs)

    def test_roi_table_recovers_the_same_rate(self, long_tree):
        base, long_dirs = long_tree
        summary = collect_change_stats(base, long_dirs)
        csv = Path(summary["roi_tables"]["lh"]).read_text().strip().split("\n")
        header = csv[0].split(",")
        assert header == ["roi", "measure", "slope", "mean", "spc", "n_timepoints"]
        thick_rows = [r.split(",") for r in csv[1:] if r.split(",")[1] == "ThickAvg"]
        assert thick_rows, "no ThickAvg rows"
        for row in thick_rows:
            assert float(row[2]) == pytest.approx(0.1, abs=1e-5)

    def test_writes_a_qdec_table_and_summary(self, long_tree):
        base, long_dirs = long_tree
        collect_change_stats(base, long_dirs)
        assert (base / "scripts" / "long.qdec.table.dat").exists()
        summary = json.loads((base / "stats" / "long.change-stats.json").read_text())
        assert len(summary["timepoints"]) == 3
        assert summary["times"]["sub-01_ses-004_long"] == 4.0

    def test_missing_measure_is_skipped_not_fatal(self, long_tree):
        base, long_dirs = long_tree
        for d in long_dirs.values():
            (d / "surf" / "lh.curv").unlink()
        summary = collect_change_stats(base, long_dirs)
        assert "lh.curv" in summary["skipped"]
        # and the others still came through
        assert "lh.thickness-rate" in summary["vertex_outputs"]

    def test_one_timepoint_is_rejected(self, long_tree):
        base, long_dirs = long_tree
        first = sorted(long_dirs)[0]
        with pytest.raises(ValueError, match="at least 2 timepoints"):
            collect_change_stats(base, {first: long_dirs[first]})

    def test_roi_tables_follow_the_atlas_that_named_the_stats_files(self, long_tree):
        """atlas_name has to come from the recon that wrote these files.

        It is only ever used to build ?h.aparc.<X>atlas.mapped.stats, and a wrong
        value does not raise -- it finds nothing and drops every ROI table into
        summary["skipped"], which is easy to miss in a run that otherwise looks
        fine.
        """
        base, long_dirs = long_tree
        for d in long_dirs.values():
            for hemi in ("lh", "rh"):
                src = d / "stats" / f"{hemi}.aparc.ARM2atlas.mapped.stats"
                src.rename(src.with_name(f"{hemi}.aparc.ARM3atlas.mapped.stats"))

        matched = collect_change_stats(
            base, long_dirs, measures=(), hemis=("lh",), atlas_name="ARM3"
        )
        assert "lh" in matched["roi_tables"]
        assert "lh.roi" not in matched["skipped"]

        mismatched = collect_change_stats(
            base, long_dirs, measures=(), hemis=("lh",), atlas_name="ARM2"
        )
        assert "lh" not in mismatched["roi_tables"]
        assert "ARM2" in mismatched["skipped"]["lh.roi"]

    def test_the_time_variable_changes_the_rate_and_is_recorded(self, long_tree):
        """A rate is only interpretable alongside the time it is per.

        The same surfaces yield 0.1/unit against the session labels {1,2,4} and
        0.01/day against real spacing {0,10,30}. Nothing in the maps distinguishes
        those, so the summary has to say which one it fitted.
        """
        base, long_dirs = long_tree
        summary = collect_change_stats(
            base,
            long_dirs,
            times={
                "sub-01_ses-001_long": 0.0,
                "sub-01_ses-002_long": 10.0,
                "sub-01_ses-004_long": 30.0,
            },
            time_source="sub-01_sessions.tsv:age",
            measures=("thickness",),
            hemis=("lh",),
        )
        assert summary["time_source"] == "sub-01_sessions.tsv:age"
        rate = np.asanyarray(
            nib.load(summary["vertex_outputs"]["lh.thickness-rate"]).dataobj
        ).ravel()
        assert np.allclose(rate, 0.01, atol=1e-5)

        on_disk = json.loads((base / "stats" / "long.change-stats.json").read_text())
        assert on_disk["time_source"] == "sub-01_sessions.tsv:age"

    def test_time_source_names_the_fallback_when_there_is_no_tsv(self, long_tree):
        base, long_dirs = long_tree
        summary = collect_change_stats(
            base, long_dirs, measures=("thickness",), hemis=("lh",)
        )
        assert summary["time_source"] == "session label"

    def test_time_source_says_scan_order_for_unparseable_labels(
        self, tmp_path, long_tree
    ):
        """ses-preop and friends carry no number, so the rate is per scan."""
        _, long_dirs = long_tree
        base = tmp_path / "sub-02_base"
        renamed = {
            f"sub-02_ses-{label}_long": d
            for label, d in zip(("preop", "postop", "followup"), long_dirs.values())
        }
        summary = collect_change_stats(
            base, renamed, measures=("thickness",), hemis=("lh",)
        )
        assert summary["time_source"] == "scan order"
        assert sorted(summary["ordinal_time_fallback"]) == sorted(renamed)

    def test_explicit_times_override_label_parsing(self, long_tree):
        """Session labels are often scan indices, not elapsed time."""
        base, long_dirs = long_tree
        ids = sorted(long_dirs)
        # Real spacing 10x the label spacing -> rate should be 10x smaller.
        times = {ids[0]: 10.0, ids[1]: 20.0, ids[2]: 40.0}
        summary = collect_change_stats(base, long_dirs, times=times)
        rate = np.asarray(
            nib.load(summary["vertex_outputs"]["lh.thickness-rate"]).dataobj
        ).ravel()
        np.testing.assert_allclose(rate, 0.01, atol=1e-6)
