"""Unit tests for the atlas surface-projection helpers (steps/anatomical.py).

Covers combining per-depth samples (label mode with a mid-thickness tie-break,
continuous nearest-mid-thickness with paired frames) and filling label holes inside
cortex on a small mesh. Synthetic arrays only; no FreeSurfer needed.
"""

import numpy as np

from nhp_mri_prep.steps.anatomical import (
    _SURF_PROJFRACS,
    _combine_depth_samples,
    _fill_label_holes_in_cortex,
)


def _samples(rows):
    """(n_vert, n_depth) list -> (n_vert, n_depth, 1) float array."""
    return np.asarray(rows, dtype=np.float32)[:, :, None]


# --------------------------------------------------------------------------- #
# _combine_depth_samples
# --------------------------------------------------------------------------- #
def test_label_mode_across_depths():
    fracs = (0.0, 0.5, 1.0)
    out = _combine_depth_samples(_samples([[3, 3, 7]]), fracs, is_label=True)
    assert out[0, 0] == 3


def test_label_tie_goes_to_mid_depth():
    fracs = (0.0, 0.25, 0.5, 1.0)
    # 3 and 7 both appear twice; 7 is at 0.5
    out = _combine_depth_samples(_samples([[3, 3, 7, 7]]), fracs, is_label=True)
    assert out[0, 0] == 7


def test_label_ignores_zero_samples():
    fracs = (0.0, 0.5, 1.0)
    out = _combine_depth_samples(_samples([[0, 0, 5], [0, 0, 0]]), fracs, is_label=True)
    assert out[:, 0].tolist() == [5, 0]


def test_continuous_takes_nonzero_nearest_mid():
    fracs = (0.0, 0.4, 0.5, 1.0)
    rows = [[1.0, 2.0, 0.0, 9.0]]  # 0.5 empty -> 0.4 is next nearest
    out = _combine_depth_samples(_samples(rows), fracs, is_label=False)
    assert out[0, 0] == np.float32(2.0)


def test_continuous_frames_stay_paired():
    fracs = (0.0, 0.5, 1.0)
    # frame 0 empty at 0.5 -> depth 0.0 chosen for every frame
    samples = np.array(
        [[[170.0, 4.0], [0.0, 9.0], [-170.0, 6.0]]], dtype=np.float32
    )  # (1 vert, 3 depths, 2 frames)
    out = _combine_depth_samples(samples, fracs, is_label=False)
    assert out[0].tolist() == [170.0, 4.0]


def test_default_depths_cover_white_to_pial():
    assert _SURF_PROJFRACS[0] == 0.0 and _SURF_PROJFRACS[-1] == 1.0
    assert 0.5 in _SURF_PROJFRACS


# --------------------------------------------------------------------------- #
# _fill_label_holes_in_cortex
# --------------------------------------------------------------------------- #
def _strip_mesh(n):
    """Triangle strip over vertices 0..n-1 (each vertex links to its 2 neighbours on each side)."""
    return np.array([[i, i + 1, i + 2] for i in range(n - 2)], dtype=np.int64)


def test_fill_spreads_labels_inside_cortex_only():
    faces = _strip_mesh(8)
    labels = np.array([0, 4, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    cortex = np.array([False, True, True, True, True, True, True, False])
    out = _fill_label_holes_in_cortex(faces, labels, cortex)
    assert out.tolist() == [0, 4, 4, 4, 4, 4, 4, 0]


def test_fill_keeps_existing_labels():
    faces = _strip_mesh(6)
    labels = np.array([2, 2, 0, 0, 9, 9], dtype=np.int64)
    cortex = np.ones(6, dtype=bool)
    out = _fill_label_holes_in_cortex(faces, labels, cortex)
    assert out[[0, 1, 4, 5]].tolist() == [2, 2, 9, 9]
    assert np.all(out != 0)


def test_fill_stops_when_no_seed_reachable():
    faces = _strip_mesh(5)
    labels = np.zeros(5, dtype=np.int64)
    cortex = np.ones(5, dtype=bool)
    out = _fill_label_holes_in_cortex(faces, labels, cortex)
    assert np.all(out == 0)


def test_label_ignores_negative_tissue_ids():
    fracs = (0.0, 0.5, 1.0)
    # ARM marks white matter as -1; it must not outvote the cortical label
    out = _combine_depth_samples(_samples([[-1, -1, 12]]), fracs, is_label=True)
    assert out[0, 0] == 12
