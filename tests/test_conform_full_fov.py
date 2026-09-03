"""Geometry of the full-FOV conform output.

The extra ``desc-conformFullFOV`` image must be the *same* conform on a strictly
larger, voxel-aligned grid — so the standard conformed image is an exact sub-volume
of it. These tests pin that invariant, plus the two conventions it rests on: the
outer-edge corner box, and the FSL matrix re-basing (whose x component follows
``pad_right``, not ``pad_left``, on a positive-determinant reference).

Pure geometry — no FSL, no AFNI, no network.
"""

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

from nhp_mri_prep.operations.sitk_rigid_registration import (
    _sitk_affine_lps,
    _sitk_tx_to_fsl_matrix,
    _sitk_tx_to_matrix,
    fsl_mat_for_new_reference,
    fsl_mat_to_world_affine,
    reference_padding_to_cover,
    world_mat_to_vox2vox,
)
from nhp_mri_prep.utils.mri import pad_image, write_inner_box_mask

# det(RAS) > 0 -> SimpleITK sees det(LPS) > 0 too, which activates the FSL x-flip.
# Every template this pipeline ships is in that branch, so it is the default here.
_REF_AFFINE = np.diag([0.5, 0.5, 0.5, 1.0])
_REF_AFFINE[:3, 3] = [-5.0, -5.0, -5.0]
# det < 0 -> no x-flip; used to pin the other branch of _sitk_fsl_scale.
_REF_AFFINE_NEG = np.diag([-0.5, 0.5, 0.5, 1.0])
_REF_AFFINE_NEG[:3, 3] = [5.0, -5.0, -5.0]
# Deliberately coarser and wider than the reference, and offset, so it overhangs.
_MOV_AFFINE = np.diag([0.6, 0.6, 0.6, 1.0])
_MOV_AFFINE[:3, 3] = [-9.0, -9.0, -9.0]


def _save(path, shape, affine, seed=0):
    data = np.random.RandomState(seed).rand(*shape).astype(np.float32) + 1.0
    img = nib.Nifti1Image(data, affine)
    img.header.set_data_dtype(np.float32)
    nib.save(img, str(path))
    return path


def _tx():
    """A deliberately oblique rigid transform (reference-world -> moving-world)."""
    tx = sitk.Euler3DTransform()
    tx.SetCenter((1.0, 2.0, 3.0))
    tx.SetRotation(0.1, -0.2, 0.3)
    tx.SetTranslation((2.0, -3.0, 1.5))
    return tx


def _read(path):
    return sitk.ReadImage(str(path), sitk.sitkFloat32)


def _resample(moving, fixed, tx):
    return sitk.Resample(moving, fixed, tx, sitk.sitkLinear, 0.0, moving.GetPixelID())


def _setup(
    tmp_path, ref_affine=_REF_AFFINE, ref_shape=(20, 20, 20), mov_shape=(30, 30, 30)
):
    ref_f = _save(tmp_path / "ref.nii.gz", ref_shape, ref_affine, seed=1)
    mov_f = _save(tmp_path / "mov.nii.gz", mov_shape, _MOV_AFFINE, seed=2)
    tx = _tx()
    vox2vox = world_mat_to_vox2vox(_sitk_tx_to_matrix(tx), ref_f, mov_f)
    return ref_f, mov_f, tx, vox2vox


def test_expanded_output_contains_standard_as_exact_subvolume(tmp_path):
    """The load-bearing invariant: same conform, bigger box, nothing moved.

    Bit-exact here because this is the sitk path — the same transform object is applied
    and only the output grid changes. The flirt path guarantees the same *alignment* (an
    integer voxel offset) but not identical values, because FLIRT redoes its arithmetic
    from a matrix re-expressed for the enlarged grid; see :func:`_conform_full_fov`.
    """
    ref_f, mov_f, tx, vox2vox = _setup(tmp_path)
    ref_shape = np.array(nib.load(str(ref_f)).shape[:3])
    mov_shape = np.array(nib.load(str(mov_f)).shape[:3])

    pad_left, pad_right = reference_padding_to_cover(vox2vox, mov_shape, ref_shape)
    assert np.any(pad_left + pad_right), "test setup should require real padding"

    big_f = tmp_path / "ref_big.nii.gz"
    pad_image(ref_f, big_f, pad_left, pad_right, dtype=np.float32)

    standard = sitk.GetArrayFromImage(_resample(_read(mov_f), _read(ref_f), tx))
    expanded = sitk.GetArrayFromImage(_resample(_read(mov_f), _read(big_f), tx))

    # GetArrayFromImage is (z, y, x); pad_left is (x, y, z).
    pl = pad_left[::-1]
    inner = expanded[
        pl[0] : pl[0] + standard.shape[0],
        pl[1] : pl[1] + standard.shape[1],
        pl[2] : pl[2] + standard.shape[2],
    ]
    assert np.array_equal(inner, standard)
    # The whole point: the enlarged grid recovers data the target FOV threw away.
    assert expanded.sum() > standard.sum()


def test_expanded_grid_is_a_pure_integer_translation(tmp_path):
    ref_f, mov_f, _, vox2vox = _setup(tmp_path)
    ref_shape = np.array(nib.load(str(ref_f)).shape[:3])
    mov_shape = np.array(nib.load(str(mov_f)).shape[:3])
    pad_left, pad_right = reference_padding_to_cover(vox2vox, mov_shape, ref_shape)

    big_f = tmp_path / "ref_big.nii.gz"
    pad_image(ref_f, big_f, pad_left, pad_right, dtype=np.float32)

    old, new = _read(ref_f), _read(big_f)
    assert np.allclose(old.GetSpacing(), new.GetSpacing())
    assert np.allclose(old.GetDirection(), new.GetDirection())
    assert list(new.GetSize()) == list(ref_shape + pad_left + pad_right)

    shift = np.linalg.inv(_sitk_affine_lps(new)) @ _sitk_affine_lps(old)
    assert np.allclose(shift[:3, :3], np.eye(3))
    assert np.allclose(shift[:3, 3], pad_left)


def test_every_moving_corner_lands_inside_the_expanded_grid(tmp_path):
    ref_f, mov_f, tx, vox2vox = _setup(tmp_path)
    ref_shape = np.array(nib.load(str(ref_f)).shape[:3])
    mov_shape = np.array(nib.load(str(mov_f)).shape[:3])
    pad_left, pad_right = reference_padding_to_cover(vox2vox, mov_shape, ref_shape)

    big_f = tmp_path / "ref_big.nii.gz"
    pad_image(ref_f, big_f, pad_left, pad_right, dtype=np.float32)
    new_shape = np.array(nib.load(str(big_f)).shape[:3])
    v_new = world_mat_to_vox2vox(_sitk_tx_to_matrix(tx), big_f, mov_f)

    lows, highs = np.full(3, -0.5), mov_shape - 0.5
    corners = np.array(
        [
            [
                lows[0] if not i & 1 else highs[0],
                lows[1] if not i & 2 else highs[1],
                lows[2] if not i & 4 else highs[2],
                1.0,
            ]
            for i in range(8)
        ]
    ).T
    mapped = (v_new @ corners)[:3]
    assert (mapped >= -0.5).all()
    assert (mapped <= (new_shape - 0.5)[:, None]).all()


def test_edge_corners_are_load_bearing(tmp_path):
    """Voxel centres would clip a nonzero half-voxel slab — the chamber rim."""
    _, _, _, vox2vox = _setup(tmp_path)
    ref_shape, mov_shape = np.array([20, 20, 20]), np.array([30, 30, 30])
    pl_edge, pr_edge = reference_padding_to_cover(
        vox2vox, mov_shape, ref_shape, margin=0
    )

    centres = np.array(
        [
            [
                0 if not i & 1 else mov_shape[0] - 1,
                0 if not i & 2 else mov_shape[1] - 1,
                0 if not i & 4 else mov_shape[2] - 1,
                1.0,
            ]
            for i in range(8)
        ]
    ).T
    mapped = (vox2vox @ centres)[:3]
    lo = np.floor(mapped.min(axis=1)).astype(int)
    hi = np.ceil(mapped.max(axis=1)).astype(int)
    pl_centre = np.maximum(0, -lo)
    pr_centre = np.maximum(0, hi - (ref_shape - 1))

    assert (pl_edge >= pl_centre).all() and (pr_edge >= pr_centre).all()
    assert np.any(pl_edge > pl_centre) or np.any(pr_edge > pr_centre)


@pytest.mark.parametrize(
    "ref_affine, flipped", [(_REF_AFFINE, True), (_REF_AFFINE_NEG, False)]
)
def test_fsl_mat_rebasing_follows_pad_right_on_the_flipped_axis(
    tmp_path, ref_affine, flipped
):
    """FSL measures x from the far edge when det > 0, so that component tracks pad_right."""
    ref_f = _save(tmp_path / "ref.nii.gz", (20, 20, 20), ref_affine, seed=1)
    mov_f = _save(tmp_path / "mov.nii.gz", (30, 30, 30), _MOV_AFFINE, seed=2)
    tx = _tx()
    mat_old = _sitk_tx_to_fsl_matrix(tx, _read(ref_f), _read(mov_f))

    pad_left, pad_right = np.array([7, 3, 11]), np.array([2, 9, 4])
    big_f = tmp_path / "ref_big.nii.gz"
    pad_image(ref_f, big_f, pad_left, pad_right, dtype=np.float32)

    mat_new = fsl_mat_for_new_reference(mat_old, ref_f, big_f)
    assert np.allclose(mat_new[:3, :3], mat_old[:3, :3])

    sp = 0.5
    expected_x = sp * (pad_right[0] if flipped else pad_left[0])
    expected = np.array([expected_x, sp * pad_left[1], sp * pad_left[2]])
    assert np.allclose(mat_new[:3, 3] - mat_old[:3, 3], expected, atol=1e-9)


def test_world_affine_from_fsl_mat_round_trips(tmp_path):
    ref_f = _save(tmp_path / "ref.nii.gz", (20, 20, 20), _REF_AFFINE, seed=1)
    mov_f = _save(tmp_path / "mov.nii.gz", (30, 30, 30), _MOV_AFFINE, seed=2)
    tx = _tx()
    mat = _sitk_tx_to_fsl_matrix(tx, _read(ref_f), _read(mov_f))
    assert np.allclose(
        fsl_mat_to_world_affine(mat, ref_f, mov_f), _sitk_tx_to_matrix(tx), atol=1e-9
    )


def test_moving_fully_inside_needs_no_padding(tmp_path):
    """A scan already within the target FOV must not grow the grid."""
    ref_f = _save(tmp_path / "ref.nii.gz", (40, 40, 40), _REF_AFFINE, seed=1)
    small_affine = np.diag([0.5, 0.5, 0.5, 1.0])
    small_affine[:3, 3] = [-1.0, -1.0, -1.0]
    mov_f = _save(tmp_path / "mov.nii.gz", (8, 8, 8), small_affine, seed=2)

    identity = sitk.Euler3DTransform()
    vox2vox = world_mat_to_vox2vox(_sitk_tx_to_matrix(identity), ref_f, mov_f)
    pad_left, pad_right = reference_padding_to_cover(
        vox2vox, np.array([8, 8, 8]), np.array([40, 40, 40]), margin=0
    )
    assert not np.any(pad_left) and not np.any(pad_right)


def test_implausible_transform_is_rejected(tmp_path):
    ref_shape, mov_shape = np.array([20, 20, 20]), np.array([30, 30, 30])
    runaway = np.eye(4)
    runaway[:3, 3] = [5000.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="Implausible full-FOV padding"):
        reference_padding_to_cover(runaway, mov_shape, ref_shape)

    with pytest.raises(ValueError, match="Non-finite"):
        reference_padding_to_cover(np.full((4, 4), np.nan), mov_shape, ref_shape)


def test_inner_box_mask_marks_exactly_the_original_fov(tmp_path):
    ref_f = _save(tmp_path / "ref.nii.gz", (20, 20, 20), _REF_AFFINE, seed=1)
    pad_left, pad_right = np.array([7, 3, 11]), np.array([2, 9, 4])
    big_f = tmp_path / "ref_big.nii.gz"
    pad_image(ref_f, big_f, pad_left, pad_right, dtype=np.float32)

    box_f = tmp_path / "box.nii.gz"
    write_inner_box_mask(big_f, box_f, pad_left, (20, 20, 20))

    box = nib.load(str(box_f))
    data = np.asarray(box.dataobj)
    assert data.shape == tuple(np.array([20, 20, 20]) + pad_left + pad_right)
    assert data.sum() == 20**3
    assert data[7:27, 3:23, 11:31].all()
    assert np.allclose(box.affine, nib.load(str(big_f)).affine)


# ---------------------------------------------------------------------------
# QC snapshot: full-FOV underlay + white target-FOV box
# ---------------------------------------------------------------------------


def _qc_inputs(tmp_path):
    """A conformed image, its full-FOV counterpart, template and FOV box."""
    pad_left, pad_right = np.array([4, 4, 4]), np.array([4, 4, 4])
    conformed = _save(tmp_path / "conformed.nii.gz", (20, 20, 20), _REF_AFFINE, seed=3)
    template = _save(tmp_path / "template.nii.gz", (20, 20, 20), _REF_AFFINE, seed=4)

    big_f = tmp_path / "full_fov.nii.gz"
    big_template_f = tmp_path / "full_fov_template.nii.gz"
    pad_image(conformed, big_f, pad_left, pad_right, dtype=np.float32)
    pad_image(template, big_template_f, pad_left, pad_right, dtype=np.float32)

    box_f = tmp_path / "fov_box.nii.gz"
    write_inner_box_mask(big_template_f, box_f, pad_left, (20, 20, 20))
    return conformed, template, big_f, big_template_f, box_f


def test_conform_qc_renders_both_figures(tmp_path):
    """The pair: the uncropped view with the FOV boxed, and that box enlarged."""
    from nhp_mri_prep.quality_control.snapshots import create_conform_qc

    conformed, template, big_f, big_template_f, box_f = _qc_inputs(tmp_path)
    out = tmp_path / "conform.png"
    out_full = tmp_path / "conform_fullfov.png"
    result = create_conform_qc(
        conformed_file=str(conformed),
        template_file=str(template),
        save_f=out,
        modality="anat",
        full_fov_file=big_f,
        full_fov_template_file=big_template_f,
        fov_box_file=box_f,
        full_fov_save_f=out_full,
    )
    assert result.get("anat_conform_overlay") == str(out)
    assert result.get("anat_conform_fullfov_overlay") == str(out_full)
    for f in (out, out_full):
        assert f.exists() and f.stat().st_size > 0


def test_conform_qc_skips_the_full_fov_figure_without_an_output_path(tmp_path):
    """The three inputs alone are not enough; the extra figure needs somewhere to go."""
    from nhp_mri_prep.quality_control.snapshots import create_conform_qc

    conformed, template, big_f, big_template_f, box_f = _qc_inputs(tmp_path)
    out = tmp_path / "conform.png"
    result = create_conform_qc(
        conformed_file=str(conformed),
        template_file=str(template),
        save_f=out,
        modality="anat",
        full_fov_file=big_f,
        full_fov_template_file=big_template_f,
        fov_box_file=box_f,
    )
    assert set(result) == {"anat_conform_overlay"}
    assert out.exists()


def test_conform_report_entries_are_registered(tmp_path):
    """A figure the report has no mapping for renders with an empty caption."""
    from nhp_mri_prep.quality_control.reports import (
        FIGURE_DESCRIPTIONS,
        SNAPSHOT_MAPPINGS,
        SNAPSHOT_ORDER,
    )

    assert SNAPSHOT_MAPPINGS["conformFullFOV"]["key"] == "conform_fullfov_overlay"
    assert FIGURE_DESCRIPTIONS["conformFullFOV"]
    # Context first, then the blow-up.
    assert SNAPSHOT_ORDER.index("conform_fullfov_overlay") < SNAPSHOT_ORDER.index(
        "conform_overlay"
    )


def test_conform_qc_falls_back_without_full_fov_inputs(tmp_path):
    """The functional path passes none of them and must render exactly as before."""
    from nhp_mri_prep.quality_control.snapshots import create_conform_qc

    conformed, template, _, _, _ = _qc_inputs(tmp_path)
    out = tmp_path / "conform_func.png"
    result = create_conform_qc(
        conformed_file=str(conformed),
        template_file=str(template),
        save_f=out,
        modality="func",
    )
    assert set(result) == {"func_conform_overlay"}
    assert result.get("func_conform_overlay") == str(out)
    assert out.exists() and out.stat().st_size > 0


def test_outline_on_a_mismatched_grid_is_rejected(tmp_path):
    from nhp_mri_prep.quality_control.mri_plotting import create_grid_mri_image

    conformed, _, big_f, _, box_f = _qc_inputs(tmp_path)
    with pytest.raises(ValueError, match="does not match underlay"):
        create_grid_mri_image(
            underlay_data=str(conformed),  # 20^3
            outline_data=str(box_f),  # 28^3
            num_cols=2,
            perspectives=["axial"],
        )


def test_full_fov_output_is_named_in_the_conform_space():
    """The full-FOV image is on the conform grid, not the scanner grid it came from.

    The input carries ``space-scanner``; keeping that would label a T1w-space image as
    scanner space. Passing the space in the suffix both fixes the label and makes
    ``create_bids_output_filename`` drop the stale entity instead of stacking two.
    """
    from nhp_mri_prep.utils.bids import create_bids_output_filename

    for modality in ("T1w", "T2w"):
        name = create_bids_output_filename(
            f"sub-01_ses-001_run-1_space-scanner_{modality}.nii.gz",
            f"space-{modality}_desc-conformFullFOV",
            modality,
        )
        assert name == (
            f"sub-01_ses-001_run-1_space-{modality}_desc-conformFullFOV_{modality}.nii.gz"
        )
        assert "space-scanner" not in name

    # An input with no space entity at all (the plain BIDS naming template) still gets one.
    assert (
        create_bids_output_filename(
            "sub-01_ses-001_run-1_T1w.nii.gz", "space-T1w_desc-conformFullFOV", "T1w"
        )
        == "sub-01_ses-001_run-1_space-T1w_desc-conformFullFOV_T1w.nii.gz"
    )

    # The unpublished intermediate keeps its historical name, and the exact-token
    # globs in modules/anatomical.nf must still tell the two apart.
    intermediate = create_bids_output_filename(
        "sub-01_ses-001_run-1_T1w.nii.gz", "desc-conform", "T1w"
    )
    assert "_desc-conform_" in intermediate
    assert "_desc-conform_" not in (
        "sub-01_ses-001_run-1_space-T1w_desc-conformFullFOV_T1w.nii.gz"
    )


def test_full_fov_render_failure_still_produces_the_standard_figure(
    tmp_path, monkeypatch
):
    """The enlarged grid is the memory-hungry one; its failure must not cost figure 2.

    Figure 2 is the figure the report actually needs, and ``qc_conform`` records the
    path it returns — so a shared ``try`` both loses the useful figure and reports a
    PNG that was never written.
    """
    from nhp_mri_prep.quality_control import snapshots as snap

    conformed, template, big_f, big_template_f, box_f = _qc_inputs(tmp_path)
    real_render = snap.create_grid_mri_image

    def _fail_on_the_outlined_figure(*args, **kwargs):
        # Only the full-FOV figure is drawn with an outline.
        if kwargs.get("outline_data") is not None:
            raise MemoryError("enlarged grid does not fit in memory")
        return real_render(*args, **kwargs)

    monkeypatch.setattr(snap, "create_grid_mri_image", _fail_on_the_outlined_figure)

    out = tmp_path / "conform.png"
    out_full = tmp_path / "conform_fullfov.png"
    result = snap.create_conform_qc(
        conformed_file=str(conformed),
        template_file=str(template),
        save_f=out,
        modality="anat",
        full_fov_file=big_f,
        full_fov_template_file=big_template_f,
        fov_box_file=box_f,
        full_fov_save_f=out_full,
    )
    assert set(result) == {"anat_conform_overlay"}
    assert out.exists() and out.stat().st_size > 0
    assert not out_full.exists()


def test_full_fov_figure_is_skipped_when_the_grid_was_not_enlarged(tmp_path):
    """No-expansion and fallback copy the conformed image under the full-FOV name.

    The FOV box then covers the whole grid, so no outline is drawn and figure 1 is a
    pixel-for-pixel duplicate of figure 2 captioned "lavender box marks ...".
    """
    from nhp_mri_prep.quality_control.snapshots import create_conform_qc

    conformed = _save(tmp_path / "conformed.nii.gz", (20, 20, 20), _REF_AFFINE, seed=3)
    template = _save(tmp_path / "template.nii.gz", (20, 20, 20), _REF_AFFINE, seed=4)
    # What _no_expansion() produces: the standard grid, and a box covering all of it.
    box_f = tmp_path / "fov_box.nii.gz"
    write_inner_box_mask(template, box_f, np.zeros(3, dtype=int), (20, 20, 20))

    out = tmp_path / "conform.png"
    out_full = tmp_path / "conform_fullfov.png"
    result = create_conform_qc(
        conformed_file=str(conformed),
        template_file=str(template),
        save_f=out,
        modality="anat",
        full_fov_file=conformed,
        full_fov_template_file=template,
        fov_box_file=box_f,
        full_fov_save_f=out_full,
    )
    assert set(result) == {"anat_conform_overlay"}
    assert out.exists()
    assert not out_full.exists()
