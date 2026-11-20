"""Unit tests for custom-template passthrough and JSON sidecar generation.

Covers:
- is_custom_template_path / has_nifti_extension / space_label_for (utils/templates.py)
- resolve_template custom-path branch: valid file, missing file, wrong extension,
  and unchanged registered-spec behaviour
- create_bids_sidecar_filename (utils/bids.py)
- write_derivative_sidecar / write_dataset_description / template_source_block (utils/sidecar.py)
"""

import json
from pathlib import Path

import pytest

from nhp_mri_prep.utils.templates import (
    is_custom_template_path,
    has_nifti_extension,
    space_label_for,
    resolve_template,
)
from nhp_mri_prep.utils.bids import create_bids_sidecar_filename
from nhp_mri_prep.version import get_version
from nhp_mri_prep.utils.sidecar import (
    write_derivative_sidecar,
    write_dataset_description,
    template_source_block,
    engine_generated_by,
    bold_timeseries_fields,
    ENGINE_DISPLAY,
)


# --------------------------------------------------------------------------- #
# path detection / label
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [
        ("/data/tpl.nii.gz", True),
        ("/data/tpl.nii", True),
        ("rel/dir/tpl.nii.gz", True),
        ("tpl.nii", True),  # bare NIfTI filename
        ("/data/tpl.mgz", True),  # path with wrong ext still "intended as custom"
        ("/data/tpl", True),  # extensionless path
        ("NMT2Sym:res-05", False),
        ("NMT2Sym", False),
        ("T1w", False),
        ("", False),
        (None, False),
    ],
)
def test_is_custom_template_path(value, expected):
    assert is_custom_template_path(value) is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("x.nii", True),
        ("x.nii.gz", True),
        ("x.mgz", False),
        ("x", False),
        ("NMT2Sym:res-05", False),
    ],
)
def test_has_nifti_extension(value, expected):
    assert has_nifti_extension(value) is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("/data/tpl.nii.gz", "template"),
        ("tpl.nii", "template"),
        ("NMT2Sym:res-05", "NMT2Sym"),
        ("MEBRAINS", "MEBRAINS"),
        ("112RMSL:res-05", "112RMSL"),
        ("", "NMT2Sym"),
        (None, "NMT2Sym"),
    ],
)
def test_space_label_for(value, expected):
    assert space_label_for(value) == expected


# --------------------------------------------------------------------------- #
# resolve_template custom-path branch
# --------------------------------------------------------------------------- #
def test_resolve_template_valid_custom_file(tmp_path):
    tpl = tmp_path / "my_tpl.nii.gz"
    tpl.write_bytes(b"")
    assert resolve_template(str(tpl)) == str(tpl.resolve())


def test_resolve_template_missing_custom_file_raises():
    with pytest.raises(ValueError, match="Custom template file not found"):
        resolve_template("/data/does_not_exist.nii.gz")


def test_resolve_template_wrong_extension_raises(tmp_path):
    # File exists but is not .nii/.nii.gz -> must raise, never silently accept.
    bad = tmp_path / "tpl.mgz"
    bad.write_bytes(b"")
    with pytest.raises(ValueError, match="must be a .nii or .nii.gz file"):
        resolve_template(str(bad))


def test_resolve_template_extensionless_raises(tmp_path):
    bad = tmp_path / "tpl"
    bad.write_bytes(b"")
    with pytest.raises(ValueError, match="must be a .nii or .nii.gz file"):
        resolve_template(str(bad))


def test_resolve_template_registered_spec_unchanged():
    # Bundled template still resolves to a real file under template_zoo.
    resolved = resolve_template("NMT2Sym:res-05")
    assert resolved.endswith(".nii.gz")
    assert Path(resolved).is_file()


# --------------------------------------------------------------------------- #
# sidecar filename pairing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "data,sidecar",
    [
        (
            "sub-01_space-NMT2Sym_desc-preproc_T1w.nii.gz",
            "sub-01_space-NMT2Sym_desc-preproc_T1w.json",
        ),
        (
            "sub-01_from-T1w_to-template_mode-image_xfm.mat",
            "sub-01_from-T1w_to-template_mode-image_xfm.json",
        ),
        (
            "sub-01_from-T1w_to-NMT2Sym_mode-image_xfm.h5",
            "sub-01_from-T1w_to-NMT2Sym_mode-image_xfm.json",
        ),
        (
            "sub-01_hemi-L_desc-cortex_mask.label.gii",
            "sub-01_hemi-L_desc-cortex_mask.json",
        ),
        (
            "sub-01_desc-confounds_timeseries.tsv",
            "sub-01_desc-confounds_timeseries.json",
        ),
        (
            "/abs/dir/sub-01_space-template_desc-brain_mask.nii.gz",
            "sub-01_space-template_desc-brain_mask.json",
        ),
    ],
)
def test_create_bids_sidecar_filename(data, sidecar):
    assert create_bids_sidecar_filename(data) == sidecar


# --------------------------------------------------------------------------- #
# template_source_block
# --------------------------------------------------------------------------- #
def test_template_source_block_builtin():
    b = template_source_block("NMT2Sym:res-05", "/zoo/NMT2Sym.nii.gz")
    assert b == {
        "OutputSpace": "NMT2Sym:res-05",
        "TemplatePath": "/zoo/NMT2Sym.nii.gz",
        "Custom": False,
    }


def test_template_source_block_custom():
    b = template_source_block("/data/custom.nii.gz", "/data/custom.nii.gz")
    assert b["Custom"] is True
    assert b["OutputSpace"] == "/data/custom.nii.gz"


# --------------------------------------------------------------------------- #
# write_derivative_sidecar
# --------------------------------------------------------------------------- #
def _sidecar(tmp_path, name, **kw):
    img = tmp_path / name
    img.write_bytes(b"")
    p = write_derivative_sidecar(img, **kw)
    return json.loads(Path(p).read_text())


def test_sidecar_custom_template_space(tmp_path):
    data = _sidecar(
        tmp_path,
        "sub-01_space-template_desc-preproc_T1w.nii.gz",
        output_space="/data/custom.nii.gz",
        resolved_template_path="/data/custom.nii.gz",
        skull_stripped=True,
        sources=["/raw/sub-01_T1w.nii.gz"],
    )
    assert data["SpatialReference"] == "template"
    assert data["TemplateSource"]["Custom"] is True
    assert data["TemplateSource"]["OutputSpace"] == "/data/custom.nii.gz"
    assert data["SkullStripped"] is True
    assert data["Sources"] == ["/raw/sub-01_T1w.nii.gz"]
    assert data["GeneratedBy"][0]["Name"] == "brainana"
    assert data["GeneratedBy"][0]["Version"] == get_version()


def test_sidecar_builtin_template_space(tmp_path):
    data = _sidecar(
        tmp_path,
        "sub-01_space-NMT2Sym_desc-preproc_T1w.nii.gz",
        output_space="NMT2Sym:res-05",
        resolved_template_path="/zoo/NMT2Sym.nii.gz",
    )
    assert data["SpatialReference"] == "NMT2Sym"
    assert data["TemplateSource"]["Custom"] is False


def test_sidecar_native_space_has_no_template_block(tmp_path):
    # space-T1w is native; template label (NMT2Sym) does not match -> no TemplateSource.
    data = _sidecar(
        tmp_path,
        "sub-01_space-T1w_desc-brain_mask.nii.gz",
        output_space="NMT2Sym:res-05",
        resolved_template_path="/zoo/NMT2Sym.nii.gz",
        roi_type="Brain",
    )
    assert "TemplateSource" not in data
    assert "SpatialReference" not in data
    assert data["Type"] == "Brain"


def test_sidecar_transform_to_template_gets_block(tmp_path):
    data = _sidecar(
        tmp_path,
        "sub-01_from-T1w_to-NMT2Sym_mode-image_xfm.h5",
        output_space="NMT2Sym:res-05",
        resolved_template_path="/zoo/NMT2Sym.nii.gz",
    )
    assert data["TemplateSource"]["OutputSpace"] == "NMT2Sym:res-05"


def test_sidecar_written_next_to_image_with_json_ext(tmp_path):
    img = tmp_path / "sub-01_space-T1w_desc-preproc_bold.nii.gz"
    img.write_bytes(b"")
    p = write_derivative_sidecar(img)
    assert Path(p).name == "sub-01_space-T1w_desc-preproc_bold.json"
    assert Path(p).parent == tmp_path


# --------------------------------------------------------------------------- #
# engine GeneratedBy (real registration engine on xfm sidecars)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "token,expected_name",
    [
        ("flirt", "FLIRT"),
        ("sitk", "SimpleITK"),
        ("simpleitk", "SimpleITK"),
        ("fireants", "FireANTs"),
        ("ants", "ANTs"),
        ("antspyx", "ANTsPy"),
        ("ANTs", "ANTs"),  # case-insensitive
    ],
)
def test_engine_generated_by_names(token, expected_name):
    gb = engine_generated_by(token)
    assert gb[0]["Name"] == expected_name
    # Version, when present, is a string; always exactly one entry (engine only).
    assert len(gb) == 1
    assert set(gb[0]) <= {"Name", "Version"}


@pytest.mark.parametrize("token", ["", None, "bogus-engine"])
def test_engine_generated_by_unknown_falls_back_to_brainana(token):
    gb = engine_generated_by(token)
    assert gb[0]["Name"] == "brainana"


def test_engine_display_covers_all_tokens():
    # The step-metadata tokens the .nf passes must all be mapped.
    for token in ("flirt", "sitk", "fireants", "ants", "antspyx"):
        assert token in ENGINE_DISPLAY


def test_xfm_sidecar_generatedby_is_engine(tmp_path):
    data = _sidecar(
        tmp_path,
        "sub-01_from-T1w_to-NMT2Sym_mode-image_xfm.h5",
        output_space="NMT2Sym:res-05",
        resolved_template_path="/zoo/NMT2Sym.nii.gz",
        engine="flirt",
    )
    assert len(data["GeneratedBy"]) == 1
    assert data["GeneratedBy"][0]["Name"] == "FLIRT"
    # engine-only: brainana must not appear in an xfm sidecar's GeneratedBy
    assert all(e["Name"] != "brainana" for e in data["GeneratedBy"])


def test_non_engine_sidecar_generatedby_is_brainana(tmp_path):
    data = _sidecar(
        tmp_path,
        "sub-01_space-NMT2Sym_desc-preproc_T1w.nii.gz",
        output_space="NMT2Sym:res-05",
        resolved_template_path="/zoo/NMT2Sym.nii.gz",
    )
    assert data["GeneratedBy"][0]["Name"] == "brainana"


def test_xfm_sidecar_unknown_engine_falls_back(tmp_path):
    data = _sidecar(
        tmp_path,
        "sub-01_from-T1w_to-NMT2Sym_mode-image_xfm.h5",
        output_space="NMT2Sym:res-05",
        engine=None,
    )
    assert data["GeneratedBy"][0]["Name"] == "brainana"


# --------------------------------------------------------------------------- #
# write_dataset_description
# --------------------------------------------------------------------------- #
def test_write_dataset_description(tmp_path):
    p = write_dataset_description(
        tmp_path,
        output_space="NMT2Sym:res-05",
        resolved_template_path="/zoo/NMT2Sym.nii.gz",
    )
    data = json.loads(Path(p).read_text())
    assert Path(p).name == "dataset_description.json"
    assert data["DatasetType"] == "derivative"
    assert data["GeneratedBy"][0]["Name"] == "brainana"
    assert data["GeneratedBy"][0]["Version"] == get_version()
    assert data["TemplateSource"]["Custom"] is False


def test_write_dataset_description_custom(tmp_path):
    p = write_dataset_description(
        tmp_path,
        output_space="/data/custom.nii.gz",
        resolved_template_path="/data/custom.nii.gz",
    )
    data = json.loads(Path(p).read_text())
    assert data["TemplateSource"]["Custom"] is True


# --------------------------------------------------------------------------- #
# RepetitionTime / SliceTimingCorrected on 4D preproc bold sidecars
# --------------------------------------------------------------------------- #
def test_sidecar_writes_repetition_time_and_slice_timing_corrected(tmp_path):
    data = _sidecar(
        tmp_path,
        "sub-01_space-NMT2Sym_desc-preproc_bold.nii.gz",
        repetition_time=1.7,
        slice_timing_corrected=True,
    )
    assert data["RepetitionTime"] == 1.7
    assert data["SliceTimingCorrected"] is True


def test_sidecar_slice_timing_corrected_false_is_written(tmp_path):
    # bool(False) is not None -> the key is emitted (distinct from "unknown").
    data = _sidecar(
        tmp_path,
        "sub-01_space-bold_desc-preproc_bold.nii.gz",
        slice_timing_corrected=False,
    )
    assert data["SliceTimingCorrected"] is False


def _make_raw_bold(dataset_dir, *, with_session, metadata):
    """Create a raw BIDS bold pair and return the .nii.gz path for bold_timeseries_fields."""
    sub = dataset_dir / "sub-01"
    func = (sub / "ses-001" / "func") if with_session else (sub / "func")
    func.mkdir(parents=True)
    stem = (
        "sub-01_ses-001_task-rest_run-1_bold"
        if with_session
        else "sub-01_task-rest_run-1_bold"
    )
    (func / f"{stem}.json").write_text(json.dumps(metadata))
    nii = func / f"{stem}.nii.gz"
    nii.write_bytes(b"")
    return nii


_STC_ON = {"func": {"slice_timing_correction": {"enabled": True}}}
_STC_OFF = {"func": {"slice_timing_correction": {"enabled": False}}}


def test_bold_timeseries_fields_session_layout(tmp_path):
    nii = _make_raw_bold(
        tmp_path,
        with_session=True,
        metadata={
            "RepetitionTime": 1.7,
            "SliceTiming": [0.0, 0.85],
            "TaskName": "rest",
        },
    )
    assert bold_timeseries_fields(nii, _STC_ON) == {
        "repetition_time": 1.7,
        "slice_timing_corrected": True,
    }


def test_bold_timeseries_fields_no_session_layout(tmp_path):
    nii = _make_raw_bold(
        tmp_path,
        with_session=False,
        metadata={"RepetitionTime": 1.7, "SliceTiming": [0.0, 0.85]},
    )
    # dataset_dir walks up to the sub- ancestor's parent regardless of session nesting.
    assert bold_timeseries_fields(nii, _STC_ON) == {
        "repetition_time": 1.7,
        "slice_timing_corrected": True,
    }


def test_bold_timeseries_fields_stc_disabled_keeps_tr(tmp_path):
    nii = _make_raw_bold(
        tmp_path,
        with_session=True,
        metadata={"RepetitionTime": 1.7, "SliceTiming": [0.0, 0.85]},
    )
    assert bold_timeseries_fields(nii, _STC_OFF) == {
        "repetition_time": 1.7,
        "slice_timing_corrected": False,
    }


def test_bold_timeseries_fields_invalid_slice_timing(tmp_path):
    # max(SliceTiming) > TR -> pipeline auto-disables STC for the run.
    nii = _make_raw_bold(
        tmp_path,
        with_session=True,
        metadata={"RepetitionTime": 1.0, "SliceTiming": [0.0, 2.0]},
    )
    assert bold_timeseries_fields(nii, _STC_ON) == {
        "repetition_time": 1.0,
        "slice_timing_corrected": False,
    }


def test_bold_timeseries_fields_missing_json(tmp_path):
    # No sidecar anywhere -> no TR, STC False; never raises.
    sub = tmp_path / "sub-01" / "ses-001" / "func"
    sub.mkdir(parents=True)
    nii = sub / "sub-01_ses-001_task-rest_run-1_bold.nii.gz"
    nii.write_bytes(b"")
    assert bold_timeseries_fields(nii, _STC_ON) == {"slice_timing_corrected": False}


def test_bold_timeseries_fields_tr_present_no_slice_timing(tmp_path):
    # TR but no SliceTiming array -> TR carried, STC False (nothing to correct).
    nii = _make_raw_bold(tmp_path, with_session=True, metadata={"RepetitionTime": 1.7})
    assert bold_timeseries_fields(nii, _STC_ON) == {
        "repetition_time": 1.7,
        "slice_timing_corrected": False,
    }
