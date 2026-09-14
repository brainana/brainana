"""Grid comparisons used to prove a volume did not move.

These back the longitudinal stream's two geometry guards. The property that
matters is the one in `describe_geometry_mismatch`'s docstring: shape and zooms
alone accept a translation or an axis flip, and a translated destination grid is
exactly what a mis-targeted LTA produces -- so the affine has to be compared, and
these tests pin that it is.
"""

import nibabel as nib
import numpy as np
import pytest

from fastsurfer_surfrecon.utils.geometry import (
    describe_geometry_mismatch,
    volume_affine,
    volume_geometry,
)

SHAPE = (4, 4, 4)
ZOOMS = (0.8, 0.8, 0.8)


def _vol(path, shape=SHAPE, zooms=ZOOMS, translation=(0.0, 0.0, 0.0), flip=False):
    affine = np.diag([*zooms, 1.0]).astype(float)
    if flip:
        affine[0, 0] = -affine[0, 0]
    affine[:3, 3] = translation
    nib.save(nib.MGHImage(np.zeros(shape, dtype=np.uint8), affine), str(path))
    return path


def test_identical_volumes_report_no_mismatch(tmp_path):
    a = _vol(tmp_path / "a.mgz")
    b = _vol(tmp_path / "b.mgz")
    assert describe_geometry_mismatch(a, b) is None


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"shape": (4, 4, 5)}, "shape"),
        ({"zooms": (0.8, 0.8, 1.0)}, "zooms"),
        ({"translation": (0.5, 0.0, 0.0)}, "affine"),
        ({"flip": True}, "affine"),
    ],
)
def test_each_kind_of_difference_is_named(tmp_path, kwargs, expected):
    a = _vol(tmp_path / "a.mgz")
    b = _vol(tmp_path / "b.mgz", **kwargs)
    message = describe_geometry_mismatch(a, b)
    assert message is not None and expected in message


@pytest.mark.parametrize("kwargs", [{"translation": (0.5, 0.0, 0.0)}, {"flip": True}])
def test_shape_and_zooms_alone_would_miss_these(tmp_path, kwargs):
    """The reason volume_affine exists: these two agree on shape and zooms."""
    a = _vol(tmp_path / "a.mgz")
    b = _vol(tmp_path / "b.mgz", **kwargs)
    assert volume_geometry(a) == volume_geometry(b)
    assert describe_geometry_mismatch(a, b) is not None


def test_labels_appear_so_the_reader_knows_which_is_which(tmp_path):
    a = _vol(tmp_path / "a.mgz")
    b = _vol(tmp_path / "b.mgz", shape=(4, 4, 5))
    message = describe_geometry_mismatch(a, b, labels=("the timepoint", "the base"))
    assert "the timepoint" in message and "the base" in message


def test_shape_is_json_serialisable(tmp_path):
    """These land in provenance files; MGH headers hand back numpy int32."""
    import json

    shape, zooms = volume_geometry(_vol(tmp_path / "a.mgz"))
    json.dumps({"shape": list(shape), "zooms": list(zooms)})


def test_affine_is_the_full_voxel_to_world_mapping(tmp_path):
    affine = volume_affine(_vol(tmp_path / "a.mgz", translation=(1.0, 2.0, 3.0)))
    assert affine.shape == (4, 4)
    assert np.allclose(affine[:3, 3], [1.0, 2.0, 3.0])
