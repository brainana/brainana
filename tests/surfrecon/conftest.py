"""Shared fixtures for surface-reconstruction tests.

Every mesh here is synthesised in numpy and written through
``nibabel.freesurfer.write_geometry``, so the whole suite runs in well under a
second and needs neither FreeSurfer binaries nor real subject data.

The three cube variants exist to cover a distinction the pipeline used to miss:
Euler characteristic alone does not characterise a good surface. ``badwind_cube``
is closed with Euler 2 and still broken.
"""

import os
import warnings

import nibabel.freesurfer.io as fs
import numpy as np
import pytest


@pytest.fixture(autouse=True, scope="session")
def _username_for_nibabel():
    """nibabel's surface writer calls getpass.getuser(), which fails on some hosts."""
    os.environ.setdefault("USERNAME", "test")


CUBE_V = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
)

CUBE_F = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ],
    dtype=np.int32,
)


@pytest.fixture
def closed_cube():
    """Closed, consistently oriented, Euler 2 -- a healthy surface."""
    return CUBE_V.copy(), CUBE_F.copy()


@pytest.fixture
def open_cube():
    """One square face removed: open (boundary edges), Euler 1."""
    return CUBE_V.copy(), CUBE_F[2:].copy()


@pytest.fixture
def badwind_cube():
    """One triangle wound backwards: closed and Euler 2, but NOT oriented.

    This is the case that a check based only on the Euler number passes.
    """
    faces = CUBE_F.copy()
    faces[0] = faces[0][::-1]
    return CUBE_V.copy(), faces


@pytest.fixture
def write_surf(tmp_path):
    """Write (vertices, faces) to a FreeSurfer surface file and return the path."""

    def _write(name, vertices, faces, volume_info=None):
        path = tmp_path / name
        with warnings.catch_warnings():
            # nibabel warns when volume_info is absent; irrelevant for fixtures.
            warnings.simplefilter("ignore")
            fs.write_geometry(path, vertices, faces, volume_info=volume_info)
        return path

    return _write


@pytest.fixture
def freesurfer_home(tmp_path, monkeypatch):
    """An existing, empty FREESURFER_HOME so ReconSurfConfig validates without FreeSurfer."""
    fs_home = tmp_path / "fs"
    fs_home.mkdir()
    monkeypatch.setenv("FREESURFER_HOME", str(fs_home))
    return fs_home
