"""Template-initialised surfaces: carry a template's white surface into a subject.

Stage s12b uses these instead of tessellating the subject's white-matter volume
(s08-s12). The template mesh is closed and genus 0, so no topology fix is needed,
and every subject reconstructed this way shares the template's vertex numbering,
so a label defined on the template (V1) indexes the same place in every subject.

Coordinates: FreeSurfer surfaces store "surface RAS" (tkr) coordinates of the
volume they were made from. Surface RAS uses a fixed LIA-style voxel frame
whatever the volume's real orientation, so converting to scanner RAS goes through
voxels: ``scanner = vox2ras @ inv(vox2ras_tkr) @ surface``, both matrices rebuilt
from the volume-geometry block every surface carries. (Only for LIA volumes, such
as conformed orig.mgz, does that reduce to ``scanner = surface + c_ras``.) The
template surface is taken to scanner RAS (template world), moved into the
subject's T1w world with ANTs, and expressed in the subject orig.mgz's surface
RAS. The subject's orig.mgz is conformed from that T1w and keeps its world space,
which is what makes the T1w transform usable here.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from nibabel.freesurfer import read_geometry, read_label, write_geometry

from ..wrappers.base import run_fs_command

logger = logging.getLogger(__name__)


def volume_info_from_mgz(mgz_path: Path) -> dict:
    """The volume-geometry block FreeSurfer writes into a surface made from ``mgz_path``.

    Keys and types match what ``nibabel.freesurfer.read_geometry(...,
    read_metadata=True)`` returns for a FreeSurfer-written surface, so the
    result can be passed straight to ``write_geometry(volume_info=...)``.
    """
    hdr = nib.load(str(mgz_path)).header
    mdc = np.asarray(hdr["Mdc"], dtype=float)  # rows are the x/y/z direction cosines
    return {
        "head": np.array([2, 0, 20], dtype=np.int32),
        "valid": "1  # volume info valid",
        "filename": str(mgz_path),
        "volume": np.asarray(hdr["dims"][:3], dtype=np.int64),
        "voxelsize": np.asarray(hdr["delta"], dtype=float),
        "xras": mdc[0],
        "yras": mdc[1],
        "zras": mdc[2],
        "cras": np.asarray(hdr["Pxyz_c"], dtype=float),
    }


def _vox2ras(info: dict) -> np.ndarray:
    """Scanner vox2ras from a FreeSurfer volume-geometry block."""
    mdc = np.column_stack([info["xras"], info["yras"], info["zras"]]).astype(float)
    delta = np.asarray(info["voxelsize"], dtype=float)
    dims = np.asarray(info["volume"], dtype=float)
    m = np.eye(4)
    m[:3, :3] = mdc * delta
    m[:3, 3] = np.asarray(info["cras"], dtype=float) - m[:3, :3] @ (dims / 2.0)
    return m


def _vox2ras_tkr(info: dict) -> np.ndarray:
    """FreeSurfer's surface-RAS (tkregister) vox2ras for the same grid: LIA, centred."""
    dx, dy, dz = np.asarray(info["voxelsize"], dtype=float)
    nx, ny, nz = np.asarray(info["volume"], dtype=float)
    return np.array(
        [
            [-dx, 0.0, 0.0, dx * nx / 2.0],
            [0.0, 0.0, dz, -dz * nz / 2.0],
            [0.0, -dy, 0.0, dy * ny / 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def surface_to_scanner(points: np.ndarray, info: dict) -> np.ndarray:
    """Surface RAS -> scanner RAS for the volume described by ``info``."""
    m = _vox2ras(info) @ np.linalg.inv(_vox2ras_tkr(info))
    return (m @ np.c_[points, np.ones(len(points))].T)[:3].T


def scanner_to_surface(points: np.ndarray, info: dict) -> np.ndarray:
    """Scanner RAS -> surface RAS for the volume described by ``info``."""
    m = _vox2ras_tkr(info) @ np.linalg.inv(_vox2ras(info))
    return (m @ np.c_[points, np.ones(len(points))].T)[:3].T


def transform_points_ants(ras_points: np.ndarray, xfm: Path, log_file: Optional[Path] = None) -> np.ndarray:
    """Map world (RAS) points through an ANTs transform with antsApplyTransformsToPoints.

    ANTs works in LPS. With the subject-to-template *image* transform
    (from-T1w_to-<template>), points go template -> subject: an image transform
    maps fixed-space (template) points to moving-space (subject) points.
    Accepts whatever the registration wrote: a displacement field (.nii.gz),
    a composite (.h5) or an affine (.mat).
    """
    lps = np.asarray(ras_points, dtype=float) * [-1.0, -1.0, 1.0]
    with tempfile.TemporaryDirectory(prefix="tpl_points_") as tmp:
        src, dst = Path(tmp) / "in.csv", Path(tmp) / "out.csv"
        np.savetxt(src, np.c_[lps, np.zeros(len(lps))], delimiter=",",
                   header="x,y,z,t", comments="", fmt="%.6f")
        run_fs_command(
            ["antsApplyTransformsToPoints", "-d", "3", "-i", str(src), "-o", str(dst), "-t", str(xfm)],
            log_file=log_file,
            expect_outputs=[dst],
        )
        moved = np.loadtxt(dst, delimiter=",", skiprows=1)[:, :3]
    if moved.shape != lps.shape:
        raise RuntimeError(
            f"antsApplyTransformsToPoints returned {moved.shape[0]} points for {lps.shape[0]}"
        )
    return moved * [-1.0, -1.0, 1.0]


def warp_template_surface(
    template_surf: Path,
    xfm: Path,
    subject_orig_mgz: Path,
    out_surf: Path,
    log_file: Optional[Path] = None,
) -> np.ndarray:
    """Write ``template_surf`` carried into the subject, in the subject's surface RAS.

    Returns the subject scanner-RAS vertex coordinates (for records and tests).
    The face list is the template's, unchanged.
    """
    verts, faces, meta = read_geometry(str(template_surf), read_metadata=True)
    if not all(k in meta for k in ("xras", "yras", "zras", "cras", "volume", "voxelsize")):
        raise ValueError(f"{template_surf} has no volume geometry; cannot place it in world space")
    template_world = surface_to_scanner(verts, meta)
    subject_world = transform_points_ants(template_world, xfm, log_file=log_file)
    info = volume_info_from_mgz(subject_orig_mgz)
    write_geometry(str(out_surf), scanner_to_surface(subject_world, info), faces, volume_info=info)
    return subject_world


def fraction_in_mask(surface: Path, mask_mgz: Path) -> float:
    """Share of ``surface``'s vertices whose nearest voxel of ``mask_mgz`` is in the mask.

    ``surface`` must be in the surface RAS of a volume on ``mask_mgz``'s grid (any
    volume conformed with orig.mgz). Vertices off the grid count as outside. A
    fitted white surface lies wholly inside the brain mask (1.0 on every devtest
    reconstruction), so a warped template well below that was misregistered.
    """
    img = nib.load(str(mask_mgz))
    inside_mask = np.asarray(img.dataobj) > 0
    coords, _ = read_geometry(str(surface))
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(img.header.get_vox2ras_tkr()), coords)).astype(int)
    on_grid = np.all((ijk >= 0) & (ijk < inside_mask.shape[:3]), axis=1)
    inside = np.zeros(len(coords), dtype=bool)
    inside[on_grid] = inside_mask[tuple(ijk[on_grid].T)]
    return float(inside.mean())


def write_vertex_label(path: Path, vertices: np.ndarray, coords: np.ndarray, comment: str) -> None:
    """Write a FreeSurfer ASCII label of ``vertices`` (indices into ``coords``)."""
    vertices = np.asarray(vertices, dtype=int)
    with open(path, "w") as fh:
        fh.write(f"#!ascii label , {comment}\n{len(vertices)}\n")
        for i in vertices:
            x, y, z = coords[i]
            fh.write(f"{i} {x:.3f} {y:.3f} {z:.3f} 0.0000000000\n")


def write_rip_label(path: Path, surface: Path, frozen: Path, within: Optional[Path] = None) -> int:
    """Label of the vertices placement may move: ``within`` (default: all) minus ``frozen``.

    mris_place_surface --rip-label holds every vertex *outside* the label fixed,
    so passing this label freezes the ``frozen`` vertices (and anything outside
    ``within``). Returns the number of vertices in the written label.
    """
    coords, _ = read_geometry(str(surface))
    movable = np.ones(len(coords), dtype=bool)
    if within is not None:
        movable[:] = False
        movable[read_label(str(within))] = True
    movable[read_label(str(frozen))] = False
    idx = np.flatnonzero(movable)
    write_vertex_label(path, idx, coords, f"placement: movable vertices (frozen = {Path(frozen).name})")
    return int(idx.size)
