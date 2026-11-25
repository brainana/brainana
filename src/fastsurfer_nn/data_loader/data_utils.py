# Copyright 2023 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# IMPORTS
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nibabel.filebasedimages import FileBasedHeader as _Header
from numpy import typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import (
    binary_closing,
    binary_erosion,
    uniform_filter,
    zoom,
)
from skimage.measure import label

from fastsurfer_nn.atlas.atlas_manager import get_atlas_manager
from fastsurfer_nn.data_loader.conform import check_affine_in_nifti, conform, is_conform
from fastsurfer_nn.utils import logging

##
# Global Vars
##
SUPPORTED_OUTPUT_FILE_FORMATS = ("mgz", "nii", "nii.gz")
LOGGER = logging.getLogger(__name__)

##
# Helper Functions
##

##
# Image Loading and Saving
##


# Conform an MRI brain image to UCHAR, RAS orientation, and 1mm or minimal isotropic
# voxels
def load_and_conform_image(
    img_filename: Path | str,
    order: int = 1,
    logger: logging.Logger = LOGGER,
    **conform_kwargs,
) -> tuple[_Header, np.ndarray, np.ndarray]:
    """
    Load MRI image and conform it to UCHAR, RAS orientation and 1mm or minimum isotropic
    voxels size.

    Only, if it does not already have this format.

    Parameters
    ----------
    img_filename : Path, str
        Path and name of volume to read.
    order : int, default=1
        Interpolation order for image conformation (0=nearest, 1=linear(default), 2=quadratic, 3=cubic).
    logger : logging.Logger, default=<local logger>
        Logger to write output to (default = STDOUT).
    **conform_kwargs
        Additional parameters to conform and is_conform.

    Returns
    -------
    nibabel.Header header_info
        Header information of the conformed image.
    numpy.ndarray affine_info
        Affine information of the conformed image.
    numpy.ndarray orig_data
        Conformed image data.

    Raises
    ------
    RuntimeError
        If input has multiple input frames or inconsistent nifti headers.
    """
    img_file = Path(img_filename)
    orig = cast(nib.analyze.SpatialImage, nib.load(img_file))
    # is_conform and conform accept numeric values and the string 'min' instead of the bool value
    if not is_conform(orig, **conform_kwargs):
        logger.info(
            "Preprocessing: conforming image to UCHAR, RAS orientation, and minimum isotropic voxels"
        )

        if len(orig.shape) > 3 and orig.shape[3] != 1:
            raise RuntimeError(
                f"Multiple input frames ({orig.shape[3]}) not supported!"
            )

        # Check affine if image is nifti image
        if img_file.suffix == ".nii" or img_file.suffixes[-2:] == [".nii", ".gz"]:
            if not check_affine_in_nifti(
                cast(nib.nifti1.Nifti1Image | nib.nifti2.Nifti1Image, orig),
                logger=logger,
            ):
                raise RuntimeError("Inconsistency in nifti-header!")

        # conform
        orig = conform(orig, order=order, **conform_kwargs)

    # Return header and affine information
    return orig.header, orig.affine, np.asanyarray(orig.dataobj)


def load_image(
    file: str | Path,
    name: str = "image",
    **kwargs,
) -> tuple[nib.analyze.SpatialImage, np.ndarray]:
    """
    Load file 'file' with nibabel, including all data.

    Parameters
    ----------
    file : Path, str
        Path to the file to load.
    name : str, default="image"
        Name of the file (optional), only effects error messages.
    **kwargs :
        Additional keyword arguments.

    Returns
    -------
    Tuple[nib.analyze.SpatialImage, np.ndarray]
        The nibabel image object and a numpy array of the data.

    Raises
    ------
    IOError
        Failed loading the file
        nibabel releases the GIL, so the following is a parallel example.
        {
        >>> from concurrent.futures import ThreadPoolExecutor
        >>> with ThreadPoolExecutor() as pool:
        >>>     future1 = pool.submit(load_image, filename1)
        >>>     future2 = pool.submit(load_image, filename2)
        >>>     image, data = future1.result()
        >>>     image2, data2 = future2.result()
        }
    """
    try:
        img = cast(nib.analyze.SpatialImage, nib.load(file, **kwargs))
    except (OSError, FileNotFoundError) as e:
        raise OSError(
            f"Failed loading the {name} '{file}' with error: {e.args[0]}"
        ) from e
    return img, np.asarray(img.dataobj)


def load_maybe_conform(
    file: Path | str,
    alt_file: Path | str,
    **conform_kwargs,
) -> tuple[Path, nib.analyze.SpatialImage, np.ndarray]:
    """
    Load an image by file, check whether it is conformed to vox_size and conform to
    vox_size if it is not.

    Parameters
    ----------
    file : Path, str
        Path to the file to load.
    alt_file : Path, str
        Alternative file to interpolate from.
    **conform_kwargs
        Additional parameters to conform and is_conform.

    Returns
    -------
    Path
        The path to the file.
    nib.analyze.SpatialImage
        The file container object including the corrected header.
    np.ndarray
        The data loaded from the file.

    See Also
    --------
    fastsurfer_nn.data_loader.conform.conform
        For additional parameters supported via `conform_kwargs`.
    """
    file = Path(file)
    alt_file = Path(alt_file)
    conform_kwargs_is_conform = dict(conform_kwargs.items())
    del conform_kwargs_is_conform["order"]

    _is_conform, img = False, None
    if file.is_file():
        # see if the file is 1mm
        img = cast(nib.analyze.SpatialImage, nib.load(file))
        # is_conform only needs the header, not the data
        _is_conform = is_conform(
            img, **conform_kwargs_is_conform, verbose=False, vox_eps=0.1
        )

    if _is_conform:
        # calling np.asarray here, forces the load of img.dataobj into memory
        # (which is parallel with other operations, if done here)
        data = np.asarray(img.dataobj)
        dst_file = file
    else:
        # the image is not conformed to 1mm, do this now.
        fileext = [
            ext
            for ext in SUPPORTED_OUTPUT_FILE_FORMATS
            if file.name.endswith("." + ext)
        ]
        if len(fileext) != 1:
            raise RuntimeError(
                f"Invalid file extension of conf_name: {file}, must be one of {SUPPORTED_OUTPUT_FILE_FORMATS}."
            )
        file_no_fileext = str(file)[: -len(fileext[0]) - 1]
        vox_size = conform_kwargs.get("vox_size", 1.0)
        vox_suffix = (
            ".min" if vox_size == "min" else f".{str(vox_size).replace('.', '')}mm"
        )
        if not file_no_fileext.endswith(vox_suffix):
            file_no_fileext += vox_suffix
        # if the orig file is neither absolute nor in the subject path, use the conformed file
        src_file = alt_file if alt_file.is_file() else file
        if not alt_file.is_file():
            LOGGER.warning(
                f"No valid alternative file (e.g. orig, here: {alt_file}) was given to interpolate from, so we might "
                f"lose quality due to multiple chained interpolations. "
            )

        dst_file = Path(file_no_fileext + "." + fileext[0])
        # conform to 1mm
        header, affine, data = load_and_conform_image(
            src_file,
            logger=logging.getLogger(__name__ + ".conform"),
            **conform_kwargs,
        )

        # after conforming, save the conformed file
        save_image(header, affine, data, dst_file)
        img = nib.MGHImage(data, affine, header)
    return dst_file, img, data


# Save image routine
def save_image(
    header: _Header,
    affine: npt.NDArray[float],
    img_array: np.ndarray,
    output_f: str | Path,
    dtype: npt.DTypeLike | None = None,
) -> None:
    """
    Save an image (nibabel MGHImage), according to the desired output file format.

    Supported formats are defined in supported_output_file_formats. Saves predictions to save_as.

    Parameters
    ----------
    header : _Header
        Image header information.
    affine : npt.NDArray[float]
        Image affine information.
    img_array : np.ndarray
        An array containing image data.
    output_f : Path, str
        Name under which to save prediction; this determines output file format.
    dtype : npt.DTypeLike, optional
        Image array type; if provided, the image object is explicitly set to match this type.
    """
    output_f = Path(output_f)
    valid_ext = output_f.suffix[
        1:
    ] in SUPPORTED_OUTPUT_FILE_FORMATS or output_f.suffixes[-2:] == [".nii", ".gz"]
    if not valid_ext:
        raise ValueError(
            f"Output filename does not contain a supported file format {SUPPORTED_OUTPUT_FILE_FORMATS}! "
            f"Got: {output_f.suffixes}"
        )

    # Save image with header and affine
    # The header contains useful metadata beyond spatial info

    if output_f.suffix == ".mgz":
        # MGH format only supports specific dtypes: uint8, int16, int32, float32
        # Convert unsupported dtypes (like int64) to supported ones
        if img_array.dtype == np.int64:
            # Convert int64 to int32 (safe for label/mask values < 2^31)
            img_array = img_array.astype(np.int32)
        elif img_array.dtype == np.uint64:
            img_array = img_array.astype(np.uint32)
        elif img_array.dtype == np.float64:
            img_array = img_array.astype(np.float32)

        # Create MGH image with header and affine
        mgh_img = nib.MGHImage(img_array, affine, header)

    elif output_f.suffix == ".nii" or output_f.suffixes[-2:] == [".nii", ".gz"]:
        # Create NIfTI image with header and affine
        mgh_img = nib.nifti1.Nifti1Image(img_array, affine, header)
    else:
        # This should never happen due to the check above, but add for safety
        raise ValueError(f"Unsupported file format: {output_f.suffixes}")

    if dtype is not None:
        mgh_img.set_data_dtype(dtype)

    if output_f.suffix in (".mgz", ".nii"):
        nib.save(mgh_img, output_f)
    elif output_f.suffixes[-2:] == [".nii", ".gz"]:
        # For correct outputs, nii.gz files should be saved using the nifti1
        # sub-module's save():
        nib.nifti1.save(mgh_img, str(output_f))


##
# Orientation-aware spatial transformations
##


def get_plane_axes(
    orientation_code: str,
) -> dict[str, dict[str, int | tuple[int, int]]]:
    """
    Parse orientation code and return axis mappings for each anatomical plane view.

    This function determines which array axis corresponds to each anatomical plane
    by parsing the 3-letter orientation code (e.g., 'LIA', 'RAS', 'LPS').

    Anatomical planes are defined by the axis they slice through:
    - Axial: slices along Superior-Inferior (S/I) axis - horizontal brain cuts
    - Coronal: slices along Anterior-Posterior (A/P) axis - front-to-back cuts
    - Sagittal: slices along Left-Right (L/R) axis - side cuts

    Parameters
    ----------
    orientation_code : str
        3-letter orientation code where each letter indicates the direction
        that axis points to. E.g., 'LIA' means:
        - Axis 0 points Left
        - Axis 1 points Inferior
        - Axis 2 points Anterior

    Returns
    -------
    dict
        Dictionary with keys 'axial', 'coronal', 'sagittal', each containing:
        - 'slice_axis': int - which axis to iterate over for slices
        - 'shown_axes': tuple[int, int] - which two axes appear in the 2D slice

    Examples
    --------
    >>> get_plane_axes('LIA')
    {'axial': {'slice_axis': 1, 'shown_axes': (0, 2)},
     'coronal': {'slice_axis': 2, 'shown_axes': (0, 1)},
     'sagittal': {'slice_axis': 0, 'shown_axes': (1, 2)}}

    >>> get_plane_axes('RAS')
    {'axial': {'slice_axis': 2, 'shown_axes': (0, 1)},
     'coronal': {'slice_axis': 1, 'shown_axes': (0, 2)},
     'sagittal': {'slice_axis': 0, 'shown_axes': (1, 2)}}
    """
    orientation_code = orientation_code.upper()

    if len(orientation_code) != 3:
        raise ValueError(
            f"Orientation code must be 3 letters, got: '{orientation_code}'"
        )

    # Find which axis corresponds to each anatomical direction
    si_axis = None  # Superior-Inferior axis (for axial slices)
    ap_axis = None  # Anterior-Posterior axis (for coronal slices)
    lr_axis = None  # Left-Right axis (for sagittal slices)

    for i, direction in enumerate(orientation_code):
        if direction in ("S", "I"):
            si_axis = i
        elif direction in ("A", "P"):
            ap_axis = i
        elif direction in ("L", "R"):
            lr_axis = i
        else:
            raise ValueError(
                f"Invalid direction '{direction}' in orientation code '{orientation_code}'. "
                f"Must be one of: L, R, A, P, S, I"
            )

    # Validate all axes were found
    if si_axis is None or ap_axis is None or lr_axis is None:
        raise ValueError(
            f"Orientation code '{orientation_code}' must contain exactly one of "
            f"(S/I), (A/P), and (L/R)"
        )

    def get_shown_axes(slice_axis: int) -> tuple[int, int]:
        """Get the two axes shown in a 2D slice (perpendicular to slice_axis)."""
        return tuple(i for i in range(3) if i != slice_axis)

    return {
        "axial": {
            "slice_axis": si_axis,
            "shown_axes": get_shown_axes(si_axis),
        },
        "coronal": {
            "slice_axis": ap_axis,
            "shown_axes": get_shown_axes(ap_axis),
        },
        "sagittal": {
            "slice_axis": lr_axis,
            "shown_axes": get_shown_axes(lr_axis),
        },
    }


def get_plane_transform(
    plane: str,
    orientation_code: str = "lia",
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Get the axis transformation needed for a specific plane view.

    For 2D CNN processing with get_thick_slices(), we need to rearrange the 3D volume so that:
    - The slice axis is at position 2 (last axis) - get_thick_slices operates on axis 2
    - The two in-plane axes are at positions 0 and 1

    The in-plane axes are ordered so that the A-P axis (if present) comes first,
    matching the original FastSurfer transform behavior for LIA orientation.

    Parameters
    ----------
    plane : str
        One of 'axial', 'coronal', 'sagittal'.
    orientation_code : str, default='lia'
        3-letter orientation code (e.g., 'LIA', 'RAS').

    Returns
    -------
    forward_axes : tuple[int, ...]
        Axis order for np.moveaxis source argument to transform volume for plane processing.
    inverse_axes : tuple[int, ...]
        Axis order to reverse the transformation.

    Examples
    --------
    >>> fwd, inv = get_plane_transform('axial', 'LIA')
    >>> # For LIA: axial slices along axis 1 (I)
    >>> # Transform: move axis 1 to position 2 (last)
    >>> transformed = np.moveaxis(vol, fwd, (0, 1, 2))
    >>> original = np.moveaxis(transformed, inv, (0, 1, 2))
    """
    plane_axes = get_plane_axes(orientation_code)

    if plane not in plane_axes:
        raise ValueError(
            f"Invalid plane '{plane}'. Must be one of: axial, coronal, sagittal"
        )

    slice_axis = plane_axes[plane]["slice_axis"]
    shown_axes = list(plane_axes[plane]["shown_axes"])

    # For coronal plane (slicing along A-P), keep the natural order of shown axes
    # For other planes, we need to match the legacy behavior where A-P axis comes first
    # in the in-plane dimensions when it's not the slice axis.
    #
    # Original LIA transforms:
    #   - coronal (slice=A): identity, in-plane is (L, I) at positions (0, 1)
    #   - axial (slice=I): in-plane is (A, L) at positions (0, 1) → A first
    #   - sagittal (slice=L): in-plane is (A, I) at positions (0, 1) → A first
    #
    # The pattern: when A-P is in shown_axes, put it first
    ap_axis = plane_axes["coronal"]["slice_axis"]  # A-P axis

    if plane != "coronal" and ap_axis in shown_axes:
        # Put A-P axis first among the shown axes
        shown_axes.remove(ap_axis)
        shown_axes.insert(0, ap_axis)

    # Forward transform: shown_axes -> positions 0, 1 and slice_axis -> position 2
    forward_axes = (shown_axes[0], shown_axes[1], slice_axis)

    # Inverse transform: reverse the mapping
    inverse_axes = tuple(forward_axes.index(i) for i in range(3))

    return forward_axes, inverse_axes


def transform_for_plane(
    vol: npt.NDArray,
    plane: str,
    orientation_code: str = "lia",
    inverse: bool = False,
) -> np.ndarray:
    """
    Transform volume for processing a specific anatomical plane view.

    Rearranges the volume axes so that:
    - The slice axis (axis being iterated over) is at position 2 (last)
    - The two in-plane axes are at positions 0 and 1

    Works with any 3-letter orientation code (e.g., 'LIA', 'RAS', 'LPS').

    Parameters
    ----------
    vol : npt.NDArray
        Image volume to transform. Can be 3D (H, W, D) or 4D (H, W, D, C).
    plane : str
        One of 'axial', 'coronal', 'sagittal'.
    orientation_code : str, default='lia'
        3-letter orientation code (e.g., 'LIA', 'RAS').
    inverse : bool, default=False
        If True, apply the inverse transformation (back to original orientation).

    Returns
    -------
    np.ndarray
        Transformed volume with axes rearranged for the specified plane.

    Examples
    --------
    >>> # Transform for axial plane processing
    >>> axial_vol = transform_for_plane(vol, 'axial', 'LIA')
    >>> # Process slices...
    >>> # Transform back
    >>> original_vol = transform_for_plane(axial_vol, 'axial', 'LIA', inverse=True)
    """
    forward_axes, inverse_axes = get_plane_transform(plane, orientation_code)

    if inverse:
        axes = inverse_axes
    else:
        axes = forward_axes

    # np.moveaxis expects source and destination
    # We want to move axes to positions (0, 1, 2)
    return np.moveaxis(vol, axes, (0, 1, 2))


def get_zoom_indices_for_plane(
    plane: str,
    orientation_code: str = "lia",
) -> tuple[int, int]:
    """
    Get the voxel size indices for the in-plane dimensions of a specific plane.

    When processing a plane, we need to know the voxel sizes for the two
    in-plane dimensions to properly scale the network input. The indices are
    returned in the order that corresponds to positions (0, 1) in the
    transformed volume (after transform_for_plane).

    Parameters
    ----------
    plane : str
        One of 'axial', 'coronal', 'sagittal'.
    orientation_code : str, default='lia'
        3-letter orientation code (e.g., 'LIA', 'RAS').

    Returns
    -------
    tuple[int, int]
        Indices into the voxel size array for the two in-plane dimensions,
        in the order they appear after the plane transform.

    Examples
    --------
    >>> voxel_sizes = np.array([0.5, 0.5, 0.5])  # LIA orientation
    >>> idx = get_zoom_indices_for_plane('axial', 'LIA')
    >>> in_plane_voxels = voxel_sizes[list(idx)]
    """
    # Get the forward transform axes - positions 0, 1 correspond to in-plane dims
    forward_axes, _ = get_plane_transform(plane, orientation_code)
    return (forward_axes[0], forward_axes[1])


def get_permute_order_for_plane(
    plane: str,
    orientation_code: str = "lia",
) -> tuple[int, int, int, int]:
    """
    Get the permutation order for rearranging a 4D tensor after plane inference.

    After 2D CNN inference on a plane, predictions have shape:
        (slice_batch, classes, in_plane_dim1, in_plane_dim2)

    We need to permute to the canonical volume format:
        (vol_dim0, vol_dim1, vol_dim2, classes)

    This function computes the permutation indices for torch.permute().

    Parameters
    ----------
    plane : str
        One of 'axial', 'coronal', 'sagittal'.
    orientation_code : str, default='lia'
        3-letter orientation code (e.g., 'LIA', 'RAS').

    Returns
    -------
    tuple[int, int, int, int]
        Permutation indices for torch.permute(). The index `0` appears at the
        position corresponding to the slice axis.

    Examples
    --------
    For LIA orientation:
    >>> get_permute_order_for_plane('axial', 'LIA')
    (3, 0, 2, 1)  # slice axis is 1 (I), so 0 goes to position 1

    >>> get_permute_order_for_plane('coronal', 'LIA')
    (2, 3, 0, 1)  # slice axis is 2 (A), so 0 goes to position 2

    >>> get_permute_order_for_plane('sagittal', 'LIA')
    (0, 3, 2, 1)  # slice axis is 0 (L), so 0 goes to position 0

    For RAS orientation:
    >>> get_permute_order_for_plane('axial', 'RAS')
    (2, 3, 0, 1)  # slice axis is 2 (S), so 0 goes to position 2

    >>> get_permute_order_for_plane('coronal', 'RAS')
    (3, 0, 2, 1)  # slice axis is 1 (A), so 0 goes to position 1

    >>> get_permute_order_for_plane('sagittal', 'RAS')
    (0, 3, 2, 1)  # slice axis is 0 (R), so 0 goes to position 0
    """
    # Get the forward transform to understand how axes are reordered
    # forward_axes = (in_plane_0_orig_axis, in_plane_1_orig_axis, slice_orig_axis)
    forward_axes, _ = get_plane_transform(plane, orientation_code)

    in_plane_0_axis = forward_axes[0]  # Original axis at transformed position 0
    in_plane_1_axis = forward_axes[1]  # Original axis at transformed position 1
    slice_axis = forward_axes[2]  # Original axis at transformed position 2 (slice)

    # Input tensor shape after dataset processing: (slice_batch, classes, in_plane_dim0, in_plane_dim1)
    # - Position 0: slice_batch → goes to slice_axis in output
    # - Position 1: classes → goes to position 3 in output
    # - Position 2: in_plane_dim0 → goes to in_plane_0_axis in output
    # - Position 3: in_plane_dim1 → goes to in_plane_1_axis in output
    #
    # Output tensor shape: (vol_dim0, vol_dim1, vol_dim2, classes)
    #
    # We need to compute: for each output position, which input position provides it?

    permute = [0, 0, 0, 0]

    # Slice axis in output comes from input position 0
    permute[slice_axis] = 0

    # in_plane_0_axis in output comes from input position 2
    permute[in_plane_0_axis] = 2

    # in_plane_1_axis in output comes from input position 3
    permute[in_plane_1_axis] = 3

    # Classes (position 3 in output) comes from input position 1
    permute[3] = 1

    return tuple(permute)


##
# Slice Processing
##


# Thick slice generator (for eval) and blank slices filter (for training)
def get_thick_slices(img_data: npt.NDArray, slice_thickness: int = 3) -> np.ndarray:
    """
    Extract thick slices from the image.

    Feed slice_thickness preceding and succeeding slices to network,
    label only middle one.

    Parameters
    ----------
    img_data : npt.NDArray
        3D MRI image read in with nibabel.
    slice_thickness : int
        Number of slices to stack on top and below slice of interest (default=3).

    Returns
    -------
    np.ndarray
        Image data with the thick slices of the n-th axis appended into the n+1-th axis.
    """
    img_data_pad = np.pad(
        img_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode="edge"
    )

    # sliding_window_view will automatically create thick slices through a sliding window, but as this in only a view,
    # less memory copies are required
    return sliding_window_view(img_data_pad, 2 * slice_thickness + 1, axis=2)


def filter_blank_slices_thick(
    img_vol: npt.NDArray,
    label_vol: npt.NDArray,
    weight_vol: npt.NDArray,
    threshold: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter blank slices from the volume using the label volume.

    Parameters
    ----------
    img_vol : npt.NDArray
        Orig image volume.
    label_vol : npt.NDArray
        Label images (ground truth).
    weight_vol : npt.NDArray
        Weight corresponding to labels.
    threshold : int
        Threshold for number of pixels needed to keep slice (below = dropped). (Default value = 50).

    Returns
    -------
    filtered img_vol : np.ndarray
        Image volume with blank slices removed.
    label_vol : np.ndarray
        Label volume with blank slices removed.
    weight_vol : np.ndarray
        Weight volume with blank slices removed.
    """
    # Get indices of all slices with more than threshold labels/pixels
    select_slices = np.sum(label_vol, axis=(0, 1)) > threshold

    # Retain only slices with more than threshold labels/pixels
    img_vol = img_vol[:, :, select_slices, :]
    label_vol = label_vol[:, :, select_slices]
    weight_vol = weight_vol[:, :, select_slices]

    return img_vol, label_vol, weight_vol


##
# Size Management and Image Resizing
##


# Size calculation helpers
def calculate_resize_scale(
    input_h: int, input_w: int, target_size: int
) -> tuple[float, int, int]:
    """
    Calculate scale factor and new dimensions for proportional resize.

    Parameters
    ----------
    input_h, input_w : int
        Input image dimensions
    target_size : int
        Target size for the larger dimension

    Returns
    -------
    scale_factor : float
        Scale factor to apply
    new_h, new_w : int
        New dimensions after scaling (rounded)
    """
    max_dim = max(input_h, input_w)
    scale_factor = target_size / max_dim
    new_h = round(input_h * scale_factor)
    new_w = round(input_w * scale_factor)
    return scale_factor, new_h, new_w


def calculate_padding_amount(
    input_h: int, input_w: int, target_h: int, target_w: int
) -> tuple[int, int]:
    """
    Calculate padding amounts needed to reach target size.

    Parameters
    ----------
    input_h, input_w : int
        Current image dimensions
    target_h, target_w : int
        Target dimensions

    Returns
    -------
    pad_h, pad_w : int
        Padding amounts (can be negative if cropping needed)
    """
    pad_h = target_h - input_h
    pad_w = target_w - input_w
    return pad_h, pad_w


# Unified padding function (used everywhere)
def pad_to_size(
    image: npt.NDArray,
    output_size: int | tuple[int, int],
    mode: str = "edge",
    pos: str = "top_left",
) -> np.ndarray:
    """
    Pad image to target size with specified padding mode.

    Unified padding function used throughout the codebase. Supports both edge padding
    (replicates edge pixels) and zero padding (fills with zeros).

    Parameters
    ----------
    image : npt.NDArray
        Image to pad. Can be 2D (H, W) or 3D (H, W, C) or higher dimensions.
    output_size : int or tuple[int, int]
        Target size for height and width. If int, uses same size for both dimensions.
    mode : str, default='edge'
        Padding mode: 'edge' (replicates edge pixels) or 'zero' (fills with zeros).
    pos : str, default='top_left'
        Position to place the input image. Currently only 'top_left' is supported.

    Returns
    -------
    np.ndarray
        Padded image with shape (output_size[0], output_size[1], ...)
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    if mode not in ["edge", "zero"]:
        raise ValueError(f"mode must be 'edge' or 'zero', got '{mode}'")

    if len(image.shape) == 2:
        h, w = image.shape
        pad_h = output_size[0] - h
        pad_w = output_size[1] - w

        if pad_h < 0 or pad_w < 0:
            # Crop if image is larger than output_size
            if pad_h < 0:
                h = output_size[0]
            if pad_w < 0:
                w = output_size[1]
            image = image[:h, :w]
            pad_h = max(0, output_size[0] - h)
            pad_w = max(0, output_size[1] - w)

        if pad_h > 0 or pad_w > 0:
            if mode == "edge":
                padded_img = np.pad(
                    image,
                    ((0, pad_h), (0, pad_w)),
                    mode="edge",
                ).astype(image.dtype)
            else:  # mode == 'zero'
                padded_img = np.zeros(output_size, dtype=image.dtype)
                if pos == "top_left":
                    padded_img[0:h, 0:w] = image
        else:
            padded_img = image
    else:
        h, w = image.shape[:2]
        pad_h = output_size[0] - h
        pad_w = output_size[1] - w

        if pad_h < 0 or pad_w < 0:
            # Crop if image is larger than output_size
            if pad_h < 0:
                h = output_size[0]
            if pad_w < 0:
                w = output_size[1]
            # Handle different dimensionalities when cropping
            if len(image.shape) == 3:
                image = image[:h, :w, :]
            else:  # 4D or more
                slices = [slice(0, h), slice(0, w)] + [slice(None)] * (
                    len(image.shape) - 2
                )
                image = image[tuple(slices)]
            pad_h = max(0, output_size[0] - h)
            pad_w = max(0, output_size[1] - w)

        if pad_h > 0 or pad_w > 0:
            if mode == "edge":
                # Handle different dimensionalities
                if len(image.shape) == 3:
                    padded_img = np.pad(
                        image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge"
                    ).astype(image.dtype)
                else:  # 4D or more
                    pad_width = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (
                        len(image.shape) - 2
                    )
                    padded_img = np.pad(image, pad_width, mode="edge").astype(
                        image.dtype
                    )
            else:  # mode == 'zero'
                pad_shape = list(output_size) + list(image.shape[2:])
                padded_img = np.zeros(pad_shape, dtype=image.dtype)
                if pos == "top_left":
                    if len(image.shape) == 3:
                        padded_img[0:h, 0:w, :] = image
                    else:  # 4D or more
                        slices = [slice(0, h), slice(0, w)] + [slice(None)] * (
                            len(image.shape) - 2
                        )
                        padded_img[tuple(slices)] = image
        else:
            padded_img = image

    return padded_img


def pad_volume_edges_percent(
    volume: npt.NDArray, padding_percent: float, mode: str = "edge"
) -> tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    """
    Add symmetric edge padding to a 3D volume based on percentage.

    Pads each dimension by the specified percentage on both sides.
    For example, 0.05 (5%) padding on a 100x100x100 volume adds 5 voxels
    on each side, resulting in 110x110x110.

    Parameters
    ----------
    volume : npt.NDArray
        3D volume array with shape (H, W, D)
    padding_percent : float
        Padding percentage (0.0 to 1.0). Applied to each edge.
        Example: 0.05 = 5% padding on each edge (10% total per dimension)
    mode : str, default='edge'
        Padding mode: 'edge' (replicates edge voxels) or 'constant' (zeros)

    Returns
    -------
    np.ndarray
        Padded volume
    tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        Padding amounts for each dimension: ((pad_h_before, pad_h_after),
        (pad_w_before, pad_w_after), (pad_d_before, pad_d_after))
    """
    if padding_percent <= 0.0:
        return volume, ((0, 0), (0, 0), (0, 0))

    if len(volume.shape) != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

    h, w, d = volume.shape

    # Calculate padding amounts (symmetric on each side)
    pad_h = int(round(h * padding_percent))
    pad_w = int(round(w * padding_percent))
    pad_d = int(round(d * padding_percent))

    # Create padding specification for np.pad
    # Format: ((before_axis0, after_axis0), (before_axis1, after_axis1), ...)
    pad_width = ((pad_h, pad_h), (pad_w, pad_w), (pad_d, pad_d))

    # Apply padding
    if mode == "edge":
        padded_volume = np.pad(volume, pad_width, mode="edge").astype(volume.dtype)
    elif mode == "constant":
        padded_volume = np.pad(
            volume, pad_width, mode="constant", constant_values=0
        ).astype(volume.dtype)
    else:
        raise ValueError(f"mode must be 'edge' or 'constant', got '{mode}'")

    return padded_volume, pad_width


def pad_volume_to_cube(
    volume: npt.NDArray, mode: str = "constant"
) -> tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    """
    Pad a 3D volume to make it cubic (same size in all dimensions).

    Pads the volume symmetrically to match the maximum dimension.
    For example, a volume of shape (320, 144, 210) will be padded to (320, 320, 320).

    Contract for callers that adjust affines (e.g. conform with img_size='cube'):
    ``pad_widths[i][0]`` is always ``pad_total // 2`` (floor). For odd ``pad_total``,
    ``pad_widths[i][1] == pad_widths[i][0] + 1``. Use ``pad_widths[i][0]`` as the
    voxel-space offset of the original data block inside the padded volume — not
    ``pad_total / 2.0``, which differs by 0.5 voxels when ``pad_total`` is odd.

    Parameters
    ----------
    volume : npt.NDArray
        3D volume array with shape (H, W, D) or higher dimensions.
        If higher dimensions, only the first 3 dimensions are padded.
    mode : str, default='constant'
        Padding mode: 'constant' (zeros) or 'edge' (replicates edge voxels)

    Returns
    -------
    np.ndarray
        Padded volume with cubic shape (max_dim, max_dim, max_dim, ...)
    tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        Padding amounts for first 3 dimensions: ((pad_h_before, pad_h_after),
        (pad_w_before, pad_w_after), (pad_d_before, pad_d_after))
    """
    if len(volume.shape) < 3:
        raise ValueError(f"Expected at least 3D volume, got shape {volume.shape}")

    h, w, d = volume.shape[:3]
    max_dim = max(h, w, d)

    # Check if already cubic
    if h == w == d == max_dim:
        return volume, ((0, 0), (0, 0), (0, 0))

    # Calculate padding needed for each of the first 3 dimensions
    pad_widths = []
    for dim_size in [h, w, d]:
        pad_total = max_dim - dim_size
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        pad_widths.append((pad_before, pad_after))

    # Invariant: floor-half split (odd pad_total => one extra voxel after the block).
    for pa, pb in pad_widths[:3]:
        assert pb - pa in (0, 1), (
            "pad_volume_to_cube split invariant broken; affine callers assume "
            "p_before == pad_total // 2 and |p_after - p_before| <= 1"
        )

    # Add zero padding for any additional dimensions (4th, 5th, etc.)
    if len(volume.shape) > 3:
        pad_widths.extend([(0, 0)] * (len(volume.shape) - 3))

    # Apply padding
    if mode == "constant":
        padded_volume = np.pad(
            volume, pad_widths, mode="constant", constant_values=0
        ).astype(volume.dtype)
    elif mode == "edge":
        padded_volume = np.pad(volume, pad_widths, mode="edge").astype(volume.dtype)
    else:
        raise ValueError(f"mode must be 'constant' or 'edge', got '{mode}'")

    return padded_volume, tuple(pad_widths[:3])


def depad_volume(
    volume: npt.NDArray | torch.Tensor,
    pad_width: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> npt.NDArray | torch.Tensor:
    """
    Remove edge padding from a 3D or 4D volume.

    Works with both numpy arrays and torch tensors. For 4D volumes (e.g., predictions
    with shape H, W, D, num_classes), only the first 3 dimensions are depadded.

    Parameters
    ----------
    volume : npt.NDArray | torch.Tensor
        Padded volume. Can be:
        - 3D: (H, W, D) - numpy array
        - 4D: (H, W, D, num_classes) - numpy array or torch tensor
    pad_width : tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        Padding amounts from pad_volume_edges_percent:
        ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after), (pad_d_before, pad_d_after))

    Returns
    -------
    npt.NDArray | torch.Tensor
        Depadded volume with original dimensions restored
    """
    if all(pad == (0, 0) for pad in pad_width):
        return volume

    (
        (pad_h_before, pad_h_after),
        (pad_w_before, pad_w_after),
        (pad_d_before, pad_d_after),
    ) = pad_width

    # Calculate slice indices
    h_start = pad_h_before
    h_end = -pad_h_after if pad_h_after > 0 else None
    w_start = pad_w_before
    w_end = -pad_w_after if pad_w_after > 0 else None
    d_start = pad_d_before
    d_end = -pad_d_after if pad_d_after > 0 else None

    # Handle 3D and 4D cases
    if len(volume.shape) == 3:
        # 3D volume: (H, W, D)
        if isinstance(volume, torch.Tensor):
            return volume[h_start:h_end, w_start:w_end, d_start:d_end]
        else:
            return volume[h_start:h_end, w_start:w_end, d_start:d_end]
    elif len(volume.shape) == 4:
        # 4D volume: (H, W, D, num_classes)
        if isinstance(volume, torch.Tensor):
            return volume[h_start:h_end, w_start:w_end, d_start:d_end, :]
        else:
            return volume[h_start:h_end, w_start:w_end, d_start:d_end, :]
    else:
        raise ValueError(f"Expected 3D or 4D volume, got shape {volume.shape}")


# Unified image processing utilities (used in both training and inference)
def resize_to_target_size(
    image: npt.NDArray,
    target_size: int,
    order: int = 1,
) -> tuple[np.ndarray, float]:
    """
    Resize image proportionally to fit within target_size, then pad to exact dimensions.

    Works with both 2D slices (H, W) or (H, W, C) and 3D volumes (H, W, D) or (H, W, D, C).
    This is the unified resize function used in both training and inference.

    Parameters
    ----------
    image : npt.NDArray
        Image to resize. Can be:
        - 2D: (H, W) or (H, W, C)
        - 3D: (H, W, D) or (H, W, D, C)
    target_size : int
        Target size for both height and width (e.g., 256)
    order : int, default=1
        Interpolation order (0=nearest, 1=linear, 3=cubic)
        Use 0 for labels, 1 for images

    Returns
    -------
    np.ndarray
        Resized and padded image with shape (target_size, target_size, ...)
    float
        Scale factor used for resizing
    """
    h, w = image.shape[:2]
    scale_factor, new_h, new_w = calculate_resize_scale(h, w, target_size)

    if scale_factor != 1.0:
        # Calculate zoom factors for first 2 dimensions
        zoom_factors = (new_h / h, new_w / w)
        # Add 1.0 for remaining dimensions (depth, channels, etc.)
        if len(image.shape) > 2:
            zoom_factors = zoom_factors + (1.0,) * (len(image.shape) - 2)
        resized = zoom(image, zoom_factors, order=order)
    else:
        resized = image.copy()

    # Pad to exact target_size using edge padding
    padded = pad_to_size(resized, target_size, mode="edge", pos="top_left")

    return padded, scale_factor


def resize_from_target_size(
    image: npt.NDArray,
    target_size: int,
    output_h: int,
    output_w: int,
    order: int = 0,
) -> np.ndarray:
    """
    Reverse of resize_to_target_size: resize from target_size back to output dimensions.

    This is the exact inverse operation used in inference to reverse the resize applied
    in the transform pipeline. For small images that were upsampled, only the actual
    content region (not padding) is resized.

    Works with both 2D slices (H, W) or (H, W, C) and higher-dimensional arrays.
    This is the unified reverse resize function used in inference.

    Parameters
    ----------
    image : npt.NDArray
        Image at target_size. Can be:
        - 2D: (target_size, target_size) or (target_size, target_size, C)
        - Higher dims: (target_size, target_size, ...)
        For upsampled images, padding may be present (zeros in bottom/right)
    target_size : int
        Current size of image (e.g., 256)
    output_h, output_w : int
        Target output dimensions (conformed size, e.g., 366, 366)
    order : int, default=0
        Interpolation order (0=nearest for predictions, 1=linear for images)
        Use 0 for label predictions to avoid interpolation artifacts

    Returns
    -------
    np.ndarray
        Resized image with shape (output_h, output_w, ...)
    """
    # Calculate what the actual content dimensions should be (matching forward resize logic)
    # Forward resize: scale = target_size / max(h, w), new_h = round(h * scale), new_w = round(w * scale)
    # For reverse: calculate what new_h, new_w were, then only resize that region (not padding)
    max_output_dim = max(output_h, output_w)
    forward_scale = target_size / max_output_dim  # Same scale used in forward resize
    actual_content_h = round(
        output_h * forward_scale
    )  # What new_h was in forward resize
    actual_content_w = round(
        output_w * forward_scale
    )  # What new_w was in forward resize

    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        f"resize_from_target_size: target_size={target_size}, output=({output_h}, {output_w}), "
        f"max_output_dim={max_output_dim}, forward_scale={forward_scale:.6f}, "
        f"calculated content=({actual_content_h}, {actual_content_w})"
    )

    # IMPORTANT: For small images that were upsampled, the calculated actual_content_h/w
    # tells us the exact dimensions of the content region (before padding to target_size).
    # We must use these dimensions to crop before reverse resizing.

    # Determine content dimensions: use calculated values if they're less than target_size
    # This handles the case where padding was added during forward resize
    # IMPORTANT: We must use the EXACT inverse of forward resize to maintain mathematical correctness
    # Forward resize: zoom_factors = (new_h/h, new_w/w) where new_h = round(h * scale)
    # Reverse resize: zoom_factors = (h/new_h, w/new_w) = (output_h/content_h, output_w/content_w)
    # We cannot exclude rows/columns as that breaks the inverse relationship!

    if actual_content_h < target_size:
        # Padding was added - only use the actual content region
        content_h = actual_content_h
        logger.info(
            f"resize_from_target_size: Using calculated content_h={content_h} (padding detected, target_size={target_size})"
        )
    else:
        # No padding - entire image is content (use exact dimensions for perfect inverse)
        content_h = target_size

    if actual_content_w < target_size:
        # Padding was added - only use the actual content region
        content_w = actual_content_w
        logger.info(
            f"resize_from_target_size: Using calculated content_w={content_w} (padding detected, target_size={target_size})"
        )
    else:
        # No padding - entire image is content (use exact dimensions for perfect inverse)
        content_w = target_size

    # For upsampled images (target_size > max_output_dim), there may be padding
    # Only resize the actual content region (top-left corner), not the padding
    # This is critical for small images where padding was added
    if content_h < target_size or content_w < target_size:
        # Crop to actual content region before resizing (exclude padding)
        if len(image.shape) == 2:
            content_region = image[:content_h, :content_w]
        elif len(image.shape) == 3:
            content_region = image[:content_h, :content_w, :]
        else:  # 4D or more
            slices = [slice(None)] * len(image.shape)
            slices[0] = slice(0, content_h)
            slices[1] = slice(0, content_w)
            content_region = image[tuple(slices)]

        logger.info(
            f"resize_from_target_size: Cropping to content region ({content_h}, {content_w}) "
            f"from target_size ({target_size}, {target_size}) before reverse resize. "
            f"Output will be ({output_h}, {output_w})"
        )
    else:
        # No padding, entire image is content
        content_region = image
        content_h = target_size
        content_w = target_size

    # Calculate reverse scale factor to resize from content region back to output dimensions
    # Forward resize uses: zoom_factors = (new_h/h, new_w/w) where new_h = round(h * scale), new_w = round(w * scale)
    # Reverse should use: zoom_factors = (h/new_h, w/new_w) = (output_h/content_h, output_w/content_w)
    # This is the EXACT inverse of the forward resize zoom factors
    zoom_h = output_h / content_h if content_h > 0 else 1.0
    zoom_w = output_w / content_w if content_w > 0 else 1.0

    logger.info(
        f"resize_from_target_size: Resizing content region ({content_h}, {content_w}) "
        f"to output ({output_h}, {output_w}) using zoom_factors=({zoom_h:.6f}, {zoom_w:.6f})"
    )

    zoom_factors = (zoom_h, zoom_w)
    if len(content_region.shape) > 2:
        zoom_factors = zoom_factors + (1.0,) * (len(content_region.shape) - 2)

    # Resize using scipy.ndimage.zoom
    resized = zoom(content_region, zoom_factors, order=order)

    logger.info(
        f"resize_from_target_size: After zoom, resized shape={resized.shape[:2]}, "
        f"target output=({output_h}, {output_w}), zoom_factors were=({zoom_h:.6f}, {zoom_w:.6f})"
    )

    # Crop or pad to exact output dimensions (since proportional resize may not match exactly)
    if len(image.shape) == 2:
        # 2D: crop/pad to (output_h, output_w)
        if resized.shape[0] != output_h or resized.shape[1] != output_w:
            if resized.shape[0] > output_h or resized.shape[1] > output_w:
                # Crop if larger
                resized = resized[:output_h, :output_w]
            else:
                # Pad if smaller using edge padding
                resized = pad_to_size(
                    resized, (output_h, output_w), mode="edge", pos="top_left"
                )
    elif len(image.shape) == 3:
        # 3D: crop/pad to (output_h, output_w, C)
        if resized.shape[0] != output_h or resized.shape[1] != output_w:
            if resized.shape[0] > output_h or resized.shape[1] > output_w:
                resized = resized[:output_h, :output_w, :]
            else:
                # Pad if smaller using edge padding
                resized = pad_to_size(
                    resized, (output_h, output_w), mode="edge", pos="top_left"
                )
    else:  # 4D or more
        # Crop/pad first 2 dimensions
        if resized.shape[0] != output_h or resized.shape[1] != output_w:
            if resized.shape[0] > output_h or resized.shape[1] > output_w:
                slices = [slice(None)] * len(resized.shape)
                slices[0] = slice(0, output_h)
                slices[1] = slice(0, output_w)
                resized = resized[tuple(slices)]
            else:
                # Pad if smaller using edge padding
                resized = pad_to_size(
                    resized, (output_h, output_w), mode="edge", pos="top_left"
                )

    return resized


##
# Weight Map Generation
##


# weight map generator
def create_weight_mask(
    mapped_aseg: npt.NDArray,
    max_weight: int = 5,
    max_edge_weight: int = 5,
    max_hires_weight: int | None = None,
    mean_filter: bool = False,
    cortex_mask: bool = True,
    gradient: bool = True,
    cortex_labels: set | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Create weighted mask with multiple weighting strategies.

    Weight components:
    1. Base weights: Median frequency balancing (addresses class imbalance)
    2. Gradient weights: Edge detection (ALL label boundaries including subcortical)
    3. High-res weights: Challenging cortical regions (narrow sulci, thin gyri)
    4. Cortex border weights: Outer cortical boundary (pial surface)

    Note: Deep nuclei and subcortical structures are weighted through gradient (edge) weights.
          The cortex-specific masks only apply to cortical labels.

    Parameters
    ----------
    mapped_aseg : np.ndarray
        Segmentation to create weight mask from.
    max_weight : int
        Maximal weight on median frequency balancing (cap at this value). Default: 5.
    max_edge_weight : int
        Maximal weight on gradient (cap at this value). Applies to ALL boundaries. Default: 5.
    max_hires_weight : int
        Maximal weight on narrow cortical regions (cap at this value). Default: None.
    mean_filter : bool
        Flag to add mean_filter smoothing. Default: False.
    cortex_mask : bool
        Flag to create outer cortical boundary weight mask. Default: True.
    gradient : bool
        Flag to create gradient weight mask for all boundaries. Default: True.
    cortex_labels : set
        Explicit set of cortical label IDs from atlas metadata (e.g., ARM2, ARM3).
        Required for cortex-specific weight masks.

    Returns
    -------
    np.ndarray
        Weight mask for training.
    """
    unique, counts = np.unique(mapped_aseg, return_counts=True)

    # Median Frequency Balancing
    weights_per_label = np.median(counts) / counts
    weights_per_label[weights_per_label > max_weight] = max_weight

    # Create full weight array indexed by label value (not by position in unique array)
    # This is necessary because we'll index it using mapped_aseg values directly
    max_label = int(np.max(unique))
    class_wise_weights = np.zeros(max_label + 1, dtype=np.float32)
    class_wise_weights[unique.astype(int)] = weights_per_label

    (h, w, d) = mapped_aseg.shape
    weights_mask = np.reshape(
        class_wise_weights[mapped_aseg.ravel().astype(int)], (h, w, d)
    )

    # Gradient Weighting
    if gradient:
        (gx, gy, gz) = np.gradient(mapped_aseg)
        grad_weight = max_edge_weight * np.asarray(
            np.power(np.power(gx, 2) + np.power(gy, 2) + np.power(gz, 2), 0.5) > 0,
            dtype="float",
        )

        weights_mask += grad_weight

    if max_hires_weight is not None and cortex_labels is not None:
        # High-resolution weighting for challenging cortical structures
        if verbose:
            print(f"  Adding narrow cortical region mask (weight={max_hires_weight}):")
        mask1 = deep_sulci_and_wm_strand_mask(
            mapped_aseg,
            structure=np.ones((3, 3, 3)),
            cortex_labels=cortex_labels,
            verbose=verbose,
        )
        weights_mask += mask1 * max_hires_weight

        if cortex_mask:
            if verbose:
                print(
                    f"  Adding outer cortical boundary mask (weight={max_hires_weight // 2}):"
                )
            mask2 = cortex_border_mask(
                mapped_aseg,
                structure=np.ones((3, 3, 3)),
                cortex_labels=cortex_labels,
                verbose=verbose,
            )
            weights_mask += mask2 * (max_hires_weight) // 2

    if mean_filter:
        weights_mask = uniform_filter(weights_mask, size=3)

    return weights_mask


def cortex_border_mask(
    label: npt.NDArray,
    structure: npt.NDArray,
    cortex_labels: set,
    verbose: bool = False,
) -> np.ndarray:
    """
    Create a mask of the outer cortical boundary (pial surface region).

    This identifies the outermost layer of cortical gray matter, which is the
    boundary between cortex and CSF/meninges. This boundary is critical for:
    - Accurate cortical thickness measurements
    - Cortical surface reconstruction
    - Distinguishing cortex from CSF

    Note: This is about CORTICAL boundaries, not subcortical structures (deep nuclei).
    Deep nuclei boundaries (e.g., thalamus-WM) are handled by gradient weighting.

    Parameters
    ----------
    label : npt.NDArray
        Ground truth labels.
    structure : npt.NDArray
        Structuring element to erode with.
    cortex_labels : set
        Explicit set of cortical label IDs from atlas metadata (e.g., ARM2, ARM3).
    verbose : bool
        Print statistics about detected boundaries.

    Returns
    -------
    np.ndarray
        Binary mask of the outer cortical boundary (pial surface region).
    """
    # Create binary brainmask, erode it, and find the difference (outer layer)
    bm = np.clip(label, a_max=1, a_min=0)
    eroded = binary_erosion(bm, structure=structure)
    diff_im = np.logical_xor(eroded, bm)

    # Keep only the outer layer that belongs to cortical labels
    cortex_mask = np.isin(label, list(cortex_labels))
    diff_im[~cortex_mask] = 0

    # Print readable statistics
    num_detected = np.sum(diff_im)
    if verbose:
        total_voxels = diff_im.size
        percent = 100 * num_detected / total_voxels if total_voxels > 0 else 0
        print(
            f"  → Outer cortical boundary: {num_detected:,} voxels ({percent:.2f}% of volume)"
        )
        print("     (pial surface region where cortex meets CSF)")

    return diff_im


def deep_sulci_and_wm_strand_mask(
    volume: npt.NDArray,
    structure: npt.NDArray,
    cortex_labels: set,
    iteration: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    """
    Get a binary mask of narrow/thin cortical regions (deep sulci and intervening white matter).

    This function identifies challenging-to-segment cortical structures:
    - Deep cortical sulci (narrow folds in the cortex)
    - Thin gyral tips (narrow cortical protrusions)
    - Small white matter regions between closely packed cortical folds

    Note: This detects CORTICAL structures, not subcortical nuclei (deep nuclei).
    Deep nuclei boundaries are handled by gradient weighting.

    Parameters
    ----------
    volume : npt.NDArray
        Loaded image (aseg, label space).
    structure : npt.NDArray
        Structuring element (e.g. np.ones((3, 3, 3))).
    cortex_labels : set
        Explicit set of cortical label IDs from atlas metadata (e.g., ARM2, ARM3).
    iteration : int
        Number of times mask should be dilated + eroded. Defaults to 1.
    verbose : bool
        Print statistics about detected regions.

    Returns
    -------
    np.ndarray
        Binary mask of narrow/thin cortical structures.
    """
    # Create binary mask of cortical regions only (cortex = 1, everything else = 0)
    cortex_binary = np.zeros(shape=volume.shape)
    cortex_mask = np.isin(volume, list(cortex_labels))
    cortex_binary[cortex_mask] = 1

    # Apply morphological closing (erosion then dilation) to find thin cortical structures
    # This removes thin regions (deep sulci, narrow gyri) while preserving bulk cortex
    eroded = binary_closing(cortex_binary, iterations=iteration, structure=structure)

    # Get the difference: these are the narrow/thin cortical regions that were removed
    diff_image = np.logical_xor(cortex_binary, eroded)

    # Print readable statistics
    num_detected = np.sum(diff_image)
    if verbose:
        total_voxels = diff_image.size
        percent = 100 * num_detected / total_voxels if total_voxels > 0 else 0
        print(
            f"  → Narrow cortical regions: {num_detected:,} voxels ({percent:.2f}% of volume)"
        )
        print("     (deep sulci, thin gyri, and intervening WM)")

    return diff_image


##
# Label Mapping and LUT Processing
##


# Label mapping functions (to aparc (eval) and to label (train))
def read_classes_from_lut(lut_file: str | Path):
    """
    Read in a ColorLUT table with atlas label definitions.

    Supports both basic and extended ColorLUT formats:
    - Basic format: ID, LabelName, R, G, B, A
    - Extended format: ID, LabelName, Region, Hemi, R, G, B, A

    The extended format includes region type (cortex/subcortex) and hemisphere
    information, eliminating the need for a separate roiinfo file.

    Parameters
    ----------
    lut_file : Path, str
        The path and name of ColorLUT file with classes of interest.
        Example entry (basic):
        ID LabelName  R   G   B   A
        0   Unknown   0   0   0   0
        1   Left-Cerebral-Exterior 70  130 180 0

        Example entry (extended):
        ID LabelName  Region  Hemi  R   G   B   A
        2   cortex-rh-ACgG  cortex  rh  240 237 34  0
        ...

    Returns
    -------
    pandas.DataFrame
        DataFrame with label info. Always includes: ID, LabelName, R/Red, G/Green,
        B/Blue, A/Alpha. May also include: Region, Hemi (if extended format).
    """
    if not isinstance(lut_file, Path):
        lut_file = Path(lut_file)
    if lut_file.suffix == ".tsv":
        return pd.read_csv(lut_file, sep="\t")

    # Read in file
    names = {
        "ID": "int",
        "LabelName": "str",
        "Red": "int",
        "Green": "int",
        "Blue": "int",
        "Alpha": "int",
    }
    kwargs = {}
    if lut_file.suffix == ".csv":
        kwargs["sep"] = ","
    elif lut_file.suffix == ".txt":
        kwargs["sep"] = "\\s+"
    else:
        raise RuntimeError(
            f"Unknown LUT file extension {lut_file}, must be csv, txt or tsv."
        )

    # Check if the file has a header row by reading the first non-comment line
    skiprows = []
    with open(lut_file, "r") as f:
        for row_num, line in enumerate(f):
            line = line.strip()
            # Skip comments and blank lines
            if not line or line.startswith("#"):
                continue
            # Check if first column can be converted to integer
            sep = kwargs.get("sep", ",")
            if sep == "\\s+":
                first_col = line.split()[0]
            else:
                first_col = line.split(sep)[0]
            try:
                int(first_col)
                # First column is an integer, so this is data, not a header
                break
            except ValueError:
                # First column is not an integer, so it's likely a header - skip this row
                skiprows.append(row_num)
                break

    return pd.read_csv(
        lut_file,
        index_col=False,
        skip_blank_lines=True,
        comment="#",
        header=None,
        skiprows=skiprows if skiprows else None,
        names=list(names.keys()),
        dtype=names,
        **kwargs,
    )


def lut_has_extended_format(lut_df: pd.DataFrame) -> bool:
    """
    Check if a LUT DataFrame has the extended format with Region and Hemi columns.

    Parameters
    ----------
    lut_df : pandas.DataFrame
        DataFrame returned by read_classes_from_lut()

    Returns
    -------
    bool
        True if the LUT has Region and Hemi columns (extended format)
    """
    return "Region" in lut_df.columns and "Hemi" in lut_df.columns


def get_region_info_from_lut(
    lut_df: pd.DataFrame,
) -> tuple[dict[int, str], dict[int, str]]:
    """
    Extract region and hemisphere information from an extended LUT.

    Parameters
    ----------
    lut_df : pandas.DataFrame
        DataFrame returned by read_classes_from_lut() with extended format

    Returns
    -------
    tuple[dict[int, str], dict[int, str]]
        Two dictionaries: (label_id -> region_type, label_id -> hemisphere)
        Returns empty dicts if LUT doesn't have extended format.

    Examples
    --------
    >>> lut = read_classes_from_lut("ARM2_ColorLUT.tsv")
    >>> region_map, hemi_map = get_region_info_from_lut(lut)
    >>> region_map[2]  # 'cortex'
    >>> hemi_map[2]    # 'rh'
    """
    if not lut_has_extended_format(lut_df):
        return {}, {}

    region_map = dict(zip(lut_df["ID"], lut_df["Region"]))
    hemi_map = dict(zip(lut_df["ID"], lut_df["Hemi"]))
    return region_map, hemi_map


def map_label2aparc_aseg(
    mapped_aseg: torch.Tensor, labels: torch.Tensor | npt.NDArray
) -> torch.Tensor:
    """
    Perform look-up table mapping from sequential label space to LUT space.

    Parameters
    ----------
    mapped_aseg : torch.Tensor
        Label space segmentation (aparc.DKTatlas + aseg).
    labels : Union[torch.Tensor, npt.NDArray]
        List of labels defining LUT space.

    Returns
    -------
    torch.Tensor
        Labels in LUT space.
    """
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    labels = labels.to(mapped_aseg.device)
    return labels[mapped_aseg]


# FreeSurfer-specific functions removed
# These were hardcoded for FreeSurfer's specific label scheme and are not needed
# for generalized atlas support (ARM2, ARM3, etc.)


def unify_lateralized_labels(
    lut: str | pd.DataFrame, combi: tuple[str, str] = ("Left-", "Right-")
) -> Mapping:
    """
    Generate lookup dictionary of left-right labels.

    Parameters
    ----------
    lut : Union[str, pd.DataFrame]
        Either lut-file string to load or pandas dataframe
        Example entry:
        ID LabelName  R   G   B   A
        0   Unknown   0   0   0   0
        1   Left-Cerebral-Exterior 70  130 180 0.
    combi : Tuple[str, str]
        Prefix or labelnames to combine. Default: Left- and Right-.

    Returns
    -------
    Mapping
        Dictionary mapping between left and right hemispheres.
    """
    if isinstance(lut, str):
        lut = read_classes_from_lut(lut)
    left = lut[["ID", "LabelName"]][lut["LabelName"].str.startswith(combi[0])]
    right = lut[["ID", "LabelName"]][lut["LabelName"].str.startswith(combi[1])]
    left["LabelName"] = left["LabelName"].str.removeprefix(combi[0])
    right["LabelName"] = right["LabelName"].str.removeprefix(combi[1])
    mapp = left.merge(right, on="LabelName")
    return pd.Series(mapp.ID_y.values, index=mapp.ID_x).to_dict()


def get_labels_from_lut(
    lut: str | pd.DataFrame, label_extract: tuple[str, str] = ("Left-", "ctx-rh")
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract labels from the lookup tables.

    Parameters
    ----------
    lut : Union[str, pd.DataFrame]
        ColorLUT table with atlas label definitions (either path to it
        or already loaded as pandas DataFrame).
        Example entry:
        ID LabelName  R   G   B   A
        0   Unknown   0   0   0   0
        1   Left-Cerebral-Exterior 70  130 180 0.
    label_extract : Tuple[str, str]
        Suffix of label names to mask for sagittal labels
        Default: "Left-" and "ctx-rh".

    Returns
    -------
    np.ndarray
        Full label list.
    np.ndarray
        Sagittal label list.
    """
    if isinstance(lut, str):
        lut = read_classes_from_lut(lut)
    mask = lut["LabelName"].str.startswith(label_extract)
    return lut["ID"].values, lut["ID"][~mask].values


# FreeSurfer-specific label mapping functions removed
# These contained hardcoded FreeSurfer label mappings (aparc, aseg processing)
# For atlas-agnostic support, use atlas.atlas_manager methods instead


def map_prediction_sagittal2full(
    prediction_sag: npt.NDArray,
    num_classes: int,
    atlas_name: str | None = None,
    idx_list: list[int] | None = None,
) -> np.ndarray:
    """
    Remap the prediction on the sagittal network to full label space used by coronal and axial networks.

    For binary tasks (num_classes==2), no remapping needed - all planes use same label space.
    For multi-class, expands merged hemisphere labels to bilateral space.

    Parameters
    ----------
    prediction_sag : npt.NDArray
        Sagittal prediction (labels).
    num_classes : int
        Number of SAGITTAL classes.
    atlas_name : Optional[str]
        Name of the atlas, used to rebuild the mapping when idx_list is None.
    idx_list : Optional[list[int]]
        For each bilateral dense index, the sagittal dense index to copy (checkpoint key
        ``bilateral_dense_to_sagittal_dense``).

    Returns
    -------
    np.ndarray
        Remapped prediction.
    """
    # Binary brain mask mode - no hemisphere expansion needed
    if num_classes == 2:
        return (
            prediction_sag  # Already correct - brain is brain regardless of hemisphere
        )

    # Multi-class: expand hemisphere-merged sagittal dense classes to bilateral dense
    # classes. Prefer the mapping stored in the checkpoint (idx_list); otherwise rebuild it
    # from the atlas roiinfo.
    if idx_list is None:
        if atlas_name is None:
            atlas_name = os.environ.get("ATLAS_NAME")
        if not atlas_name:
            raise ValueError("map_prediction_sagittal2full needs atlas_name or idx_list")
        idx_list = get_atlas_manager(atlas_name).get_sagittal_to_bilateral_expansion()
    return prediction_sag[:, idx_list, :, :]


##
# Utility Functions
##


# Clean up and class separation
def bbox_3d(
    img: npt.NDArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the three-dimensional bounding box coordinates.

    Parameters
    ----------
    img : npt.NDArray
        Mri image.

    Returns
    -------
    np.ndarray
        Rmin.
    np.ndarray
        Rmax.
    np.ndarray
        Cmin.
    np.ndarray
        Cmax.
    np.ndarray
        Zmin.
    np.ndarray
        Zmax.
    """
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax


def get_largest_cc(segmentation: npt.NDArray) -> np.ndarray:
    """
    Find the largest connected component of segmentation.

    Parameters
    ----------
    segmentation : npt.NDArray
        Segmentation.

    Returns
    -------
    np.ndarray
        Largest connected component of segmentation (binary mask).
    """
    labels = label(segmentation, connectivity=3, background=0)

    bincount = np.bincount(labels.flat)
    background = np.argmax(bincount)
    bincount[background] = -1

    largest_cc = labels == np.argmax(bincount)

    return largest_cc
