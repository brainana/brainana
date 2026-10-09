"""
Morphological operations for medical image processing.

This module provides functions for post-processing binary labels,
including connected component analysis, hole filling, and 
morphological operations.
"""

import numpy as np
import scipy.ndimage as snd
from typing import Optional


def extract_largest_component(label: np.ndarray) -> np.ndarray:
    """
    Extract the largest connected component from a binary label.

    Args:
        label: Binary label array

    Returns:
        Binary label containing only the largest connected component

    Raises:
        ValueError: If label is not binary
    """
    if not np.all(np.isin(label, [0, 1])):
        raise ValueError("Input label must be binary (containing only 0s and 1s)")

    if not np.any(label):
        return label  # Return empty label if no positive voxels

    # Label connected components
    labels, num_labels = snd.label(label)

    if num_labels == 0:
        return label

    # Find largest component (excluding background label 0)
    component_sizes = np.bincount(labels.reshape(-1))
    component_sizes[0] = 0  # Ignore background

    if len(component_sizes) <= 1:
        return label

    largest_component = component_sizes.argmax()
    return (labels == largest_component).astype(label.dtype)


# A brain-sized blob the network half-believes in can sit in the background
# noise beside the head (PRIME-DE carmenlyon sub-12: 578k voxels at mean
# probability 0.61 against a 552k-voxel brain at 0.95). Its voxels never pass
# this probability; the brain's mostly do (88% there).
CONFIDENT_CORE_THRESHOLD = 0.9

# A runner-up at least this fraction of the kept component's size is a near-tie
# worth a warning.
NEAR_TIE_RATIO = 0.5


def select_confident_component(
    label: np.ndarray,
    prob: np.ndarray,
    core_threshold: float = CONFIDENT_CORE_THRESHOLD,
) -> tuple:
    """Keep the connected component of ``label`` holding the most confident voxels.

    The component with the most voxels above ``core_threshold`` in ``prob`` wins,
    not the largest one, so a faint background blob cannot outvote the brain by
    size alone. Without any voxel above the threshold this falls back to the
    largest component. The kept component's voxels are unchanged: only which
    component is kept differs.

    Args:
        label: Binary label array (``prob`` thresholded)
        prob: Foreground probability, same shape as ``label``
        core_threshold: Probability a voxel needs to count as confident

    Returns:
        ``(mask, info)``: the kept component with ``label``'s dtype, and a dict
        with ``components``, ``kept_voxels``, ``largest_voxels``,
        ``runner_up_voxels`` and ``by`` ("core", "size" or "empty").
    """
    if label.shape != prob.shape:
        raise ValueError(f"label {label.shape} and prob {prob.shape} differ in shape")
    if not np.all(np.isin(label, [0, 1])):
        raise ValueError("Input label must be binary (containing only 0s and 1s)")

    labels, num_labels = snd.label(label)
    if num_labels == 0:
        return label, {"components": 0, "by": "empty"}

    sizes = np.bincount(labels.reshape(-1), minlength=num_labels + 1)
    sizes[0] = 0
    core = np.bincount(
        labels[prob > core_threshold], minlength=num_labels + 1
    )
    core[0] = 0
    if core.any():
        kept, by = int(core.argmax()), "core"
    else:
        kept, by = int(sizes.argmax()), "size"
    others = np.delete(sizes, [0, kept])
    info = {
        "components": int(num_labels),
        "kept_voxels": int(sizes[kept]),
        "largest_voxels": int(sizes.max()),
        "runner_up_voxels": int(others.max()) if others.size else 0,
        "by": by,
    }
    return (labels == kept).astype(label.dtype), info


def fill_label_holes(label: np.ndarray) -> np.ndarray:
    """
    Fill holes in a binary label by extracting the largest background
    component and inverting.

    This approach identifies the main background region and considers
    everything else as foreground, effectively filling holes.

    Args:
        label: Binary label array

    Returns:
        Binary label with holes filled

    Raises:
        ValueError: If label is not binary
    """
    if not np.all(np.isin(label, [0, 1])):
        raise ValueError("Input label must be binary (containing only 0s and 1s)")

    # Find largest background component
    background = label == 0
    largest_background = extract_largest_component(background)

    # Everything not in largest background is considered foreground
    filled_label = (largest_background == 0).astype(label.dtype)
    return filled_label


def morphological_erosion_dilation(
    label: np.ndarray, structure: Optional[np.ndarray] = None, iterations: int = 1
) -> np.ndarray:
    """
    Apply erosion followed by dilation to a binary label.

    This operation (opening) helps remove noise and smooth boundaries
    while preserving the main structure size.

    Args:
        label: Binary label array
        structure: Structuring element for morphological operations.
                  If None, uses 3D cross-shaped structure.
        iterations: Number of iterations for each operation

    Returns:
        Processed binary label

    Raises:
        ValueError: If label is not binary or iterations < 0
    """
    if not np.all(np.isin(label, [0, 1])):
        raise ValueError("Input label must be binary (containing only 0s and 1s)")

    if iterations < 0:
        raise ValueError("Iterations must be non-negative")

    if iterations == 0:
        return label

    # Default to 3D cross-shaped structuring element
    if structure is None:
        structure = snd.generate_binary_structure(label.ndim, 1)

    # Erosion
    eroded = snd.binary_erosion(
        label, structure=structure, iterations=iterations
    ).astype(label.dtype)

    # Extract largest component after erosion to remove small fragments
    eroded = extract_largest_component(eroded)

    # Dilation to restore size
    dilated = snd.binary_dilation(
        eroded, structure=structure, iterations=iterations
    ).astype(label.dtype)

    return dilated


def get_bounding_box(label: np.ndarray, margin: int = 0) -> tuple:
    """
    Get the bounding box of a binary label with optional margin.

    Args:
        label: Binary label array
        margin: Additional margin to add around the bounding box

    Returns:
        Tuple of slices defining the bounding box
    """
    if not np.any(label):
        return tuple(slice(0, s) for s in label.shape)

    # Find coordinates of non-zero elements
    coords = np.where(label)

    # Get min and max coordinates for each dimension
    min_coords = [np.min(coord) for coord in coords]
    max_coords = [np.max(coord) for coord in coords]

    # Add margin
    min_coords = [max(0, c - margin) for c in min_coords]
    max_coords = [min(s, c + margin + 1) for s, c in zip(label.shape, max_coords)]

    # Create slices
    slices = tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))
    return slices


def crop_to_label(label: np.ndarray, margin: int = 0) -> tuple[np.ndarray, tuple]:
    """
    Crop an array to the bounding box of a binary label.

    Args:
        label: Binary label array
        margin: Additional margin to add around the bounding box

    Returns:
        Tuple of (cropped_label, slices_used)
    """
    slices = get_bounding_box(label, margin)
    cropped = label[slices]
    return cropped, slices
