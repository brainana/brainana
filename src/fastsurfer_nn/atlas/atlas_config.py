#!/usr/bin/env python3
"""
Atlas configuration: region classification and label mapping.

Terminology used throughout fastsurfer_nn (keep it consistent):

* **sparse** labels = atlas label IDs exactly as stored in label NIfTI files and in the
  ColorLUT, e.g. 0 (background), 2, 16, 1002, 502, -1, -1501. Non-contiguous, may be negative.
* **dense** labels = contiguous network class indices 0..N-1 with 0 = background. This is
  what the CNN predicts and what the HDF5 ``aseg_dataset`` stores.
  Axial/coronal (bilateral): N = number of atlas labels + background (ARM2: 71).
  Sagittal (left/right merged by ``key_nohemi``): ARM2: 36.

(Not the TensorFlow sense of "sparse = integer class vs dense = one-hot".)

Conversions: sparse -> dense ``map_labels_to_dense`` / ``map_labels_to_sagittal_dense``;
dense -> sparse ``map_labels_to_sparse`` with ``create_dense_to_sparse_mapping``. The plane-
aware wrappers live on ``AtlasManager`` (``to_dense``, ``dense_to_sparse``, ``dense_ids_of``).
Region sets on ``AtlasConfig`` (``cortex_labels`` etc.) hold SPARSE IDs.

Supports roiinfo-based atlases (ARM levels) and FreeSurfer (threshold-based).
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import random


# ============================================================================
# ATLAS CONFIGURATION - Region Classification
# ============================================================================


class AtlasConfig:
    """
    Atlas configuration class that handles region classification.

    Attributes
    ----------
    name : str
        Atlas name (e.g., "ARM3", "FreeSurfer")
    ctx_thresh : int
        Cortex threshold for threshold-based classification
    cortex_labels : Set[int]
        Set of label IDs that are cortical
    subcortex_labels : Set[int]
        Set of label IDs that are subcortical
    cerebral_wm_labels : Set[int]
        Set of label IDs that are cerebral white matter
    cerebellar_wm_labels : Set[int]
        Set of label IDs that are cerebellar white matter
    use_threshold : bool
        If True, use threshold-based classification (FreeSurfer)
    """

    def __init__(
        self,
        name: str,
        ctx_thresh: Optional[int] = None,
        cortex_labels: Optional[Set[int]] = None,
        subcortex_labels: Optional[Set[int]] = None,
        cerebral_wm_labels: Optional[Set[int]] = None,
        cerebellar_wm_labels: Optional[Set[int]] = None,
        use_threshold: bool = False,
    ):
        """
        Initialize atlas configuration.

        Parameters
        ----------
        name : str
            Atlas name
        ctx_thresh : int, optional
            Threshold for cortex classification (labels > thresh = cortex)
        cortex_labels : Set[int], optional
            Explicit set of cortical label IDs
        subcortex_labels : Set[int], optional
            Explicit set of subcortical label IDs
        cerebral_wm_labels : Set[int], optional
            Explicit set of cerebral white matter label IDs
        cerebellar_wm_labels : Set[int], optional
            Explicit set of cerebellar white matter label IDs
        use_threshold : bool
            If True, use threshold-based (FreeSurfer), else metadata-based (ARM3)
        """
        self.name = name
        self.ctx_thresh = ctx_thresh
        self.cortex_labels = cortex_labels or set()
        self.subcortex_labels = subcortex_labels or set()
        self.cerebral_wm_labels = cerebral_wm_labels or set()
        self.cerebellar_wm_labels = cerebellar_wm_labels or set()
        self.use_threshold = use_threshold

    def is_cortex(self, label: int) -> bool:
        """Check if a label is cortical."""
        if self.use_threshold and self.ctx_thresh is not None:
            return label > self.ctx_thresh
        return label in self.cortex_labels

    def is_subcortex(self, label: int) -> bool:
        """Check if a label is subcortical."""
        if self.use_threshold and self.ctx_thresh is not None:
            return 0 < label <= self.ctx_thresh
        return label in self.subcortex_labels

    def is_cerebral_wm(self, label: int) -> bool:
        """Check if a label is cerebral white matter."""
        return label in self.cerebral_wm_labels

    def is_cerebellar_wm(self, label: int) -> bool:
        """Check if a label is cerebellar white matter."""
        return label in self.cerebellar_wm_labels

    def get_region_type(self, label: int) -> str:
        """
        Get region type for a label.

        Returns: 'cortex', 'subcortex', 'cerebral_WM', 'cerebellar_WM', 'background', or 'unknown'
        """
        if label == 0:
            return "background"
        elif self.is_cortex(label):
            return "cortex"
        elif self.is_subcortex(label):
            return "subcortex"
        elif self.is_cerebral_wm(label):
            return "cerebral_WM"
        elif self.is_cerebellar_wm(label):
            return "cerebellar_WM"
        else:
            return "unknown"

    def get_ctx_thresh_for_plane(self, plane: str = "coronal") -> Optional[int]:
        """
        Get appropriate ctx_thresh for given plane.

        Returns None for metadata-based atlases (ARM3).
        """
        if self.use_threshold and self.ctx_thresh is not None:
            if plane == "sagittal":
                return 19
            else:
                return 33
        return None


def load_atlas_config_from_roiinfo(roiinfo_path: Path) -> AtlasConfig:
    """
    Load atlas configuration from a roiinfo.txt file.

    Uses metadata (region column), NOT thresholds.
    Labels can be in any order (positive, negative, non-sequential).

    Expected format (tab-separated):
    key_nohemi  key  region        name          name_full                    hemi
    3           3    cortex        ACC           anterior_cingulate_cortex    rh
    503         503  subcortex     LPal          lateral_pallium              rh
    -1          -1   cerebral_WM   cerebral_WM   cerebral_white_matter        rh
    -501        -501 cerebellar_WM cerebellar_WM cerebellar_white_matter      rh
    """
    cortex_labels = set()
    subcortex_labels = set()
    cerebral_wm_labels = set()
    cerebellar_wm_labels = set()

    with open(roiinfo_path, "r") as f:
        lines = f.readlines()

    # Skip header
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue

        try:
            label = int(parts[1])  # key column
            region = parts[2].lower()  # region column

            if region == "cortex":
                cortex_labels.add(label)
            elif region == "subcortex":
                subcortex_labels.add(label)
            elif region == "wm":
                # Differentiate WM types by name column (ctxWM vs cbWM)
                if len(parts) >= 4:
                    name = parts[3]  # name column
                    if "ctx" in name.lower() or "cerebral" in name.lower():
                        cerebral_wm_labels.add(label)
                    elif "cb" in name.lower() or "cerebell" in name.lower():
                        cerebellar_wm_labels.add(label)
                    else:
                        # Default to cerebral WM if unclear
                        cerebral_wm_labels.add(label)
        except (ValueError, IndexError):
            continue

    # Background (0) is not a region: it is in none of these sets.

    atlas_name = roiinfo_path.stem

    return AtlasConfig(
        name=atlas_name,
        ctx_thresh=None,
        cortex_labels=cortex_labels,
        subcortex_labels=subcortex_labels,
        cerebral_wm_labels=cerebral_wm_labels,
        cerebellar_wm_labels=cerebellar_wm_labels,
        use_threshold=False,
    )


def create_freesurfer_config(plane: str = "coronal") -> AtlasConfig:
    """
    Create FreeSurfer atlas configuration using threshold approach.

    FreeSurfer: labels <= ctx_thresh = subcortex, labels > ctx_thresh = cortex
    """
    ctx_thresh = 19 if plane == "sagittal" else 33

    return AtlasConfig(name="FreeSurfer", ctx_thresh=ctx_thresh, use_threshold=True)


# ============================================================================
# SAGITTAL HEMISPHERE MAPPING - For Multi-View Training
# ============================================================================


def load_sagittal_hemisphere_mapping(
    roiinfo_path: Optional[Path] = None,
) -> Tuple[Dict[int, int], List[int], int]:
    """
    Load hemisphere mapping for sagittal plane training from roiinfo.txt.

    Maps bilateral structures (left/right) to single hemisphere-merged IDs.

    Parameters
    ----------
    roiinfo_path : Path, optional
        Path to roiinfo.txt. Required.

    Returns
    -------
    Tuple[Dict[int, int], List[int], int]
        - bilateral_to_merged: Dict mapping full keys (3, 1003) -> key_nohemi (3)
        - sagittal_labels: List of unique key_nohemi values (sorted)
        - num_sagittal_classes: Total classes including background

    Examples
    --------
    >>> mapping, labels, num_classes = load_sagittal_hemisphere_mapping()
    >>> mapping[3]     # 3 (right ACC)
    >>> mapping[1003]  # 3 (left ACC, merged to same)
    >>> num_classes    # 75 (74 unique key_nohemi + background)
    """
    if roiinfo_path is None:
        raise ValueError("roiinfo_path is required (e.g. AtlasManager('ARM2').roiinfo_path)")

    bilateral_to_merged = {}
    key_nohemi_set = set()

    with open(roiinfo_path, "r") as f:
        lines = f.readlines()

    # Skip header
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue

        try:
            key_nohemi = int(parts[0])  # Column 0: hemisphere-merged ID
            key = int(parts[1])  # Column 1: full bilateral ID

            # Map both hemispheres to the same key_nohemi
            bilateral_to_merged[key] = key_nohemi
            key_nohemi_set.add(key_nohemi)

        except (ValueError, IndexError):
            continue

    # Background (0) is NOT in roiinfo.txt, but we need it
    bilateral_to_merged[0] = 0
    key_nohemi_set.add(0)

    # Create sorted list of sagittal labels (for dense mapping)
    # Sort: positive first, then negative (by absolute value)
    sagittal_labels = sorted(
        [k for k in key_nohemi_set if k != 0], key=lambda x: (x < 0, abs(x))
    )

    num_sagittal_classes = len(sagittal_labels) + 1  # +1 for background

    return bilateral_to_merged, sagittal_labels, num_sagittal_classes


def create_sagittal_sparse_to_dense_mapping(
    roiinfo_path: Optional[Path] = None,
) -> Dict[int, int]:
    """
    Create sparse-to-dense mapping for sagittal training.

    Maps bilateral labels -> dense indices for sagittal model.
    Both hemispheres of same structure map to same dense index.

    Parameters
    ----------
    roiinfo_path : Path, optional
        Path to roiinfo.txt. Required.

    Returns
    -------
    Dict[int, int]
        Mapping from bilateral labels -> dense indices

    Examples
    --------
    >>> mapping = create_sagittal_sparse_to_dense_mapping()
    >>> mapping[0]     # 0 (background)
    >>> mapping[3]     # 1 (ACC)
    >>> mapping[1003]  # 1 (ACC, same as right hemisphere)
    >>> mapping[11]    # 2 (MCC)
    >>> mapping[1011]  # 2 (MCC, same as right hemisphere)
    """
    bilateral_to_merged, sagittal_labels, _ = load_sagittal_hemisphere_mapping(
        roiinfo_path
    )

    # Create dense mapping: key_nohemi -> dense_index
    nohemi_to_dense = {0: 0}  # Background
    for dense_idx, key_nohemi in enumerate(sagittal_labels, start=1):
        nohemi_to_dense[key_nohemi] = dense_idx

    # Create final mapping: bilateral_key -> dense_index
    # This goes through: bilateral_key -> key_nohemi -> dense_index
    bilateral_to_dense = {}
    for bilateral_key, key_nohemi in bilateral_to_merged.items():
        bilateral_to_dense[bilateral_key] = nohemi_to_dense[key_nohemi]

    return bilateral_to_dense


def map_labels_to_sagittal_dense(
    label_array: np.ndarray, roiinfo_path: Optional[Path] = None
) -> np.ndarray:
    """
    Map bilateral labels to sagittal hemisphere-merged dense indices.

    This is used during sagittal training data generation.

    Parameters
    ----------
    label_array : np.ndarray
        Array with bilateral label IDs (e.g., 3, 1003, 11, 1011, ...)
    roiinfo_path : Path, optional
        Path to roiinfo.txt. Required.

    Returns
    -------
    np.ndarray
        Array with hemisphere-merged dense indices (0 to num_sagittal_classes-1)

    Examples
    --------
    >>> # Input has both hemispheres
    >>> bilateral_labels = np.array([0, 3, 1003, 11, 1011])
    >>> sagittal_dense = map_labels_to_sagittal_dense(bilateral_labels)
    >>> # Output: [0, 1, 1, 2, 2] - left and right map to same index
    """
    bilateral_to_dense = create_sagittal_sparse_to_dense_mapping(roiinfo_path)

    dense_array = np.zeros_like(label_array, dtype=np.int32)

    unique_labels = np.unique(label_array)
    for bilateral_label in unique_labels:
        bilateral_label_int = int(bilateral_label)
        if bilateral_label_int in bilateral_to_dense:
            mask = label_array == bilateral_label
            dense_array[mask] = bilateral_to_dense[bilateral_label_int]
        # Unknown labels stay as 0 (background)

    return dense_array


# ============================================================================
# LABEL MAPPING - Sparse ↔ Dense Conversion
# ============================================================================

def create_sparse_to_dense_mapping(sparse_labels: List[int]) -> Dict[int, int]:
    """
    Create mapping from sparse labels to dense indices.

    Maps background (0) to dense index 0, and all other labels to dense indices 1+.
    This ensures that dense index 0 always represents background, which is the
    standard convention in segmentation tasks.

    Parameters
    ----------
    sparse_labels : List[int]
        List of sparse label IDs (including 0 if present).

    Returns
    -------
    Dict[int, int]
        Dictionary mapping sparse label ID -> dense index
        - 0 -> 0 (background)
        - Other labels -> 1, 2, 3, ... (regions)

    Examples
    --------
    >>> # sparse_labels = [0, 3, 11, 17, ..., -1501] (including 0)
    >>> mapping = create_sparse_to_dense_mapping([0, 3, 11, 17])
    >>> mapping[0]     # 0 (background)
    >>> mapping[3]     # 1 (first region)
    >>> mapping[11]    # 2 (second region)
    >>> mapping[17]    # 3 (third region)
    """
    if not sparse_labels:
        raise ValueError("sparse_labels cannot be empty")

    mapping_dict = {}

    # First, ensure background (0) maps to dense index 0
    mapping_dict[0] = 0

    # Then map all other labels to dense indices 1+
    dense_idx = 1
    for sparse_label in sparse_labels:
        if sparse_label != 0:  # Skip background, already mapped
            mapping_dict[int(sparse_label)] = dense_idx
            dense_idx += 1

    return mapping_dict


def create_dense_to_sparse_mapping(sparse_labels: List[int]) -> np.ndarray:
    """
    Create mapping from dense indices to sparse labels.

    Returns array where index i gives the sparse label for dense index i.
    Dense index 0 maps to background (0), dense indices 1+ map to regions.

    Parameters
    ----------
    sparse_labels : List[int]
        List of sparse label IDs (including 0 if present).

    Returns
    -------
    np.ndarray
        Array where:
        - Index 0 = 0 (background)
        - Index 1+ = region labels (non-zero sparse labels)

    Examples
    --------
    >>> # sparse_labels = [0, 3, 11, 17, ..., -1501] (including 0)
    >>> mapping = create_dense_to_sparse_mapping([0, 3, 11, 17])
    >>> mapping[0]     # 0 (background)
    >>> mapping[1]     # 3 (first region)
    >>> mapping[2]     # 11 (second region)
    >>> mapping[3]     # 17 (third region)
    """
    if not sparse_labels:
        raise ValueError("sparse_labels cannot be empty")

    # Create array with background at index 0, then regions
    dense_to_sparse = [0]  # Background at index 0

    # Add all non-zero labels as regions
    for sparse_label in sparse_labels:
        if sparse_label != 0:  # Skip background, already added
            dense_to_sparse.append(sparse_label)

    return np.array(dense_to_sparse, dtype=np.int32)


def map_labels_to_dense(
    label_array: np.ndarray, sparse_to_dense_map: Dict[int, int]
) -> np.ndarray:
    """
    Map sparse labels to dense indices.

    Works with any label values (positive, negative, zero).
    Unknown labels are mapped to 0 (background).

    Parameters
    ----------
    label_array : np.ndarray
        Array with sparse label IDs
    sparse_to_dense_map : Dict[int, int]
        Mapping dictionary from sparse labels to dense indices.

    Returns
    -------
    np.ndarray
        Array with dense indices (0 to num_classes-1)
    """
    if not sparse_to_dense_map:
        raise ValueError("sparse_to_dense_map cannot be empty")

    dense_array = np.zeros_like(label_array, dtype=np.int32)

    unique_labels = np.unique(label_array)
    for sparse_label in unique_labels:
        sparse_label_int = int(sparse_label)
        if sparse_label_int in sparse_to_dense_map:
            mask = label_array == sparse_label
            dense_array[mask] = sparse_to_dense_map[sparse_label_int]

    return dense_array


def map_labels_to_sparse(
    dense_array: np.ndarray, dense_to_sparse_map: np.ndarray
) -> np.ndarray:
    """
    Map dense indices back to sparse labels.

    Parameters
    ----------
    dense_array : np.ndarray
        Array with dense indices (0 to num_classes-1)
    dense_to_sparse_map : np.ndarray
        Mapping array where index i gives the sparse label for dense index i.

    Returns
    -------
    np.ndarray
        Array with sparse label IDs
    """
    if len(dense_to_sparse_map) == 0:
        raise ValueError("dense_to_sparse_map cannot be empty")

    return dense_to_sparse_map[dense_array.astype(np.int32)]


def create_mapping_from_labels(labels: List[int]) -> Tuple[Dict[int, int], np.ndarray]:
    """
    Create both mappings from a label list.

    Parameters
    ----------
    labels : List[int]
        List of sparse label IDs (including 0 if present).

    Returns
    -------
    Tuple[Dict[int, int], np.ndarray]
        (sparse_to_dense_dict, dense_to_sparse_array)
    """
    if not labels:
        raise ValueError("labels cannot be empty")

    sparse_to_dense = create_sparse_to_dense_mapping(labels)
    dense_to_sparse = create_dense_to_sparse_mapping(labels)
    return sparse_to_dense, dense_to_sparse


def load_labels_from_json(json_path: Path) -> Tuple[List[int], int]:
    """
    Load labels from a JSON file.

    Expected format:
    {
        "labels": [3, 11, 17, ..., -1, -1001],
        "num_labels": 148
    }

    Returns
    -------
    Tuple[List[int], int]
        (list of label IDs, number of classes)
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["labels"], data["num_labels"]


# ============================================================================
# LABEL GENERATION - Create JSON and ColorLUT from roiinfo.txt
# ============================================================================

# DEPRECATED: labels.json generation removed - use ColorLUT.tsv instead


def generate_colorlut_from_roiinfo(roiinfo_path: Path, output_tsv_path: Path):
    """
    Generate extended ColorLUT.tsv from roiinfo.txt.

    Creates deterministic colors for each label with proper names matching global_var.py.
    Includes background (0) and uses region-hemi-name format for label names.

    The extended format includes Region and Hemi columns, eliminating the need
    for a separate roiinfo file.
    """
    import pandas as pd

    # Read roiinfo file
    df = pd.read_csv(roiinfo_path, sep="\t")

    # Create label name in format: region-hemi-name (matches global_var.py)
    df["LabelName"] = df["region"] + "-" + df["hemi"] + "-" + df["name"]

    # Rename 'key' column to 'ID' for ColorLUT
    df = df.rename(columns={"key": "ID"})

    # Generate deterministic colors for each label
    def generate_color(label_id):
        random.seed(abs(int(label_id)))
        return pd.Series(
            {
                "R": random.randint(20, 255),
                "G": random.randint(20, 255),
                "B": random.randint(20, 255),
                "A": 0,
            }
        )

    # Apply color generation
    colors = df["ID"].apply(generate_color)
    df = pd.concat([df, colors], axis=1)

    # Sort: positive first, then negative (by absolute value)
    df["sort_key"] = df["ID"].apply(lambda x: (x < 0, abs(x)))
    df = df.sort_values("sort_key").drop("sort_key", axis=1)

    # Add background row at the beginning
    bg_row = pd.DataFrame(
        [{"ID": 0, "LabelName": "Background", "R": 0, "G": 0, "B": 0, "A": 0}]
    )

    # Reorder columns: ID, LabelName, then all roiinfo columns, then RGBA
    color_cols = ["R", "G", "B", "A"]
    id_cols = ["ID", "LabelName"]
    other_cols = [col for col in df.columns if col not in id_cols + color_cols]
    column_order = id_cols + other_cols + color_cols

    df = df[column_order]
    bg_row = bg_row.reindex(columns=column_order, fill_value="")

    # Combine background and labels
    result = pd.concat([bg_row, df], ignore_index=True)

    # Save to TSV
    result.to_csv(output_tsv_path, sep="\t", index=False)

    print(f"✅ Generated {output_tsv_path}")
    print(f"   Total entries: {len(result)} (including background)")
    print("   Format: Extended (includes all roiinfo columns)")


# ============================================================================
# VERIFICATION - Check cortex detection
# ============================================================================
