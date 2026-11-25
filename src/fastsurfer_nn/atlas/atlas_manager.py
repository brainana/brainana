#!/usr/bin/env python3
"""
Atlas Manager - Flexible Atlas Configuration System

This module provides a unified interface for managing different atlases in fastsurfer_nn.
It replaces hardcoded atlas references with a flexible configuration system that can
work with any atlas as long as a roiinfo.txt file is provided.

Key Features:
- Load any atlas from roiinfo.txt files
- Generate required atlas files (labels.json, ColorLUT.tsv)
- Provide unified interface for atlas configuration
- Support both metadata-based (ARM2, ARM3) and threshold-based (FreeSurfer) atlases
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .atlas_config import (
    AtlasConfig,
    load_atlas_config_from_roiinfo,
    create_freesurfer_config,
    generate_colorlut_from_roiinfo,
    load_sagittal_hemisphere_mapping,
    map_labels_to_sagittal_dense,
    create_sparse_to_dense_mapping,
    create_dense_to_sparse_mapping,
    map_labels_to_dense,
    map_labels_to_sparse,
)


class AtlasManager:
    """
    Unified atlas management system for fastsurfer_nn.

    This class provides a single interface for managing different atlases,
    replacing hardcoded references with flexible configuration.
    """

    def __init__(self, atlas_name: str, atlas_dir: Optional[Path] = None):
        """
        Initialize atlas manager.

        Parameters
        ----------
        atlas_name : str
            Name of the atlas (e.g., 'ARM2', 'ARM3', 'FreeSurfer')
        atlas_dir : Path, optional
            Directory containing atlas files. If None, uses default location.
        """
        self.atlas_name = atlas_name  # Keep original case as provided
        self.atlas_dir = (
            atlas_dir or Path(__file__).parent / f"atlas-{atlas_name.upper()}"
        )

        # Atlas file paths
        self.roiinfo_path = self.atlas_dir / f"atlas-{atlas_name.upper()}_roiinfo.txt"
        self.labels_json_path = self.atlas_dir / f"{atlas_name.lower()}_labels.json"
        self.colorlut_path = self.atlas_dir / f"{atlas_name.upper()}_ColorLUT.tsv"

        # Load or create atlas configuration
        self._load_atlas_config()

    def _load_atlas_config(self):
        """Load atlas configuration from roiinfo.txt or create default."""
        if self.atlas_name == "freesurfer":
            self.config = create_freesurfer_config()
            self.labels = []
            self.num_classes = 0
        elif self.roiinfo_path.exists():
            # Load from roiinfo.txt
            self.config = load_atlas_config_from_roiinfo(self.roiinfo_path)

            # Auto-generate ColorLUT if it doesn't exist
            if not self.colorlut_path.exists():
                print(
                    f"ColorLUT not found at {self.colorlut_path}, generating from roiinfo.txt..."
                )
                self.generate_atlas_files()

            # Load or generate labels
            if self.labels_json_path.exists():
                with open(self.labels_json_path, "r") as f:
                    labels_data = json.load(f)
                    self.labels = labels_data["labels"]
                    self.num_classes = labels_data["num_labels"]
            else:
                # Generate labels from roiinfo.txt
                self.labels = self._generate_labels_from_roiinfo()
                self.num_classes = len(self.labels)
                self._save_labels_json()
        else:
            raise FileNotFoundError(
                f"Atlas roiinfo file not found: {self.roiinfo_path}"
            )

    def _generate_labels_from_roiinfo(self) -> List[int]:
        """Generate labels list from roiinfo.txt file."""
        labels = []

        with open(self.roiinfo_path, "r") as f:
            lines = f.readlines()

        # Parse header to get column names
        header = lines[0].strip().split("\t")
        col_idx = {name: idx for idx, name in enumerate(header)}

        # Process data rows
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) < len(col_idx):
                continue

            try:
                label = int(parts[col_idx["key"]])
                labels.append(label)
            except (ValueError, IndexError, KeyError):
                continue

        # Add background (0) if not present
        if 0 not in labels:
            labels.append(0)

        # CRITICAL: Return labels in the order they appear in roiinfo.txt (UNSORTED)
        # Sorting was added on Oct 23 22:28, but training data was created earlier with unsorted labels
        # This caused label flipping issues where RH/LH were swapped
        # Keep labels in file order to match training data!
        return labels

    def _save_labels_json(self):
        """Save labels to JSON file - DEPRECATED: Use ColorLUT.tsv instead."""
        # No longer needed - all label info is in ColorLUT.tsv
        pass

    def generate_atlas_files(self):
        """Generate all required atlas files from roiinfo.txt."""
        if not self.roiinfo_path.exists():
            raise FileNotFoundError(f"roiinfo.txt not found: {self.roiinfo_path}")

        # Ensure atlas directory exists
        self.atlas_dir.mkdir(parents=True, exist_ok=True)

        # Generate ColorLUT.tsv (labels.json is no longer needed)
        generate_colorlut_from_roiinfo(self.roiinfo_path, self.colorlut_path)

        print(f"✅ Generated atlas files for {self.atlas_name.upper()}")
        print(f"   ColorLUT: {self.colorlut_path}")

    def get_atlas_config(self, plane: str = "coronal") -> AtlasConfig:
        """Get atlas configuration for specified plane."""
        if self.atlas_name == "freesurfer":
            return create_freesurfer_config(plane)
        else:
            return self.config

    def get_num_classes(self) -> int:
        """Get total number of classes (including background)."""
        return self.num_classes

    def get_sparse_labels(self) -> List[int]:
        """Sparse label IDs (atlas IDs, incl. background 0) in roiinfo FILE order.

        This is NOT the dense (network class) order: background is appended LAST here,
        while dense index 0 is background. Use get_dense_to_sparse_mapping() /
        dense_to_sparse(plane) whenever you need "sparse ID of dense index i".
        """
        return self.labels

    # ------------------------------------------------------------------ plane-aware helpers
    # Terminology (see atlas_config module docstring):
    #   sparse = atlas label IDs as stored in label NIfTIs / ColorLUT (0, 2, 1002, -1, ...)
    #   dense  = contiguous network class indices 0..N-1 (0 = background)
    # Axial/coronal models are bilateral (N = all labels); sagittal models merge left/right
    # into key_nohemi classes (N = unique key_nohemi + background).

    def num_dense_classes(self, plane: str = "coronal") -> int:
        if plane == "sagittal":
            return self.get_sagittal_mapping()[2]
        return self.get_num_classes()

    def dense_to_sparse(self, plane: str = "coronal") -> np.ndarray:
        """Array d2s with d2s[dense] = sparse ID, for the given plane.

        Sagittal classes are hemisphere-merged; each maps to its key_nohemi, which in ARM
        atlases is the right-hemisphere ID (so it has a ColorLUT entry).
        """
        if plane != "sagittal":
            return self.get_dense_to_sparse_mapping()
        _, sagittal_labels, _ = self.get_sagittal_mapping()
        return np.array([0] + list(sagittal_labels), dtype=np.int32)

    def sparse_to_dense(self, plane: str = "coronal") -> Dict[int, int]:
        """Dict sparse ID -> dense index for the given plane (bilateral IDs for sagittal)."""
        if plane != "sagittal":
            return self.get_sparse_to_dense_mapping()
        from .atlas_config import create_sagittal_sparse_to_dense_mapping

        return create_sagittal_sparse_to_dense_mapping(self.roiinfo_path)

    def to_dense(self, sparse_array: np.ndarray, plane: str = "coronal") -> np.ndarray:
        """Map a sparse label array to dense indices for the given plane."""
        if plane == "sagittal":
            return self.map_labels_to_sagittal_dense(sparse_array)
        return self.map_labels_to_dense(sparse_array)

    def dense_ids_of(self, sparse_ids, plane: str = "coronal") -> List[int]:
        """Dense indices of a set of sparse IDs (background and unknown IDs dropped)."""
        lut = self.sparse_to_dense(plane)
        return sorted({lut[int(s)] for s in sparse_ids if int(s) != 0 and int(s) in lut} - {0})

    def cortex_dense(self, plane: str = "coronal") -> List[int]:
        """Dense indices of cortical classes for the given plane."""
        return self.dense_ids_of(self.get_atlas_config(plane).cortex_labels, plane)

    def dense_class_names(self, plane: str = "coronal") -> List[str]:
        """Class names in dense order (index 0 = 'background')."""
        names = {0: "background"}
        if self.roiinfo_path.exists():
            df = pd.read_csv(self.roiinfo_path, sep="\t")
            for _, r in df.iterrows():
                key = int(r["key_nohemi"] if plane == "sagittal" else r["key"])
                hemi = "both" if plane == "sagittal" else str(r.get("hemi", ""))
                names.setdefault(key, f"{r['region']}-{hemi}-{r['name']}")
        return [names.get(int(s), str(int(s))) for s in self.dense_to_sparse(plane)]

    def get_sparse_to_dense_mapping(self) -> Dict[int, int]:
        """Get sparse to dense label mapping."""
        return create_sparse_to_dense_mapping(self.labels)

    def get_dense_to_sparse_mapping(self) -> np.ndarray:
        """Get dense to sparse label mapping."""
        return create_dense_to_sparse_mapping(self.labels)

    def map_labels_to_dense(self, label_array: np.ndarray) -> np.ndarray:
        """Map sparse labels to dense indices."""
        sparse_to_dense = self.get_sparse_to_dense_mapping()
        return map_labels_to_dense(label_array, sparse_to_dense)

    def map_labels_to_sparse(self, dense_array: np.ndarray) -> np.ndarray:
        """Map dense indices back to sparse labels."""
        dense_to_sparse = self.get_dense_to_sparse_mapping()
        return map_labels_to_sparse(dense_array, dense_to_sparse)

    def get_sagittal_mapping(self) -> Tuple[Dict[int, int], List[int], int]:
        """Get sagittal hemisphere mapping."""
        return load_sagittal_hemisphere_mapping(self.roiinfo_path)

    def map_labels_to_sagittal_dense(self, label_array: np.ndarray) -> np.ndarray:
        """Map bilateral labels to sagittal hemisphere-merged dense indices."""
        return map_labels_to_sagittal_dense(label_array, self.roiinfo_path)

    def get_sagittal_to_bilateral_expansion(self) -> List[int]:
        """Get expansion mapping for sagittal to bilateral conversion."""
        (
            bilateral_to_merged,
            sagittal_labels,
            num_sagittal_classes,
        ) = self.get_sagittal_mapping()

        # Create sagittal sparse -> dense index mapping
        nohemi_to_dense = {0: 0}  # Background
        for dense_idx, key_nohemi in enumerate(sagittal_labels, start=1):
            nohemi_to_dense[key_nohemi] = dense_idx

        # Create bilateral sparse -> dense mapping
        bilateral_sparse_to_dense = self.get_sparse_to_dense_mapping()

        # Create the expansion index list
        num_bilateral_classes = len(bilateral_sparse_to_dense)
        bilateral_dense_to_sparse = {v: k for k, v in bilateral_sparse_to_dense.items()}

        idx_list = []
        for bilateral_dense_idx in range(num_bilateral_classes):
            bilateral_sparse = bilateral_dense_to_sparse.get(bilateral_dense_idx, 0)
            sagittal_sparse = bilateral_to_merged.get(bilateral_sparse, 0)
            sagittal_dense_idx = nohemi_to_dense.get(sagittal_sparse, 0)
            idx_list.append(sagittal_dense_idx)

        return idx_list

    def get_region_names(self, plane: str = "coronal") -> Dict[str, List[str]]:
        """Get region names organized by type and plane.

        Returns organized region names for both sagittal (hemisphere-merged)
        and not_sagittal (bilateral) views. Sagittal view uses key_nohemi for
        merging left/right hemispheres.
        """
        if not self.roiinfo_path.exists():
            return {
                "cortex": [],
                "subcortex": [],
                "wm": [],
                "csf": [],
                "labels_id_map": {},
            }

        # Read the TSV file using pandas
        df = pd.read_csv(self.roiinfo_path, sep="\t")

        # Filter: Skip negative keys except for WM and CSF
        df["key_str"] = df["key"].astype(str)
        mask = ~(df["key_str"].str.startswith("-") & ~df["region"].isin(["WM", "CSF"]))
        df = df[mask].copy()

        # Initialize result
        result = {
            "cortex_sagittal": [],
            "cortex_not_sagittal": [],
            "subcortex_sagittal": [],
            "subcortex_not_sagittal": [],
            "wm_sagittal": [],
            "wm_not_sagittal": [],
            "csf_sagittal": [],
            "csf_not_sagittal": [],
            "labels_id_map": {},
            "region_types": list(df["region"].unique()),
        }

        # Build labels_id_map (ID -> region-hemi-name)
        df["region_name_with_hemi"] = df["region"] + "-" + df["hemi"] + "-" + df["name"]
        result["labels_id_map"] = df.set_index("key")["region_name_with_hemi"].to_dict()

        # Process each region type
        for region in ["cortex", "subcortex", "WM", "CSF"]:
            region_df = df[df["region"] == region]

            if len(region_df) == 0:
                continue

            # NOT_SAGITTAL: Include all rows (bilateral, with hemispheres)
            result[f"{region.lower()}_not_sagittal"] = region_df[
                "region_name_with_hemi"
            ].tolist()

            # SAGITTAL: Hemisphere-merged (drop duplicates by key_nohemi)
            # This efficiently handles the hemisphere merging using pandas!
            sagittal_df = region_df.drop_duplicates(subset=["key_nohemi"])

            if region == "cortex":
                # Format: ctx-both-{name}
                result["cortex_sagittal"] = (
                    sagittal_df["name"].apply(lambda x: f"ctx-both-{x}")
                ).tolist()
            else:
                # Format: {region}-{name}
                result[f"{region.lower()}_sagittal"] = (
                    sagittal_df["name"].apply(lambda x: f"{region.lower()}-{x}")
                ).tolist()

        return result

    def get_class_dict(self) -> Dict[str, Dict[str, List[str]]]:
        """Get class dictionary following FreeSurfer tradition."""
        region_names = self.get_region_names()
        region_types = list(region_names["region_types"])

        # "aseg" = subcortex + WM + CSF (non-cortical regions)
        # "aparc" = cortex (cortical regions)
        aseg_sagittal = []
        aseg_not_sagittal = []
        aparc_sagittal = []
        aparc_not_sagittal = []

        for region_type in region_types:
            region_key = region_type.lower()
            if region_key in ["subcortex", "wm", "csf"]:
                # Add to aseg (non-cortical regions)
                aseg_sagittal.extend(region_names.get(f"{region_key}_sagittal", []))
                aseg_not_sagittal.extend(
                    region_names.get(f"{region_key}_not_sagittal", [])
                )
            elif region_key == "cortex":
                # Add to aparc (cortical regions)
                aparc_sagittal.extend(region_names.get(f"{region_key}_sagittal", []))
                aparc_not_sagittal.extend(
                    region_names.get(f"{region_key}_not_sagittal", [])
                )

        # Create class dictionary with only aseg and aparc
        class_dict = {
            "sagittal": {
                "aseg": aseg_sagittal,
                "aparc": aparc_sagittal,
                self.atlas_name: aseg_sagittal + aparc_sagittal,  # Combined view
            },
            "not_sagittal": {
                "aseg": aseg_not_sagittal,
                "aparc": aparc_not_sagittal,
                self.atlas_name: aseg_not_sagittal
                + aparc_not_sagittal,  # Combined view
            },
        }

        return class_dict

    def verify_atlas_files(self) -> bool:
        """Verify that all required atlas files exist."""
        required_files = [self.roiinfo_path, self.labels_json_path, self.colorlut_path]
        missing_files = [f for f in required_files if not f.exists()]

        if missing_files:
            print(f"❌ Missing atlas files for {self.atlas_name.upper()}:")
            for f in missing_files:
                print(f"   - {f}")
            return False

        print(f"✅ All atlas files found for {self.atlas_name.upper()}")
        return True


def get_atlas_manager(
    atlas_name: str, atlas_dir: Optional[Path] = None
) -> AtlasManager:
    """
    Get atlas manager instance.

    Parameters
    ----------
    atlas_name : str
        Name of the atlas (e.g., 'ARM2', 'ARM3', 'FreeSurfer')
    atlas_dir : Path, optional
        Directory containing atlas files

    Returns
    -------
    AtlasManager
        Atlas manager instance
    """
    return AtlasManager(atlas_name, atlas_dir)


# Convenience functions for backward compatibility
def get_atlas_config(atlas_name: str = "arm3", plane: str = "coronal") -> AtlasConfig:
    """Get atlas configuration by name."""
    manager = get_atlas_manager(atlas_name)
    return manager.get_atlas_config(plane)


def get_num_classes(atlas_name: str = "arm3") -> int:
    """Get number of classes for atlas."""
    manager = get_atlas_manager(atlas_name)
    return manager.get_num_classes()


def get_class_dict(atlas_name: str = "arm3") -> Dict[str, Dict[str, List[str]]]:
    """Get class dictionary for atlas."""
    manager = get_atlas_manager(atlas_name)
    return manager.get_class_dict()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Atlas Manager CLI")
    parser.add_argument("atlas_name", help="Name of the atlas (e.g., ARM2, ARM3)")
    parser.add_argument("--generate", action="store_true", help="Generate atlas files")
    parser.add_argument("--verify", action="store_true", help="Verify atlas files")
    parser.add_argument("--atlas-dir", type=Path, help="Atlas directory path")

    args = parser.parse_args()

    try:
        manager = get_atlas_manager(args.atlas_name, args.atlas_dir)

        if args.generate:
            manager.generate_atlas_files()

        if args.verify:
            manager.verify_atlas_files()

        if not args.generate and not args.verify:
            print(f"Atlas Manager for {args.atlas_name.upper()}")
            print(f"  Classes: {manager.get_num_classes()}")
            print(f"  Labels: {len(manager.get_sparse_labels())}")
            print(f"  Config: {manager.get_atlas_config().name}")

    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
