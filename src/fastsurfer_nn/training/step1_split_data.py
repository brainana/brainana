#!/usr/bin/env python3

"""
Split data into training and validation sets

This script uses the config_utils module to resolve paths from YAML config.
All paths are derived from training_data_dir and output_dir in the YAML file.
Works with both binary and multi-class segmentation tasks.
"""

import argparse
from pathlib import Path
import random
import json
import sys

# Add src/ to path for fastsurfer_nn imports (training/ -> fastsurfer_nn -> src)
_src_dir = Path(__file__).resolve().parent.parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
from fastsurfer_nn.utils.config_utils import (
    load_yaml_config,
    get_paths_from_config,
)


def animal_key(stem: str) -> str:
    """'site-ohsu_sub-032217_ses-002_run-1_T1w' -> 'site-ohsu_sub-032217'."""
    parts = stem.split("_")
    sub = next((p for p in parts if p.startswith("sub-")), parts[0])
    site = parts[0] if parts[0].startswith("site-") else ""
    return f"{site}_{sub}" if site else sub


def split_data(data_dir, val_split=0.2, seed=42, min_train_subjects=5, force=False):
    """
    Split data into train/val sets

    Parameters
    ----------
    data_dir : Path
        Directory containing 'images' and 'labels' subdirectories
    val_split : float
        Fraction of data to use for validation
    seed : int
        Random seed for reproducibility
    min_train_subjects : int
        Minimum number of subjects to keep for training
    """
    data_dir = Path(data_dir)
    image_dir = data_dir / "images"
    label_dir = data_dir / "labels"

    # Get all image files (any .nii.gz files)
    image_files = sorted(image_dir.glob("*.nii.gz"))
    # Extract subject names (remove .nii.gz extension)
    subjects = [f.name.replace(".nii.gz", "") for f in image_files]

    # Remove any subjects without corresponding labels
    # Label files have suffix: image "abcde.nii.gz" -> label "abcde_xxx.nii.gz"
    valid_subjects = []
    for subj in subjects:
        # Look for label file that starts with subject name followed by underscore
        label_pattern = f"{subj}_*.nii.gz"
        label_matches = list(label_dir.glob(label_pattern))

        if len(label_matches) == 0:
            # No label file found - skip this subject
            continue
        elif len(label_matches) == 1:
            # Exactly one match - valid
            valid_subjects.append(subj)
        else:
            # Multiple matches found - print warning and skip
            print(f"Warning: Multiple label files found for subject {subj}:")
            for match in label_matches:
                print(f"  - {match.name}")
            print(f"  Skipping subject {subj} (expected exactly one match)")

    print(f"Total subjects: {len(subjects)}")
    print(f"Valid subjects (with labels): {len(valid_subjects)}")

    if len(valid_subjects) < min_train_subjects + 1:
        print(
            f"Warning: Not enough subjects for splitting (need at least {min_train_subjects + 1})"
        )
        print("Using all subjects for training")
        train_subjects = valid_subjects
        val_subjects = []
    else:
        # Split by ANIMAL (site + sub-), so sessions/runs of one monkey never end up on both
        # sides (that would leak anatomy into validation).
        by_animal = {}
        for s in valid_subjects:
            by_animal.setdefault(animal_key(s), []).append(s)
        animals = sorted(by_animal)
        random.seed(seed)
        random.shuffle(animals)

        num_val = max(1, int(round(len(animals) * val_split)))
        if len(animals) - num_val < 1:
            num_val = len(animals) - 1
        val_animals = set(animals[:num_val])
        val_subjects = [s for s in valid_subjects if animal_key(s) in val_animals]
        train_subjects = [s for s in valid_subjects if animal_key(s) not in val_animals]
        if len(train_subjects) < min_train_subjects:
            print(f"Warning: only {len(train_subjects)} training subjects")

    # print(f"\nTraining subjects: {len(train_subjects)}")
    # print(f"Validation subjects: {len(val_subjects)}")

    # Save split info
    split_info = {
        "train": sorted(train_subjects),
        "val": sorted(val_subjects),
        "total": len(valid_subjects),
        "seed": seed,
        "val_split": val_split,
    }

    split_file = data_dir / "data_split.json"
    if split_file.exists() and not force:
        raise SystemExit(f"{split_file} exists (it decides which subjects are validation); "
                         "pass --force to replace it")
    with open(split_file, "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nSplit information saved to: {split_file}")
    # print("\nTraining subjects:")
    # for subj in train_subjects:
    #     print(f"  {subj}")

    # if val_subjects:
    #     print("\nValidation subjects:")
    #     for subj in val_subjects:
    #         print(f"  {subj}")

    return train_subjects, val_subjects


def main():
    parser = argparse.ArgumentParser(
        description="Split monkey data into train/val sets"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (SINGLE SOURCE OF TRUTH)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction for validation (default: 0.2 = 20%%, can override YAML)",
    )
    parser.add_argument("--force", action="store_true",
                        help="replace an existing data_split.json")
    args = parser.parse_args()

    # Load config from YAML using unified config utilities
    cfg = load_yaml_config(args.config)
    paths = get_paths_from_config(cfg)

    # Get data directory and seed from resolved paths
    data_dir = paths["data_dir"]
    seed = cfg.get("RNG_SEED", 42)

    print(f"Configuration from YAML: {args.config}")
    print(f"  Data directory:  {data_dir}")
    print(f"  Val split:       {args.val_split}")
    print(f"  Random seed:     {seed}")
    print()

    split_data(
        data_dir=data_dir, val_split=args.val_split, seed=seed, min_train_subjects=5,
        force=args.force,
    )


if __name__ == "__main__":
    main()
