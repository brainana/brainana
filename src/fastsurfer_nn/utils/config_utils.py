#!/usr/bin/env python3

"""
Config Utilities - Path Resolution for fastsurfer_nn
=====================================================

This module provides utilities for resolving paths from YAML configuration files.
It implements a direct paths approach using training_data_dir and output_dir.

Key Features:
- Automatic path derivation based on conventions
- HDF5 file naming: {train|val}_{PLANE}_{ORIENTATION}.hdf5 (e.g., train_axial_RAS.hdf5)
- Clear error messages

Usage:
------
from config_utils import get_paths_from_config

cfg = yaml.safe_load(open('config.yaml'))
paths = get_paths_from_config(cfg)

# Access paths
train_hdf5 = paths['train_hdf5']
log_dir = paths['log_dir']
"""

from pathlib import Path
import yaml


def get_paths_from_config(cfg):
    """
    Get all paths from configuration with smart resolution.

    This function requires the following configuration format:

        training_data_dir: "/path/to/training_data"
        output_dir: "/path/to/output"
        DATA:
          PLANE: "coronal"

    Parameters
    ----------
    cfg : dict
        Configuration dictionary loaded from YAML

    Returns
    -------
    dict
        Dictionary with resolved paths:
        - data_dir: Directory containing source data (images/, labels/)
        - train_hdf5: Path to training HDF5 file
        - val_hdf5: Path to validation HDF5 file
        - log_dir: Path to training output directory

    Examples
    --------
    >>> cfg = yaml.safe_load(open('config.yaml'))
    >>> paths = get_paths_from_config(cfg)
    >>> print(paths['train_hdf5'])
    /path/to/training_data/train_coronal_RAS.hdf5

    Note: HDF5 files are named as {train|val}_{PLANE}_{ORIENTATION}.hdf5
    If ORIENTATION is not specified in DATA.PREPROCESSING, falls back to {split}_{PLANE}.hdf5
    """

    if "TRAINING_DATA_DIR" not in cfg or "OUTPUT_DIR" not in cfg:
        raise ValueError(
            "Invalid configuration format. Config must have:\n"
            '    TRAINING_DATA_DIR: "/path/to/training_data"\n'
            '    OUTPUT_DIR: "/path/to/output"\n'
            "    DATA:\n"
            '      PLANE: "coronal"'
        )

    plane = cfg["DATA"]["PLANE"]

    # Direct paths format
    data_dir = Path(cfg["TRAINING_DATA_DIR"])
    log_dir = Path(cfg["OUTPUT_DIR"])

    # Get orientation from preprocessing config if available
    orientation = None
    if "PREPROCESSING" in cfg.get("DATA", {}):
        orientation = cfg["DATA"]["PREPROCESSING"].get("ORIENTATION")

    # HDF5 files follow naming convention: {split}_{plane}_{ORIENTATION}.hdf5
    # Example: train_axial_RAS.hdf5, val_coronal_RAS.hdf5, train_mixed_RAS.hdf5
    # Orientation is always saved in uppercase (e.g., RAS) but recognized case-insensitively
    # If orientation is not specified, fall back to old format: {split}_{plane}.hdf5
    if orientation:
        # Convert to uppercase for consistent file naming (always save as RAS, not ras)
        # This ensures files are saved with uppercase and found regardless of config case
        orientation_upper = orientation.upper()
        train_hdf5 = data_dir / f"train_{plane}_{orientation_upper}.hdf5"
        val_hdf5 = data_dir / f"val_{plane}_{orientation_upper}.hdf5"
    else:
        train_hdf5 = data_dir / f"train_{plane}.hdf5"
        val_hdf5 = data_dir / f"val_{plane}.hdf5"

    return {
        "data_dir": data_dir,
        "train_hdf5": train_hdf5,
        "val_hdf5": val_hdf5,
        "log_dir": log_dir,
    }


def load_yaml_config(yaml_path):
    """
    Load configuration from YAML file.

    Parameters
    ----------
    yaml_path : str or Path
        Path to YAML configuration file

    Returns
    -------
    dict
        Configuration dictionary

    Examples
    --------
    >>> cfg = load_yaml_config('config/T1w_ARM2_coronal_lia.yaml')
    >>> print(cfg['DATA']['PLANE'])
    coronal
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def print_paths_summary(paths):
    """
    Print a summary of resolved paths for debugging.

    Parameters
    ----------
    paths : dict
        Dictionary of paths from get_paths_from_config

    Examples
    --------
    >>> paths = get_paths_from_config(cfg)
    >>> print_paths_summary(paths)
    """
    print("\n" + "=" * 70)
    print("Resolved Paths from Configuration")
    print("=" * 70)
    print(f"Data directory:     {paths['data_dir']}")
    print(f"Training HDF5:      {paths['train_hdf5']}")
    print(f"Validation HDF5:    {paths['val_hdf5']}")
    print(f"Output directory:   {paths['log_dir']}")
    print("=" * 70 + "\n")


def validate_paths(paths, check_existence=False):
    """
    Validate that paths are properly configured.

    Parameters
    ----------
    paths : dict
        Dictionary of paths from get_paths_from_config
    check_existence : bool, optional
        If True, check if data_dir actually exists (default: False)

    Returns
    -------
    bool
        True if paths are valid

    Raises
    ------
    ValueError
        If paths are invalid or don't exist (when check_existence=True)
    """
    # Check that all required keys exist
    required_keys = ["data_dir", "train_hdf5", "val_hdf5", "log_dir"]
    for key in required_keys:
        if key not in paths:
            raise ValueError(f"Missing required path key: {key}")

    # Check path types
    for key, value in paths.items():
        if not isinstance(value, Path):
            raise ValueError(f"Path '{key}' must be a Path object, got {type(value)}")

    # Optionally check existence
    if check_existence:
        if not paths["data_dir"].exists():
            raise ValueError(
                f"Data directory does not exist: {paths['data_dir']}\n"
                f"Expected to find: images/ and labels/ subdirectories"
            )

    return True


if __name__ == "__main__":
    """
    Test the path resolution with sample configs.
    """
    import sys

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        print(f"Testing path resolution with: {config_path}")

        cfg = load_yaml_config(config_path)
        paths = get_paths_from_config(cfg)
        print_paths_summary(paths)

        try:
            validate_paths(paths, check_existence=True)
            print("✅ All paths validated successfully!")
        except ValueError as e:
            print(f"⚠️  Warning: {e}")
    else:
        print("Usage: python config_utils.py <config.yaml>")
        print("\nExample:")
        print("  python config_utils.py config/T1w_ARM2_coronal_lia.yaml")
