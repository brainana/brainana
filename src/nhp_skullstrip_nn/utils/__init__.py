"""
Utility module for nhp_skullstrip_nn.

This module contains I/O utilities, GPU utilities, morphological operations,
and general plotting functions.
"""

from .io import load_nifti, save_nifti, load_pickle, save_pickle
from .gpu import get_device, setup_device, resolve_device
from .plot import plot_slice, plot_volume
from .log import setup_logging, get_logger, MacacaLogger
from .morphology import (
    extract_largest_component,
    select_confident_component,
    fill_label_holes,
    morphological_erosion_dilation,
    get_bounding_box,
    crop_to_label,
)

__all__ = [
    "load_nifti",
    "save_nifti",
    "load_pickle",
    "save_pickle",
    "get_device",
    "setup_device",
    "resolve_device",
    "plot_slice",
    "plot_volume",
    # Logging utilities
    "setup_logging",
    "get_logger",
    "MacacaLogger",
    # Morphology functions
    "extract_largest_component",
    "select_confident_component",
    "fill_label_holes",
    "morphological_erosion_dilation",
    "get_bounding_box",
    "crop_to_label",
    # Legacy names
    "extract_large_comp",
    "fill_holes",
    "erosion_dilation",
]
