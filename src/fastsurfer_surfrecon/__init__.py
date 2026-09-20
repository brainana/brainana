"""
FastSurfer Surface Reconstruction Pipeline.

A Python-based surface reconstruction pipeline for neuroimaging,
designed for non-human primate (macaque) brain MRI processing.

Based on FastSurfer's recon_surf module.
"""

__version__ = "0.1.0"

from .config import ReconSurfConfig, AtlasConfig, ProcessingConfig
from .pipeline import ReconSurfPipeline

__all__ = [
    "__version__",
    "ReconSurfConfig",
    "AtlasConfig",
    "ProcessingConfig",
    "ReconSurfPipeline",
]
