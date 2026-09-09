"""
FreeSurfer binary wrappers.

Provides Python interfaces to FreeSurfer command-line tools.
"""

from .base import run_fs_command, run_recon_all, FreeSurferError, get_fs_home

# Import all wrappers
from .mri import (
    mri_convert,
    mri_pretess,
    mri_mc,
    mri_mask,
    mri_normalize,
    mri_cc,
    mri_surf2volseg,
    mri_add_xform_to_header,
)

from .mris import (
    mris_info,
    mris_extract_main_component,
    mris_remesh,
    mris_smooth,
    mris_place_surface,
    mris_place_surface_curv_map,
    mris_place_surface_area_map,
    mris_place_surface_thickness,
    mris_register,
    mris_ca_label,
    mris_anatomical_stats,
)


from .registration import (
    talairach_avi,
    lta_convert,
    pctsurfcon as pctsurfcon_wrapper,
)
from .longitudinal import (
    mri_robust_template,
    mri_convert_apply_lta,
    make_upright,
    mri_concatenate_lta,
)

__all__ = [
    # Base utilities
    "run_fs_command",
    "run_recon_all",
    "FreeSurferError",
    "get_fs_home",
    # mri_* commands
    "mri_convert",
    "mri_pretess",
    "mri_mc",
    "mri_mask",
    "mri_normalize",
    "mri_cc",
    "mri_surf2volseg",
    "mri_add_xform_to_header",
    # mris_* commands
    "mris_info",
    "mris_extract_main_component",
    "mris_remesh",
    "mris_smooth",
    "mris_place_surface",
    "mris_place_surface_curv_map",
    "mris_place_surface_area_map",
    "mris_place_surface_thickness",
    "mris_register",
    "mris_ca_label",
    "mris_anatomical_stats",
    # Registration
    "talairach_avi",
    "lta_convert",
    "pctsurfcon_wrapper",
    # Longitudinal stream
    "mri_robust_template",
    "mri_convert_apply_lta",
    "make_upright",
    "mri_concatenate_lta",
]
