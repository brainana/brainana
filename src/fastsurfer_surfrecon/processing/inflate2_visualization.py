"""
Post-topology visualization smoothing and inflation (inflate2).

Applied in stage 14 after parcellation mapping; can also be run standalone
to refresh existing subject trees without re-mapping parcellation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from ..wrappers.mris import mris_inflate, mris_smooth

if TYPE_CHECKING:
    from ..config import ReconSurfConfig
    from ..io.subjects_dir import SubjectsDir

logger = logging.getLogger(__name__)

HemiType = Literal["lh", "rh"]


def apply_inflate2_visualization(
    config: "ReconSurfConfig",
    sd: "SubjectsDir",
    hemi: HemiType,
) -> bool:
    """
    Re-smooth smoothwm and re-inflate for visualization (stage-14 inflate2 block).

    Returns True if adjustments were applied, False if skipped (None iterations
    or missing inputs).
    """
    if config.processing.inflate2_iterations is None:
        return False

    smoothwm = sd.hemi_surf(hemi, "smoothwm")
    inflated = sd.hemi_surf(hemi, "inflated")
    white_preaparc = sd.hemi_surf(hemi, "white.preaparc")

    if not smoothwm.exists() or not inflated.exists() or not white_preaparc.exists():
        logger.warning(
            "Cannot apply inflate2 visualization for %s: missing surfaces "
            "(smoothwm=%s, inflated=%s, white.preaparc=%s)",
            hemi,
            smoothwm.exists(),
            inflated.exists(),
            white_preaparc.exists(),
        )
        return False

    logger.info(
        "Applying post-processing adjustments for %s (visualization smoothing/inflation)...",
        hemi,
    )
    logger.info(
        "Re-smoothing %s.smoothwm with %s iterations (visualization smoothing)...",
        hemi,
        config.processing.smooth2_iterations,
    )

    smoothwm_adjusted = sd.hemi_surf(hemi, "smoothwm.adjusted")
    mris_smooth(
        input_surf=white_preaparc,
        output_surf=smoothwm_adjusted,
        n_iterations=config.processing.smooth2_iterations,
        nw=True,
        seed=1234,
        log_file=config.log_file,
        subject_dir=sd.subject_dir,
    )
    smoothwm_adjusted.replace(smoothwm)
    logger.info(
        "Replaced %s.smoothwm with adjusted version (%s iterations)",
        hemi,
        config.processing.smooth2_iterations,
    )

    smoothwm = sd.hemi_surf(hemi, "smoothwm")
    n_smooth = config.processing.inflate2_smooth_iterations
    if n_smooth and n_smooth > 0:
        logger.info(
            "Re-smoothing %s.smoothwm with %s iterations (inflation input only)...",
            hemi,
            n_smooth,
        )
        smoothwm_forinflate = sd.hemi_surf(hemi, "smoothwm.forinflate")
        mris_smooth(
            input_surf=smoothwm,
            output_surf=smoothwm_forinflate,
            n_iterations=n_smooth,
            nw=True,
            seed=1234,
            log_file=config.log_file,
            subject_dir=sd.subject_dir,
        )
        inflate_input = smoothwm_forinflate
    else:
        inflate_input = smoothwm

    logger.info(
        "Re-inflating %s.inflated with %s iterations (visualization inflation)...",
        hemi,
        config.processing.inflate2_iterations,
    )
    inflated_adjusted = sd.hemi_surf(hemi, "inflated.adjusted")
    mris_inflate(
        input_surf=inflate_input,
        output_surf=inflated_adjusted,
        n_iterations=config.processing.inflate2_iterations,
        no_save_sulc=False,
        log_file=config.log_file,
        subject_dir=sd.subject_dir,
    )
    inflated_adjusted.replace(inflated)
    logger.info(
        "Replaced %s.inflated with adjusted version (%s iterations)",
        hemi,
        config.processing.inflate2_iterations,
    )

    if n_smooth and n_smooth > 0:
        smoothwm_forinflate = sd.hemi_surf(hemi, "smoothwm.forinflate")
        if smoothwm_forinflate.exists():
            smoothwm_forinflate.unlink()
            logger.debug("Removed temporary %s.smoothwm.forinflate", hemi)

    return True
