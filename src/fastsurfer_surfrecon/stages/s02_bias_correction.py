"""
Stage 02: Bias Field Correction

Performs N4 bias field correction and WM intensity normalization.
"""

import logging

from .base import PipelineStage
from ..processing.bias_correction import bias_correct_and_normalize

logger = logging.getLogger(__name__)


class BiasCorrection(PipelineStage):
    """Bias field correction using N4 algorithm."""

    name = "bias_correction"
    description = "N4 bias field correction and WM normalization"

    def _run(self) -> None:
        """Run N4 bias correction."""
        logger.info("Running N4 bias field correction...")

        # Get mask if available
        mask_path = self.config.mask or self.sd.mask
        if not mask_path.exists():
            mask_path = None
            logger.warning("No mask available, running without mask")

        # WM normalization takes WM to be aseg labels 2/41, so it needs the
        # aseg-mapped volume. The atlas volume (aparc.<atlas>atlas+aseg.orig)
        # keeps raw atlas IDs, where 2 is right anterior cingulate cortex: norm
        # then put cortex at 105 and true WM at ~125-175 instead of ~105.
        aseg_nocc = self.sd.mri("aseg.auto_noCCseg.mgz")
        aseg_path = aseg_nocc if aseg_nocc.exists() else None

        bias_correct_and_normalize(
            input_path=self.sd.orig,
            output_path=self.sd.orig_nu,
            mask_path=mask_path,
            aseg_path=aseg_path,
            shrink_factor=self.config.processing.n4_shrink_factor,
            num_levels=self.config.processing.n4_levels,
            num_iterations=self.config.processing.n4_num_iterations,
            threads=self.config.processing.threads,
        )

    def expected_outputs(self) -> list:
        """Bias-corrected, WM-normalised volume."""
        return [self.sd.orig_nu]
