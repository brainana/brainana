"""
Stage 09: Surface Smoothing

Smooths the initial surface (smooth1).
Creates smoothwm.nofix from orig.nofix, before topology fix.
"""

import logging

from ..processing.surface_fix import assert_surface_invariants
from .base import HemisphereStage
from ..wrappers.mris import mris_smooth

logger = logging.getLogger(__name__)


class Smoothing(HemisphereStage):
    """Smooth initial surface (smooth1, before topology fix)."""

    name = "smoothing"
    description = "Surface smoothing (smoothwm.nofix)"

    def _run(self) -> None:
        """Smooth surface (smooth1, before topology fix).

        Uses smooth_iterations parameter. Creates smoothwm.nofix from orig.nofix.
        """
        orig_nofix = self.hemi_path("orig.nofix")
        smoothwm_nofix = self.hemi_path("smoothwm.nofix")

        logger.info(
            f"Smoothing {self.hemi} surface (smooth1, n={self.config.processing.smooth_iterations})..."
        )
        mris_smooth(
            input_surf=orig_nofix,
            output_surf=smoothwm_nofix,
            n_iterations=self.config.processing.smooth_iterations,
            nw=True,
            seed=1234,
            log_file=self.config.log_file,
            subject_dir=self.sd.subject_dir,
        )

    def is_disabled(self) -> bool:
        """Longitudinal timepoints inherit this geometry from the base template.

        See stages/s00_long_init.py: the base's surfaces are seeded into this
        timepoint, so recomputing them here would discard the shared topology
        that makes cross-timepoint vertex correspondence exact.
        """
        return self.config.longitudinal

    def expected_outputs(self) -> list:
        """Smoothed surface, before topology correction."""
        return [self.hemi_path("smoothwm.nofix")]

    def verify_outputs(self) -> None:
        """Smoothing moves vertices; it must not change connectivity."""
        super().verify_outputs()
        assert_surface_invariants(
            self.hemi_path("smoothwm.nofix"),
            closed=True,
            oriented=True,
            euler=None,
            context=f"{self.hemi} s09 smoothing",
            strict=self.config.processing.strict_surface_checks,
        )
