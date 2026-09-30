"""
Stage 10: Surface Inflation

Inflates surface to sphere (inflate1).
This is the first inflation, performed before topology fix.
For high-resolution data, sufficient inflation (e.g., 100 iterations) is critical
for correct surface mapping onto sphere and subsequent defect labeling.
"""

import logging

from .base import HemisphereStage
from ..wrappers.mris import mris_inflate

logger = logging.getLogger(__name__)


class Inflation(HemisphereStage):
    """Inflate surface to sphere (inflate1, before topology fix)."""

    name = "inflation"
    description = "Surface inflation (inflate1)"

    def _run(self) -> None:
        """Inflate surface (inflate1, before topology fix).

        Uses inflate_iterations parameter. For high-resolution data (0.75mm isotropic),
        use 20-50 or even 100 iterations to ensure sufficient inflation.
        """
        # Input is smoothwm.nofix (before topology fix)
        smoothwm_nofix = self.hemi_path("smoothwm.nofix")
        inflated_nofix = self.hemi_path("inflated.nofix")
        if not smoothwm_nofix.exists():
            raise FileNotFoundError(
                f"{self.hemi}.smoothwm.nofix not found. "
                "This should be created in stage 09 (smoothing)."
            )

        logger.info(
            f"Inflating {self.hemi} surface (inflate1, n={self.config.processing.inflate_iterations})..."
        )
        mris_inflate(
            input_surf=smoothwm_nofix,
            output_surf=inflated_nofix,
            n_iterations=self.config.processing.inflate_iterations,
            no_save_sulc=self.config.processing.inflate_no_save_sulc,
            log_file=self.config.log_file,
            subject_dir=self.sd.subject_dir,
        )

    def is_disabled(self) -> bool:
        """Off when the geometry comes from elsewhere.

        Longitudinal timepoints inherit it from the base template (see
        stages/s00_long_init.py): recomputing it here would discard the shared
        topology that makes cross-timepoint vertex correspondence exact.
        Template-initialised runs build it in s12b from the template's white
        surface instead of tessellating the white-matter volume.
        """
        return self.config.longitudinal or self.config.template_init

    def should_skip(self) -> bool:
        """Skip if inflated exists, or if a later stage's output proves it ran.

        Custom rather than expected_outputs(): this deliberately ORs across
        s11/s12 side effects, because s12 deletes inflated.nofix after
        consuming it -- so this stage's own output legitimately disappears.
        """
        if self.hemi_path("inflated.nofix").exists():
            return True
        # With the FreeSurfer qsphere, s11 reads inflated.nofix. If s11 is
        # about to run again (qsphere.nofix gone) after s12 deleted
        # inflated.nofix, the later outputs below are no reason to skip:
        # s11 would fail on the missing input.
        if (
            self.config.processing.use_fs_qsphere
            and not self.hemi_path("qsphere.nofix").exists()
        ):
            return False
        # Otherwise a later stage's output proves this stage ran; s12 deletes
        # inflated.nofix after using it.
        return (
            self.hemi_path("inflated").exists()
            or self.hemi_path("inflated.nofix").exists()
            or self.hemi_path("sphere").exists()
            or self.hemi_path("qsphere.nofix").exists()
        )
