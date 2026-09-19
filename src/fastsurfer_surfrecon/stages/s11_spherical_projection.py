"""
Stage 11: Spherical Projection

Projects surface to sphere (qsphere or spectral projection).
"""

import logging
import shutil

from .base import HemisphereStage
from ..processing.surface_fix import assert_surface_invariants
from ..processing.spherical import spherically_project_surface

logger = logging.getLogger(__name__)


class SphericalProjection(HemisphereStage):
    """Project surface to sphere."""

    name = "spherical_projection"
    description = "Spherical projection (qsphere)"

    def _run(self) -> None:
        """Project to sphere."""
        sphere = self.hemi_path("sphere")
        smoothwm_nofix = self.hemi_path("smoothwm.nofix")
        qsphere_nofix = self.hemi_path("qsphere.nofix")

        # if self.config.processing.fsqsphere:
        #     # Use FreeSurfer qsphere
        #     logger.info(f"Using FreeSurfer qsphere for {self.hemi}")
        #     flags = []
        #     if self.config.hires:
        #         flags.append("-hires")
        #     run_recon_all(
        #         subject=self.config.subject_id,
        #         hemi=self.hemi,
        #         steps=["-qsphere"],
        #         flags=flags,
        #         threads=self.threads,
        #         log_file=self.config.log_file,
        #         subjects_dir=self.config.subjects_dir,
        #     )
        # else:

        # Use spectral projection
        logger.info(f"Using spectral projection for {self.hemi}")
        # FastSurfer uses smoothwm.nofix as input for spherical projection

        if not smoothwm_nofix.exists():
            raise FileNotFoundError(
                f"{smoothwm_nofix} not found. " "Smoothing stage must run first."
            )

        # Entry gate. spherically_project() rejects non-closed meshes with a
        # bare "Can only project closed meshes", which names neither the file
        # nor the defect. Checking here reports the offending surface and its
        # actual V/F/closed/oriented/euler state instead.
        assert_surface_invariants(
            smoothwm_nofix,
            closed=True,
            oriented=False,  # orientation is not required to project
            euler=None,  # not topology-corrected yet
            context=f"{self.hemi} s11 input",
            strict=self.config.processing.strict_surface_checks,
        )

        # FastSurfer creates qsphere.nofix directly, so we do the same
        # Also create sphere for consistency with FreeSurfer naming
        spherically_project_surface(
            input_path=smoothwm_nofix,
            output_path=qsphere_nofix,
            threads=self.threads,
        )
        # Also create sphere as an alias (copy for compatibility)
        if not sphere.exists():
            shutil.copy(qsphere_nofix, sphere)

    def is_disabled(self) -> bool:
        """Longitudinal timepoints inherit this geometry from the base template.

        See stages/s00_long_init.py: the base's surfaces are seeded into this
        timepoint, so recomputing them here would discard the shared topology
        that makes cross-timepoint vertex correspondence exact.
        """
        return self.config.longitudinal

    def expected_outputs(self) -> list:
        """Both spheres this stage writes.

        Previously only `sphere` was checked, but s12 rewrites `sphere` too --
        so a half-finished s11 could be masked by a later stage's output.
        `qsphere.nofix` is written only here.
        """
        return [
            self.hemi_path("qsphere.nofix"),
            self.hemi_path("sphere"),
        ]
