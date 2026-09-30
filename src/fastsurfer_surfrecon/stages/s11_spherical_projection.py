"""
Stage 11: Spherical Projection

Projects surface to sphere (qsphere or spectral projection).
"""

import logging
import shutil

from .base import HemisphereStage
from ..processing.surface_fix import assert_surface_invariants
from ..processing.spherical import spherically_project_surface
from ..wrappers.mris import mris_sphere_quick

logger = logging.getLogger(__name__)


class SphericalProjection(HemisphereStage):
    """Project surface to sphere."""

    name = "spherical_projection"
    description = "Spherical projection (qsphere)"

    def _run(self) -> None:
        """Project to sphere."""
        sphere = self.hemi_path("sphere")
        qsphere_nofix = self.hemi_path("qsphere.nofix")

        # mris_fix_topology (s12) marks as defects the faces that overlap on
        # this map. FreeSurfer's quasi-homeomorphic sphere is the map it was
        # designed around; the spectral projection folds more, and the folds
        # enlarge and merge defects that the topology fix then cuts away --
        # on the test cohort most visibly in thin occipital/V1 white matter.
        if self.config.processing.use_fs_qsphere:
            # recon-all -qsphere: mris_sphere -q from the inflated surface
            source = self.hemi_path("inflated.nofix")
            method = "FreeSurfer qsphere (mris_sphere -q)"
            prerequisite = "Inflation stage (s10)"
        else:
            # FastSurfer: spectral projection of smoothwm.nofix
            source = self.hemi_path("smoothwm.nofix")
            method = "spectral projection"
            prerequisite = "Smoothing stage (s09)"
        logger.info(f"Using {method} for {self.hemi}")

        if not source.exists():
            raise FileNotFoundError(
                f"{source} not found. {prerequisite} must run first."
            )

        # Entry gate. A non-closed input fails both methods with a message that
        # names neither the file nor the defect (spectral: "Can only project
        # closed meshes"). Checking here reports the offending surface and its
        # actual V/F/closed/oriented/euler state instead.
        assert_surface_invariants(
            source,
            closed=True,
            oriented=False,  # orientation is not required to project
            euler=None,  # not topology-corrected yet
            context=f"{self.hemi} s11 input",
            strict=self.config.processing.strict_surface_checks,
        )

        if self.config.processing.use_fs_qsphere:
            mris_sphere_quick(
                input_surf=source,
                output_surf=qsphere_nofix,
                log_file=self.config.log_file,
                subject_dir=self.sd.subject_dir,
            )
        else:
            spherically_project_surface(
                input_path=source,
                output_path=qsphere_nofix,
                threads=self.threads,
            )
        # Also create sphere as an alias (copy for compatibility)
        if not sphere.exists():
            shutil.copy(qsphere_nofix, sphere)

    def is_disabled(self) -> bool:
        """Off when the geometry comes from elsewhere.

        Longitudinal timepoints inherit it from the base template (see
        stages/s00_long_init.py): recomputing it here would discard the shared
        topology that makes cross-timepoint vertex correspondence exact.
        Template-initialised runs build it in s12b from the template's white
        surface instead of tessellating the white-matter volume.
        """
        return self.config.longitudinal or self.config.template_init

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
