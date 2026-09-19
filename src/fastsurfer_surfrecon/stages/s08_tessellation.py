"""
Stage 08: Surface Tessellation

Creates initial surface using marching cubes or FreeSurfer tessellation.
"""

import logging
import numpy as np

from ..processing.surface_fix import (
    assert_surface_invariants,
    verify_surface_ras,
)
from .base import HemisphereStage
from ..wrappers.mri import mri_pretess, mri_mc
from ..wrappers.mris import mris_extract_main_component, mris_remesh
from ..wrappers.base import run_recon_all, FreeSurferError
from ..processing.surface_fix import fix_mc_surface_header

logger = logging.getLogger(__name__)


class Tessellation(HemisphereStage):
    """Create initial surface tessellation."""

    name = "tessellation"
    description = "Surface tessellation (orig.nofix)"

    def _run(self) -> None:
        """Create initial surface."""
        filled = self.sd.mri("filled.mgz")
        brain = self.sd.mri("brainmask.mgz")
        orig_nofix = self.hemi_path("orig.nofix")

        if self.config.processing.fstess:
            # Use FreeSurfer tessellation
            logger.info(f"Using FreeSurfer tessellation for {self.hemi}")
            flags = []
            if self.config.hires:
                flags.append("-hires")
            run_recon_all(
                subject=self.config.subject_id,
                hemi=self.hemi,
                steps=["-tessellate"],
                flags=flags,
                threads=self.threads,
                log_file=self.config.log_file,
                subjects_dir=self.config.subjects_dir,
            )
        else:
            # Use marching cubes
            # Pretessellate
            logger.info(f"Pretessellating {self.hemi}")
            pretess = self.sd.mri(f"filled-pretess{self.hemi_value}.mgz")
            if not pretess.exists():
                mri_pretess(
                    input_vol=filled,
                    label=self.hemi_value,
                    norm=brain,
                    output_vol=pretess,
                    log_file=self.config.log_file,
                    subject_dir=self.sd.subject_dir,
                )

            # Marching cubes
            logger.info(f"Using marching cubes for {self.hemi}")
            hires_suffix = ".predec" if self.config.hires else ""
            orig_nofix = self.hemi_path(f"orig.nofix{hires_suffix}")
            mri_mc(
                input_vol=pretess,
                label=self.hemi_value,
                output_surf=orig_nofix,
                log_file=self.config.log_file,
                subject_dir=self.sd.subject_dir,
            )

            # Fix surface header (scannerRAS -> surfaceRAS)
            logger.info(f"Fixing surface header for {self.hemi}")
            fix_mc_surface_header(
                surface_path=orig_nofix,
                pretess_path=pretess,
                output_path=orig_nofix,
            )

            # Verify surfaceRAS header
            verify_surface_ras(
                orig_nofix,
                log_file=self.config.log_file,
                subject_dir=self.sd.subject_dir,
            )

            # Extract main component
            logger.info(f"Extracting main component for {self.hemi}")
            mris_extract_main_component(
                input_surf=orig_nofix,
                output_surf=orig_nofix,
                log_file=self.config.log_file,
                subject_dir=self.sd.subject_dir,
            )

            # Re-fix header after extraction (mris_extract_main_component may reset it)
            logger.info(f"Re-fixing surface header after extraction for {self.hemi}")
            fix_mc_surface_header(
                surface_path=orig_nofix,
                pretess_path=pretess,
                output_path=orig_nofix,
            )

            # Verify surfaceRAS header again after extraction
            verify_surface_ras(
                orig_nofix,
                log_file=self.config.log_file,
                subject_dir=self.sd.subject_dir,
            )

            # Decimate for hires (try less decimation first on "too small" errors)
            if self.config.hires:
                orig_nofix_final = self.hemi_path("orig.nofix")
                desired_face_areas = np.arange(0.1, 0.5, 0.02)
                last_error = None
                for desired_face_area in desired_face_areas:
                    try:
                        logger.info(
                            f"Decimating surface for {self.hemi} "
                            f"(desired_face_area={desired_face_area})"
                        )
                        mris_remesh(
                            input_surf=orig_nofix,
                            output_surf=orig_nofix_final,
                            desired_face_area=desired_face_area,
                            log_file=self.config.log_file,
                            subject_dir=self.sd.subject_dir,
                            iters=5,
                        )
                        orig_nofix = orig_nofix_final
                        break
                    except FreeSurferError as e:
                        last_error = e
                        logger.warning(
                            f"mris_remesh failed with desired_face_area={desired_face_area}: "
                            f"{e}; trying next value"
                        )
                else:
                    if last_error is not None:
                        raise last_error

    def is_disabled(self) -> bool:
        """Longitudinal timepoints inherit this geometry from the base template.

        See stages/s00_long_init.py: the base's surfaces are seeded into this
        timepoint, so recomputing them here would discard the shared topology
        that makes cross-timepoint vertex correspondence exact.
        """
        return self.config.longitudinal

    def expected_outputs(self) -> list:
        """Tessellated surface, before topology correction."""
        return [self.hemi_path("orig.nofix")]

    def verify_outputs(self) -> None:
        """Marching-cubes output must be a closed, oriented manifold.

        euler is deliberately unchecked: this surface has not been topology
        corrected yet, so a non-genus-0 result here is expected and normal
        (sub-01's rh was euler -32 at this point and still recoverable).
        """
        super().verify_outputs()
        assert_surface_invariants(
            self.hemi_path("orig.nofix"),
            closed=True,
            oriented=True,
            euler=None,
            context=f"{self.hemi} s08 tessellation",
            strict=self.config.processing.strict_surface_checks,
        )
