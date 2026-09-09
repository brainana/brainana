"""
Stage 15: Surface Placement

Places white and pial surfaces.
"""

import logging
import shutil

from ..processing.surface_fix import assert_surface_invariants
from .base import HemisphereStage
from ..io.surface import convert_fs_surface_to_gifti
from ..wrappers.mris import mris_place_surface

logger = logging.getLogger(__name__)


class SurfacePlacement(HemisphereStage):
    """Place white and pial surfaces."""

    name = "surface_placement"
    description = "White and pial surface placement"

    def _run(self) -> None:
        """Place white and pial surfaces.

        This stage performs the final surface placement:
        1. Place white surface from white.preaparc
        2. Place pial surface from white surface

        The white and pial surfaces are the final cortical boundaries used for
        statistics and analysis.
        """
        white = self.hemi_path("white")
        pial = self.hemi_path("pial")
        pial_t1 = self.hemi_path("pial.T1")

        # Determine which parcellation annotation to use for surface placement
        # This helps guide surface placement by providing cortical region information
        if self.config.processing.fsaparc:
            aparc = self.hemi_label("aparc.annot")  # FreeSurfer aparc
        else:
            aparc = self.hemi_label(
                f"aparc.{self.config.atlas.name}atlas.mapped.annot"
            )  # Mapped atlas parcellation

        # Get cortex labels for surface placement
        # cortex.label: cortical ribbon (used for white surface)
        # cortex+hipamyg.label: cortical ribbon + hippocampus + amygdala (used for pial surface)
        cortex_label = self.hemi_label("cortex.label")
        cortex_hipamyg_label = self.hemi_label("cortex+hipamyg.label")

        # Longitudinal timepoints anchor both passes to the base template's
        # surfaces and cap how far vertices may travel, exactly as
        # `recon-all -long` does (recon-all:4224-4235 for white,
        # :4289-4304 for pial). That anchoring is what converts a shared mesh
        # into reduced across-timepoint variance; without it the timepoints
        # inherit the topology but drift freely, and the variance reduction --
        # the entire point of the longitudinal stream -- is much smaller.
        #
        # Note it trades bias for variance: constraining movement toward the
        # base also damps genuine change. That is the right trade in the regime
        # this stream is designed for (near-static brains, where true change is
        # small), and it is why the behaviour is gated on config.longitudinal
        # rather than applied unconditionally.
        longitudinal = self.config.longitudinal
        long_max_cbv_dist = self.config.long_max_cbv_dist

        # Step 1: Place white surface
        # The white surface is placed from white.preaparc, which was created in stage 13.
        # This is the final white matter surface boundary.
        if not white.exists():
            logger.info(f"Placing {self.hemi} white surface...")
            # FreeSurfer starts the longitudinal white pass from the base's
            # white (copied to orig_white by stage 00) rather than this
            # timepoint's white.preaparc, and follows it with --rip-surf.
            white_input = self.hemi_path("white.preaparc")
            if longitudinal:
                orig_white = self.hemi_path("orig_white")
                if orig_white.exists():
                    white_input = orig_white
                else:
                    logger.warning(
                        "longitudinal=True but %s is missing; falling back to "
                        "white.preaparc. Surfaces will not be anchored to the "
                        "base and across-timepoint variance will be higher.",
                        orig_white,
                    )
            mris_place_surface(
                input_surf=white_input,
                output_surf=white,
                hemi=self.hemi,
                wm=self.sd.mri("wm.mgz"),
                invol=self.sd.mri("brain.finalsurfs.mgz"),
                aseg=self.sd.mri("aseg.presurf.mgz"),
                adgws_in=self.sdir / f"autodet.gw.stats.{self.hemi}.dat",
                white=True,
                threads=self.threads,
                rip_label=cortex_label,
                rip_bg=True,
                rip_surf=white_input,
                aparc=aparc if aparc.exists() else None,
                max_cbv_dist=long_max_cbv_dist if longitudinal else None,
                log_file=self.config.log_file,
                subject_dir=self.sd.subject_dir,
                subjects_dir=self.config.subjects_dir,
            )

        # Step 2: Place pial surface
        # The pial surface is placed from the white surface, extending outward to the
        # pial boundary. It uses cortex+hipamyg.label to include hippocampus and amygdala
        # regions. The pial surface is initially created as pial.T1, then copied to pial.
        if not pial_t1.exists():
            logger.info(f"Placing {self.hemi} pial surface...")
            # As above, but the longitudinal pial pass additionally blends a
            # quarter of the way toward this timepoint's own white surface
            # (recon-all:4304). repulse_surf / white_surf stay the timepoint's
            # white either way -- only the starting surface changes.
            pial_input = white
            blend_surf = None
            if longitudinal:
                orig_pial = self.hemi_path("orig_pial")
                if orig_pial.exists():
                    pial_input = orig_pial
                    blend_surf = (self.config.long_pial_blend_weight, white)
                else:
                    logger.warning(
                        "longitudinal=True but %s is missing; starting the "
                        "pial pass from this timepoint's white instead.",
                        orig_pial,
                    )
            mris_place_surface(
                input_surf=pial_input,
                output_surf=pial_t1,
                hemi=self.hemi,
                wm=self.sd.mri("wm.mgz"),
                invol=self.sd.mri("brain.finalsurfs.mgz"),
                aseg=self.sd.mri("aseg.presurf.mgz"),
                adgws_in=self.sdir / f"autodet.gw.stats.{self.hemi}.dat",
                pial=True,
                threads=self.threads,
                rip_label=cortex_hipamyg_label
                if cortex_hipamyg_label.exists()
                else cortex_label,
                pin_medial_wall=cortex_label,  # Pin medial wall to prevent expansion
                repulse_surf=white,  # Repulse from white surface
                white_surf=white,  # Reference white surface
                aparc=aparc if aparc.exists() else None,
                max_cbv_dist=long_max_cbv_dist if longitudinal else None,
                blend_surf=blend_surf,
                log_file=self.config.log_file,
                subject_dir=self.sd.subject_dir,
                subjects_dir=self.config.subjects_dir,
            )

        # Copy pial.T1 to pial (standard naming convention)
        if not pial.exists():
            if pial_t1.exists():
                shutil.copy(pial_t1, pial)
            else:
                raise FileNotFoundError(f"pial.T1 not found for {self.hemi}")

        # Create GIFTI surfaces for downstream QC (with CRAS-applied coordinates).
        # Guard on the binary surface existing: on a resume over an older subject tree,
        # `inflated` may be absent (s12/s14 skip when their primary outputs already exist),
        # and converting a missing surface would crash mris_convert.
        for surf_name in ("white", "pial", "inflated"):
            in_surf = self.hemi_path(surf_name)
            out_gii = self.hemi_path(f"{surf_name}.surf.gii")
            if not out_gii.exists():
                if not in_surf.exists():
                    logger.warning(
                        f"{self.hemi}.{surf_name} surface not found; skipping GIFTI conversion."
                    )
                    continue
                convert_fs_surface_to_gifti(in_surf, out_gii, apply_cras=True)

    def expected_outputs(self) -> list:
        """Final surfaces plus their GIFTI counterparts."""
        return [
            self.hemi_path("white"),
            self.hemi_path("pial"),
            self.hemi_path("inflated"),
            self.hemi_path("white.surf.gii"),
            self.hemi_path("pial.surf.gii"),
            self.hemi_path("inflated.surf.gii"),
        ]

    def verify_outputs(self) -> None:
        """white and pial inherit orig's connectivity from mris_place_surface.

        Given the s12 gate on orig, a violation here means placement itself
        damaged the mesh. Warn-only by default: these are the surfaces most
        likely to have historically tolerated deviations, so the cohort audit
        should inform whether to make them strict.
        """
        super().verify_outputs()
        for name in ("white", "pial"):
            assert_surface_invariants(
                self.hemi_path(name),
                closed=True,
                oriented=True,
                euler=2,
                context=f"{self.hemi} s15 {name}",
                strict=self.config.processing.strict_surface_checks,
            )
