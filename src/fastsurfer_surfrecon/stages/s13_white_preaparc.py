"""
Stage 13: White Preaparc Surface

Creates white.preaparc surface for parcellation mapping.
"""

import logging

from .base import HemisphereStage
from .s12b_template_init import frozen_label
from ..processing.template_init import write_rip_label
from ..wrappers.mris import mris_autodet_gwstats, mris_place_surface

logger = logging.getLogger(__name__)


class WhitePreaparc(HemisphereStage):
    """Create white.preaparc surface."""

    name = "white_preaparc"
    description = "White preaparc surface"

    def _run(self) -> None:
        """Create white.preaparc.

        This stage performs two tasks:
        1. Auto-detect gray/white matter statistics (autodet.gw.stats.{hemi}.dat)
        2. Create white.preaparc surface from orig surface

        Note: autodet_stats is created FIRST, even if white.preaparc already exists,
        because it's required by later stages (s15 surface placement) regardless
        of whether white.preaparc needs to be recreated.
        """
        white_preaparc = self.hemi_path("white.preaparc")

        # Step 1: Auto-detect gray/white stats
        # This file is required by stage 15 (surface placement) for placing white
        # and pial surfaces. It must exist even if white.preaparc already exists.
        autodet_stats = self.sdir / f"autodet.gw.stats.{self.hemi}.dat"
        if not autodet_stats.exists():
            logger.info(f"Auto-detecting GW stats for {self.hemi}...")
            # When claustrum fix runs, stats should be computed from the pre-fix
            # intensity volume while placement still uses the fixed volume.
            stats_input_vol = self.sd.mri("brain.finalsurfs_orig.mgz")
            if not stats_input_vol.exists():
                stats_input_vol = self.sd.mri("brain.finalsurfs.mgz")
            logger.info(f"Using stats input volume: {stats_input_vol.name}")
            mris_autodet_gwstats(
                output_stats=autodet_stats,
                input_vol=stats_input_vol,
                wm_vol=self.sd.mri("wm.mgz"),
                # `orig`, not `orig.premesh` (upstream recon-surf.sh uses the
                # premesh here). The two are interchangeable for this purpose --
                # measured on real subjects, gwstats output is byte-identical
                # from either, including on hemispheres where the premesh was
                # non-oriented or where pymeshfix changed the vertex count.
                # `orig` is preferred because it is the surface this stage then
                # places from, and because s12 guarantees it is closed,
                # outward-oriented and genus 0. `orig.premesh` carries no such
                # guarantee, and after a pymeshfix repair it is by definition
                # the *unrepaired* mesh. gwstats samples intensities along
                # surface normals, so an inverted or torn scaffold silently
                # inverts the gray/white estimates rather than failing.
                surface=self.hemi_path("orig"),
                log_file=self.config.log_file,
            )

        # Step 2: Create white.preaparc
        # This is the pre-parcellation white surface used for parcellation mapping
        # in stage 14, and as input for final white surface placement in stage 15.
        logger.info(f"Creating {self.hemi}.white.preaparc...")
        # Place white.preaparc surface using mris_place_surface
        # This creates a white matter surface from the fixed orig surface (after topology fix)
        # Parameters:
        #   - max_cbv_dist=5: Maximum distance for cortical boundary value search
        #   - nsmooth=3: Number of smoothing iterations during placement
        # On a template mesh the frozen label (V1) keeps the template's white
        # surface: --rip-label holds every vertex outside the label it is given,
        # so pass "all vertices except the frozen ones". Note mris_place_surface
        # applies --nsmooth to the whole surface *before* ripping, so the frozen
        # vertices still get that light smoothing (up to ~1 mm).
        rip_label = None
        frozen = frozen_label(self)
        if frozen is not None:
            rip_label = self.hemi_label("template.movable.label")
            n = write_rip_label(rip_label, self.hemi_path("orig"), frozen)
            logger.info("Holding %s at the template (%d vertices movable)", frozen.name, n)
        mris_place_surface(
            input_surf=self.hemi_path("orig"),
            output_surf=white_preaparc,
            hemi=self.hemi,
            wm=self.sd.mri("wm.mgz"),
            invol=self.sd.mri("brain.finalsurfs.mgz"),
            aseg=self.sd.mri("aseg.presurf.mgz"),
            adgws_in=autodet_stats,
            white=True,
            threads=self.threads,
            max_cbv_dist=5,
            nsmooth=3,
            rip_label=rip_label,
            log_file=self.config.log_file,
            subject_dir=self.sd.subject_dir,
            subjects_dir=self.config.subjects_dir,
        )

    def expected_outputs(self) -> list:
        """Pre-parcellation white surface and the GW intensity stats it used."""
        return [
            self.hemi_path("white.preaparc"),
            self.sdir / f"autodet.gw.stats.{self.hemi}.dat",
        ]
