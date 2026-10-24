"""
Stage 14: Parcellation Mapping

Maps volume labels to surface parcellation.
"""

import logging

import nibabel.freesurfer.io as fs

from .base import HemisphereStage
from ..processing.inflate2_visualization import apply_inflate2_visualization
from ..processing.parcellation import sample_parcellation, smooth_aparc_files
from ..wrappers.base import run_recon_all

logger = logging.getLogger(__name__)


class Parcellation(HemisphereStage):
    """Map volume labels to surface."""

    name = "parcellation"
    description = "Surface parcellation mapping"

    def _run(self) -> None:
        """Map parcellation to surface."""
        # Create cortex label
        cortex_label = self.hemi_label("cortex.label")
        if not cortex_label.exists():
            logger.info(f"Creating {self.hemi}.cortex.label...")
            flags = []
            if self.config.hires:
                flags.append("-hires")
            run_recon_all(
                subject=self.config.subject_id,
                hemi=self.hemi,
                steps=["-cortex-label"],
                flags=flags,
                threads=self.threads,
                log_file=self.config.log_file,
                subjects_dir=self.config.subjects_dir,
            )

        # Create inflated and curvHK surfaces
        inflated = self.hemi_path("inflated")
        curv = self.hemi_path("curv")
        if not inflated.exists() or not curv.exists():
            logger.info(f"Creating inflated and curvHK for {self.hemi}...")
            flags = []
            if self.config.hires:
                flags.append("-hires")
            run_recon_all(
                subject=self.config.subject_id,
                hemi=self.hemi,
                steps=["-curvHK"],
                flags=flags,
                threads=self.threads,
                log_file=self.config.log_file,
                subjects_dir=self.config.subjects_dir,
            )

        # Map parcellation
        aparc_mapped = self.hemi_label(
            f"aparc.{self.config.atlas.name}atlas.mapped.annot"
        )
        logger.info(f"Mapping parcellation to {self.hemi} surface...")

        # Get lookup tables
        seg_lut = self.config.atlas.get_hemi_lut(self.hemi)
        surf_lut = self.config.atlas.get_lut()

        # Load surface once for reuse
        white_preaparc_path = self.hemi_path("white.preaparc")
        logger.info(f"Loading surface: {white_preaparc_path}")
        surface_data = fs.read_geometry(white_preaparc_path, read_metadata=True)
        surface = (surface_data[0], surface_data[1])

        # Sample parcellation
        aparc_prefix = self.hemi_label(
            f"aparc.{self.config.atlas.name}atlas.mapped.prefix.annot"
        )
        aseg_orig = self.sd.mri(f"aparc.{self.config.atlas.name}atlas+aseg.orig.mgz")

        sample_parcellation(
            surface_path=white_preaparc_path,
            segmentation_path=aseg_orig,
            cortex_path=cortex_label,
            output_path=aparc_prefix,
            volume_lut=seg_lut,
            surface_lut=surf_lut,
            proj_mm=0.6,
            search_radius=2.0,
            surface=surface,
        )

        # Smooth parcellation (reuse the same surface)
        smooth_aparc_files(
            insurf=white_preaparc_path,
            inaparc=aparc_prefix,
            incort=cortex_label,
            outaparc=aparc_mapped,
            surface=surface,
        )

        # Optional: Apply additional smoothing and inflation adjustments for visualization
        #
        # This section allows fine-tuning of smoothwm and inflated surfaces for non-human
        # data (e.g., monkey data) where different smoothing/inflation parameters may be
        # needed for optimal visualization. This is applied AFTER the initial smooth2/inflate2
        # steps in stage 12 (topology fix).
        #
        # IMPORTANT: These files are replaced in-place. This is safe because:
        #   - smoothwm and inflated are created in stage 12 (topology fix)
        #   - This stage (14) runs after stage 12
        #   - Stage 15 (surface placement) uses white.preaparc, not smoothwm/inflated
        #   - These are visualization surfaces, not used for surface placement
        #
        apply_inflate2_visualization(self.config, self.sd, self.hemi)

    def should_skip(self) -> bool:
        """Skip if mapped parcellation exists.

        Kept as a custom check rather than expected_outputs(): the later
        visualisation block rewrites `smoothwm` and `inflated` *in place*
        rather than creating new files, so declaring them would not
        distinguish a complete run from one that died before that block.
        Known limitation -- a crash between the annot write and the viz block
        leaves this stage looking complete.
        """
        aparc_mapped = self.hemi_label(
            f"aparc.{self.config.atlas.name}atlas.mapped.annot"
        )
        return aparc_mapped.exists()
