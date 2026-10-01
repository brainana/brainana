"""
Stage 12b: Template-initialised surface

Replaces s08-s12 (tessellation .. topology fix) when config.template_init is on:
the template's white surface, carried into the subject by the subject-to-template
registration, becomes this subject's orig. The template mesh is closed and genus 0,
so no topology fix is needed, and the subject shares the template's vertex
numbering. That shared numbering is what lets s13 and s15 hold the template's
freeze label (V1) at the template.

Also writes the files s08-s12 would have left behind (orig.nofix, smoothwm.nofix,
qsphere.nofix), so a longitudinal timepoint can be seeded from a
template-initialised base exactly as from a tessellated one (see s00_long_init).
"""

import json
import logging
import shutil

from .base import HemisphereStage
from ..processing.surface_fix import assert_surface_invariants, fix_surface_orientation
from ..processing.template_init import fraction_in_mask, warp_template_surface
from ..wrappers.mris import mris_inflate, mris_remove_intersection, mris_smooth

logger = logging.getLogger(__name__)

# Below this share of warped vertices inside mask.mgz the registration is
# suspect. A fitted white surface is wholly inside the mask (1.0 on every devtest
# reconstruction); the warped template sits a little off the fit, so allow some.
MIN_FRACTION_IN_MASK = 0.95


def freeze_label_name(config) -> str | None:
    """Label file (in the subject's label/ dir, without hemi) holding the frozen vertices."""
    if not config.template_freeze_label:
        return None
    return f"template.{config.template_freeze_label}.label"


def frozen_label(stage):
    """This hemisphere's freeze label if white-surface placement must hold it, else None.

    Held when the mesh is the template's: a template-initialised run (s12b wrote
    the label) or a longitudinal timepoint seeded from such a base (s00 copied
    it). Gated on those modes, not on the file alone, so a stale label in a
    reused tree cannot freeze a tessellated mesh whose vertex numbering it does
    not match.
    """
    name = freeze_label_name(stage.config)
    if not name or not (stage.config.template_init or stage.config.longitudinal):
        return None
    path = stage.hemi_label(name)
    return path if path.exists() else None


class TemplateInit(HemisphereStage):
    """Build orig/smoothwm/inflated/sphere from the template's white surface."""

    name = "template_init"
    description = "Template-initialised surface (instead of s08-s12)"

    def _run(self) -> None:
        tpl = self.config.template_subject_dir
        orig_nofix = self.hemi_path("orig.nofix")
        orig = self.hemi_path("orig")

        # 1. Template white surface, carried into this subject.
        logger.info(
            "Warping %s/surf/%s.white into the subject with %s",
            tpl, self.hemi, self.config.template_xfm,
        )
        warp_template_surface(
            template_surf=tpl / "surf" / f"{self.hemi}.white",
            xfm=self.config.template_xfm,
            subject_orig_mgz=self.sd.mri("orig.mgz"),
            out_surf=orig_nofix,
            log_file=self.config.log_file,
            post=self.config.template_xfm_post,
        )

        # A misregistered template is still a valid mesh, and V1 is then held in
        # the wrong place with nothing to pull it back. Placement would not
        # correct it, so say so here rather than leave it for the QC figures.
        in_mask = fraction_in_mask(orig_nofix, self.sd.mri("mask.mgz"))
        if in_mask < MIN_FRACTION_IN_MASK:
            logger.warning(
                "%s: only %.1f%% of the warped template white surface lies inside "
                "mask.mgz (expected >= %.0f%%); check the subject-to-template "
                "registration (%s)",
                self.hemi, 100 * in_mask, 100 * MIN_FRACTION_IN_MASK, self.config.template_xfm,
            )
        else:
            logger.info("%s: %.1f%% of the warped template lies inside mask.mgz", self.hemi, 100 * in_mask)

        # 2. A smooth warp preserves the template's topology, but it can bring
        #    opposite banks of a tight sulcus through each other.
        mris_remove_intersection(
            input_surf=orig_nofix,
            output_surf=orig,
            log_file=self.config.log_file,
            subject_dir=self.sd.subject_dir,
        )
        fix_surface_orientation(surface_path=orig, backup_path=self.hemi_path("orig.noorient"))

        # 3. Visualisation / inflation surfaces, as s12 makes them from its orig.
        smoothwm = self.hemi_path("smoothwm")
        mris_smooth(
            input_surf=orig,
            output_surf=smoothwm,
            n_iterations=self.config.processing.smooth_iterations,
            nw=True,
            seed=1234,
            log_file=self.config.log_file,
            subject_dir=self.sd.subject_dir,
        )
        shutil.copy2(smoothwm, self.hemi_path("smoothwm.nofix"))
        mris_inflate(
            input_surf=smoothwm,
            output_surf=self.hemi_path("inflated"),
            n_iterations=self.config.processing.inflate_iterations,
            no_save_sulc=False,
            log_file=self.config.log_file,
            subject_dir=self.sd.subject_dir,
        )

        # 4. The spherical map needs no computing: the mesh is the template's.
        tpl_sphere = tpl / "surf" / f"{self.hemi}.sphere"
        for name in ("sphere", "qsphere", "qsphere.nofix"):
            shutil.copy2(tpl_sphere, self.hemi_path(name))

        # 5. The vertices s13/s15 hold at the template (vertex indices are shared).
        label = freeze_label_name(self.config)
        if label:
            shutil.copy2(
                tpl / "label" / f"{self.hemi}.{self.config.template_freeze_label}.label",
                self.hemi_label(label),
            )

        (self.sd.scripts_dir / f"{self.hemi}.template_init.json").write_text(
            json.dumps(
                {
                    "template_subject_dir": str(tpl),
                    "template_xfm": str(self.config.template_xfm),
                    "template_xfm_post": (
                        str(self.config.template_xfm_post) if self.config.template_xfm_post else None
                    ),
                    "freeze_label": self.config.template_freeze_label,
                    "warped_fraction_in_mask": round(in_mask, 4),
                    "registration_suspect": in_mask < MIN_FRACTION_IN_MASK,
                },
                indent=2,
            )
            + "\n"
        )

    def is_disabled(self) -> bool:
        """Only for template-initialised runs; timepoints inherit the base's mesh (s00)."""
        return not self.config.template_init or self.config.longitudinal

    def expected_outputs(self) -> list:
        """The files s13 onward (and s00 seeding from this run as a base) rely on."""
        out = [
            self.hemi_path("orig"),
            self.hemi_path("orig.nofix"),
            self.hemi_path("smoothwm"),
            self.hemi_path("smoothwm.nofix"),
            self.hemi_path("inflated"),
            self.hemi_path("sphere"),
            self.hemi_path("qsphere"),
            self.hemi_path("qsphere.nofix"),
        ]
        label = freeze_label_name(self.config)
        if label:
            out.append(self.hemi_label(label))
        return out

    def verify_outputs(self) -> None:
        """The template mesh is closed, oriented and genus 0; the warp must keep it so."""
        super().verify_outputs()
        assert_surface_invariants(
            self.hemi_path("orig"),
            closed=True,
            oriented=True,
            euler=2,
            context=f"{self.hemi} s12b orig",
            strict=self.config.processing.strict_surface_checks,
        )
