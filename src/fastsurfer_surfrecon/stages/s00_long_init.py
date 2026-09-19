"""
Stage 00: Longitudinal timepoint initialization.

Seeds a longitudinal timepoint from its subject's base template, so that the
timepoint inherits the base's *geometry* while every intensity-derived volume is
still recomputed from the timepoint's own scan.

This is brainana's equivalent of FreeSurfer's ``rca-long-tp-init`` plus
``longmc``. Two facts about that design carry the whole stage:

1. The timepoint is reconstructed **in base space**. ``longmc`` resamples the
   cross-sectional ``orig.mgz`` through the timepoint-to-base transform, after
   which volumes and base surfaces share one geometry. That is why the base's
   surfaces below are *copied* and never transformed, and why vertex
   correspondence across timepoints is exact without any spherical registration.
2. Seeding the base's surfaces satisfies the completion rules of s08-s12, and
   those stages additionally disable themselves under ``config.longitudinal``.
   Placement (s13 onward) then refines the inherited mesh against this
   timepoint's intensities.

What is deliberately *not* seeded matters as much as what is:

- ``norm.mgz`` / ``T1.mgz`` / ``brainmask.mgz`` -- s05 rebuilds these by masking
  *this* timepoint's ``nu.mgz`` with the base's ``mask.mgz``. Seeding them would
  freeze the base's intensities into every timepoint and erase the very signal
  being measured.
- ``filled.mgz`` / ``brain.finalsurfs.mgz`` -- s07 gates the creation of
  ``brain.mgz`` *and* ``brain.finalsurfs.mgz`` behind ``if not filled.exists()``,
  so seeding ``filled.mgz`` (as ``recon-all`` does) would leave
  ``brain.finalsurfs.mgz`` missing and s13/s15 would fail. ``filled.mgz`` is only
  read by s08, which is disabled here, so letting s07 regenerate it is correct.
- ``?h.curv`` -- s14 gates ``recon-all -curvHK`` on ``curv`` being absent;
  seeding it would suppress ``?h.white.H``/``?h.white.K``, which ``-curvstats``
  in s19 needs.
- ``rawavg.mgz`` and ``transforms/talairach.xfm`` -- s01 symlinks the former and
  s04 writes an identity dummy for the latter under ``skip_talairach``.
"""

import json
import logging
import shutil

from .base import PipelineStage, StageOutputError
from ..utils.geometry import describe_geometry_mismatch, volume_geometry
from ..wrappers.longitudinal import mri_convert_apply_lta

logger = logging.getLogger(__name__)

# Volumes copied from the base so every timepoint shares one label scaffold.
# s07 derives wm.mgz/filled.mgz from the aseg via the ColorLUT, so a shared aseg
# yields an identical scaffold across timepoints while the intensity volumes
# stay timepoint-specific.
_BASE_VOLUMES = (
    "mask.mgz",
    "aseg.auto_noCCseg.mgz",
    "aseg.presurf.mgz",
    "aparc+aseg.orig.mgz",
)

# Copied when present. Absent ARM6 simply disables the claustrum fix, exactly as
# in a cross-sectional run.
_BASE_VOLUMES_OPTIONAL = ("aparc.ARM6atlas+aseg.orig.mgz",)

# Per-hemisphere surface seeds, as {destination: source-in-base}.
# The pivot is `orig <- white`: the base's *final* white surface becomes the
# timepoint's starting mesh, which is FreeSurfer's own longitudinal convention
# (rca-long-tp-init copies base surf/?h.white to the timepoint's surf/?h.orig).
_SURFACE_SEEDS = {
    # Satisfies s08 (tessellation).
    "orig.nofix": "orig.nofix",
    # Satisfies s09 (smoothing).
    "smoothwm.nofix": "smoothwm.nofix",
    # Satisfies s11 (spherical projection); s10's hand-written skip rule also
    # ORs on `sphere`, so this covers inflation too.
    "qsphere.nofix": "qsphere.nofix",
    "sphere": "sphere",
    # Satisfies s12 (topology fix).
    "orig": "white",
    "smoothwm": "smoothwm",
    "inflated": "inflated",
    "qsphere": "qsphere",
    # Not required by any skip rule, but they are what lets surface placement
    # anchor each pass to the base under config.longitudinal.
    "orig_white": "white",
    "orig_pial": "pial",
}

# Stages whose geometry the seeds replace. Checked live in verify_outputs() so
# that a future change to any of their completion rules fails loudly here
# instead of silently re-tessellating a base-inherited mesh.
_INHERITED_STAGE_MODULES = (
    ("s08_tessellation", "Tessellation"),
    ("s09_smoothing", "Smoothing"),
    ("s10_inflation", "Inflation"),
    ("s11_spherical_projection", "SphericalProjection"),
    ("s12_topology_fix", "TopologyFix"),
)


class LongTimepointInit(PipelineStage):
    """Seed a longitudinal timepoint from its within-subject base template."""

    name = "long_init"
    description = "Longitudinal init (seed timepoint from base template)"

    # ------------------------------------------------------------------
    # Skip / disable logic
    # ------------------------------------------------------------------

    def is_disabled(self) -> bool:
        """Only runs for longitudinal timepoints."""
        return not self.config.longitudinal

    def expected_outputs(self) -> list:
        """Everything this stage guarantees, so a partial run is not cached.

        `long.base` is written first and the surface seeds last, so declaring
        both ends means an interrupted seeding cannot look complete.
        """
        outputs = [
            self.sd.scripts_dir / "long.base",
            self.sd.mri("orig.mgz"),
        ]
        outputs += [self.sd.mri(v) for v in _BASE_VOLUMES]
        for hemi in ("lh", "rh"):
            outputs += [self.sd.hemi_surf(hemi, dst) for dst in _SURFACE_SEEDS]
        return outputs

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _run(self) -> None:
        base_dir = self.config.base_subject_dir
        cross_dir = self.config.cross_subject_dir
        base_id = self.config.base_subject_id
        cross_id = self.config.cross_subject_id
        lta = self.config.tp_to_base_lta

        # The config validator guarantees these, but this stage is the one place
        # that would silently produce a broken tree if they were absent.
        if base_dir is None or cross_dir is None or lta is None:
            raise ValueError(
                "LongTimepointInit requires base_subject_id, cross_subject_id "
                "and tp_to_base_lta on the config"
            )

        provenance: dict[str, object] = {
            "base_subject_id": base_id,
            "cross_subject_id": cross_id,
            "tp_to_base_lta": str(lta),
            "seeded": {},
        }

        # 1. Bookkeeping. These files, not the directory name, are what marks a
        #    tree as a longitudinal timepoint -- for FreeSurfer's long_* tools
        #    and for anyone reading the directory later.
        self.sd.scripts_dir.mkdir(parents=True, exist_ok=True)
        (self.sd.scripts_dir / "long.base").write_text(f"{base_id}\n")
        (self.sd.scripts_dir / "long.cross").write_text(f"{cross_id}\n")
        base_tps = base_dir / "scripts" / "base-tps"
        if base_tps.exists():
            shutil.copy2(base_tps, self.sd.scripts_dir / "long.base-tps")
        else:
            logger.warning(
                "%s not found; the timepoint list of the base is not recorded "
                "in this timepoint's provenance",
                base_tps,
            )

        # 2. Put this timepoint's volume into base space (longmc equivalent).
        cross_orig = cross_dir / "mri" / "orig.mgz"
        if not cross_orig.exists():
            raise FileNotFoundError(
                f"Cross-sectional orig.mgz not found at {cross_orig}. The "
                f"cross-sectional reconstruction of {cross_id} must complete "
                "before its longitudinal timepoint can be initialized."
            )
        self.sd.mri_dir.mkdir(parents=True, exist_ok=True)
        mri_convert_apply_lta(
            input_vol=cross_orig,
            output_vol=self.sd.mri("orig.mgz"),
            lta=lta,
            odt="uchar",
            resample="cubic",
            log_file=self.config.log_file,
            cmd_log_file=self.config.cmd_log_file,
        )
        provenance["orig_from"] = str(cross_orig)

        # The one check that stands between a mis-targeted LTA and silently
        # misaligned surfaces. `mri_convert -at` makes the output adopt the
        # *LTA's destination* geometry, while everything copied in below -- the
        # asegs, and every surface in _SURFACE_SEEDS -- is on the *base's*
        # geometry. If those two disagree, nothing downstream notices:
        # collect_change_stats would still fit clean per-vertex rates, and they
        # would be measuring the wrong vertices.
        #
        # Before step 3 on purpose, so a bad tree fails with nothing seeded
        # rather than half-seeded. Two real volumes are compared rather than the
        # LTA's header, so this also catches a transform mis-wired by the
        # workflow's staging, not just one built on the wrong grid.
        base_orig = base_dir / "mri" / "orig.mgz"
        if not base_orig.exists():
            raise FileNotFoundError(
                f"Base template is missing mri/orig.mgz (looked in {base_orig}); "
                f"the base reconstruction of {base_id} must complete before its "
                "timepoints can be initialized."
            )
        mismatch = describe_geometry_mismatch(
            self.sd.mri("orig.mgz"),
            base_orig,
            labels=(
                "this timepoint, resampled through the LTA",
                f"base {base_id}",
            ),
        )
        if mismatch:
            raise StageOutputError(
                f"{lta} does not target the base's voxel grid.\n\n"
                + mismatch
                + "\n\nThe surfaces and segmentations about to be inherited are "
                "defined on the base's grid, so this timepoint would be "
                "misaligned against its own mesh by exactly this difference -- "
                "and nothing downstream would report it. The transform must come "
                "from the build_base_template run that produced this base, and "
                "the base's orig.mgz must not have been re-gridded since."
            )
        shape, zooms = volume_geometry(base_orig)
        provenance["geometry_checked_against"] = str(base_orig)
        provenance["grid"] = {"shape": list(shape), "zooms": list(zooms)}

        # 3. Inherit the label scaffold from the base. Both trees are in base
        #    space, so these are straight copies -- no resampling.
        atlas_seg = f"aparc.{self.config.atlas.name}atlas+aseg.orig.mgz"
        required = list(_BASE_VOLUMES)
        if atlas_seg not in required:
            required.append(atlas_seg)
        for name in required:
            src = base_dir / "mri" / name
            if not src.exists():
                raise FileNotFoundError(
                    f"Base template is missing {name} (looked in {src}). The "
                    f"base reconstruction of {base_id} must complete before its "
                    "timepoints can be initialized."
                )
            dst = self.sd.mri(name)
            shutil.copy2(src, dst, follow_symlinks=True)
            provenance["seeded"][f"mri/{name}"] = str(src)

        for name in _BASE_VOLUMES_OPTIONAL:
            src = base_dir / "mri" / name
            if src.exists():
                shutil.copy2(src, self.sd.mri(name), follow_symlinks=True)
                provenance["seeded"][f"mri/{name}"] = str(src)
            else:
                logger.info("Base has no %s; skipping (optional)", name)

        # FreeSurfer keeps the base aseg under a base-stamped name for
        # provenance. Nothing in brainana reads it; it is written so the tree is
        # recognisable to FreeSurfer users and tools.
        base_aseg = base_dir / "mri" / "aseg.mgz"
        if base_aseg.exists():
            shutil.copy2(
                base_aseg,
                self.sd.mri(f"aseg_{base_id}.mgz"),
                follow_symlinks=True,
            )

        # 4. Inherit the mesh. Copied, not transformed -- see the module
        #    docstring.
        self.sd.surf_dir.mkdir(parents=True, exist_ok=True)
        for hemi in ("lh", "rh"):
            for dst_name, src_name in _SURFACE_SEEDS.items():
                src = base_dir / "surf" / f"{hemi}.{src_name}"
                if not src.exists():
                    raise FileNotFoundError(
                        f"Base template is missing surf/{hemi}.{src_name} "
                        f"(looked in {src}), needed to seed "
                        f"{hemi}.{dst_name} for this timepoint."
                    )
                dst = self.sd.hemi_surf(hemi, dst_name)
                shutil.copy2(src, dst, follow_symlinks=True)
                provenance["seeded"][f"surf/{hemi}.{dst_name}"] = str(src)

        # 5. Record what was seeded from where. An inherited surface is
        #    otherwise indistinguishable from a computed one.
        (self.sd.scripts_dir / "long_init.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        logger.info(
            "Seeded %d file(s) from base %s", len(provenance["seeded"]), base_id
        )

    def verify_outputs(self) -> None:
        """Postcondition: outputs exist AND s08-s12 will not re-run.

        The second half is the important one. Seeding is only correct if it
        actually prevents the geometry stages from running, and that depends on
        their completion rules, which live in those stages. So query them live
        rather than trusting the seed table in this module to stay in sync.
        """
        super().verify_outputs()

        import importlib

        # Deliberately NOT `is_disabled() or should_skip()`. is_disabled() for
        # these stages is config.longitudinal, which is always True while this
        # stage is running, so including it would make the check vacuous. What
        # must hold is that the seeds satisfy each stage's own completion rule --
        # that is what a resumed run and the surface QC step depend on, and it is
        # what would silently break if a stage's expected_outputs() gained an
        # entry the seed table does not cover.
        problems: list[str] = []
        for module_name, class_name in _INHERITED_STAGE_MODULES:
            module = importlib.import_module(f".{module_name}", __package__)
            stage_cls = getattr(module, class_name)
            for hemi in ("lh", "rh"):
                stage = stage_cls(self.config, self.sd, hemi)
                if stage.should_skip():
                    continue
                missing = [str(p) for p in stage.expected_outputs() if not p.exists()]
                problems.append(
                    f"{class_name} ({hemi}) is not satisfied by the seeds"
                    + (f"; missing: {', '.join(missing)}" if missing else "")
                )

        if problems:
            raise StageOutputError(
                "Longitudinal seeding is incomplete -- the geometry these "
                "stages would otherwise compute is not fully inherited from "
                "the base:\n  " + "\n  ".join(problems)
            )
