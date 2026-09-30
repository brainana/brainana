"""
Stage 12: Topology Fix

Fixes topological defects in surface.
"""

import json
import logging
import shutil

from .base import HemisphereStage
from ..wrappers.base import FreeSurferError
from ..wrappers.mris import mris_fix_topology, mris_remove_intersection
from ..wrappers.mris import mris_smooth, mris_inflate
from ..processing.surface_fix import (
    SurfaceInvariantError,
    assert_surface_invariants,
    fix_surface_orientation,
    validate_surface,
)
from ..processing.spherical import spherically_project_surface
from ..processing.topology_fix import repair_surface_pymeshfix

logger = logging.getLogger(__name__)

# pymeshfix repairs a defect by deleting the faces around it and filling the
# hole. Orientation slips cost a few dozen vertices (~0.1 %); a repair that
# drops more than this is cutting away cortex, and the non-GA search is tried
# before such a mesh is accepted. The case that set it: a -ga premesh that came
# back open (euler 0) lost 4 % of its vertices to pymeshfix -- the occipital
# pole -- where the same hemisphere's clean -ga run lost none.
MAX_PYMESHFIX_VERTEX_LOSS = 0.02

# Overlays mris_fix_topology writes next to the premesh. A second (non-GA) run
# overwrites them, so the GA run's copies are kept for when GA's mesh is used.
_DEFECT_OVERLAYS = ("defect_labels", "defect_borders", "defect_chull")


class TopologyFix(HemisphereStage):
    """Fix topological defects."""

    name = "topology_fix"
    description = "Topology fix (fix)"

    def _run(self) -> None:
        """Fix topological defects in surface.

        This stage performs a multi-step topology correction process:
        1. Fix topology using mris_fix_topology (creates orig.premesh)
        2. Copy premesh to orig (if needed)
        3. Remove surface intersections
        4. Clean up temporary files (inflated.nofix)
        5. Fix surface orientation
        6. Recreate smoothwm from fixed orig
        7. Recreate inflated from smoothwm
        8. Recreate sphere from smoothwm

        The topology fix is critical for ensuring the surface has correct topology
        (genus 0, no handles) required for spherical mapping and parcellation.
        """
        orig = self.hemi_path("orig")
        # What this run actually did, written to scripts/<hemi>.topology_fix.json
        # and folded into surface_qc.json. "mode" stays None when a resume
        # reuses an existing premesh, so the record never claims a path it did
        # not observe.
        self._record = {"mode": None, "pymeshfix": False, "flipped_inside_out": False}
        self._prior = self._read_record()
        # Prepare inputs for topology fix
        # FreeSurfer's mris_fix_topology expects qsphere.nofix as input.
        # Our spectral projection (stage 11) creates qsphere.nofix directly, but this
        # fallback handles edge cases (e.g., if using FreeSurfer qsphere or legacy data
        # where only sphere exists).
        qsphere_nofix = self.hemi_path("qsphere.nofix")
        sphere = self.hemi_path("sphere")
        if not qsphere_nofix.exists() and sphere.exists():
            logger.info(
                f"Creating {self.hemi}.qsphere.nofix from {self.hemi}.sphere (fallback for FreeSurfer qsphere)"
            )
            shutil.copy(sphere, qsphere_nofix)

        inflated_nofix = self.hemi_path("inflated.nofix")
        orig_nofix = self.hemi_path("orig.nofix")

        logger.info(f"Fixing topology for {self.hemi}...")

        # Step 1: Fix topology using mris_fix_topology
        # This command identifies and fixes topological defects (handles, holes) in the surface.
        # It uses the spherical representation (qsphere.nofix) and inflated surface to guide
        # the topology correction, with the genetic-algorithm search (-ga) unless it fails
        # or processing.topology_fix_ga is off (see _fix_topology).
        # Output: orig.premesh (preliminary mesh with fixed topology)
        premesh = self.hemi_path("orig.premesh")
        premesh_resumed = premesh.exists()
        if not premesh_resumed:
            # Inputs are verified here, not at the top of the stage, because they
            # are needed *only* by mris_fix_topology. Step 4 below deletes
            # inflated.nofix once it has been consumed, so an unconditional check
            # would make every resume of this stage fail on a file the stage
            # itself removed.
            #   - qsphere.nofix: stage 11 (spherical projection)
            #   - inflated.nofix: stage 10 (inflation)
            #   - orig.nofix:     stage 08 (tessellation)
            if not qsphere_nofix.exists():
                raise FileNotFoundError(
                    f"{self.hemi}.qsphere.nofix not found. "
                    "This should be created in stage 11 (spherical_projection)."
                )
            if not inflated_nofix.exists():
                raise FileNotFoundError(
                    f"{self.hemi}.inflated.nofix not found. It is created in "
                    "stage 10 (inflation) and consumed here. If you are re-running "
                    f"this stage, delete {self.hemi}.orig.premesh and re-run stage 10."
                )
            if not orig_nofix.exists():
                raise FileNotFoundError(
                    f"{self.hemi}.orig.nofix not found. "
                    "This should be created in stage 08 (tessellation)."
                )

            self._record["mode"] = self._fix_topology(
                qsphere_nofix, inflated_nofix, orig_nofix, premesh
            )
            rescue_eligible = self._record["mode"] == "ga"
        else:
            # Resume. The rescue needs the stage inputs, which step 4 deletes,
            # so it can only still run when the earlier run died before step 4.
            # It is skipped when that run already compared the two searches
            # ("ga" is in its record) or did not use -ga at all.
            prior = self._prior or {}
            rescue_eligible = "ga" not in prior and (
                prior.get("mode") == "ga"
                or (prior.get("mode") is None and self.config.processing.topology_fix_ga)
            )

        if (self._prior or {}).get("mode") == "no_ga_rescue" and premesh_resumed:
            # An earlier run already chose the non-GA mesh. Re-deriving from
            # orig.premesh (the GA mesh) would promote the very surface that
            # rescue rejected, so take the non-GA mesh back instead.
            premesh_for_orig = self._resume_no_ga_rescue()
        else:
            # A -ga premesh that is open or not genus 0, or that pymeshfix can
            # only rescue by deleting a sizeable piece of it, is a failed GA
            # search rather than a small slip; the non-GA search on the same
            # inputs usually fixes those defects without losing cortex, so both
            # are tried and the mesh that keeps more of the surface wins.
            premesh_for_orig, ga_info, ga_error = self._try_repair(
                premesh, self.hemi_path("orig.premesh.pymeshfix")
            )
            if rescue_eligible and self._ga_failed(ga_info, ga_error):
                premesh_for_orig = self._no_ga_rescue(
                    premesh_for_orig,
                    ga_info,
                    ga_error,
                    qsphere_nofix,
                    inflated_nofix,
                    orig_nofix,
                )
            elif ga_error is not None:
                raise ga_error
            else:
                self._record["pymeshfix"] = ga_info["pymeshfix"]
                self._record["pymeshfix_vertex_loss"] = ga_info[
                    "pymeshfix_vertex_loss"
                ]

        # A premesh can be closed, consistently wound and genus 0 and still be
        # entirely inside-out -- the no--ga fallback above has produced exactly
        # that. The pymeshfix predicate does not look at the sign, so such a
        # mesh reaches the gate unrepaired and the gate (which requires outward
        # normals) aborts the hemisphere over something fix_surface_orientation
        # corrects losslessly. Flip it here; the gate below stays the authority.
        if fix_surface_orientation(
            surface_path=premesh_for_orig,
            backup_path=premesh_for_orig.with_name(
                premesh_for_orig.name + ".insideout"
            ),
        ):
            logger.warning(f"{self.hemi} premesh was inside-out; flipped before gate")
            self._record["flipped_inside_out"] = True

        # Gate: nothing defective may become orig. mris_place_surface preserves
        # connectivity, so white/pial inherit this mesh's topology exactly --
        # which makes this the single highest-value check in the pipeline.
        assert_surface_invariants(
            premesh_for_orig,
            closed=True,
            oriented=True,
            euler=2,
            context=f"{self.hemi} pre-orig",
        )
        self._write_record()

        # Step 2: Copy premesh to orig (final fixed surface)
        # The premesh (or pymeshfix result) is the topology-fixed version that becomes the final orig surface.
        #
        # Always re-copy rather than skipping when orig exists. orig is written
        # here at step 2 of 8, so a run that died later leaves an orig that does
        # not correspond to premesh_for_orig. Tracking whether it was rewritten
        # lets the later steps know their inputs changed.
        orig_regenerated = False
        if not orig.exists() or not self._same_file(premesh_for_orig, orig):
            logger.info(f"Copying {premesh_for_orig.name} to {self.hemi}.orig...")
            shutil.copy(premesh_for_orig, orig)
            orig_regenerated = True

        # Step 3: Remove surface intersections
        # Even after topology fix, the surface may have self-intersections.
        # This step removes any remaining intersections to ensure a clean surface.
        #
        # Guarded on orig_regenerated: this is an in-place, non-idempotent
        # operation, so re-running it on an already-processed orig would keep
        # eroding the surface on every resume.
        if orig_regenerated:
            logger.info(f"Removing intersections from {self.hemi}.orig...")
            mris_remove_intersection(
                input_surf=orig,
                output_surf=orig,  # In-place operation
                log_file=self.config.log_file,
                subject_dir=self.sd.subject_dir,
            )
        else:
            logger.info(
                f"{self.hemi}.orig already current; skipping intersection removal"
            )

        # Step 4: Clean up temporary files
        # inflated.nofix is no longer needed after topology fix (it was only needed as input).
        # This matches recon-all behavior and frees up disk space.
        inflated_nofix = self.hemi_path("inflated.nofix")
        if inflated_nofix.exists():
            logger.info(
                f"Removing {self.hemi}.inflated.nofix (no longer needed after fix)"
            )
            inflated_nofix.unlink()

        # Step 5: Fix surface orientation
        # Ensure the surface has correct vertex ordering (consistent normal direction).
        # This creates a backup (orig.noorient) before fixing if needed.
        fix_surface_orientation(
            surface_path=orig,
            backup_path=self.hemi_path("orig.noorient"),
        )

        # Step 6: re-create smoothwm from fixed orig after topology fix
        # Regenerated whenever orig changed: reusing a smoothwm derived from a
        # superseded orig silently mixes two different meshes.
        smoothwm = self.hemi_path("smoothwm")
        if orig_regenerated or not smoothwm.exists():
            logger.info(
                f"Creating {self.hemi}.smoothwm from fixed {self.hemi}.orig (smooth, {self.config.processing.smooth_iterations} iterations)..."
            )
            mris_smooth(
                input_surf=orig,
                output_surf=smoothwm,
                n_iterations=self.config.processing.smooth_iterations,
                nw=True,
                seed=1234,
                log_file=self.config.log_file,
                subject_dir=self.sd.subject_dir,
            )

        # Step 7: re-create inflated from smoothwm after topology fix
        inflated = self.hemi_path("inflated")
        if orig_regenerated or not inflated.exists():
            logger.info(
                f"Creating {self.hemi}.inflated from {self.hemi}.smoothwm (inflate2, {self.config.processing.inflate2_iterations or 'default'} iterations)..."
            )
            mris_inflate(
                input_surf=smoothwm,
                output_surf=inflated,
                n_iterations=self.config.processing.inflate_iterations,
                no_save_sulc=False,  # Save sulc file for visualization
                log_file=self.config.log_file,
                subject_dir=self.sd.subject_dir,
            )

        # Step 8: Re-create sphere from smoothwm after topology fix
        # So sphere has the same vertex count as orig/smoothwm/white/pial (post-fix mesh).
        smoothwm = self.hemi_path("smoothwm")
        sphere = self.hemi_path("sphere")
        qsphere = self.hemi_path("qsphere")

        logger.info(
            f"Re-creating {self.hemi}.sphere from {self.hemi}.smoothwm (post-topology-fix)..."
        )
        spherically_project_surface(
            input_path=smoothwm,
            output_path=sphere,
            threads=self.threads,
        )
        shutil.copy(sphere, qsphere)

    def _try_repair(self, premesh, pymeshfix_out):
        """Make ``premesh`` a clean genus-0 sphere, with pymeshfix if needed.

        Returns ``(path, info, error)``: the clean mesh, a record of what the
        repair cost, and ``None`` -- or ``(None, info, error)`` when pymeshfix
        did not converge, so the caller can still try another premesh.

        The predicate is closed AND oriented AND euler == 2, not euler alone:
        a mesh with one triangle wound backwards is closed with euler 2 but not
        oriented, and mris_fix_topology can leave boundary edges behind.
        """
        info = validate_surface(premesh)
        logger.info(
            "%s: V=%d F=%d closed=%s oriented=%s euler=%s",
            premesh.name,
            info["n_vertices"],
            info["n_faces"],
            info["is_closed"],
            info["is_oriented"],
            info["euler"],
        )
        record = {
            "premesh_vertices": info["n_vertices"],
            "premesh_closed": info["is_closed"],
            "premesh_euler": info["euler"],
            "vertices": info["n_vertices"],
            "pymeshfix": False,
            "pymeshfix_vertex_loss": 0.0,
        }
        if info["is_closed"] and info["is_oriented"] and info["euler"] == 2:
            logger.info(f"{premesh.name} topology OK, skipping pymeshfix")
            return premesh, record, None

        max_iterations = 5
        logger.warning(
            f"{premesh.name} has defective topology. "
            f"Running pymeshfix up to {max_iterations} iterations..."
        )
        current_input = premesh
        repaired = None
        for iteration in range(max_iterations):
            # Use temp output when input and output would be the same path
            if current_input.resolve() == pymeshfix_out.resolve():
                output_path = pymeshfix_out.parent / (pymeshfix_out.name + ".tmp")
            else:
                output_path = pymeshfix_out
            repaired = repair_surface_pymeshfix(current_input, output_path)
            if output_path.suffix == ".tmp":
                shutil.move(output_path, pymeshfix_out)
                repaired = validate_surface(pymeshfix_out)
            logger.info(
                "  Iteration %d: V=%d F=%d closed=%s oriented=%s euler=%s",
                iteration + 1,
                repaired["n_vertices"],
                repaired["n_faces"],
                repaired["is_closed"],
                repaired["is_oriented"],
                repaired["euler"],
            )
            if (
                repaired["is_closed"]
                and repaired["is_oriented"]
                and repaired["euler"] == 2
            ):
                loss = 1.0 - repaired["n_vertices"] / info["n_vertices"]
                logger.info(
                    f"  Topology corrected after {iteration + 1} iteration(s), "
                    f"{info['n_vertices'] - repaired['n_vertices']} vertices "
                    f"removed ({loss:.1%})"
                )
                record.update(
                    vertices=repaired["n_vertices"],
                    pymeshfix=True,
                    pymeshfix_vertex_loss=round(loss, 4),
                )
                return pymeshfix_out, record, None
            current_input = pymeshfix_out

        # Never promote a still-defective mesh to orig. Everything downstream
        # (surface placement, parcellation, morphometry) inherits this mesh's
        # connectivity, so continuing only moves the failure somewhere less
        # diagnosable.
        error = SurfaceInvariantError(
            pymeshfix_out,
            repaired or {},
            ["topology repair did not converge"],
            context=f"{self.hemi} pymeshfix, {max_iterations} iterations",
        )
        return None, record, error

    @staticmethod
    def _ga_failed(info, error) -> bool:
        """True when the -ga result is a failed search, not a small slip."""
        return (
            error is not None
            or not info["premesh_closed"]
            or info["premesh_euler"] != 2
            or info["pymeshfix_vertex_loss"] > MAX_PYMESHFIX_VERTEX_LOSS
        )

    def _no_ga_rescue(
        self, ga_path, ga_info, ga_error, qsphere_nofix, inflated_nofix, orig_nofix
    ):
        """Run the non-GA search too and keep whichever mesh retains more surface."""
        self._record["ga"] = ga_info
        if not (
            qsphere_nofix.exists() and inflated_nofix.exists() and orig_nofix.exists()
        ):
            logger.warning(
                f"{self.hemi}: -ga result looks like a failed search but the "
                "mris_fix_topology inputs are gone; keeping it"
            )
            if ga_error is not None:
                raise ga_error
            return ga_path

        logger.warning(
            f"{self.hemi}: -ga premesh is a failed search (closed="
            f"{ga_info['premesh_closed']}, euler={ga_info['premesh_euler']}, "
            f"pymeshfix removed {ga_info['pymeshfix_vertex_loss']:.1%}); "
            "trying mris_fix_topology without -ga"
        )
        saved = {}
        for name in _DEFECT_OVERLAYS:
            overlay = self.hemi_path(name)
            if overlay.exists():
                saved[overlay] = overlay.with_name(overlay.name + ".ga")
                shutil.copy(overlay, saved[overlay])

        noga_premesh = self.hemi_path("orig.premesh.noga")
        noga_path, noga_info, noga_error = None, None, None
        try:
            mris_fix_topology(
                **self._fix_topology_kwargs(
                    qsphere_nofix, inflated_nofix, orig_nofix, noga_premesh
                ),
                ga=False,
            )
            noga_path, noga_info, noga_error = self._try_repair(
                noga_premesh, self.hemi_path("orig.premesh.noga.pymeshfix")
            )
        except FreeSurferError as e:
            logger.warning(f"{self.hemi}: mris_fix_topology without -ga failed ({e})")
        if noga_info is not None:
            self._record["no_ga"] = noga_info

        use_noga = (
            noga_error is None
            and noga_path is not None
            and (ga_error is not None or noga_info["vertices"] > ga_info["vertices"])
        )
        if use_noga:
            logger.warning(
                f"{self.hemi}: using the non-GA mesh ({noga_info['vertices']} "
                f"vertices vs {ga_info['vertices']} from -ga)"
            )
            self._record["mode"] = "no_ga_rescue"
            self._record["pymeshfix"] = noga_info["pymeshfix"]
            for kept in saved.values():
                kept.unlink(missing_ok=True)
            return noga_path

        # GA's mesh stays: put its defect overlays back.
        for overlay, kept in saved.items():
            shutil.move(kept, overlay)
        if ga_error is not None:
            raise ga_error
        logger.warning(
            f"{self.hemi}: keeping the -ga mesh ({ga_info['vertices']} vertices"
            + (
                f" vs {noga_info['vertices']} without -ga)"
                if noga_info
                else "; the non-GA search did not produce a usable mesh)"
            )
        )
        self._record["pymeshfix"] = ga_info["pymeshfix"]
        return ga_path

    def _fix_topology_kwargs(self, qsphere_nofix, inflated_nofix, orig_nofix, premesh):
        return dict(
            subject=self.config.subject_id,
            hemi=self.hemi,
            sphere=qsphere_nofix,
            inflated=inflated_nofix,
            orig=orig_nofix,
            output_premesh=premesh,
            mgz=True,
            seed=1234,  # Fixed seed for reproducibility
            log_file=self.config.log_file,
            subjects_dir=self.config.subjects_dir,
        )

    def _fix_topology(self, qsphere_nofix, inflated_nofix, orig_nofix, premesh) -> str:
        """Run mris_fix_topology, falling back to the non-GA search if -ga fails.

        FreeSurfer 7.4.1's genetic-algorithm search can abort outright on some
        large defects ("stack smashing detected"), taking the hemisphere with
        it, while the default search fixes the same defects. The mesh either
        path produces is held to the same pre-orig gate, so the fallback trades
        GA's (usually better) patch choice for a surface instead of none.

        Returns the path taken: "ga", "no_ga_fallback" or "no_ga".
        """
        kwargs = self._fix_topology_kwargs(
            qsphere_nofix, inflated_nofix, orig_nofix, premesh
        )
        if not self.config.processing.topology_fix_ga:
            logger.info(f"Running mris_fix_topology for {self.hemi} (-ga disabled)...")
            mris_fix_topology(ga=False, **kwargs)
            return "no_ga"

        logger.info(f"Running mris_fix_topology -ga for {self.hemi}...")
        try:
            mris_fix_topology(ga=True, **kwargs)
            return "ga"
        except FreeSurferError as e:
            logger.warning(
                f"{self.hemi}: mris_fix_topology -ga failed ({e}); retrying without -ga"
            )
            premesh.unlink(missing_ok=True)
            mris_fix_topology(ga=False, **kwargs)
            return "no_ga_fallback"

    def _resume_no_ga_rescue(self):
        """Return the non-GA mesh an earlier run chose, re-validated."""
        noga_premesh = self.hemi_path("orig.premesh.noga")
        if not noga_premesh.exists():
            raise FileNotFoundError(
                f"{self.hemi}.topology_fix.json says the non-GA mesh was chosen, "
                f"but {noga_premesh.name} is gone. Delete {self.hemi}.orig.premesh "
                "and re-run from stage 10 to redo the topology fix."
            )
        path, info, error = self._try_repair(
            noga_premesh, self.hemi_path("orig.premesh.noga.pymeshfix")
        )
        if error is not None:
            raise error
        for key in ("mode", "ga", "no_ga"):
            if key in self._prior:
                self._record[key] = self._prior[key]
        self._record["pymeshfix"] = info["pymeshfix"]
        return path

    def _record_path(self):
        return self.sd.scripts_dir / f"{self.hemi}.topology_fix.json"

    def _read_record(self):
        """The record an earlier run of this stage left, or None."""
        try:
            return json.loads(self._record_path().read_text())
        except (OSError, ValueError):
            return None

    def _write_record(self) -> None:
        """Record which topology path was taken (read by surface_qc.json)."""
        path = self._record_path()
        if self._record["mode"] is None and self._prior:
            # Resumed with an existing premesh: keep the mode the earlier run
            # observed rather than overwrite it with "unknown".
            self._record["mode"] = self._prior.get("mode")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._record, indent=2))

    @staticmethod
    def _same_file(a, b) -> bool:
        """True if two paths hold identical bytes (cheap size check first)."""
        if not (a.exists() and b.exists()):
            return False
        if a.stat().st_size != b.stat().st_size:
            return False
        return a.read_bytes() == b.read_bytes()

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
        """Everything this stage guarantees on success.

        Note `sphere` is deliberately absent: stage 11 also writes it, so
        including it here would let stage 11's output satisfy this stage's skip
        check. `qsphere` is written last by this stage and by no other, which
        makes it the honest completion marker.
        """
        return [
            self.hemi_path("orig"),
            self.hemi_path("smoothwm"),
            self.hemi_path("inflated"),
            self.hemi_path("qsphere"),
        ]

    def verify_outputs(self) -> None:
        """Postcondition: outputs exist AND orig is a clean genus-0 sphere."""
        super().verify_outputs()
        assert_surface_invariants(
            self.hemi_path("orig"),
            closed=True,
            oriented=True,
            # outward matters as much as consistent: pymeshfix can return a
            # consistently-wound but inverted mesh, which every other topology
            # check passes and which silently inverts normal-based sampling
            # downstream (e.g. the gray/white intensity estimates in s13).
            outward=True,
            euler=2,
            context=f"{self.hemi} s12 output",
        )
