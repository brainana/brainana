"""
Stage 12: Topology Fix

Fixes topological defects in surface.
"""

import logging
import shutil

from .base import HemisphereStage
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
        # the topology correction. The -ga flag enables automatic genus adjustment.
        # Output: orig.premesh (preliminary mesh with fixed topology)
        premesh = self.hemi_path("orig.premesh")
        if not premesh.exists():
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

            logger.info(f"Running mris_fix_topology for {self.hemi}...")
            mris_fix_topology(
                subject=self.config.subject_id,
                hemi=self.hemi,
                sphere=qsphere_nofix,
                inflated=inflated_nofix,
                orig=orig_nofix,
                output_premesh=premesh,
                mgz=True,  # Use mgz format for volumes
                ga=True,  # Enable automatic genus adjustment
                seed=1234,  # Fixed seed for reproducibility
                log_file=self.config.log_file,
                subjects_dir=self.config.subjects_dir,
            )

        # If the premesh is not a clean genus-0 sphere, run pymeshfix iteratively
        # (max 5) to close boundary edges and fix orientation.
        #
        # The predicate is closed AND oriented AND euler == 2, not euler alone.
        # Euler is not sufficient: a mesh with one triangle wound backwards is
        # closed with euler 2 but is not oriented, and mris_fix_topology can
        # leave boundary edges behind. All three are computed in-process, so
        # there is no path where the check silently does not happen.
        premesh_for_orig = premesh
        info = validate_surface(premesh)
        logger.info(
            "%s.orig.premesh: V=%d F=%d closed=%s oriented=%s euler=%s",
            self.hemi,
            info["n_vertices"],
            info["n_faces"],
            info["is_closed"],
            info["is_oriented"],
            info["euler"],
        )

        if not (info["is_closed"] and info["is_oriented"] and info["euler"] == 2):
            premesh_pymeshfix = self.hemi_path("orig.premesh.pymeshfix")
            max_iterations = 5
            logger.warning(
                f"{self.hemi} premesh has defective topology. "
                f"Running pymeshfix up to {max_iterations} iterations..."
            )
            current_input = premesh
            repaired = None
            for iteration in range(max_iterations):
                # Use temp output when input and output would be the same path
                if current_input.resolve() == premesh_pymeshfix.resolve():
                    output_path = premesh_pymeshfix.parent / (
                        premesh_pymeshfix.name + ".tmp"
                    )
                else:
                    output_path = premesh_pymeshfix
                repaired = repair_surface_pymeshfix(current_input, output_path)
                if output_path.suffix == ".tmp":
                    shutil.move(output_path, premesh_pymeshfix)
                    repaired = validate_surface(premesh_pymeshfix)
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
                    logger.info(
                        f"  Topology corrected after {iteration + 1} iteration(s)"
                    )
                    premesh_for_orig = premesh_pymeshfix
                    break
                current_input = premesh_pymeshfix
            else:
                # Do not promote a still-defective mesh to orig. Everything
                # downstream (surface placement, parcellation, morphometry)
                # inherits this mesh's connectivity, so continuing here only
                # moves the failure somewhere less diagnosable.
                raise SurfaceInvariantError(
                    premesh_pymeshfix,
                    repaired or {},
                    ["topology repair did not converge"],
                    context=f"{self.hemi} pymeshfix, {max_iterations} iterations",
                )
        else:
            logger.info(
                f"{self.hemi} premesh topology OK (closed, oriented, euler=2), "
                "skipping pymeshfix"
            )

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

    @staticmethod
    def _same_file(a, b) -> bool:
        """True if two paths hold identical bytes (cheap size check first)."""
        if not (a.exists() and b.exists()):
            return False
        if a.stat().st_size != b.stat().st_size:
            return False
        return a.read_bytes() == b.read_bytes()

    def is_disabled(self) -> bool:
        """Longitudinal timepoints inherit this geometry from the base template.

        See stages/s00_long_init.py: the base's surfaces are seeded into this
        timepoint, so recomputing them here would discard the shared topology
        that makes cross-timepoint vertex correspondence exact.
        """
        return self.config.longitudinal

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
