"""
Main pipeline orchestrator for FastSurfer surface reconstruction.

Coordinates the execution of all pipeline stages.
"""

from typing import Optional, Callable
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import ReconSurfConfig
from .io.subjects_dir import SubjectsDir
from .stages import (
    LongTimepointInit,
    # Volume stages
    VolumePrep,
    BiasCorrection,
    MaskAseg,
    Talairach,
    NormT1,
    CCSegmentation,
    WMFilled,
    ClaustrumFix,
    # Surface stages
    Tessellation,
    Smoothing,
    Inflation,
    SphericalProjection,
    TopologyFix,
    WhitePreaparc,
    Parcellation,
    SurfacePlacement,
    ComputeMorphometry,
    Registration,
    Statistics,
    CorticalRibbon,
    AsegRefinement,
    AparcMapping,
    WMParcMapping,
)

logger = logging.getLogger(__name__)


class ReconSurfPipeline:
    """
    Main surface reconstruction pipeline.

    Orchestrates the execution of all pipeline stages, handling
    parallelization, logging, and error recovery.

    Parameters
    ----------
    config : ReconSurfConfig
        Pipeline configuration

    Examples
    --------
    >>> config = ReconSurfConfig(
    ...     subject_id="sub-001",
    ...     subjects_dir=Path("/data/subjects"),
    ...     t1_input=Path("/data/input/T1.mgz"),
    ...     segmentation=Path("/data/input/aparc+aseg.mgz"),
    ... )
    >>> pipeline = ReconSurfPipeline(config)
    >>> pipeline.run()
    """

    def __init__(self, config: ReconSurfConfig):
        self.config = config
        self.sd = SubjectsDir(config.subjects_dir, config.subject_id)

        # Timing
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def run(self) -> None:
        """
        Run the complete surface reconstruction pipeline.

        Executes all phases:
        1. Volume preprocessing
        2. Surface creation (per hemisphere)
        3. Statistics and finalization
        """
        self.start_time = datetime.now()
        logger.info(f"Starting recon-surf for {self.config.subject_id}")
        logger.info(f"Start time: {self.start_time}")

        # Setup directories
        self.sd.setup()
        self._setup_logging()

        try:
            # Phase 1: Volume Processing
            self._run_volume_phase()

            # Phase 2: Surface Creation (per hemisphere)
            self._run_surface_phase()

            # Phase 3: Statistics and Finalization
            self._run_stats_phase()

            # Write done file
            self._write_done_file()

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

        self.end_time = datetime.now()
        elapsed = self.end_time - self.start_time
        logger.info(f"Completed recon-surf for {self.config.subject_id}")
        logger.info(f"Total time: {elapsed}")

    def _setup_logging(self) -> None:
        """Set up file logging."""
        if self.config.log_file:
            log_path = self.config.log_file
        else:
            log_path = self.sd.log_file

        # Add file handler
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logging.getLogger("fastsurfer_surfrecon").addHandler(file_handler)

        # Write header
        with open(log_path, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write("FastSurfer Recon Pipeline Log\n")
            f.write(f"Subject: {self.config.subject_id}\n")
            f.write(f"Start: {self.start_time}\n")
            f.write(f"{'='*80}\n\n")

        # Initialize cmd log file (fastsurfer_recon.cmd on disk)
        cmd_log_path = self.config.cmd_log_file
        cmd_log_path.parent.mkdir(parents=True, exist_ok=True)
        from .wrappers.base import set_cmd_log_file

        with open(cmd_log_path, "a") as f:
            timestamp = datetime.now().strftime("%a %b %d %H:%M:%S %Z %Y")
            f.write("\n\n#---------------------------------\n")
            f.write(f"# New invocation of fastsurfer-recon {timestamp} \n")
            f.write("#--------------------------------------------\n")
        # Set global cmd log file so all commands are logged
        set_cmd_log_file(cmd_log_path)

    # ------------------------------------------------------------------
    # Stage ordering. These are the single source of truth: the phase
    # methods below and the driver scripts in scripts/ all build their
    # stage lists from here, so a stage added in one place cannot silently
    # go missing from another.
    # ------------------------------------------------------------------

    def volume_stages(self) -> list:
        """Volume-processing stages, in execution order (s00-s07b).

        LongTimepointInit is first and disables itself unless
        config.longitudinal, so a cross-sectional run is unaffected. It must
        stay first: ordering.py derives VOLUME_STEPS from stage filenames, so
        "s00" sorts ahead of "s01" and test_volume_stage_order_matches_step_list
        asserts this list agrees with that order.
        """
        return [
            LongTimepointInit(self.config, self.sd),
            VolumePrep(self.config, self.sd),
            BiasCorrection(self.config, self.sd),
            MaskAseg(self.config, self.sd),
            Talairach(self.config, self.sd),
            NormT1(self.config, self.sd),
            CCSegmentation(self.config, self.sd),
            WMFilled(self.config, self.sd),
            ClaustrumFix(self.config, self.sd),
        ]

    def surface_stages(self, hemi: str) -> list:
        """Per-hemisphere surface stages, in execution order (s08-s17).

        Statistics is deliberately excluded: mris_anatomical_stats needs both
        hemispheres' pial surfaces, so it runs sequentially after both
        hemispheres finish.
        """
        return [
            Tessellation(self.config, self.sd, hemi),
            Smoothing(self.config, self.sd, hemi),
            Inflation(self.config, self.sd, hemi),
            SphericalProjection(self.config, self.sd, hemi),
            TopologyFix(self.config, self.sd, hemi),
            WhitePreaparc(self.config, self.sd, hemi),
            Parcellation(self.config, self.sd, hemi),
            SurfacePlacement(self.config, self.sd, hemi),
            ComputeMorphometry(self.config, self.sd, hemi),
            Registration(self.config, self.sd, hemi),
        ]

    def post_surface_stages(self) -> list:
        """Volume stages that need both hemispheres and the ribbon (s20-s22)."""
        return [
            AsegRefinement(self.config, self.sd),
            AparcMapping(self.config, self.sd),
            WMParcMapping(self.config, self.sd),
        ]

    def _run_volume_phase(self) -> None:
        """
        Volume preprocessing phase.

        Stages:
        - s01: Volume preparation (orig, conformance check)
        - s02: Bias correction (N4)
        - s03: Mask and aseg processing
        - s04: Talairach registration (optional)
        - s05: Normalization and T1 creation
        - s06: CC segmentation (optional)
        - s07: WM segmentation and filled creation
        - s07b: Claustrum fix (optional, runs only if ARM6 atlas is present)
        """
        logger.info("=" * 60)
        logger.info("Phase 1: Volume Processing")
        logger.info("=" * 60)

        for stage in self.volume_stages():
            stage.run()

    def _run_surface_phase(self) -> None:
        """
        Surface creation phase (per hemisphere).

        Stages:
        - s08: Tessellation
        - s09: Smoothing
        - s10: Inflation
        - s11: Spherical projection
        - s12: Topology fix (optional)
        - s13: White preaparc
        - s14: Parcellation mapping
        - s15: Surface placement
        - s16: Compute morphometry
        - s17: Registration (optional)
        - s18: Cortical ribbon
        - s19: Statistics
        """
        logger.info("=" * 60)
        logger.info("Phase 2: Surface Creation")
        logger.info("=" * 60)

        def process_hemisphere(hemi: str) -> None:
            """Process a single hemisphere."""
            logger.info(f"Processing hemisphere: {hemi}")

            for stage in self.surface_stages(hemi):
                stage.run()

        hemis = ["lh", "rh"]

        # Run hemispheres
        if (
            self.config.processing.parallel_hemis
            and self.config.processing.threads >= 2
        ):
            logger.info("Running hemispheres in parallel")
            self._run_parallel(process_hemisphere, hemis)
        else:
            logger.info("Running hemispheres sequentially")
            for hemi in hemis:
                process_hemisphere(hemi)

        # Cortical ribbon must be created before statistics because brainvol.stats
        # needs ribbon.mgz to compute cortical gray matter volume
        logger.info("Creating cortical ribbon (needs both hemispheres' surfaces)")
        CorticalRibbon(self.config, self.sd).run()

        # Statistics must run sequentially after cortical ribbon is created
        # because brainvol.stats needs ribbon.mgz for cortical volume computation
        logger.info("Computing statistics for both hemispheres (sequential)")
        for hemi in hemis:
            Statistics(self.config, self.sd, hemi).run()

        # Post-surface volume stages (need both hemispheres' surfaces and ribbon)
        for stage in self.post_surface_stages():
            stage.run()

        self._write_surface_qc(hemis)

    def _write_surface_qc(self, hemis: list) -> None:
        """Record the topology of every key surface to scripts/surface_qc.json.

        Gives a per-subject record of what the meshes actually looked like,
        which is what makes a cohort audit possible -- e.g. deciding whether
        strict_surface_checks can be turned on by default without rejecting
        subjects that have always been considered successful.

        Never fails the run: this is diagnostics, not a gate.
        """
        import json

        from .processing.surface_fix import validate_surface

        try:
            report = {"subject": self.config.subject_id, "surfaces": {}}
            for hemi in hemis:
                for name in (
                    "orig.nofix",
                    "smoothwm.nofix",
                    "orig.premesh",
                    "orig",
                    "white",
                    "pial",
                ):
                    path = self.sd.surf_dir / f"{hemi}.{name}"
                    if path.exists():
                        report["surfaces"][f"{hemi}.{name}"] = validate_surface(path)

            out = self.sd.scripts_dir / "surface_qc.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2))
            logger.info(f"Wrote surface QC report: {out}")
        except Exception as e:
            logger.warning(f"Could not write surface QC report: {e}")

    def _run_stats_phase(self) -> None:
        """
        Statistics and finalization phase.

        Note: Statistics are computed per-hemisphere in the surface phase.
        This phase handles any final cross-hemisphere operations.
        """
        logger.info("=" * 60)
        logger.info("Phase 3: Finalization")
        logger.info("=" * 60)

        # Statistics are already computed per-hemisphere in surface phase
        # This phase can be used for cross-hemisphere operations if needed
        logger.info("Statistics computed per hemisphere in surface phase")

    def _run_parallel(
        self,
        func: Callable[[str], None],
        items: list[str],
    ) -> None:
        """
        Run a function on multiple items in parallel.

        Uses ThreadPoolExecutor to process items concurrently. If any item
        fails, the exception is logged and re-raised, stopping all processing.

        Parameters
        ----------
        func : callable
            Function to call for each item. Must accept a single string argument.
        items : list[str]
            Items to process (e.g., hemisphere names like ['lh', 'rh'])

        Raises
        ------
        Exception
            Re-raises any exception from item processing, stopping all parallel tasks.
        """
        # Collect every item's outcome before raising.
        #
        # Raising from inside the `with` does not cancel the other workers --
        # ThreadPoolExecutor.__exit__ calls shutdown(wait=True), so they run to
        # completion anyway and their results (or their own exceptions) were
        # simply discarded. That made a right-hemisphere failure look like it
        # had also destroyed the left hemisphere's work, and hid the second
        # error entirely when both failed.
        failures: dict = {}

        with ThreadPoolExecutor(max_workers=len(items)) as executor:
            futures = {executor.submit(func, item): item for item in items}

            for future in as_completed(futures):
                item = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing {item}: {e}")
                    failures[item] = e

        for item in items:
            status = "FAILED" if item in failures else "ok"
            logger.info("  %-4s %s", str(item), status)

        if failures:
            summary = "; ".join(
                f"{item}: {type(exc).__name__}: {exc}" for item, exc in failures.items()
            )
            succeeded = [i for i in items if i not in failures]
            raise RuntimeError(
                f"{len(failures)} of {len(items)} items failed -- {summary}"
                + (
                    f" (completed: {', '.join(map(str, succeeded))})"
                    if succeeded
                    else ""
                )
            ) from next(iter(failures.values()))

    def _write_done_file(self) -> None:
        """Write the done file with run information."""
        import os
        import platform

        done_path = self.sd.done_file

        # Get username safely
        try:
            username = os.getlogin()
        except OSError:
            username = os.environ.get("USER", str(os.getuid()))

        elapsed = self.end_time - self.start_time if self.end_time else None
        elapsed_hours = elapsed.total_seconds() / 3600 if elapsed else 0

        content = f"""------------------------------
SUBJECT {self.config.subject_id}
START_TIME {self.start_time}
END_TIME {self.end_time}
RUNTIME_HOURS {elapsed_hours:.3f}
USER {username}
HOST {platform.node()}
PROCESSOR {platform.machine()}
OS {platform.system()}
UNAME {platform.platform()}
VERSION fastsurfer-recon
CMDPATH fastsurfer-recon
"""

        done_path.write_text(content)
        logger.info(f"Wrote done file: {done_path}")
