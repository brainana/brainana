#!/usr/bin/env python3
"""
Generate per-subject QC reports for a finished (or aborted) pipeline run.

This is invoked from the Nextflow ``workflow.onComplete`` handler so that a
report is ALWAYS produced, regardless of whether the run succeeded or aborted
early. Snapshots are published incrementally during the run, so the report
contains whatever completed before a failure. A run-status section (success vs
early abort + error detail) is injected into every report from --status-file.

Usage:
    python3 generate_reports.py --output-dir DIR --config-file FILE \
        [--status-file STATUS_JSON]

Always exits 0: report generation must never turn a successful run into a
failure, nor mask the real error of an aborted one.
"""

import sys
import json
import argparse
import logging
from pathlib import Path

# Add src/ to path for nhp_mri_prep imports (nextflow_scripts/ -> nhp_mri_prep -> src)
_src_dir = Path(__file__).resolve().parent.parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from nhp_mri_prep.steps.qc import qc_generate_report
from nhp_mri_prep.quality_control.run_status import run_status_log_label
# The merging loader, not utils.nextflow.load_config: main.nf falls back to the
# user's raw --config when the effective config is missing, and a partial file
# must still be read over the package defaults.
from nhp_mri_prep.config.config_io import load_config

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("generate_reports")


def discover_subjects(output_dir: Path) -> list:
    """Find subject IDs to report on.

    Primary source is published ``sub-*`` output directories. Falls back to the
    discovery job lists in ``nextflow_reports/`` so that a run which aborts
    before publishing any subject output still yields a report.
    """
    subjects = {
        p.name[len("sub-") :]
        for p in output_dir.glob("sub-*")
        if p.is_dir() and p.name.startswith("sub-")
    }
    if subjects:
        return sorted(subjects)

    nfr = output_dir / "nextflow_reports"
    for jobs_name in ("anatomical_jobs.json", "functional_jobs.json"):
        jobs_path = nfr / jobs_name
        if not jobs_path.is_file():
            continue
        try:
            with open(jobs_path, encoding="utf-8") as jf:
                jobs = json.load(jf)
            subjects.update(str(j["subject_id"]) for j in jobs if j.get("subject_id"))
        except (OSError, json.JSONDecodeError, TypeError) as e:
            logger.warning("Report: could not read %s - %s", jobs_path, e)

    return sorted(subjects)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config-file", required=True, type=Path)
    parser.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help="JSON file with the pipeline run status (success/abort + error info)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()

    # Load run status (best-effort: a missing/garbled status just omits the section)
    run_status = None
    if args.status_file and args.status_file.is_file():
        try:
            with open(args.status_file, encoding="utf-8") as sf:
                run_status = json.load(sf)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                "Report: could not read status file %s - %s", args.status_file, e
            )

    try:
        config = load_config(str(args.config_file))
    except Exception as e:
        logger.error(
            "Report: could not load config %s - %s; skipping reports",
            args.config_file,
            e,
        )
        return 0

    subjects = discover_subjects(output_dir)
    if not subjects:
        logger.warning(
            "Report: no subjects found under %s; nothing to generate", output_dir
        )
        return 0

    status_label = "unknown"
    if run_status is not None:
        status_label = run_status_log_label(run_status)
    logger.info(
        "Report: generating reports for %d subject(s) [run status: %s]",
        len(subjects),
        status_label,
    )

    for subject_id in subjects:
        snapshot_dir = output_dir / f"sub-{subject_id}" / "figures"
        report_path = output_dir / f"sub-{subject_id}.html"
        try:
            qc_generate_report(
                snapshot_dir=snapshot_dir,
                report_path=report_path,
                config=config,
                snapshot_paths=None,  # Auto-discover from directory
                run_status=run_status,
            )
            logger.info("Report: wrote %s", report_path)
        except Exception as e:
            # Never let one subject's failure stop the others or fail the run.
            logger.warning("Report: failed for sub-%s - %s", subject_id, e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
