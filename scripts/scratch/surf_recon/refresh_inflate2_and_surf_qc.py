#!/usr/bin/env python3
"""
Refresh stage-14 inflate2 visualization surfaces, corticalSurfAndMeasures QC,
and tSNR QC (surface panel uses inflated meshes).

For flat BIDS datasets (subject dirs at dataset root, not site-*).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

_src_dir = Path(__file__).resolve().parents[3] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from fastsurfer_surfrecon.config import ReconSurfConfig
from fastsurfer_surfrecon.io.subjects_dir import SubjectsDir
from fastsurfer_surfrecon.io.surface import convert_fs_surface_to_gifti
from fastsurfer_surfrecon.processing.inflate2_visualization import (
    apply_inflate2_visualization,
)
from nhp_mri_prep.quality_control.dataset_report import (
    DATASET_REPORT_NAME,
    write_dataset_report,
)
from nhp_mri_prep.quality_control.reports import build_report_data, write_html_report
from nhp_mri_prep.steps.qc import qc_cortical_surf_and_measures, qc_tsnr
from nhp_mri_prep.steps.tsnr import project_tsnr_to_surface
from nhp_mri_prep.utils.bids import create_bids_filename, parse_bids_entities
from nhp_mri_prep.utils.nextflow import load_config

logger = logging.getLogger(__name__)

CORTICAL_QC_GLOB = "*_desc-corticalSurfAndMeasures_T1w.png"
SESSION_AVG_TSNR_GLOB = "*_space-T1w_desc-preproc_stat-tsnr_boldmap.nii.gz"


def list_subjects(dataset_root: Path) -> list[Path]:
    return sorted(p for p in dataset_root.glob("sub-*") if p.is_dir())


def pick_cortical_qc_path(figures_dir: Path) -> Path:
    matches = sorted(figures_dir.glob(CORTICAL_QC_GLOB))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No cortical QC PNG matching {CORTICAL_QC_GLOB} in {figures_dir}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple cortical QC PNGs in {figures_dir}: "
            + ", ".join(m.name for m in matches)
        )
    return matches[0]


def backup_file_if_exists(path: Path, dry_run: bool) -> str | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}_pre_inflate2refresh{path.suffix}")
    if backup.exists():
        return "SKIP_ALREADY_BACKED_UP"
    if dry_run:
        return f"DRY_RUN copy {path} -> {backup}"
    shutil.copy2(path, backup)
    return f"Copied {path} -> {backup}"


def backup_png_if_exists(png_path: Path, dry_run: bool) -> str | None:
    if not png_path.exists():
        return None
    backup_png = png_path.with_name(f"{png_path.stem}_todelete{png_path.suffix}")
    if backup_png.exists():
        return "SKIP_ALREADY_BACKED_UP"
    if dry_run:
        return f"DRY_RUN move {png_path} -> {backup_png}"
    shutil.move(png_path, backup_png)
    return f"Moved {png_path} -> {backup_png}"


def atlas_name_from_config(config: dict) -> str:
    return (
        config.get("anat", {})
        .get("skullstripping_segmentation", {})
        .get("atlas_name", "ARM2")
    )


def validate_subject(fs_sub_dir: Path) -> None:
    wmparc = fs_sub_dir / "mri" / "wmparc.mgz"
    if not wmparc.is_file():
        raise FileNotFoundError(f"Missing complete surf recon marker: {wmparc}")
    surf = fs_sub_dir / "surf"
    for hemi in ("lh", "rh"):
        for name in ("white.preaparc", "inflated"):
            p = surf / f"{hemi}.{name}"
            if not p.is_file():
                raise FileNotFoundError(f"Missing {p}")


def discover_session_avg_tsnr_volumes(sub_dir: Path) -> list[tuple[str, Path]]:
    """Return (session_id, nii_path) for session-average T1w tSNR maps."""
    found: list[tuple[str, Path]] = []
    for ses_dir in sorted(sub_dir.glob("ses-*")):
        if not ses_dir.is_dir():
            continue
        session_id = ses_dir.name.replace("ses-", "", 1)
        func_dir = ses_dir / "func"
        if not func_dir.is_dir():
            continue
        for nii in sorted(func_dir.glob(SESSION_AVG_TSNR_GLOB)):
            if "task-" in nii.name or "_run-" in nii.name:
                continue
            found.append((session_id, nii))
    return found


def pick_tsnr_qc_path(figures_dir: Path, session_id: str) -> Path | None:
    pattern = f"*_ses-{session_id}_desc-tSNR_bold.png"
    matches = sorted(figures_dir.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple tSNR QC PNGs for ses-{session_id} in {figures_dir}: "
            + ", ".join(m.name for m in matches)
        )
    return None


def tsnr_qc_output_path(session_avg_nii: Path, figures_dir: Path) -> Path:
    """BIDS QC basename aligned with QC_TSNR in modules/qc.nf."""
    parsed = parse_bids_entities(session_avg_nii.name)
    entities: dict[str, str] = {}
    for key in ("sub", "ses", "space"):
        if key in parsed:
            entities[key] = parsed[key]
    entities["desc"] = "tSNR"
    return figures_dir / create_bids_filename(entities, "bold", extension=".png")


def refresh_tsnr_qc_for_subject(
    sub_dir: Path,
    fs_sub_dir: Path,
    config: dict,
    dry_run: bool,
) -> int:
    """Re-project session tSNR to surface and rewrite tSNR QC PNGs. Returns count refreshed."""
    figures_dir = sub_dir / "figures"
    sessions = discover_session_avg_tsnr_volumes(sub_dir)
    if not sessions:
        return 0

    count = 0
    for session_id, session_avg_nii in sessions:
        qc_path = pick_tsnr_qc_path(figures_dir, session_id)
        if qc_path is None:
            qc_path = tsnr_qc_output_path(session_avg_nii, figures_dir)
            logger.info(
                "%s ses-%s: no existing tSNR QC PNG; will write %s",
                sub_dir.name,
                session_id,
                qc_path.name,
            )
        elif not qc_path.exists():
            qc_path = tsnr_qc_output_path(session_avg_nii, figures_dir)

        backup_msg = backup_png_if_exists(qc_path, dry_run)
        if backup_msg:
            logger.info("%s ses-%s: %s", sub_dir.name, session_id, backup_msg)

        if dry_run:
            logger.info(
                "DRY_RUN would refresh tSNR QC for %s ses-%s",
                sub_dir.name,
                session_id,
            )
            count += 1
            continue

        figures_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tsnr_surf_") as tmp:
            work = Path(tmp)
            proj = project_tsnr_to_surface(session_avg_nii, fs_sub_dir, work)
            lh_gii = proj.get("lh_gii")
            rh_gii = proj.get("rh_gii")
            if proj.get("skipped") or not lh_gii or not rh_gii:
                logger.warning(
                    "%s ses-%s: tSNR surface projection skipped (%s)",
                    sub_dir.name,
                    session_id,
                    proj.get("reason", "unknown"),
                )
                lh_path = None
                rh_path = None
            else:
                lh_path = Path(lh_gii)
                rh_path = Path(rh_gii)

            qc_tsnr(
                session_tsnr_vol=session_avg_nii,
                lh_surf_gii=lh_path,
                rh_surf_gii=rh_path,
                fs_subject_dir=fs_sub_dir,
                output_path=qc_path,
                config=config,
            )
        count += 1
        logger.info("%s ses-%s: tSNR QC written to %s", sub_dir.name, session_id, qc_path)

    return count


def refresh_inflated_gifti(sd: SubjectsDir, dry_run: bool) -> None:
    for hemi in ("lh", "rh"):
        in_surf = sd.hemi_surf(hemi, "inflated")
        out_gii = sd.hemi_surf(hemi, "inflated.surf.gii")
        if dry_run:
            logger.info("DRY_RUN would convert %s -> %s", in_surf, out_gii)
            continue
        convert_fs_surface_to_gifti(in_surf, out_gii, apply_cras=True)


def regenerate_html_reports(dataset_root: Path, config: dict, dry_run: bool) -> None:
    sublist = list_subjects(dataset_root)
    if dry_run:
        logger.info(
            "DRY_RUN would regenerate HTML for %d subjects + dataset report",
            len(sublist),
        )
        return

    built = []
    log = logging.getLogger("generate_report_only")
    for sub in sublist:
        data = build_report_data(
            sub / "figures",
            sub.parent / f"{sub.name}.html",
            config,
            log,
        )
        built.append((sub.name, data))

    dataset_href = None
    if len(built) > 1:
        write_dataset_report(
            built, dataset_root / DATASET_REPORT_NAME, log
        )
        dataset_href = DATASET_REPORT_NAME

    for name, data in built:
        data["metadata"]["dataset_report_href"] = dataset_href
        write_html_report(data, dataset_root / f"{name}.html", log)


def process_subject(
    dataset_root: Path,
    sub_dir: Path,
    config: dict,
    dry_run: bool,
    skip_qc: bool,
    skip_inflate2: bool,
) -> None:
    sub = sub_dir.name
    fs_sub_dir = dataset_root / "fastsurfer" / sub

    if not skip_inflate2:
        validate_subject(fs_sub_dir)

        for hemi in ("lh", "rh"):
            inflated = fs_sub_dir / "surf" / f"{hemi}.inflated"
            msg = backup_file_if_exists(inflated, dry_run)
            if msg:
                logger.info("%s / %s: %s", sub, hemi, msg)

        figures_dir = sub_dir / "figures"
        cortical_qc = pick_cortical_qc_path(figures_dir)
        backup_msg = backup_png_if_exists(cortical_qc, dry_run)
        if backup_msg:
            logger.info("%s: %s", sub, backup_msg)

        if dry_run:
            logger.info("DRY_RUN would apply inflate2 for %s", sub)
        else:
            atlas_name = atlas_name_from_config(config)
            threads = config.get("processing", {}).get("threads", 8)
            if threads == "auto":
                threads = 8

            recon_config = ReconSurfConfig.with_defaults(
                subject_id=sub,
                subjects_dir=str(dataset_root / "fastsurfer"),
                atlas={"name": atlas_name},
                processing={
                    "threads": threads,
                    "skip_cc": True,
                    "skip_talairach": True,
                },
                verbose=1,
            )
            sd = SubjectsDir(recon_config.subjects_dir, sub)

            for hemi in ("lh", "rh"):
                applied = apply_inflate2_visualization(recon_config, sd, hemi)
                if not applied:
                    raise RuntimeError(
                        f"inflate2 visualization did not run for {sub} {hemi}"
                    )

            refresh_inflated_gifti(sd, dry_run=False)

            if not skip_qc:
                qc_cortical_surf_and_measures(
                    fs_subject_dir=fs_sub_dir,
                    output_path=cortical_qc,
                    atlas_name=atlas_name,
                    modality="anat",
                    config=config,
                )
    elif fs_sub_dir.is_dir():
        validate_subject(fs_sub_dir)

    if not skip_qc and fs_sub_dir.is_dir():
        refresh_tsnr_qc_for_subject(sub_dir, fs_sub_dir, config, dry_run=dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Preprocessed dataset root (contains sub-*, fastsurfer/, nextflow_reports/)",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="Optional subject IDs (e.g. sub-m018); default all sub-*",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-qc", action="store_true")
    parser.add_argument(
        "--skip-inflate2",
        action="store_true",
        help="Only refresh tSNR (and cortical if not skipped); skip inflate2 surfaces",
    )
    parser.add_argument("--skip-reports", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("fastsurfer_surfrecon").setLevel(logging.INFO)

    dataset_root = args.dataset_root.resolve()
    config_path = dataset_root / "nextflow_reports" / "config.yaml"
    if not config_path.is_file():
        logger.error("Config not found: %s", config_path)
        return 1
    config = load_config(config_path)

    subjects = list_subjects(dataset_root)
    if args.subjects:
        wanted = set(args.subjects)
        subjects = [s for s in subjects if s.name in wanted]
        missing = wanted - {s.name for s in subjects}
        if missing:
            logger.error("Unknown subjects: %s", sorted(missing))
            return 1

    stats: dict[str, int] = defaultdict(int)
    for sub_dir in subjects:
        try:
            process_subject(
                dataset_root,
                sub_dir,
                config,
                args.dry_run,
                args.skip_qc,
                args.skip_inflate2,
            )
            stats["ok"] += 1
            logger.info("Finished %s", sub_dir.name)
        except Exception as exc:
            stats["errors"] += 1
            logger.error("Failed %s: %s", sub_dir.name, exc)

    if not args.skip_reports and stats["errors"] == 0:
        try:
            regenerate_html_reports(dataset_root, config, args.dry_run)
            stats["reports"] += 1
        except Exception as exc:
            logger.error("Report regeneration failed: %s", exc)
            return 1
    elif stats["errors"]:
        logger.error("Skipping HTML reports due to subject errors")

    logger.info("Summary: %s", dict(stats))
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
