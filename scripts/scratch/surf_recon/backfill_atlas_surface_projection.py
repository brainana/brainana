"""Re-project atlases onto fsnative surfaces for an existing brainana output.

Backfills ``anat/atlas_space-fsnative/*.func.gii`` (and the status JSON) with the
current ``anat_project_atlases_to_surface``: depth sampling, cortex mask, label
hole fill. Inputs are the published T1w-space atlases and the FastSurfer subject
dir of the same run, so no registration is redone. The fsnative volumes and the
per-file sidecars are left as they are.

Usage:
    python backfill_atlas_surface_projection.py --root <output_dir> --dry-run
    python backfill_atlas_surface_projection.py --root <output_dir> --only site-bordeaux/sub-m05
    python backfill_atlas_surface_projection.py --root <output_dir> --workers 4
"""

import argparse
import datetime
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add src/ to path for nhp_mri_prep imports (scripts/scratch/surf_recon -> brainana)
_src_dir = Path(__file__).resolve().parents[3] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from nhp_mri_prep.steps.anatomical import anat_project_atlases_to_surface  # noqa: E402

STATUS_NAME = "desc-atlasSurfaceProjection_status.json"


def find_jobs(root: Path):
    """One job per anat/atlas_space-T1w dir, paired with site-*/fastsurfer/sub-<id>."""
    jobs, problems = [], []
    for t1w_dir in sorted(root.glob("site-*/sub-*/**/anat/atlas_space-T1w")):
        rel = t1w_dir.relative_to(root)
        site, sub = rel.parts[0], rel.parts[1]
        fs_dir = root / site / "fastsurfer" / sub
        out_dir = t1w_dir.parent / "atlas_space-fsnative"
        atlases = sorted(t1w_dir.glob("atlas-*_space-T1w_*.nii.gz"))
        if not out_dir.is_dir():
            problems.append(f"{rel}: no atlas_space-fsnative (surface step did not run)")
            continue
        if not fs_dir.is_dir():
            problems.append(f"{rel}: missing {fs_dir}")
            continue
        if not atlases:
            problems.append(f"{rel}: no T1w atlases")
            continue
        stem = atlases[0].name.split("_space-T1w_", 1)[1][: -len(".nii.gz")]
        jobs.append(
            {
                "key": "/".join(rel.parts[:-2]),
                "atlases": [str(p) for p in atlases],
                "fs_dir": str(fs_dir),
                "out_dir": str(out_dir),
                "stem": stem,
            }
        )
    # Subject-level surfaces: two sessions of one subject would share a FastSurfer dir.
    by_fs = {}
    for job in jobs:
        by_fs.setdefault(job["fs_dir"], []).append(job["key"])
    for fs_dir, keys in by_fs.items():
        if len(keys) > 1:
            problems.append(f"{fs_dir} is shared by {keys}; pair them by hand")
    return jobs, problems


def backup(root: Path, jobs, tar_path: Path) -> None:
    """Tar the current surface maps and status JSON of every job."""
    with tarfile.open(tar_path, "w:gz") as tar:
        for job in jobs:
            out_dir = Path(job["out_dir"])
            for f in sorted(out_dir.glob("*.func.gii")) + [out_dir / STATUS_NAME]:
                if f.is_file():
                    tar.add(f, arcname=str(f.relative_to(root)))


def run_job(job, commit: str):
    out_dir = Path(job["out_dir"])
    with tempfile.TemporaryDirectory(prefix="atlas_backfill_") as tmp:
        result = anat_project_atlases_to_surface(
            atlas_files=[Path(p) for p in job["atlases"]],
            fs_subject_dir=Path(job["fs_dir"]),
            bids_name=Path(f"{job['stem']}_T1w.nii.gz"),
            working_dir=Path(tmp),
        )
        meta = result.metadata
        if meta.get("skipped"):
            return job["key"], f"skipped: {meta.get('reason')}", meta
        new_gii = sorted((Path(tmp) / "atlas").glob("*.func.gii"))
        for f in new_gii:
            shutil.copyfile(f, out_dir / f.name)
    meta["sidecars"] = [f"atlas/{Path(p).name}" for p in meta.get("sidecars", [])]
    meta["target_grid"] = str(Path(meta.get("target_grid", "")).relative_to(Path(job["fs_dir"]).parent))
    meta["backfill"] = {
        "date": datetime.date.today().isoformat(),
        "script": "scripts/scratch/surf_recon/backfill_atlas_surface_projection.py",
        "brainana_commit": commit,
    }
    (out_dir / STATUS_NAME).write_text(json.dumps(meta, indent=2))
    return job["key"], f"ok ({len(new_gii)} maps)", meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="brainana output dir")
    parser.add_argument("--dry-run", action="store_true", help="list jobs and exit")
    parser.add_argument("--only", nargs="+", help="job keys, e.g. site-bordeaux/sub-m05")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--backup",
        type=Path,
        default=None,
        help="tar.gz of the current maps (default: <root>/_backup_atlas_surf_<date>.tar.gz; "
        "written once, reused if it exists)",
    )
    args = parser.parse_args()

    jobs, problems = find_jobs(args.root)
    for p in problems:
        print(f"WARN {p}")
    if args.only:
        jobs = [j for j in jobs if j["key"] in set(args.only)]
    print(f"{len(jobs)} jobs")
    if args.dry_run:
        for job in jobs:
            print(f"  {job['key']}: {len(job['atlases'])} atlases, fs={job['fs_dir']}")
        return
    if any("shared by" in p for p in problems):
        sys.exit("Stop: a FastSurfer dir is shared by several sessions.")

    tar_path = args.backup or (
        args.root / f"_backup_atlas_surf_{datetime.date.today():%Y%m%d}.tar.gz"
    )
    if tar_path.exists():
        print(f"backup exists, not rewritten: {tar_path}")
    else:
        # always back up every job, so a later --only run is still covered
        backup(args.root, find_jobs(args.root)[0], tar_path)
        print(f"backup written: {tar_path}")

    commit = subprocess.run(
        ["git", "-C", str(_src_dir.parent), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()

    n_fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, job, commit): job["key"] for job in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            key = futures[fut]
            try:
                _, status, _ = fut.result()
            except Exception:
                n_fail += 1
                status = "FAILED\n" + traceback.format_exc()
            print(f"[{i}/{len(jobs)}] {key}: {status}", flush=True)
    print(f"done; {n_fail} failed")


if __name__ == "__main__":
    main()
