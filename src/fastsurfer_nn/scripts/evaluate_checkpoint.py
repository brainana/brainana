#!/usr/bin/env python3
"""Evaluate segmentation checkpoints on labelled volumes, the way the pipeline runs them.

Runs ``fastsurfer_nn.inference.api.segmentation`` (conform -> 1-3 views -> view fusion ->
WM-island fix -> back to the image grid) on each image and compares the result with its
label file (sparse atlas IDs) per label, per region (cortex / subcortex / WM / CSF), per
subject and per site.

Models: --model T1w (= pretrained_model/T1w_seg-ARM2_{axial,coronal,sagittal}.pkl) or
--checkpoints a.pkl b.pkl ... (plane read from each checkpoint; one per plane).

Images: --data-dir with images/, labels/ and data_split.json (--split val|train|all), or
--images/--labels folders. Labels are found as <image stem>_*.nii.gz.

Parity (T2w): --parity-model T1w --parity-swap T2w:T1w --parity-images <T1w_images dir>
also segments the paired T1w (same grid) with the T1w model and reports Dice between the
two models' segmentations, per subject.

    python -m fastsurfer_nn.scripts.evaluate_checkpoint --model T1w \
        --images .../T1w_images --labels .../T1w_atlas-ARM2/labels --subjects-file unseen.txt \
        --out .../eval_T1w_shipped_unseen
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

_src = Path(__file__).resolve().parents[2]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from fastsurfer_nn.utils.checkpoint import extract_atlas_metadata  # noqa: E402
from fastsurfer_nn.utils.constants import PRETRAINED_MODEL_DIR  # noqa: E402

log = logging.getLogger("evaluate")


def model_checkpoints(model: str | None, paths: list[Path] | None) -> dict[str, Path]:
    if paths:
        out = {}
        for p in paths:
            meta = extract_atlas_metadata(p) or {}
            plane = meta.get("plane")
            if plane not in ("axial", "coronal", "sagittal"):
                raise ValueError(f"cannot tell the plane of {p}")
            out[plane] = Path(p)
        return out
    out = {pl: PRETRAINED_MODEL_DIR / f"{model}_seg-ARM2_{pl}.pkl"
           for pl in ("axial", "coronal", "sagittal")}
    missing = [str(p) for p in out.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")
    return out


def segment(image: Path, ckpts: dict[str, Path], workdir: Path) -> np.ndarray:
    from fastsurfer_nn.inference.api import segmentation

    res = segmentation(input_image=image, output_dir=workdir, ckpt_ax=ckpts.get("axial"),
                       ckpt_cor=ckpts.get("coronal"), ckpt_sag=ckpts.get("sagittal"),
                       device="auto", viewagg_device="auto", create_hemimask=False,
                       output_data_format="nifti")
    return np.asarray(nib.load(str(res["segmentation"])).dataobj).astype(np.int32)


def to_working_grid(img: Path, lab: Path, res: float, tmp: Path):
    """Resample to isotropic ``res`` mm when the image is finer (as the pipeline's conform to the
    template grid would). Returns (image path, label array) on the grid that is scored."""
    im = nib.load(str(img))
    if res <= 0 or min(im.header.get_zooms()[:3]) >= res - 1e-3:
        return img, np.rint(np.asarray(nib.load(str(lab)).dataobj)).astype(np.int32)
    from nibabel.processing import resample_to_output

    im2 = resample_to_output(im, voxel_sizes=(res, res, res), order=1)
    lb2 = resample_to_output(nib.load(str(lab)), voxel_sizes=(res, res, res), order=0)
    out = tmp / "image_resampled.nii.gz"
    nib.save(im2, str(out))
    return out, np.rint(np.asarray(lb2.dataobj)).astype(np.int32)


def region_table(atlas: str = "ARM2") -> pd.DataFrame:
    from fastsurfer_nn.atlas.atlas_manager import get_atlas_manager

    df = pd.read_csv(get_atlas_manager(atlas).roiinfo_path, sep="\t")
    return df[["key", "name", "region", "hemi"]].rename(columns={"key": "label"})


def dice_per_label(pred: np.ndarray, ref: np.ndarray, labels) -> dict[int, tuple]:
    out = {}
    for lab in labels:
        p, r = pred == lab, ref == lab
        sp, sr = int(p.sum()), int(r.sum())
        if sp + sr == 0:
            continue
        out[int(lab)] = (2 * int((p & r).sum()) / (sp + sr), sr, sp)
    return out


def list_subjects(args) -> list[tuple[str, Path, Path]]:
    if args.data_dir:
        img_dir, lab_dir = args.data_dir / "images", args.data_dir / "labels"
        names = None
        if args.split != "all":
            split = json.loads((args.data_dir / "data_split.json").read_text())
            names = split[args.split]
    else:
        img_dir, lab_dir = args.images, args.labels
        names = None
    if args.subjects_file:
        names = [l.strip() for l in args.subjects_file.read_text().splitlines() if l.strip()]
    if names is None:
        names = sorted(p.name[:-7] for p in img_dir.glob("*.nii.gz") if not p.name.startswith("._"))
    out = []
    for n in names:
        img = img_dir / f"{n}.nii.gz"
        labs = sorted(lab_dir.glob(f"{n}_*.nii.gz"))
        if img.exists() and len(labs) == 1:
            out.append((n, img, labs[0]))
        else:
            log.warning(f"skip {n}: image {'ok' if img.exists() else 'missing'}, {len(labs)} labels")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="contrast key of shipped checkpoints, e.g. T1w or T2w")
    g.add_argument("--checkpoints", nargs="+", type=Path)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--split", default="val", choices=["val", "train", "all"])
    p.add_argument("--images", type=Path)
    p.add_argument("--labels", type=Path)
    p.add_argument("--subjects-file", type=Path, help="one image stem per line")
    p.add_argument("--parity-model", default=None)
    p.add_argument("--parity-swap", default="T2w:T1w", help="stem suffix swap, FROM:TO")
    p.add_argument("--parity-images", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--keep-segmentations", action="store_true")
    p.add_argument("--atlas", default="ARM2")
    p.add_argument("--max-res", type=float, default=0.5,
                   help="resample images finer than this (mm, isotropic) before segmenting; "
                        "0 = never (0.3 mm volumes need >30 GB for view fusion)")
    args = p.parse_args()
    if not args.data_dir and not (args.images and args.labels):
        p.error("give --data-dir or --images and --labels")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    for name in ("fastsurfer_nn", "nhp_mri_prep"):
        logging.getLogger(name).setLevel(logging.WARNING)

    ckpts = model_checkpoints(args.model, args.checkpoints)
    parity = model_checkpoints(args.parity_model, None) if args.parity_model else None
    regions = region_table(args.atlas)
    labels = regions["label"].tolist()
    subjects = list_subjects(args)
    args.out.mkdir(parents=True, exist_ok=True)
    seg_dir = args.out / "segmentations"
    log.info(f"{len(subjects)} subjects; model {', '.join(f'{k}={v.name}' for k, v in ckpts.items())}")

    rows, subj_rows = [], []
    for i, (name, img, lab) in enumerate(subjects, 1):
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="eval_") as tmp:
            work_img, ref = to_working_grid(img, lab, args.max_res, Path(tmp))
            pred = segment(work_img, ckpts, Path(tmp) / "seg")
            if args.keep_segmentations:
                seg_dir.mkdir(exist_ok=True)
                ref_img = nib.load(str(work_img))
                nib.save(nib.Nifti1Image(pred.astype(np.int16), ref_img.affine),
                         str(seg_dir / f"{name}_pred.nii.gz"))
        if ref.shape != pred.shape:
            log.error(f"{name}: prediction {pred.shape} vs label {ref.shape}; skipped")
            continue
        site = name.split("_", 1)[0]
        d = dice_per_label(pred, ref, labels)
        for labid, (dv, vr, vp) in d.items():
            rows.append(dict(subject=name, site=site, label=labid, dice=dv, voxels_label=vr, voxels_pred=vp))
        mask_dice = 2 * ((pred != 0) & (ref != 0)).sum() / max((pred != 0).sum() + (ref != 0).sum(), 1)
        srow = dict(subject=name, site=site, mean_dice=float(np.mean([v[0] for v in d.values()])),
                    brain_mask_dice=float(mask_dice))
        if parity is not None:
            src, dst = args.parity_swap.split(":")
            pimg = (args.parity_images or img.parent) / f"{name[: -len(src)]}{dst}.nii.gz"
            if pimg.exists():
                with tempfile.TemporaryDirectory(prefix="eval_par_") as tmp:
                    pwork, _ = to_working_grid(pimg, lab, args.max_res, Path(tmp))
                    ppred = segment(pwork, parity, Path(tmp) / "seg")
                if ppred.shape == pred.shape:
                    pd_ = dice_per_label(pred, ppred, labels)
                    srow["parity_dice"] = float(np.mean([v[0] for v in pd_.values()]))
                    srow["parity_vs_label_dice"] = float(np.mean(
                        [v[0] for v in dice_per_label(ppred, ref, labels).values()]))
            else:
                log.warning(f"{name}: no paired image {pimg.name}")
        subj_rows.append(srow)
        log.info(f"[{i}/{len(subjects)}] {name}: mean Dice {srow['mean_dice']:.4f}"
                 + (f", parity {srow['parity_dice']:.4f}" if 'parity_dice' in srow else "")
                 + f" ({time.time() - t0:.0f}s)")

    per_label = pd.DataFrame(rows).merge(regions, on="label", how="left")
    per_subject = pd.DataFrame(subj_rows)
    reg = per_label.groupby(["subject", "region"])["dice"].mean().unstack()
    per_subject = per_subject.merge(reg, left_on="subject", right_index=True, how="left")
    per_label.to_csv(args.out / "per_label.tsv", sep="\t", index=False)
    per_subject.to_csv(args.out / "per_subject.tsv", sep="\t", index=False)
    by_site = per_subject.drop(columns=["subject"]).groupby("site").mean(numeric_only=True)
    by_site.insert(0, "n", per_subject.groupby("site").size())
    by_site.to_csv(args.out / "per_site.tsv", sep="\t")
    by_name = per_label.groupby(["region", "name", "hemi"])["dice"].agg(["mean", "std", "min"])
    by_name.to_csv(args.out / "per_structure.tsv", sep="\t")

    md = [f"# Evaluation {time.strftime('%Y-%m-%d %H:%M')}", "",
          f"Model: {', '.join(f'{k} `{v}`' for k, v in ckpts.items())}",
          f"Subjects: {len(per_subject)}", "",
          "| | mean Dice | cortex | subcortex | WM | CSF | brain mask |" + (" parity |" if parity else ""),
          "|---|---|---|---|---|---|---|" + ("---|" if parity else "")]
    def line(lbl, df):
        vals = [df["mean_dice"].mean()] + [df[c].mean() if c in df else np.nan
                                          for c in ("cortex", "subcortex", "WM", "CSF")] + [df["brain_mask_dice"].mean()]
        if parity:
            vals.append(df["parity_dice"].mean() if "parity_dice" in df else np.nan)
        return f"| {lbl} | " + " | ".join(f"{v:.3f}" for v in vals) + " |"
    md.append(line("**all**", per_subject))
    for site, df in per_subject.groupby("site"):
        md.append(line(f"{site} (n={len(df)})", df))
    md += ["", "Weakest structures (mean over subjects):", "",
           "```", by_name.sort_values("mean").head(10).round(3).to_string(), "```"]
    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
