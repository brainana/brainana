# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training loop for the 2.5-D segmentation networks (FastSurferVINN).

What happens per epoch:
  train    - batches from the HDF5 (uint8 slices) -> GPU -> GPU augmentation -> bf16 forward ->
             combined Dice + weighted CE loss (fp32) -> optimizer step -> LR step (fractional epoch)
  validate - no augmentation; per-subject 3-D Dice from overlap counts (DiceAccumulator)
  select   - TRAIN.SELECTION_METRIC (default val_dice = mean per-subject foreground Dice)
             picks Best_training_state.pkl AND drives early stopping (one criterion)
  record   - monitor/metrics.jsonl + status.json + snapshot PNG (live dashboard),
             training_metrics.csv, logs/training.log, optional TensorBoard

Resume (TRAIN.RESUME) continues from the newest checkpoints/Epoch_*.pkl with optimizer,
scheduler, early-stopping state and best score. Fine-tuning (TRAIN.FINE_TUNE +
TRAIN.PRETRAINED_MODEL) loads weights only and FAILS if they cannot be loaded.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yacs.config

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard is optional; the monitor files are the primary log
    SummaryWriter = None

from fastsurfer_nn.atlas.atlas_manager import get_atlas_manager
from fastsurfer_nn.data_loader import loader
from fastsurfer_nn.data_loader.gpu_augment import GpuAugment
from fastsurfer_nn.models.losses import get_loss_func
from fastsurfer_nn.models.networks import build_model
from fastsurfer_nn.models.optimizer import get_optimizer
from fastsurfer_nn.utils import checkpoint as cp
from fastsurfer_nn.utils import logging
from fastsurfer_nn.utils.constants import PRETRAINED_MODEL_DIR, REPO_ROOT
from fastsurfer_nn.utils.gpu_utils import get_device
from fastsurfer_nn.utils.lr_scheduler import get_lr_scheduler
from fastsurfer_nn.utils.metrics import DiceAccumulator
from fastsurfer_nn.utils.misc import update_num_steps
from fastsurfer_nn.utils.monitor import TrainingMonitor

logger = logging.getLogger(__name__)


def resolve_pretrained(path: str) -> Path:
    """Pretrained checkpoint path. A bare file name (e.g. "T1w_seg-ARM2_coronal.pkl") means the
    shipped model in PRETRAINED_MODEL_DIR; a stale directory also falls back there (warned)."""
    p = Path(path)
    if p.exists():
        return p
    alt = Path(PRETRAINED_MODEL_DIR) / p.name
    if alt.exists():
        if str(p.parent) not in ("", "."):  # a stale directory, not a bare file name
            logger.warning(f"PRETRAINED_MODEL {path} not found; using {alt}")
        return alt
    raise FileNotFoundError(
        f"TRAIN.PRETRAINED_MODEL not found: {path} (also looked for {alt})")


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True,
                               timeout=10).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def site_of(subject: str) -> str:
    return subject.split("_", 1)[0] if subject.startswith("site-") else "unknown"


class Trainer:
    def __init__(self, cfg: yacs.config.CfgNode, profile_iters: int = 0):
        self.cfg = cfg
        self.profile_iters = int(profile_iters)
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        deterministic = bool(getattr(cfg.TRAIN, "DETERMINISTIC", False))
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.log_dir = Path(cfg.LOG_DIR)
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "logs").mkdir(parents=True, exist_ok=True)
        logging.setup_logging(str(self.log_dir / "logs" / "training.log"))

        self.device = get_device()
        self.model = build_model(cfg)
        self.loss_func = get_loss_func(cfg)
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.plane = cfg.DATA.PLANE
        self.rescale = float(cfg.DATA.PREPROCESSING.RESCALE)
        self.accum = max(1, int(getattr(cfg.TRAIN, "ACCUM_STEPS", 1)))

        amp = str(getattr(cfg.TRAIN, "AMP", "none")).lower()
        if self.device.type != "cuda":
            amp = "none"
        if amp == "bf16" and not torch.cuda.is_bf16_supported():
            logger.warning("bf16 not supported on this GPU; using fp16")
            amp = "fp16"
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
        self.scaler = torch.amp.GradScaler("cuda") if amp == "fp16" else None

        # class names / colours in DENSE order for this plane
        if self.num_classes == 2:
            self.class_names = ["background", "brain"]
            self.dense_rgb = np.array([[0, 0, 0], [255, 140, 0]])
        else:
            atlas = cfg.DATA.CLASS_OPTIONS[0] if cfg.DATA.CLASS_OPTIONS else "ARM2"
            am = get_atlas_manager(atlas)
            self.class_names = am.dense_class_names(self.plane)
            self.dense_rgb = self._dense_rgb(am)
            if len(self.class_names) != self.num_classes:
                raise ValueError(
                    f"MODEL.NUM_CLASSES={self.num_classes} but atlas {atlas} has "
                    f"{len(self.class_names)} dense classes for plane {self.plane}")

        self.gpu_aug = None
        if getattr(cfg.DATA, "AUG_DEVICE", "gpu") == "gpu" and cfg.DATA.AUG and "None" not in cfg.DATA.AUG:
            gen = torch.Generator().manual_seed(int(cfg.RNG_SEED))
            self.gpu_aug = GpuAugment(cfg, generator=gen)

        metric = getattr(cfg.TRAIN, "SELECTION_METRIC", "val_dice")
        if metric not in ("val_dice", "val_loss"):
            raise ValueError(f"TRAIN.SELECTION_METRIC must be val_dice or val_loss, not {metric}")
        self.metric = metric
        self.higher_is_better = metric == "val_dice"
        self._summary_logged = False
        # optional per-subject groups for reporting (e.g. UNC age bins), written by the data prep
        self.age_bin_of = {}
        ab = Path(cfg.TRAINING_DATA_DIR) / "age_bins.tsv" if cfg.TRAINING_DATA_DIR else None
        if ab is not None and ab.exists():
            for line in ab.read_text().splitlines()[1:]:
                parts = line.split("\t")
                if len(parts) >= 3 and parts[2] != "n/a":
                    self.age_bin_of[parts[0]] = parts[2]

    # ------------------------------------------------------------------ setup helpers
    def _dense_rgb(self, am) -> np.ndarray:
        import pandas as pd

        rgb = np.zeros((self.num_classes, 3))
        try:
            lut = pd.read_csv(am.colorlut_path, sep="\t")
            cols = {c.lower(): c for c in lut.columns}
            by_id = {int(r[cols["id"]]): (r[cols["r"]], r[cols["g"]], r[cols["b"]])
                     for _, r in lut.iterrows()}
            for d, s in enumerate(am.dense_to_sparse(self.plane)):
                rgb[d] = by_id.get(int(s), (200, 200, 200))
        except Exception as e:
            logger.warning(f"ColorLUT not readable ({e}); snapshot colours are random")
            rgb = np.random.default_rng(0).integers(40, 255, (self.num_classes, 3))
        rgb[0] = 0
        return rgb

    def _config_summary(self) -> dict:
        c = self.cfg
        return {
            "plane": c.DATA.PLANE, "contrast": getattr(c.DATA, "CONTRAST", ""),
            "classes": c.MODEL.NUM_CLASSES, "batch": c.TRAIN.BATCH_SIZE, "accum": self.accum,
            "epochs": c.TRAIN.NUM_EPOCHS, "lr": c.OPTIMIZER.BASE_LR,
            "scheduler": c.OPTIMIZER.LR_SCHEDULER, "optimizer": c.OPTIMIZER.OPTIMIZING_METHOD,
            "amp": str(self.amp_dtype).replace("torch.", "") if self.amp_dtype else "none",
            "aug_device": getattr(c.DATA, "AUG_DEVICE", "gpu"),
            "aug": self.gpu_aug.describe() if self.gpu_aug else ", ".join(c.DATA.AUG),
            "pretrained": Path(c.TRAIN.PRETRAINED_MODEL).name if c.TRAIN.FINE_TUNE else "",
            "selection": self.metric, "train_hdf5": c.DATA.PATH_HDF5_TRAIN,
            "val_hdf5": c.DATA.PATH_HDF5_VAL, "out": c.LOG_DIR,
        }

    def _log_summary(self, n_train, n_val, s_train, s_val):
        s = self._config_summary()
        logger.info("=" * 80)
        for k, v in s.items():
            logger.info(f"{k:<12}{v}")
        logger.info(f"{'device':<12}{self.device}")
        logger.info(f"{'data':<12}train {n_train} slices / {s_train} subjects, "
                    f"val {n_val} slices / {s_val} subjects")
        logger.info("=" * 80)

    def _provenance(self, train_loader, val_loader) -> dict:
        split_f = Path(self.cfg.TRAINING_DATA_DIR) / "data_split.json" if self.cfg.TRAINING_DATA_DIR else None
        split_hash = ""
        if split_f is not None and split_f.exists():
            split_hash = hashlib.sha256(split_f.read_bytes()).hexdigest()[:16]
        return {
            "git_commit": git_commit(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "training_data_dir": str(self.cfg.TRAINING_DATA_DIR),
            "data_split_sha256_16": split_hash,
            "train_subjects": list(train_loader.dataset.subjects),
            "val_subjects": list(val_loader.dataset.subjects),
            "pretrained": str(self.cfg.TRAIN.PRETRAINED_MODEL) if self.cfg.TRAIN.FINE_TUNE else "",
        }

    def _autocast(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=self.amp_dtype)

    def _to_device(self, batch, train: bool):
        images = batch["image"].to(self.device, non_blocking=True)
        if images.dtype == torch.uint8:
            images = images.float().div_(self.rescale)
        labels = batch["label"].to(self.device, non_blocking=True).long()
        weights = batch["weight"].to(self.device, non_blocking=True).float()
        scale = batch["scale_factor"].float()  # stays on CPU (see gpu_augment docstring)
        if train and self.gpu_aug is not None:
            images, labels, weights, scale = self.gpu_aug(images, labels, weights, scale)
        if self.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        return images, labels, weights, scale, batch["subject"]

    # ------------------------------------------------------------------ epochs
    def train_epoch(self, epoch, train_loader, optimizer, scheduler):
        self.model.train()
        n_iter = len(train_loader)
        acc = DiceAccumulator(self.num_classes, 1, self.device)
        log_every = max(1, int(self.cfg.TRAIN.LOG_INTERVAL))
        sums = torch.zeros(3, device=self.device)  # total, dice, ce since last log
        loss_epoch = torch.zeros((), device=self.device)
        n_since = 0
        t_epoch = time.perf_counter()
        t_last_log = t_epoch
        data_wait = 0.0
        data_wait_window = 0.0
        prof = []
        optimizer.zero_grad(set_to_none=True)
        t_end = time.perf_counter()
        for it, batch in enumerate(train_loader):
            t_got = time.perf_counter()
            wait = t_got - t_end
            data_wait += wait
            data_wait_window += wait
            images, labels, weights, scale, _ = self._to_device(batch, train=True)
            with self._autocast():
                pred = self.model(images, scale)
            loss_total, loss_dice, loss_ce = self.loss_func(pred, labels, weights)
            loss = loss_total / self.accum
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            last = it == n_iter - 1
            if (it + 1) % self.accum == 0 or last:
                if self.scaler is not None:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step((epoch - 1) + (it + 1) / n_iter)
            with torch.no_grad():
                acc.update(pred.argmax(1), labels)
                sums += torch.stack([loss_total.detach(), loss_dice.detach(), loss_ce.detach()])
                loss_epoch += loss_total.detach()
            n_since += 1
            if self.profile_iters:
                torch.cuda.synchronize(self.device)
                prof.append((wait, time.perf_counter() - t_got))
                if len(prof) >= self.profile_iters:
                    self._report_profile(prof)
                    return None
            if (it + 1) % log_every == 0 or last:
                vals = (sums / n_since).tolist()  # one sync per log interval
                now = time.perf_counter()
                window = now - t_last_log
                lr = optimizer.param_groups[0]["lr"]
                self.global_step = (epoch - 1) * n_iter + it + 1
                self.monitor.log_iter(
                    epoch=epoch, iter=it + 1, iters=n_iter, step=self.global_step,
                    loss=vals[0], loss_dice=vals[1], loss_ce=vals[2], lr=lr,
                    data_s=data_wait_window / n_since, step_s=window / n_since,
                    img_per_s=n_since * self.cfg.TRAIN.BATCH_SIZE / max(window, 1e-9),
                    data_wait_frac=data_wait_window / max(window, 1e-9),
                )
                elapsed = now - t_epoch
                eta_epoch = elapsed / (it + 1) * (n_iter - it - 1)
                self.monitor.status(state="training", epoch=epoch, iter=it + 1, iters=n_iter,
                                    eta_epoch_s=eta_epoch,
                                    eta_s=eta_epoch + self._epoch_eta(epoch))
                if self.writer is not None:
                    self.writer.add_scalar("Train/total_loss", vals[0], self.global_step)
                    self.writer.add_scalar("Train/lr", lr, self.global_step)
                logger.info(f"train epoch {epoch} iter {it + 1}/{n_iter} loss {vals[0]:.4f} "
                            f"(dice {vals[1]:.4f} ce {vals[2]:.4f}) lr {lr:.2e} "
                            f"{window / n_since:.3f}s/it data-wait {data_wait_window / max(window, 1e-9):.0%}")
                sums.zero_()
                n_since = 0
                data_wait_window = 0.0
                t_last_log = now
            t_end = time.perf_counter()
        train_s = time.perf_counter() - t_epoch
        return {
            "train_loss": float(loss_epoch) / n_iter,
            "train_dice": acc.mean_pooled_dice(),
            "train_s": train_s,
            "data_wait_frac": data_wait / max(train_s, 1e-9),
        }

    @torch.no_grad()
    def validate(self, epoch, val_loader):
        self.model.eval()
        ds = val_loader.dataset
        acc = DiceAccumulator(self.num_classes, len(ds.subjects), self.device)
        loss_sum = torch.zeros((), device=self.device)
        t0 = time.perf_counter()
        for batch in val_loader:
            images, labels, weights, scale, subj = self._to_device(batch, train=False)
            with self._autocast():
                pred = self.model(images, scale)
            loss_total, _, _ = self.loss_func(pred, labels, weights)
            loss_sum += loss_total.detach()
            acc.update(pred.argmax(1), labels, subj)
        subj_mean = acc.subject_mean().cpu().numpy()
        per_class = acc.per_class_dice().cpu().numpy()
        per_subject = {s: float(v) for s, v in zip(ds.subjects, subj_mean)}
        per_site = {}
        for s, v in per_subject.items():
            per_site.setdefault(site_of(s), []).append(v)
        per_age = {}
        for s, v in per_subject.items():
            if s in self.age_bin_of:
                per_age.setdefault(self.age_bin_of[s], []).append(v)
        return {
            "per_age": {k: float(np.nanmean(v)) for k, v in per_age.items()},
            "val_loss": float(loss_sum) / len(val_loader),
            "val_dice": acc.mean_subject_dice(),
            "val_dice_pooled": acc.mean_pooled_dice(),
            "per_class": {n: (None if np.isnan(v) else float(v))
                          for n, v in zip(self.class_names, per_class)},
            "per_subject": per_subject,
            "per_site": {k: float(np.nanmean(v)) for k, v in per_site.items()},
            "val_s": time.perf_counter() - t0,
        }

    def _snapshot_batch(self, ds):
        k = int(getattr(self.cfg.TRAIN, "SNAPSHOT_SLICES", 4))
        picks, titles = [], []
        for s in range(min(k, len(ds.subjects))):
            idx = np.nonzero(ds.subject_of == s)[0]
            if len(idx):
                picks.append(int(idx[len(idx) // 2]))
                titles.append(ds.subjects[s])
        if not picks:
            return None
        items = [ds[i] for i in picks]
        batch = {key: torch.stack([torch.as_tensor(it[key]) for it in items])
                 for key in ("image", "label", "weight", "scale_factor")}
        batch["subject"] = torch.as_tensor([it["subject"] for it in items])
        return batch, titles

    @torch.no_grad()
    def _snapshot(self, epoch, snap):
        if snap is None:
            return
        batch, titles = snap
        self.model.eval()
        images, labels, _, scale, _ = self._to_device(batch, train=False)
        with self._autocast():
            pred = self.model(images, scale).argmax(1)
        self.monitor.snapshot(epoch, images, labels, pred, self.dense_rgb, titles)

    def _report_profile(self, prof):
        arr = np.array(prof[5:] if len(prof) > 10 else prof)  # skip warm-up iterations
        wait, comp = arr[:, 0], arr[:, 1]
        res = {
            "iterations": len(arr), "batch_size": self.cfg.TRAIN.BATCH_SIZE,
            "data_wait_s_mean": float(wait.mean()), "gpu_step_s_mean": float(comp.mean()),
            "s_per_iter": float((wait + comp).mean()),
            "data_wait_frac": float(wait.sum() / (wait + comp).sum()),
            "img_per_s": float(self.cfg.TRAIN.BATCH_SIZE / (wait + comp).mean()),
            "workers": int(self.cfg.DATA_LOADER.NUM_WORKERS),
            "aug_device": getattr(self.cfg.DATA, "AUG_DEVICE", "gpu"),
            "amp": str(self.amp_dtype),
            "max_mem_gb": torch.cuda.max_memory_allocated(self.device) / 1024**3
            if self.device.type == "cuda" else None,
        }
        out = self.log_dir / "monitor" / "profile.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        logger.info("PROFILE " + json.dumps(res))

    def _epoch_eta(self, epoch) -> float:
        if not self.epoch_times:
            return 0.0
        return float(np.mean(self.epoch_times[-5:])) * (self.cfg.TRAIN.NUM_EPOCHS - epoch)

    # ------------------------------------------------------------------ main loop
    def run(self):
        cfg = self.cfg
        if cfg.NUM_GPUS > 1:
            assert cfg.NUM_GPUS <= torch.cuda.device_count(), "Cannot use more GPUs than available"
            self.model = torch.nn.DataParallel(self.model)

        val_loader = loader.get_dataloader(cfg, "val")
        train_loader = loader.get_dataloader(cfg, "train")
        update_num_steps(train_loader, cfg)
        self._log_summary(len(train_loader.dataset), len(val_loader.dataset),
                          len(train_loader.dataset.subjects), len(val_loader.dataset.subjects))

        self.model = self.model.to(self.device)
        self.channels_last = bool(getattr(cfg.TRAIN, "CHANNELS_LAST", True)) and self.device.type == "cuda"
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        optimizer = get_optimizer(self.model, cfg)
        scheduler = get_lr_scheduler(optimizer, cfg)

        start_epoch, best, best_epoch, wait_count = 1, None, 0, 0
        history = []
        if cfg.TRAIN.RESUME and cp.get_checkpoint_path(cfg.LOG_DIR):
            ckpt = cp.get_checkpoint_path(cfg.LOG_DIR)[-1]
            done, best, state = cp.restore_model_state_from_checkpoint(
                ckpt, self.model, optimizer, scheduler, fine_tune=False)
            start_epoch = done + 1
            best_epoch = state.get("best_epoch", 0)
            wait_count = state.get("wait_count", 0)
            history = state.get("history", [])
            logger.info(f"Resume: {ckpt.name}, continuing at epoch {start_epoch}, best {best}")
        elif cfg.TRAIN.RESUME:
            logger.warning(f"TRAIN.RESUME set but no checkpoint in {self.checkpoint_dir}; starting fresh")
        if start_epoch == 1 and cfg.TRAIN.FINE_TUNE:
            if not cfg.TRAIN.PRETRAINED_MODEL:
                raise ValueError("TRAIN.FINE_TUNE is set but TRAIN.PRETRAINED_MODEL is empty")
            path = resolve_pretrained(cfg.TRAIN.PRETRAINED_MODEL)
            state = cp.read_checkpoint_file(path, map_location="cpu")["model_state"]
            n_pre = state["classifier.conv.weight"].shape[0] if "classifier.conv.weight" in state else None
            drop = n_pre != self.num_classes
            cp.restore_model_state_from_checkpoint(path, self.model, fine_tune=True,
                                                   drop_classifier=drop)
            logger.info(f"Fine-tune: loaded {path.name}"
                        + (f" (classifier dropped: {n_pre} -> {self.num_classes} classes)" if drop else ""))

        if getattr(cfg.TRAIN, "COMPILE", False):
            self.model = torch.compile(self.model)

        run_name = Path(cfg.LOG_DIR).name
        self.monitor = TrainingMonitor(cfg.LOG_DIR, run_name, self._config_summary(), self.device,
                                       enabled=bool(getattr(cfg.TRAIN, "MONITOR", True)))
        self.writer = SummaryWriter(cfg.SUMMARY_PATH, flush_secs=30) if SummaryWriter else None
        provenance = self._provenance(train_loader, val_loader)
        snap = self._snapshot_batch(val_loader.dataset)
        self.epoch_times = []
        self.global_step = (start_epoch - 1) * len(train_loader)

        es = bool(cfg.TRAIN.EARLY_STOPPING)
        patience = int(cfg.TRAIN.EARLY_STOPPING_PATIENCE)
        es_wait = int(cfg.TRAIN.EARLY_STOPPING_WAIT)
        delta = float(cfg.TRAIN.EARLY_STOPPING_DELTA)
        keep_last = 3
        csv_path = self.log_dir / "training_metrics.csv"
        new_csv = not csv_path.exists() or start_epoch == 1
        csv_fh = open(csv_path, "w" if new_csv else "a", newline="")
        csv_w = csv.writer(csv_fh)
        if new_csv:
            csv_w.writerow(["epoch", "lr", "train_loss", "train_dice", "val_loss", "val_dice",
                            "val_dice_pooled", "epoch_s", "data_wait_frac", "best"])
        self.monitor.event(f"start epoch {start_epoch}" + (" (resume)" if start_epoch > 1 else ""))
        stopped_early = False
        try:
            for epoch in range(start_epoch, cfg.TRAIN.NUM_EPOCHS + 1):
                t0 = time.perf_counter()
                tr = self.train_epoch(epoch, train_loader, optimizer, scheduler)
                if tr is None:  # profile mode
                    self.monitor.close("profiled")
                    return
                self.monitor.status(state="validating", epoch=epoch)
                va = self.validate(epoch, val_loader)
                self._snapshot(epoch, snap)
                score = va[self.metric]
                score = score if not (isinstance(score, float) and math.isnan(score)) else None
                improved = score is not None and (
                    best is None
                    or (self.higher_is_better and score > best + delta)
                    or (not self.higher_is_better and score < best - delta))
                if improved:
                    best, best_epoch = score, epoch
                    wait_count = 0
                elif epoch > es_wait:
                    wait_count += 1
                epoch_s = time.perf_counter() - t0
                self.epoch_times.append(epoch_s)
                lr = optimizer.param_groups[0]["lr"]
                rec = {"epoch": epoch, "lr": lr, **tr, **va, "epoch_s": epoch_s,
                       "best": bool(improved), "best_epoch": best_epoch, "best_score": best,
                       "early_stop_count": wait_count}
                history.append({k: rec[k] for k in ("epoch", "train_loss", "train_dice",
                                                     "val_loss", "val_dice", "val_dice_pooled")})
                self.monitor.log_epoch(**rec)
                csv_w.writerow([epoch, lr, tr["train_loss"], tr["train_dice"], va["val_loss"],
                                va["val_dice"], va["val_dice_pooled"], epoch_s,
                                tr["data_wait_frac"], int(improved)])
                csv_fh.flush()
                if self.writer is not None:
                    for k in ("val_loss", "val_dice", "val_dice_pooled"):
                        self.writer.add_scalar(f"Val/{k}", va[k], epoch)
                logger.info(
                    f"epoch {epoch}: train loss {tr['train_loss']:.4f} dice {tr['train_dice']:.4f} | "
                    f"val loss {va['val_loss']:.4f} dice(3D/subject) {va['val_dice']:.4f} "
                    f"pooled {va['val_dice_pooled']:.4f} | {epoch_s / 60:.1f} min, "
                    f"data-wait {tr['data_wait_frac']:.0%}" + ("  * best" if improved else ""))

                trainer_state = {"best_epoch": best_epoch, "wait_count": wait_count,
                                 "history": history, "metric": self.metric}
                if improved:
                    cp.save_best_checkpoint(self.checkpoint_dir, epoch, best, cfg.NUM_GPUS, cfg,
                                            self.model, optimizer, scheduler, trainer_state,
                                            provenance)
                if epoch % max(1, int(cfg.TRAIN.CHECKPOINT_PERIOD)) == 0 or epoch == cfg.TRAIN.NUM_EPOCHS:
                    cp.save_checkpoint(self.checkpoint_dir, epoch, best, cfg.NUM_GPUS, cfg,
                                       self.model, optimizer, scheduler, extra_state=trainer_state,
                                       provenance=provenance)
                    for old in cp.get_checkpoint_path(cfg.LOG_DIR)[:-keep_last]:
                        old.unlink()
                self.monitor.status(state="training", epoch=epoch, best_epoch=best_epoch,
                                    best_score=best, early_stop_count=wait_count,
                                    early_stop_patience=patience if es else None,
                                    eta_s=self._epoch_eta(epoch))
                if es and wait_count >= patience:
                    logger.info(f"Early stopping at epoch {epoch}: no {self.metric} improvement "
                                f"for {patience} epochs (best {best:.4f} at epoch {best_epoch})")
                    self.monitor.event(f"early stop at epoch {epoch}")
                    cp.save_checkpoint(self.checkpoint_dir, epoch, best, cfg.NUM_GPUS, cfg,
                                       self.model, optimizer, scheduler, extra_state=trainer_state,
                                       provenance=provenance)
                    stopped_early = True
                    break
        except BaseException:
            self.monitor.close("failed")
            raise
        finally:
            csv_fh.close()
            if self.writer is not None:
                self.writer.close()
        self.monitor.event("finished" + (" (early stop)" if stopped_early else ""))
        self.monitor.close("finished")
        logger.info(f"Training finished. Best {self.metric} {best} at epoch {best_epoch}: "
                    f"{self.checkpoint_dir / 'Best_training_state.pkl'}")
