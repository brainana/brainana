"""Training monitor: machine-readable progress for the live dashboard.

Writes into ``<LOG_DIR>/monitor/``:

* ``metrics.jsonl`` - one JSON object per line, appended (survives resume):
    {"type": "iter", "time", "epoch", "iter", "iters", "step", "loss", "loss_ce",
     "loss_dice", "lr", "data_s", "step_s", "img_per_s", "gpu_mem_gb", "gpu_util"}
    {"type": "epoch", "time", "epoch", "train_loss", "train_dice", "val_loss",
     "val_dice", "val_dice_2d", "per_class": {name: dice}, "per_site": {site: dice},
     "per_subject": {subject: dice}, "lr", "epoch_s", "train_s", "val_s", "best", "data_wait_frac"}
    {"type": "event", "time", "message"}
* ``status.json`` - current state, rewritten atomically (state, pid, host, epoch, iter, eta_s,
  best_epoch, best_score, early_stop_count, config summary, last_update).
* ``snapshots/epoch_NNN.png`` - fixed validation slices: image | label | prediction.

The dashboard (``fastsurfer_nn.scripts.training_dashboard``) only reads these files, so it can
run anywhere the output folder is visible and never touches the training process.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class GpuStats:
    """GPU memory/utilisation for one torch device, sampled at most every ``period`` s."""

    def __init__(self, device: torch.device, period: float = 10.0):
        self.device = device
        self.period = period
        self._last = 0.0
        self._cache = {"gpu_mem_gb": None, "gpu_util": None}
        self._uuid = None
        if device.type == "cuda":
            try:
                uuid = torch.cuda.get_device_properties(device).uuid
                self._uuid = f"GPU-{uuid}" if not str(uuid).startswith("GPU-") else str(uuid)
            except Exception:
                self._uuid = None

    def sample(self) -> dict:
        if self.device.type != "cuda":
            return self._cache
        now = time.time()
        if now - self._last < self.period:
            return self._cache
        self._last = now
        self._cache["gpu_mem_gb"] = round(torch.cuda.max_memory_reserved(self.device) / 1024**3, 2)
        if self._uuid:
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--id={self._uuid}", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                self._cache["gpu_util"] = float(out.stdout.strip().splitlines()[0])
            except Exception:
                pass
        return self._cache


class TrainingMonitor:
    def __init__(self, log_dir: str | Path, run_name: str, config_summary: dict,
                 device: torch.device, enabled: bool = True):
        self.enabled = enabled
        self.dir = Path(log_dir) / "monitor"
        self.snap_dir = self.dir / "snapshots"
        self.run_name = run_name
        self.config_summary = config_summary
        self.gpu = GpuStats(device)
        self._status: dict = {}
        if enabled:
            self.snap_dir.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.dir / "metrics.jsonl", "a", buffering=1)
            self.status(state="starting")

    # ------------------------------------------------------------------ writing
    def _write(self, record: dict) -> None:
        if not self.enabled:
            return
        record.setdefault("time", time.time())
        self._fh.write(json.dumps(record, default=_json_default) + "\n")

    def event(self, message: str) -> None:
        self._write({"type": "event", "message": message})

    def log_iter(self, **fields) -> None:
        fields.update(self.gpu.sample())
        self._write({"type": "iter", **fields})

    def log_epoch(self, **fields) -> None:
        self._write({"type": "epoch", **fields})

    def status(self, **fields) -> None:
        if not self.enabled:
            return
        self._status.update(fields)
        self._status.update(
            run=self.run_name, pid=os.getpid(), host=socket.gethostname(),
            config=self.config_summary, last_update=time.time(),
        )
        _atomic_write_text(self.dir / "status.json",
                           json.dumps(self._status, indent=1, default=_json_default))

    def close(self, state: str = "finished") -> None:
        if self.enabled:
            self.status(state=state)
            self._fh.close()
            self.enabled = False

    # ------------------------------------------------------------------ snapshots
    def snapshot(self, epoch: int, images: torch.Tensor, labels: torch.Tensor,
                 preds: torch.Tensor, dense_to_rgb: np.ndarray, titles=None) -> Path | None:
        """Save image | label | prediction for a few slices (dense labels, RGB LUT by dense idx)."""
        if not self.enabled:
            return None
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        imgs = images.detach().float().cpu().numpy()
        labs = labels.detach().cpu().numpy().astype(int)
        prds = preds.detach().cpu().numpy().astype(int)
        n = imgs.shape[0]
        mid = imgs.shape[1] // 2
        fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.5 * n), squeeze=False,
                                 facecolor="black")
        lut = np.clip(dense_to_rgb, 0, 255).astype(np.uint8)
        for i in range(n):
            base = imgs[i, mid]
            for j, (arr, name) in enumerate(((None, "image"), (labs[i], "label"),
                                             (prds[i], "prediction"))):
                ax = axes[i, j]
                ax.imshow(base, cmap="gray", vmin=0, vmax=1)
                if arr is not None:
                    rgba = np.zeros(arr.shape + (4,), dtype=np.uint8)
                    rgba[..., :3] = lut[np.clip(arr, 0, len(lut) - 1)]
                    rgba[..., 3] = np.where(arr > 0, 150, 0)
                    ax.imshow(rgba)
                ax.set_axis_off()
                if i == 0:
                    ax.set_title(name, color="white", fontsize=9)
            if titles is not None:
                axes[i, 0].text(2, 10, str(titles[i])[:40], color="yellow", fontsize=6)
        fig.suptitle(f"{self.run_name} - epoch {epoch}", color="white", fontsize=10)
        fig.tight_layout()
        out = self.snap_dir / f"epoch_{epoch:03d}.png"
        fig.savefig(out, dpi=90, facecolor="black")
        plt.close(fig)
        return out


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
