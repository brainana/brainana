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
"""Learning-rate schedules stepped with a FRACTIONAL EPOCH every optimizer step.

All lengths (T_ZERO, STEP_SIZE, WARMUP_EPOCHS, NUM_EPOCHS) are in epochs, so a schedule
means the same thing whatever the dataset size or batch size. The trainer calls
``scheduler.step(epoch_float)`` with ``epoch_float = completed_epochs + iter / iters_per_epoch``
(0-based: the first step of training is 0.0).
"""

from __future__ import annotations

import math

import torch.optim
import yacs.config


class FractionalEpochLR:
    """LR = BASE_LR * factor(epoch) for every param group; resumable via state_dict()."""

    def __init__(self, optimizer: torch.optim.Optimizer, factor_fn, name: str):
        self.optimizer = optimizer
        self.factor_fn = factor_fn
        self.name = name
        self.base_lrs = [g.get("initial_lr", g["lr"]) for g in optimizer.param_groups]
        for g, lr in zip(optimizer.param_groups, self.base_lrs):
            g["initial_lr"] = lr
        self.last_epoch = 0.0
        self.step(0.0)

    def step(self, epoch: float) -> None:
        self.last_epoch = float(epoch)
        f = self.factor_fn(self.last_epoch)
        for g, lr in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = lr * f

    def get_last_lr(self) -> list[float]:
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {"last_epoch": self.last_epoch, "base_lrs": self.base_lrs, "name": self.name}

    def load_state_dict(self, state: dict) -> None:
        if "base_lrs" in state:
            self.base_lrs = list(state["base_lrs"])
        self.step(state.get("last_epoch", 0.0))


def cosine_factor(num_epochs: float, warmup: float, min_factor: float):
    def f(e: float) -> float:
        if warmup > 0 and e < warmup:
            return 0.1 + 0.9 * e / warmup  # start at 10% of BASE_LR
        span = max(num_epochs - warmup, 1e-8)
        p = min(max((e - warmup) / span, 0.0), 1.0)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * p))

    return f


def warm_restarts_factor(t0: float, t_mult: float, min_factor: float):
    def f(e: float) -> float:
        t_i, start = t0, 0.0
        while e >= start + t_i:
            start += t_i
            t_i *= t_mult
        p = (e - start) / t_i
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * p))

    return f


def step_factor(step_size: float, gamma: float):
    return lambda e: gamma ** math.floor(e / step_size)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer, cfg: yacs.config.CfgNode
) -> FractionalEpochLR | None:
    """Build the scheduler named by cfg.OPTIMIZER.LR_SCHEDULER (see module docstring)."""
    o = cfg.OPTIMIZER
    kind = o.LR_SCHEDULER
    min_factor = o.ETA_MIN / o.BASE_LR if o.BASE_LR > 0 else 0.0
    if kind == "cosine":
        fn = cosine_factor(cfg.TRAIN.NUM_EPOCHS, getattr(o, "WARMUP_EPOCHS", 0.0), min_factor)
    elif kind == "cosineWarmRestarts":
        fn = warm_restarts_factor(o.T_ZERO, o.T_MULT, min_factor)
    elif kind == "step_lr":
        fn = step_factor(o.STEP_SIZE, o.GAMMA)
    elif kind in ("NoScheduler", None, ""):
        return None
    else:
        raise ValueError(f"{kind} lr scheduler is not supported")
    return FractionalEpochLR(optimizer, fn, kind)
