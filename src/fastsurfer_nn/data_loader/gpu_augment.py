"""Batched data augmentation on the GPU (replaces per-slice torchio in the loader workers).

Applied by the trainer to a whole batch already on the device:

    images  [B, C, H, W] float in [0, 1]  (C = 2.5-D slab channels)
    labels  [B, H, W]    long (dense class indices)
    weights [B, H, W]    float (loss weight map)
    scale   [B, 2]       float (network scale factor = 1 / in-plane voxel size) - kept on
                         the CPU: FastSurferVINN groups samples by scale factor in Python,
                         which would force a GPU sync per comparison if it lived on the GPU.

Each augmentation is applied per sample with its own probability (cfg.DATA.AUG_PROBABILITIES):

* Scaling / Rotation / Translation - one combined affine per sample, applied with
  ``affine_grid`` + ``grid_sample`` (bilinear for images, nearest for labels and weights,
  zeros outside the field = background / weight 0). Scaling by s magnifies the anatomy, so
  the effective voxel size becomes zoom / s and the scale factor is multiplied by s
  (same convention as the torchio path: ``img_zoom *= 1 / scale_aug``).
* BiasField - multiplicative exp(polynomial) field of order AUG_BIASFIELD.ORDER, coefficients
  ~ U(-c, c) with c = max(AUG_BIASFIELD.COEFFICIENTS) (as torchio's RandomBiasField).
* RGamma - img ** exp(U(LOG_GAMMA)).
* Noise - additive Gaussian image noise, std ~ U(0, AUG_NOISE.STD).
* Gaussian - legacy FastSurfer jitter of the SCALE FACTOR input: + N(0, 0.1), clamped >= 0.1.

Elastic and RAnisotropy are not implemented here; use DATA.AUG_DEVICE = "cpu" for them.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

GEOMETRIC = ("Scaling", "Rotation", "Translation")
SUPPORTED = GEOMETRIC + ("BiasField", "RGamma", "Noise", "Gaussian")
CPU_ONLY = ("Elastic", "RAnisotropy")


def _cfg_get(node, key, default):
    return getattr(node, key, default) if node is not None else default


class GpuAugment:
    def __init__(self, cfg, generator: torch.Generator | None = None):
        d = cfg.DATA
        augs = [a for a in d.AUG if a != "None"]
        bad = [a for a in augs if a in CPU_ONLY]
        if bad:
            raise ValueError(f"{bad} need DATA.AUG_DEVICE='cpu' (not available on GPU)")
        unknown = [a for a in augs if a not in SUPPORTED]
        if unknown:
            raise ValueError(f"unknown augmentation(s) {unknown}; supported: {SUPPORTED}")
        probs = d.AUG_PROBABILITIES
        self.p = {a: float(_cfg_get(probs, a, 0.8)) for a in augs}
        self.p = {a: p for a, p in self.p.items() if p > 0}
        sc = _cfg_get(d, "AUG_SCALING", None)
        self.scales = tuple(_cfg_get(sc, "SCALES", (0.8, 1.15)))
        self.degrees = float(_cfg_get(_cfg_get(d, "AUG_ROTATION", None), "DEGREES", 5))
        tr = tuple(_cfg_get(_cfg_get(d, "AUG_TRANSLATION", None), "TRANSLATION", (15.0, 15.0, 0)))
        self.translation = (float(tr[0]), float(tr[1]))
        bf = _cfg_get(d, "AUG_BIASFIELD", None)
        coef = _cfg_get(bf, "COEFFICIENTS", (0.3, 0.7))
        self.bias_coef = float(max(coef)) if isinstance(coef, (list, tuple)) else float(coef)
        self.bias_order = int(_cfg_get(bf, "ORDER", 3))
        self.log_gamma = tuple(_cfg_get(_cfg_get(d, "AUG_RGAMMA", None), "LOG_GAMMA", (-0.1, 0.1)))
        self.noise_std = float(_cfg_get(_cfg_get(d, "AUG_NOISE", None), "STD", 0.03))
        self.g = generator

    def describe(self) -> str:
        return ", ".join(f"{a} (p={p:.2f})" for a, p in self.p.items()) or "none"

    # ------------------------------------------------------------------ helpers
    # all random numbers are drawn on the CPU (seeded generator); small tensors only
    def _rand(self, n, lo=0.0, hi=1.0):
        return lo + (hi - lo) * torch.rand(n, generator=self.g)

    def _mask(self, name, n):
        p = self.p.get(name, 0.0)
        if p <= 0:
            return torch.zeros(n, dtype=torch.bool)
        return self._rand(n) < p

    # ------------------------------------------------------------------ main
    @torch.no_grad()
    def __call__(self, images, labels, weights, scale):
        b, _, h, w = images.shape
        dev = images.device
        scale = scale.detach().to("cpu", torch.float32).clone()

        # ---- geometric: one affine per sample (parameters on CPU, warp on GPU)
        if any(a in self.p for a in GEOMETRIC):
            s = torch.ones(b)
            m = self._mask("Scaling", b)
            s[m] = self._rand(int(m.sum()), *self.scales)
            theta = torch.zeros(b)
            m = self._mask("Rotation", b)
            theta[m] = self._rand(int(m.sum()), -self.degrees, self.degrees) * math.pi / 180
            tx, ty = torch.zeros(b), torch.zeros(b)
            m = self._mask("Translation", b)
            k = int(m.sum())
            tx[m] = self._rand(k, -self.translation[0], self.translation[0]) * 2 / w
            ty[m] = self._rand(k, -self.translation[1], self.translation[1]) * 2 / h
            active = (s != 1) | (theta != 0) | (tx != 0) | (ty != 0)
            if active.any():
                cos, sin = torch.cos(theta) / s, torch.sin(theta) / s
                a = torch.zeros(b, 2, 3)
                a[:, 0, 0], a[:, 0, 1], a[:, 0, 2] = cos, -sin, tx
                a[:, 1, 0], a[:, 1, 1], a[:, 1, 2] = sin, cos, ty
                idx = active.nonzero(as_tuple=True)[0]
                idx_d = idx.to(dev)
                grid = F.affine_grid(a[idx].to(dev), (len(idx), 1, h, w), align_corners=False)
                images[idx_d] = F.grid_sample(images[idx_d], grid.to(images.dtype),
                                              mode="bilinear", padding_mode="zeros",
                                              align_corners=False)
                lw = torch.stack([labels[idx_d].float(), weights[idx_d].float()], dim=1)
                lw = F.grid_sample(lw, grid, mode="nearest", padding_mode="zeros",
                                   align_corners=False)
                labels[idx_d] = lw[:, 0].round().long()
                weights[idx_d] = lw[:, 1].to(weights.dtype)
                scale = scale * s[:, None]

        # ---- intensity
        if "BiasField" in self.p:
            m = self._mask("BiasField", b)
            if m.any():
                md = m.to(dev)
                images[md] = images[md] * self._bias_field(int(m.sum()), h, w, dev)[:, None]
        if "RGamma" in self.p:
            m = self._mask("RGamma", b)
            if m.any():
                md = m.to(dev)
                g = torch.exp(self._rand(int(m.sum()), *self.log_gamma)).to(dev)
                images[md] = images[md].clamp_min(0) ** g[:, None, None, None]
        if "Noise" in self.p:
            m = self._mask("Noise", b)
            if m.any():
                md = m.to(dev)
                std = self._rand(int(m.sum()), 0, self.noise_std).to(dev)[:, None, None, None]
                images[md] = images[md] + torch.randn_like(images[md]) * std
        images.clamp_(0.0, 1.0)

        if "Gaussian" in self.p:  # legacy: jitter the scale-factor input, not the image
            m = self._mask("Gaussian", b)
            if m.any():
                jitter = torch.randn(int(m.sum()), 1, generator=self.g) * 0.1
                scale[m] = (scale[m] + jitter).clamp_min(0.1)
        return images, labels, weights, scale

    def _bias_field(self, n, h, w, dev):
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=dev),
                                torch.linspace(-1, 1, w, device=dev), indexing="ij")
        field = torch.zeros(n, h, w, device=dev)
        for i in range(self.bias_order + 1):
            for j in range(self.bias_order + 1 - i):
                c = self._rand(n, -self.bias_coef, self.bias_coef).to(dev)
                field = field + c[:, None, None] * (xx ** i) * (yy ** j)
        return torch.exp(field)
