# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
"""Training/validation data loaders.

Augmentation runs on the GPU by default (``DATA.AUG_DEVICE = "gpu"``, see
``gpu_augment.GpuAugment``, applied by the trainer). With ``"cpu"`` the torchio transforms
below run in the loader workers instead; each transform has its own probability
(``DATA.AUG_PROBABILITIES``) - earlier versions passed a dict to ``tio.Compose``, which
silently applied every transform with p = 1.
"""

import numpy as np
import torch
import torchio as tio
import yacs.config
from torch.utils.data import DataLoader

from fastsurfer_nn.data_loader import dataset as dset
from fastsurfer_nn.utils import logging

logger = logging.getLogger(__name__)


def _get(node, key, default):
    return getattr(node, key, default) if node is not None else default


def build_cpu_transform(cfg: yacs.config.CfgNode) -> tio.Compose | None:
    """torchio augmentation pipeline with per-transform probabilities."""
    d = cfg.DATA
    probs = d.AUG_PROBABILITIES
    inc = ["img", "label", "weight"]
    make = {
        "Scaling": lambda p: tio.RandomAffine(
            scales=tuple(_get(d.AUG_SCALING, "SCALES", (0.8, 1.15))), degrees=0,
            translation=0, isotropic=True, center="image", default_pad_value="minimum",
            image_interpolation="linear", include=inc, p=p),
        "Rotation": lambda p: tio.RandomAffine(
            scales=(1.0, 1.0), degrees=(0, 0, 0, 0, -_get(d.AUG_ROTATION, "DEGREES", 5),
                                         _get(d.AUG_ROTATION, "DEGREES", 5)),
            translation=0, center="image", default_pad_value="minimum",
            image_interpolation="linear", include=inc, p=p),
        "Translation": lambda p: tio.RandomAffine(
            scales=(1.0, 1.0), degrees=0,
            translation=tuple(_get(d.AUG_TRANSLATION, "TRANSLATION", (15.0, 15.0, 0))),
            center="image", default_pad_value="minimum", image_interpolation="linear",
            include=inc, p=p),
        "Elastic": lambda p: tio.RandomElasticDeformation(
            num_control_points=_get(d.AUG_ELASTIC, "NUM_CONTROL_POINTS", 5),
            max_displacement=tuple(_get(d.AUG_ELASTIC, "MAX_DISPLACEMENT", (20, 20, 0))),
            locked_borders=2, image_interpolation="linear", include=inc, p=p),
        "RAnisotropy": lambda p: tio.RandomAnisotropy(
            axes=tuple(_get(d.AUG_RANISOTROPY, "AXES", (0, 1))),
            downsampling=tuple(_get(d.AUG_RANISOTROPY, "DOWNSAMPLING", (1.1, 1.3))),
            image_interpolation="linear", include=["img"], p=p),
        "BiasField": lambda p: tio.RandomBiasField(
            coefficients=tuple(_get(d.AUG_BIASFIELD, "COEFFICIENTS", (0.3, 0.7))),
            order=_get(d.AUG_BIASFIELD, "ORDER", 3), include=["img"], p=p),
        "RGamma": lambda p: tio.RandomGamma(
            log_gamma=tuple(_get(d.AUG_RGAMMA, "LOG_GAMMA", (-0.1, 0.1))), include=["img"], p=p),
        "Noise": lambda p: tio.RandomNoise(
            std=(0, 255.0 * _get(_get(d, "AUG_NOISE", None), "STD", 0.03)), include=["img"], p=p),
    }
    # Order: geometric first, then intensity (Gaussian = scale-factor jitter, in the dataset)
    order = ["Scaling", "Rotation", "Translation", "Elastic", "RAnisotropy",
             "BiasField", "RGamma", "Noise"]
    tfs = []
    for name in order:
        if name in d.AUG and name in make:
            p = float(_get(probs, name, 0.8))
            if p > 0:
                tfs.append(make[name](p))
    unknown = [a for a in d.AUG if a not in make and a not in ("Gaussian", "None")]
    if unknown:
        raise ValueError(f"unknown augmentation(s): {unknown}")
    return tio.Compose(tfs) if tfs else None


def site_balance_weights(dataset, mode: str) -> np.ndarray | None:
    """Per-slice sampling weights so that site s gets a share of draws proportional to
    sqrt(n_subjects_s) ("sqrt") or 1 ("equal"); slices within a site stay equally likely."""
    if mode in ("none", "", None):
        return None
    if mode not in ("sqrt", "equal"):
        raise ValueError(f"DATA.SITE_BALANCE must be none, sqrt or equal, not {mode}")
    subj_site = np.array([s.split("_", 1)[0] if s.startswith("site-") else "unknown"
                          for s in dataset.subjects])
    slice_site = subj_site[dataset.subject_of]
    w = np.zeros(len(slice_site), dtype=np.float64)
    for site in np.unique(slice_site):
        m = slice_site == site
        n_subj = len(np.unique(dataset.subject_of[m]))
        share = np.sqrt(n_subj) if mode == "sqrt" else 1.0
        w[m] = share / m.sum()
    return w / w.sum()


def _worker_init(worker_id: int) -> None:
    # distinct numpy streams per worker (torch's own RNG is already seeded per worker)
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)


def get_dataloader(cfg: yacs.config.CfgNode, mode: str) -> DataLoader:
    """Data loader for "train" (shuffled) or "val" (ordered, never augmented)."""
    assert mode in ["train", "val"], f"dataloader mode is incorrect {mode}"
    path = cfg.DATA.PATH_HDF5_TRAIN if mode == "train" else cfg.DATA.PATH_HDF5_VAL
    cache_mb = int(_get(cfg.DATA_LOADER, "HDF5_CACHE_MB", 64))
    cpu_tf = None
    if mode == "train" and _get(cfg.DATA, "AUG_DEVICE", "gpu") == "cpu" and "None" not in cfg.DATA.AUG:
        cpu_tf = build_cpu_transform(cfg)
    dataset = dset.SliceDataset(path, cfg, cpu_transform=cpu_tf, cache_mb=cache_mb)
    if len(dataset) == 0:
        raise ValueError(
            f"DataLoader: no slices in {path} (mode={mode}, plane={cfg.DATA.PLANE}, "
            f"sizes={cfg.DATA.SIZES}). Re-run step2_create_hdf5.py --split_type {mode}."
        )
    if mode == "train":
        workers = int(cfg.DATA_LOADER.NUM_WORKERS)
        batch = cfg.TRAIN.BATCH_SIZE
    else:
        workers = int(_get(cfg.DATA_LOADER, "VAL_NUM_WORKERS", 1))
        batch = int(_get(cfg.TEST, "BATCH_SIZE", cfg.TRAIN.BATCH_SIZE)) or cfg.TRAIN.BATCH_SIZE
        batch = max(batch, cfg.TRAIN.BATCH_SIZE)
    logger.info(f"DataLoader: {mode} {len(dataset)} slices, {len(dataset.subjects)} subjects, "
                f"batch {batch}, {workers} workers, augmentation "
                f"{'cpu (torchio)' if cpu_tf is not None else ('gpu' if mode == 'train' else 'none')}")
    sampler = None
    if mode == "train":
        w = site_balance_weights(dataset, _get(cfg.DATA, "SITE_BALANCE", "none"))
        if w is not None:
            from torch.utils.data import WeightedRandomSampler

            gen = torch.Generator().manual_seed(int(cfg.RNG_SEED))
            sampler = WeightedRandomSampler(torch.as_tensor(w), num_samples=len(dataset),
                                            replacement=True, generator=gen)
            sites = sorted({s.split("_", 1)[0] for s in dataset.subjects})
            logger.info(f"DataLoader: site-balanced sampling ({cfg.DATA.SITE_BALANCE}) over "
                        f"{len(sites)} sites")
    kwargs = dict(batch_size=batch, num_workers=workers,
                  shuffle=(mode == "train" and sampler is None), sampler=sampler,
                  pin_memory=bool(_get(cfg.DATA_LOADER, "PIN_MEMORY", True)),
                  drop_last=(mode == "train"), worker_init_fn=_worker_init)
    if workers > 0:
        kwargs.update(prefetch_factor=int(_get(cfg.DATA_LOADER, "PREFETCH_FACTOR", 2)),
                      persistent_workers=True)
    return DataLoader(dataset, **kwargs)
