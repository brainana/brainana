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

# IMPORTS
from collections.abc import Callable, Sequence
from typing import Optional

import h5py
from pathlib import Path
import numpy as np
import numpy.typing as npt
import torch
import torchio as tio
import yacs.config
from torch.utils.data import Dataset

from fastsurfer_nn.data_loader import data_utils as data_ultils
from fastsurfer_nn.utils import logging

logger = logging.getLogger(__name__)


# Operator to load imaged for inference
class MultiScaleOrigDataThickSlices(Dataset):
    """
    Load MRI-Image and process it to correct format for network inference.
    """

    zoom: npt.NDArray[float]

    def __init__(
        self,
        orig_data: npt.NDArray,
        orig_zoom: npt.NDArray[float] | Sequence[float],
        cfg: yacs.config.CfgNode,
        transforms: Callable[[npt.NDArray[float]], npt.NDArray[float]] | None = None,
    ):
        """
        Construct object.

        Parameters
        ----------
        orig_data : npt.NDArray
            Original Data.
        orig_zoom : npt.NDArray
            Original zoom factors.
        cfg : yacs.config.CfgNode
            Configuration Node.
        transforms : callable[[npt.NDArray[float]], npt.NDArray[float]], optional
            Transforms for the image, defaults to no transformation.
        """
        orig_max = orig_data.max()
        assert orig_max > 0.8, f"Multi Dataset - orig fail, max removed {orig_max}"
        self.plane = cfg.DATA.PLANE
        self.slice_thickness = cfg.MODEL.NUM_CHANNELS // 2
        self.base_res = 1.0

        # Handle mixed plane mode - should not reach here with "mixed", but safety check
        if self.plane == "mixed":
            raise ValueError(
                "MultiScaleOrigDataThickSlices received PLANE='mixed'. "
                "For mixed-plane models, plane should be set to specific plane (axial/coronal/sagittal) "
                "before creating dataset. This is handled automatically in Inference.run()."
            )

        # Get orientation from config (default to 'lia' for backward compatibility)
        orientation = getattr(cfg.DATA.PREPROCESSING, "ORIENTATION", "lia")

        # Use orientation-aware transform for any plane
        orig_data = data_ultils.transform_for_plane(orig_data, self.plane, orientation)
        zoom_indices = data_ultils.get_zoom_indices_for_plane(self.plane, orientation)
        self.zoom = np.asarray(orig_zoom)[list(zoom_indices)]
        logger.info(
            f"Dataset: loading {self.plane} plane with voxelsize {self.zoom} (orientation: {orientation})"
        )

        # Create thick slices
        orig_thick = data_ultils.get_thick_slices(orig_data, self.slice_thickness)
        orig_thick = np.transpose(orig_thick, (2, 0, 1, 3))
        self.images = orig_thick
        self.count = self.images.shape[0]
        self.transforms = transforms

    def _get_scale_factor(self) -> npt.NDArray[float]:
        """
        Get scaling factor to match original resolution of input image to final resolution of FastSurfer base network.

        Input resolution is taken from voxel size in image header.

        Returns
        -------
        npt.NDArray[float]
            Scale factor along x and y dimension.
        """
        # Safeguard against division by zero or invalid zoom values
        zoom = np.asarray(self.zoom)
        # Replace zero or very small values with base_res to get scale factor of 1.0
        zoom = np.where(np.abs(zoom) < 1e-6, self.base_res, zoom)
        scale = self.base_res / zoom

        return scale

    def __getitem__(self, index: int) -> dict:
        """
        Return a single image and its scale factor.

        Parameters
        ----------
        index : int
            Index of image to get.

        Returns
        -------
        dict
            Dictionary of image and scale factor.
        """
        img = self.images[index]

        scale_factor = self._get_scale_factor()
        if self.transforms is not None:
            img = self.transforms(img)

        return {"image": img, "scale_factor": scale_factor}

    def __len__(self) -> int:
        """
        Return length.

        Returns
        -------
        int
            Count.
        """
        return self.count


# ----------------------------------------------------------------------------- training data
class SliceDataset(Dataset):
    """2.5-D training/validation slices from an HDF5 file written by step2_create_hdf5.

    HDF5 layout (one group per image size, e.g. "256"):
        orig_dataset   [N, H, W, C]  uint8 (new files) or float32 (older files), 0..RESCALE
        aseg_dataset   [N, H, W]     dense class indices
        weight_dataset [N, H, W]     loss weights (float16 or float32)
        zoom_dataset   [N, 2]        in-plane voxel size (mm) after the resize to target size
        subject        [N]           subject (image stem) of each slice

    Each item is a dict:
        image        [C, H, W] uint8 (raw, normalised on the GPU) - or float32 in [0, 1] when
                     a CPU torchio transform is used or the file stores float images
        label        [H, W] uint8 dense indices
        weight       [H, W] float32
        scale_factor [2] float32 = 1 / zoom (FastSurferVINN input)
        subject      int index into ``self.subjects``

    The HDF5 file is opened lazily once per worker process with a small chunk cache
    (``cache_mb``), so several workers do not multiply a large cache in RAM.
    """

    def __init__(self, dataset_path: str, cfg: yacs.config.CfgNode,
                 cpu_transform=None, cache_mb: int = 64):
        self.dataset_path = str(dataset_path)
        self.max_size = cfg.DATA.PADDED_SIZE
        self.rescale = float(cfg.DATA.PREPROCESSING.RESCALE)
        self.cpu_transform = cpu_transform
        self.scale_jitter = cpu_transform is not None and "Gaussian" in cfg.DATA.AUG
        self.cache_bytes = int(cache_mb) * 1024**2
        self._hdf5_file = None

        self.index = []  # (size, idx)
        subjects, subj_idx = [], []
        with h5py.File(self.dataset_path, "r") as hf:
            for size in cfg.DATA.SIZES:
                key = f"{size}"
                if key not in hf:
                    logger.warning(f"Dataset: size {size} not in {self.dataset_path}")
                    continue
                grp = hf[key]
                n = len(grp["orig_dataset"])
                self.index += [(key, i) for i in range(n)]
                names = [s.decode() if isinstance(s, bytes) else str(s)
                         for s in grp["subject"][:]]
                subjects += names
                self.image_dtype = grp["orig_dataset"].dtype
                logger.info(f"Dataset: {n} slices of size {size} in {Path(self.dataset_path).name} "
                            f"(image dtype {self.image_dtype})")
        self.subjects = sorted(set(subjects))
        lut = {s: i for i, s in enumerate(self.subjects)}
        self.subject_of = np.array([lut[s] for s in subjects], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.index)

    def _file(self):
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.dataset_path, "r", rdcc_nbytes=self.cache_bytes,
                                        rdcc_nslots=10007)
        return self._hdf5_file

    def _pad(self, a: np.ndarray) -> np.ndarray:
        from fastsurfer_nn.data_loader.data_utils import pad_to_size

        return pad_to_size(a, self.max_size, mode="edge", pos="top_left")

    def __getitem__(self, index: int) -> dict:
        key, i = self.index[index]
        grp = self._file()[key]
        img = self._pad(grp["orig_dataset"][i])  # H, W, C
        label = self._pad(grp["aseg_dataset"][i])
        weight = self._pad(grp["weight_dataset"][i])
        zoom = np.asarray(grp["zoom_dataset"][i], dtype=np.float32)
        zoom = np.where(np.abs(zoom) < 1e-6, 1.0, zoom)
        scale = 1.0 / zoom

        if self.cpu_transform is not None:
            image, label_t, weight_t, scale = self._cpu_augment(img, label, weight, scale)
        else:
            image = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
            if image.dtype != torch.uint8:
                image = (image.float() / self.rescale).clamp_(0, 1)
            label_t = torch.from_numpy(np.ascontiguousarray(label)).to(torch.uint8)
            weight_t = torch.from_numpy(np.ascontiguousarray(weight)).float()
        return {
            "image": image,
            "label": label_t,
            "weight": weight_t,
            "scale_factor": torch.as_tensor(scale, dtype=torch.float32),
            "subject": int(self.subject_of[index]),
        }

    def _cpu_augment(self, img, label, weight, scale):
        """torchio path (DATA.AUG_DEVICE = 'cpu')."""
        subject = tio.Subject({
            "img": tio.ScalarImage(tensor=np.expand_dims(img.transpose(2, 0, 1), 3).astype(np.float32)),
            "label": tio.LabelMap(tensor=label[np.newaxis, :, :, np.newaxis].astype(np.int16)),
            "weight": tio.LabelMap(tensor=weight[np.newaxis, :, :, np.newaxis].astype(np.float32)),
        })
        out = self.cpu_transform(subject)
        s = 1.0
        for tf in out.get_composed_history():
            args = tf._get_reproducing_arguments()
            if "scales" in args:  # RandomAffine applied as Affine: in-plane scale
                s *= float(np.asarray(args["scales"])[0])
        image = (torch.squeeze(out["img"].data, -1).float() / self.rescale).clamp_(0, 1)
        label_t = torch.squeeze(out["label"].data).to(torch.uint8)
        weight_t = torch.squeeze(out["weight"].data).float()
        scale = scale * s
        if self.scale_jitter:
            scale = np.maximum(scale + np.random.randn() * 0.1, 0.1)
        return image, label_t, weight_t, scale

    def __del__(self):
        if getattr(self, "_hdf5_file", None) is not None:
            try:
                self._hdf5_file.close()
            except Exception:
                pass
