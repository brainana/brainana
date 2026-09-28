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

import os

# IMPORTS
import time
from typing import Optional

import numpy as np
import torch
import yacs.config
from numpy import typing as npt
from pandas import DataFrame
from torch.utils.data import DataLoader
from torchvision import transforms

from fastsurfer_nn.data_loader.data_transforms import ToTensorTest
from fastsurfer_nn.data_loader.data_utils import (
    depad_volume,
    get_permute_order_for_plane,
    map_prediction_sagittal2full,
    pad_volume_edges_percent,
)
from fastsurfer_nn.data_loader.dataset import MultiScaleOrigDataThickSlices
from fastsurfer_nn.models.networks import build_model
from fastsurfer_nn.utils import logging
from fastsurfer_nn.utils.checkpoint import extract_atlas_metadata, read_checkpoint_file
from fastsurfer_nn.utils.constants import LARGE_IMAGE_THRESHOLD
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)


class Inference:
    """Model evaluation class to run inference using fastsurfer_nn.

    Attributes
    ----------
    permute_order : Dict[str, Tuple[int, int, int, int]]
        Permutation order for axial, coronal, and sagittal
    device : Optional[torch.device])
        Device specification for distributed computation usage.
    default_device : torch.device
        Default device specification for distributed computation usage.
    cfg : yacs.config.CfgNode
        Configuration Node
    model_parallel : bool
        Option for parallel run
    model : torch.nn.Module
        Neural network model
    model_name : str
        Name of the model
    alpha : Dict[str, float]
        Alpha values for different planes.
    post_prediction_mapping_hook
        Hook for post prediction mapping.

    Methods
    -------
    setup_model
        Set up the initial model
    set_cfg
        Set configuration node
    to
        Moves and/or casts the parameters and buffers.
    load_checkpoint
        Load the checkpoint
    eval
        Evaluate predictions
    run
        Run the loaded model
    """

    permute_order: dict[str, tuple[int, int, int, int]]
    device: torch.device | None
    default_device: torch.device

    def __init__(
        self,
        cfg: yacs.config.CfgNode,
        device: torch.device,
        ckpt: str = "",
        lut: None | str | np.ndarray | DataFrame = None,
    ):
        """
        Construct Inference object.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            Configuration Node.
        device : torch.device
            Device specification for distributed computation usage.
        ckpt : str
            String or os.PathLike object containing the name to the checkpoint file (Default value = "").
        lut : str, np.ndarray, DataFrame, optional
             Lookup table for mapping.
        """
        # Set random seed from configs.
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        self.cfg = cfg

        # Switch on denormal flushing for faster CPU processing
        # seems to have less of an effect on VINN than old CNN
        torch.set_flush_denormal(True)

        self.default_device = device

        # Options for parallel run
        # Device-type checks first: device_count() initializes the CUDA driver.
        self.model_parallel = (
            self.default_device.type == "cuda"
            and self.default_device.index is None
            and torch.cuda.device_count() > 1
        )

        # Initial model setup
        self.model = None
        self._model_not_init = None
        self.setup_model(cfg, device=self.default_device)
        self.model_name = self.cfg.MODEL.MODEL_NAME

        # Initialize plane weights from config
        self.alpha = {
            "coronal": cfg.MULTIVIEW.PLANE_WEIGHTS.CORONAL,
            "axial": cfg.MULTIVIEW.PLANE_WEIGHTS.AXIAL,
            "sagittal": cfg.MULTIVIEW.PLANE_WEIGHTS.SAGITTAL,
        }

        # Compute permute_order dynamically based on orientation
        # This allows the model to work with any orientation (LIA, RAS, etc.)
        orientation = getattr(cfg.DATA.PREPROCESSING, "ORIENTATION", "lia")
        self.orientation = orientation
        self.permute_order = {
            "axial": get_permute_order_for_plane("axial", orientation),
            "coronal": get_permute_order_for_plane("coronal", orientation),
            "sagittal": get_permute_order_for_plane("sagittal", orientation),
        }
        logger.debug(
            f"Inference: using orientation '{orientation}', permute_order={self.permute_order}"
        )
        self.lut = lut

        # Initial checkpoint loading
        if ckpt:
            # this also moves the model to the para
            self.load_checkpoint(ckpt)

    def setup_model(self, cfg=None, device: torch.device = None):
        """
        Set up the model.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            Configuration Node (Default value = None).
        device : torch.device
            Device specification for distributed computation usage. (Default value = None).
        """
        if cfg is not None:
            self.cfg = cfg
        if device is None:
            device = self.default_device

        # Set up model
        self._model_not_init = build_model(
            self.cfg
        )  # ~ model = fastsurfer_nn(params_network)
        self._model_not_init.to(device)
        self.device = None

    def set_cfg(self, cfg: yacs.config.CfgNode):
        """
        Set the configuration node.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            Configuration node.
        """
        self.cfg = cfg

    def to(self, device: torch.device | None = None):
        """
        Move and/or cast the parameters and buffers.

        Parameters
        ----------
        device : Optional[torch.device]
            The desired device of the parameters and buffers in this module (Default value = None).
        """
        if self.model_parallel:
            raise RuntimeError(
                "Moving the model to other devices is not supported for multi-device models."
            )
        _device = self.default_device if device is None else device
        self.device = _device
        self.model.to(device=_device)

    def load_checkpoint(self, ckpt: str | os.PathLike):
        """
        Load the checkpoint and set device and model.

        This method now also extracts atlas metadata from the checkpoint,
        which is critical for correctly mapping model outputs to label IDs.

        Parameters
        ----------
        ckpt : Union[str, os.PathLike]
            String or os.PathLike object containing the name to the checkpoint file.

        Returns
        -------
        dict, None
            Atlas metadata extracted from checkpoint, or None if not available.
        """
        logger.info(f"Checkpoint: loading {ckpt}")

        self.model = self._model_not_init
        # If device is None, the model has never been loaded (still in random initial configuration)
        if self.device is None:
            self.device = self.default_device
        load_device = self.device

        # workaround for mps (directly loading to map_location=mps results in zeros)
        if self.device.type == "mps":
            load_device = "cpu"
        # make sure the model is, where it is supposed to be
        self.model.to(load_device)

        # Load checkpoint using centralized function
        checkpoint = read_checkpoint_file(ckpt, map_location=load_device)
        self.model.load_state_dict(checkpoint["model_state"])

        # Phase 1: Extract atlas metadata from checkpoint
        # This ensures we use the EXACT same label mapping as during training
        atlas_metadata = None
        try:
            atlas_metadata = extract_atlas_metadata(ckpt)

            if atlas_metadata:
                logger.info(
                    f"Checkpoint: loaded with atlas={atlas_metadata['atlas_name']}, "
                    f"classes={atlas_metadata['num_classes']}, plane={atlas_metadata['plane']}"
                )
                if "source" in atlas_metadata:
                    logger.info(
                        f"Checkpoint: atlas metadata source={atlas_metadata['source']}"
                    )

                # Store atlas metadata for later use
                self.atlas_metadata = atlas_metadata
            else:
                logger.warning(
                    "Checkpoint: could not extract atlas metadata, will need manual specification"
                )
                self.atlas_metadata = None
        except Exception:
            logger.warning(f"Checkpoint: No atlas metadata found in {ckpt}")
            self.atlas_metadata = None

        # workaround for mps (move the model back to mps)
        if self.device.type == "mps":
            self.model.to(self.device)

        if self.model_parallel:
            self.model = torch.nn.DataParallel(self.model)

        return atlas_metadata

    def get_modelname(self) -> str:
        """
        Return the model name.

        Returns
        -------
        str
            The name of the model.
        """
        return self.model_name

    def get_cfg(self) -> yacs.config.CfgNode:
        """
        Return the configurations.

        Returns
        -------
        yacs.config.CfgNode
            Configuration node.
        """
        return self.cfg

    def get_num_classes(self) -> int:
        """
        Return the number of classes.

        Returns
        -------
        int
            The number of classes.
        """
        return self.cfg.MODEL.NUM_CLASSES

    def get_plane(self) -> str:
        """
        Return the plane.

        Returns
        -------
        str
            The plane used in the model. Returns "mixed" if model is plane-agnostic.
        """
        plane = self.cfg.DATA.PLANE
        if plane == "mixed":
            # For mixed mode, model is plane-agnostic but we need to know it's mixed
            return "mixed"
        return plane

    def get_model_height(self) -> int:
        """
        Return the model height.

        Returns
        -------
        int
            The height of the model.
        """
        return self.cfg.MODEL.HEIGHT

    def get_model_width(self) -> int:
        """
        Return the model width.

        Returns
        -------
        int
            The width of the model.
        """
        return self.cfg.MODEL.WIDTH

    def get_max_size(self) -> int | tuple[int, int]:
        """
        Return the max size.

        Returns
        -------
        int | tuple[int, int]
            The maximum size, either a single value or a tuple (width, height).
        """
        if self.cfg.MODEL.OUT_TENSOR_WIDTH == self.cfg.MODEL.OUT_TENSOR_HEIGHT:
            return self.cfg.MODEL.OUT_TENSOR_WIDTH
        else:
            return self.cfg.MODEL.OUT_TENSOR_WIDTH, self.cfg.MODEL.OUT_TENSOR_HEIGHT

    def get_device(self) -> torch.device:
        """
        Return the device.

        Returns
        -------
        torch.device
            The device used for computation.
        """
        return self.device

    def get_atlas_metadata(self) -> dict | None:
        """
        Return the atlas metadata extracted from the checkpoint.

        Returns
        -------
        dict, None
            Atlas metadata dictionary with keys:
            - atlas_name: str
            - num_classes: int
            - plane: str
            - dense_to_sparse_mapping: np.ndarray
            - source: str
            Returns None if no metadata is available.
        """
        return getattr(self, "atlas_metadata", None)

    @torch.no_grad()
    def eval(
        self,
        init_pred: torch.Tensor,
        val_loader: DataLoader,
        *,
        out_scale: Optional = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform prediction and inplace-aggregate views into pred_prob.

        Parameters
        ----------
        init_pred : torch.Tensor
            Initial prediction.
        val_loader : DataLoader
            Validation loader.
        out_scale : Optional
            Output scale (Default value = None).
        out : torch.Tensor, optional
            Previous prediction tensor (Default value = None).

        Returns
        -------
        torch.Tensor
            Prediction probability tensor.
        """
        self.model.eval()
        # we should check here, whether the DataLoader is a Random or a SequentialSampler, but we cannot easily.
        if not isinstance(val_loader.sampler, torch.utils.data.SequentialSampler):
            logger.warning(
                "Inference: validation loader does not use SequentialSampler, may interfere with batch sorting"
            )

        # ========================================================================
        # Setup: Predictions should match conformed image size directly
        # ========================================================================
        # init_pred has shape matching the conformed image (e.g., 96×96×Z×num_classes)
        # Model processes images at conformed size (no resize, no padding)
        # Predictions should match conformed dimensions directly

        # Setup for aggregating predictions into output tensor
        plane = self.cfg.DATA.PLANE
        # For mixed mode, plane is temporarily set to current plane in run() method
        if plane == "mixed":
            # This shouldn't happen in eval() - plane should be set to specific plane by run()
            raise ValueError(
                "eval() called with PLANE='mixed'. This should be handled by run() method."
            )

        index_of_current_plane = self.permute_order[plane].index(0)
        output_slice_indices = [
            slice(None) for _ in range(4)
        ]  # Will be filled per batch

        if out is None:
            out = init_pred.detach().clone()

        start_index = 0
        log_batch_idx = None
        with logging_redirect_tqdm():
            try:
                for batch_idx, batch in tqdm(
                    enumerate(val_loader),
                    total=len(val_loader),
                    unit="batch",
                    disable=not logger.isEnabledFor(logging.INFO),
                ):
                    log_batch_idx = batch_idx
                    # move data to the model device
                    images, scale_factors = batch["image"].to(self.device), batch[
                        "scale_factor"
                    ].to(self.device)

                    # predict the current batch, outputs logits
                    # Model processes images at conformed size (no resize, no padding)
                    # Output predictions should match conformed dimensions directly
                    pred = self.model(images, scale_factors, out_scale)
                    batch_size = pred.shape[0]
                    end_index = start_index + batch_size

                    # check if we need a special mapping (e.g. as for sagittal)
                    # Note: For mixed mode, plane is set temporarily in run(), so this works correctly
                    current_plane = self.cfg.DATA.PLANE
                    if current_plane == "sagittal":
                        # Determine atlas name from config or environment
                        atlas_name = None
                        if (
                            hasattr(self.cfg.DATA, "CLASS_OPTIONS")
                            and self.cfg.DATA.CLASS_OPTIONS
                        ):
                            # Extract atlas name from class options (e.g., ["arm2"] -> "ARM2")
                            atlas_name = self.cfg.DATA.CLASS_OPTIONS[0].upper()

                        pred = map_prediction_sagittal2full(
                            pred,
                            num_classes=self.get_num_classes(),
                            atlas_name=atlas_name,
                        )

                    # permute the prediction into the out slice order
                    pred = pred.permute(*self.permute_order[plane]).to(
                        out.device
                    )  # the to-operation is implicit

                    # ========================================================================
                    # Predictions are already at conformed image size (no cropping needed)
                    # ========================================================================
                    # Model processes images at conformed size, so predictions match directly
                    # add prediction logits into the output (same as multiplying probabilities)
                    output_slice_indices[index_of_current_plane] = slice(
                        start_index, end_index
                    )
                    out[tuple(output_slice_indices)].add_(pred, alpha=self.alpha[plane])
                    start_index = end_index

                    # Explicitly delete intermediate tensors to free GPU memory
                    del images, scale_factors, pred
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

            except:
                batch_num = (
                    log_batch_idx + 1 if log_batch_idx is not None else "unknown"
                )
                logger.exception(
                    f"Exception in batch {batch_num} of {plane} inference."
                )
                raise
            else:
                batch_num = log_batch_idx + 1 if log_batch_idx is not None else 0
                logger.info(
                    f"Inference: completed {batch_num} batches for {plane} plane"
                )

        return out

    @torch.no_grad()
    def run(
        self,
        init_pred: torch.Tensor,
        img_filename: str,
        orig_data: npt.NDArray,
        orig_zoom: npt.NDArray,
        out: torch.Tensor | None = None,
        out_res: int | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """
        Run the loaded model on the data (T1) from orig_data and
        img_filename (for messages only) with scale factors orig_zoom.

        For mixed-plane models (PLANE="mixed"), processes all 3 planes and aggregates predictions.

        Parameters
        ----------
        init_pred : torch.Tensor
            Initial prediction.
        img_filename : str
            Original image filename.
        orig_data : npt.NDArray
            Original image data.
        orig_zoom : npt.NDArray
            Original zoom.
        out : Optional[torch.Tensor]
            Updated output tensor (Default = None).
        out_res : Optional[int]
            Output resolution (Default value = None).
        batch_size : int, optional
            Batch size.

        Returns
        -------
        torch.Tensor
            Prediction probability tensor.
        """
        plane = self.cfg.DATA.PLANE

        # Apply edge padding if enabled (inference only)
        padding_percent = getattr(self.cfg.TEST, "EDGE_PADDING_PERCENT", 0.0)
        pad_width = ((0, 0), (0, 0), (0, 0))
        orig_data_padded = orig_data

        # Disable padding for large images to avoid OOM
        max_dim = max(orig_data.shape)
        if padding_percent > 0.0 and max_dim > LARGE_IMAGE_THRESHOLD:
            logger.warning(
                f"Inference: Disabling edge padding for large image (max_dim={max_dim} > {LARGE_IMAGE_THRESHOLD}). "
                f"This saves significant GPU memory. Edge predictions may be slightly less accurate."
            )
            padding_percent = 0.0

        if padding_percent > 0.0:
            logger.info(
                f"Inference: Applying {padding_percent*100:.1f}% edge padding to help recognize brain tissue near boundaries"
            )
            orig_data_padded, pad_width = pad_volume_edges_percent(
                orig_data, padding_percent, mode="edge"
            )

            # Create padded prediction tensor matching padded data dimensions
            padded_shape = orig_data_padded.shape + (
                init_pred.shape[3],
            )  # (H, W, D, num_classes)

            # Calculate memory requirements for padded tensor
            padded_size_gb = (
                np.prod(padded_shape) * init_pred.element_size() / (1024**3)
            )

            # Warn if padding creates an extremely large tensor
            if padded_size_gb > 20.0:
                logger.warning(
                    f"Inference: Padding will create a very large tensor (~{padded_size_gb:.2f} GB). "
                    f"This may cause memory issues. Consider reducing EDGE_PADDING_PERCENT or disabling padding."
                )

            if out is not None and out.shape[:3] == orig_data.shape:
                # Pad the provided out tensor to match padded dimensions
                pad_h, pad_w, pad_d = pad_width[0][0], pad_width[1][0], pad_width[2][0]

                # Check GPU memory before padding
                if out.device.type == "cuda":
                    device_idx = out.device.index if out.device.index is not None else 0
                    free_memory = torch.cuda.get_device_properties(
                        device_idx
                    ).total_memory - torch.cuda.memory_allocated(device_idx)
                    free_memory_gb = free_memory / (1024**3)

                    logger.info(
                        f"Inference: Padding requires ~{padded_size_gb:.2f} GB, "
                        f"available GPU memory: {free_memory_gb:.2f} GB"
                    )

                    if free_memory_gb < padded_size_gb * 1.2:  # Need 20% buffer
                        logger.warning(
                            f"Inference: Insufficient GPU memory for padding ({padded_size_gb:.2f} GB required, "
                            f"{free_memory_gb:.2f} GB available). Moving padding operation to CPU."
                        )
                        # Move to CPU for padding, then move back
                        # CRITICAL: Delete original GPU tensor before creating padded version
                        # to avoid needing space for both simultaneously
                        target_device = out.device  # Save device before moving
                        out_cpu = out.cpu()
                        del out  # Explicitly delete original GPU tensor to free memory
                        torch.cuda.empty_cache()  # Free GPU memory
                        out_cpu = torch.nn.functional.pad(
                            out_cpu,
                            (0, 0, pad_d, pad_d, pad_w, pad_w, pad_h, pad_h),
                            mode="constant",
                            value=0,
                        )
                        out = out_cpu.to(target_device)  # Move back to original device
                        logger.info(
                            "Inference: Padding completed on CPU and moved back to GPU"
                        )
                    else:
                        # Enough memory, pad on GPU
                        out = torch.nn.functional.pad(
                            out,
                            (0, 0, pad_d, pad_d, pad_w, pad_w, pad_h, pad_h),
                            mode="constant",
                            value=0,
                        )
                else:
                    # CPU device, pad directly
                    out = torch.nn.functional.pad(
                        out,
                        (0, 0, pad_d, pad_d, pad_w, pad_w, pad_h, pad_h),
                        mode="constant",
                        value=0,
                    )
            else:
                # Create new padded tensor - check memory first
                if init_pred.device.type == "cuda":
                    device_idx = (
                        init_pred.device.index
                        if init_pred.device.index is not None
                        else 0
                    )
                    free_memory = torch.cuda.get_device_properties(
                        device_idx
                    ).total_memory - torch.cuda.memory_allocated(device_idx)
                    free_memory_gb = free_memory / (1024**3)

                    logger.info(
                        f"Inference: Creating padded tensor requires ~{padded_size_gb:.2f} GB, "
                        f"available GPU memory: {free_memory_gb:.2f} GB"
                    )

                    if free_memory_gb < padded_size_gb * 1.2:  # Need 20% buffer
                        logger.warning(
                            f"Inference: Insufficient GPU memory for padded tensor ({padded_size_gb:.2f} GB required, "
                            f"{free_memory_gb:.2f} GB available). Creating on CPU."
                        )
                        # Create on CPU first, then move to GPU
                        out = torch.zeros(
                            padded_shape,
                            dtype=init_pred.dtype,
                            device="cpu",
                            requires_grad=False,
                        )
                        out = out.to(init_pred.device)
                        logger.info(
                            "Inference: Padded tensor created on CPU and moved to GPU"
                        )
                    else:
                        # Enough memory, create on GPU
                        out = torch.zeros(
                            padded_shape,
                            dtype=init_pred.dtype,
                            device=init_pred.device,
                            requires_grad=False,
                        )
                else:
                    # CPU device, create directly
                    out = torch.zeros(
                        padded_shape,
                        dtype=init_pred.dtype,
                        device=init_pred.device,
                        requires_grad=False,
                    )

        # Ensure out is set (for no-padding case)
        if out is None:
            out = init_pred.detach().clone()

        # Handle mixed-plane mode: process all 3 planes and aggregate
        if plane == "mixed":
            logger.info(
                "Inference: Mixed-plane mode - processing all 3 planes (axial, coronal, sagittal)"
            )
            planes_to_process = ["axial", "coronal", "sagittal"]

            start = time.time()
            for current_plane in planes_to_process:
                logger.info(f"Inference: Processing {current_plane} plane...")

                # Temporarily set plane in config for this iteration
                original_plane = self.cfg.DATA.PLANE
                self.cfg.DATA.PLANE = current_plane

                try:
                    # Set up DataLoader for this plane
                    rescale = self.cfg.DATA.PREPROCESSING.RESCALE
                    test_dataset = MultiScaleOrigDataThickSlices(
                        orig_data_padded,
                        orig_zoom,
                        self.cfg,
                        transforms=transforms.Compose([ToTensorTest(rescale=rescale)]),
                    )

                    test_data_loader = DataLoader(
                        dataset=test_dataset,
                        shuffle=False,
                        batch_size=self.cfg.TEST.BATCH_SIZE
                        if batch_size is None
                        else batch_size,
                    )

                    # Run evaluation for this plane (aggregates into out)
                    self.eval(init_pred, test_data_loader, out=out, out_scale=out_res)

                finally:
                    # Restore original plane setting
                    self.cfg.DATA.PLANE = original_plane

            time_delta = time.time() - start
            logger.info(
                f"Inference: Mixed-plane mode on {img_filename} completed in {time_delta:.4f}s "
                f"(processed all 3 planes)"
            )

            # Depad output if padding was applied
            if padding_percent > 0.0:
                logger.info("Inference: Removing edge padding from predictions")
                out = depad_volume(out, pad_width)

            return out

        # Single-plane mode (original behavior)
        # Set up DataLoader
        rescale = self.cfg.DATA.PREPROCESSING.RESCALE
        test_dataset = MultiScaleOrigDataThickSlices(
            orig_data_padded,
            orig_zoom,
            self.cfg,
            transforms=transforms.Compose([ToTensorTest(rescale=rescale)]),
        )

        test_data_loader = DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=self.cfg.TEST.BATCH_SIZE if batch_size is None else batch_size,
        )

        # Run evaluation
        start = time.time()
        out = self.eval(init_pred, test_data_loader, out=out, out_scale=out_res)
        time_delta = time.time() - start
        logger.info(
            f"Inference: {plane} plane on {img_filename} completed in {time_delta:.4f}s"
        )

        # Depad output if padding was applied (unified for both modes)
        if padding_percent > 0.0:
            logger.info("Inference: Removing edge padding from predictions")
            out = depad_volume(out, pad_width)

        return out
