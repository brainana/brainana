"""
Model loading utilities for nhp_skullstrip_nn.
"""

import torch
import torch.nn as nn
from typing import Union, Optional
from pathlib import Path

from .unet import UNet2d
from ..utils.log import get_logger
from ..utils.gpu import setup_device


class ModelLoader:
    """Handles model loading with simplified error handling."""

    @staticmethod
    def load_model_from_file(
        model_path: Union[str, Path],
        device_id: Union[int, str] = "auto",
        config: Optional[object] = None,
        logger: Optional[object] = None,
    ) -> nn.Module:
        """Load a trained model from file.

        Args:
            model_path: Path to the model checkpoint file
            device_id: Device specification ('auto', -1 for CPU, or specific GPU index)
            config: Configuration object (optional, will be extracted from checkpoint if available)
            logger: Logger instance

        Returns:
            Loaded and configured UNet2d model
        """
        logger = logger or get_logger(__name__)

        device = setup_device(device_id)
        logger.info(f"Loading model from: {model_path} on device: {device}")

        # Load checkpoint first to extract config if needed
        logger.info("Loading checkpoint...")
        # Load on CPU, then move (model.to(device) below): loading straight onto MPS
        # is known to yield zeros (see fastsurfer_nn Inference.load_checkpoint).
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Extract config from checkpoint if not provided
        if config is None and "config" in checkpoint:
            checkpoint_config = checkpoint["config"]
            logger.info("Extracting config from checkpoint")
            # Create a minimal config with required attributes
            config = type(
                "Config",
                (),
                {
                    "num_input_slices": checkpoint_config.get("num_input_slices", 3),
                    "num_conv_block": checkpoint_config.get("num_conv_block", 5),
                    "kernel_root": checkpoint_config.get("kernel_root", 16),
                    "use_inst_norm": checkpoint_config.get("use_inst_norm", True),
                    "num_classes": checkpoint_config.get("num_classes", 2),
                },
            )()
        elif config is None:
            # Use default values if no config available
            logger.info("No config provided or found in checkpoint, using defaults")
            config = type(
                "Config",
                (),
                {
                    "num_input_slices": 3,
                    "num_conv_block": 5,
                    "kernel_root": 16,
                    "use_inst_norm": True,
                    "num_classes": 2,
                    "rescale_dim": 256,
                },
            )()

        # Create model architecture using config
        unet_model = UNet2d(
            dim_in=getattr(config, "num_input_slices", 3),
            num_conv_block=getattr(config, "num_conv_block", 5),
            kernel_root=getattr(config, "kernel_root", 16),
            use_inst_norm=getattr(config, "use_inst_norm", True),
            num_classes=getattr(config, "num_classes", 2),
        )

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            logger.info("Using 'model_state_dict' from checkpoint")
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            logger.info("Using 'state_dict' from checkpoint")
            state_dict = checkpoint["state_dict"]
        else:
            logger.info("Using checkpoint directly as state_dict")
            state_dict = checkpoint

        # Load state dict with automatic key cleaning
        try:
            unet_model.load_state_dict(state_dict, strict=False)
        except RuntimeError:
            logger.info("Attempting to clean state_dict keys...")
            state_dict = ModelLoader._clean_state_dict_keys(state_dict)

            # Check if we need to adapt the model for different number of classes
            checkpoint_num_classes = ModelLoader._detect_num_classes_from_state_dict(
                state_dict
            )
            current_num_classes = unet_model.out_layer.out_channels
            target_num_classes = getattr(config, "num_classes", current_num_classes)

            if checkpoint_num_classes != target_num_classes:
                logger.info(
                    f"Adapting model from {checkpoint_num_classes} to {target_num_classes} classes (from config)"
                )

                # Remove output layer weights from state_dict since they won't match
                state_dict_filtered = {
                    k: v
                    for k, v in state_dict.items()
                    if not k.startswith("out_layer.")
                }

                # Load the filtered state dict (all layers except output)
                unet_model.load_state_dict(state_dict_filtered, strict=False)

                # Replace the output layer to match the config's target classes
                kernel_root = getattr(config, "kernel_root", 16)
                unet_model.out_layer = nn.Conv2d(
                    kernel_root, target_num_classes, 3, 1, 1
                )
                unet_model.num_classes = target_num_classes

                # Initialize the new output layer weights
                nn.init.xavier_uniform_(unet_model.out_layer.weight)
                if unet_model.out_layer.bias is not None:
                    nn.init.constant_(unet_model.out_layer.bias, 0)

                logger.info(
                    f"✅ Model adapted to {target_num_classes} classes as specified in config"
                )
            else:
                # Load normally if classes match
                unet_model.load_state_dict(state_dict, strict=False)

            logger.info("Model loaded after cleaning state_dict keys")

        unet_model.to(device)
        # eval only when inference
        # unet_model.eval()
        logger.info("Model loaded")

        return unet_model

    @staticmethod
    def _clean_state_dict_keys(state_dict: dict) -> dict:
        """Clean state_dict keys by removing common prefixes and problematic buffers."""
        cleaned = {}
        for key, value in state_dict.items():
            # Remove 'module.' prefix if present
            clean_key = key[7:] if key.startswith("module.") else key

            # Skip running stats buffers for InstanceNorm2d (they cause warnings in modern PyTorch)
            if any(
                buffer_name in clean_key
                for buffer_name in ["running_mean", "running_var"]
            ):
                continue

            cleaned[clean_key] = value
        return cleaned

    @staticmethod
    def _detect_num_classes_from_state_dict(state_dict: dict) -> int:
        """Detect the number of classes from the state dict by examining the output layer."""
        for key, value in state_dict.items():
            if "out_layer.weight" in key:
                # out_layer.weight has shape [num_classes, kernel_root, kernel_size, kernel_size]
                return value.shape[0]
            elif "out_layer.bias" in key:
                # out_layer.bias has shape [num_classes]
                return value.shape[0]

        # Default fallback
        return 2
