"""
Prediction module for nhp_skullstrip_nn.
"""

import os
import shutil
import tempfile
import torch
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output, resample_from_to
from typing import Optional, Dict, Union, Any
import logging
from pathlib import Path
import sys
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from matplotlib import pyplot as plt

from ..utils.morphology import (
    extract_largest_component,
    fill_label_holes,
    morphological_erosion_dilation,
)
from ..utils.io import write_nifti
from ..model import ModelLoader
from ..utils.plot import create_mri_image_3xN
from ..data.datasets import VolumeDataset, BlockDataset
from ..train.metrics import create_metrics_tracker


# %%
def _is_anisotropic(zooms, decimals: int = 2) -> bool:
    """Return True if the voxel spacings are not equal at ``decimals`` precision.

    The 2.5D U-Net rescales a volume by a single scalar ``256/max(voxel_count)``
    applied to all three axes (see ``data/datasets.py`` ``BlockDataset``), which
    implicitly assumes isotropic voxels. Anisotropic input therefore reaches the
    network with a distorted aspect ratio and produces a broken mask.

    We are deliberately lenient: spacings that agree once rounded to ``decimals``
    (default 2, matching conform's ``np.round(voxel, 2)`` convention) are treated as
    isotropic and skip resampling, so near-isotropic acquisitions (e.g. 0.499 vs
    0.501) keep the exact current behavior with no needless reinterpolation.
    """
    z = np.round(np.asarray(zooms[:3], dtype=float), decimals)
    if not np.all(np.isfinite(z)) or z.max() <= 0:
        return False
    return bool(z.max() - z.min() > 1e-9)


def _collapse_to_3d(img):
    """Return a 3D version of ``img``, dropping or averaging any frame axis.

    Some scanners emit anatomicals with a trailing singleton frame axis, e.g.
    ``(144, 144, 60, 1)``. nibabel's ``resample_to_output`` rejects anything that is
    not exactly 3D, so the anisotropic guard below has to collapse first.

    Mirrors ``VolumeDataset.__getitem__`` (``data/datasets.py``), which averages a 4D
    image over its last axis, so the isotropic and anisotropic paths agree on what
    the network is fed. Kept self-contained (rather than importing nhp_mri_prep's
    ``ensure_3d``) so this inference package stays independently installable.
    """
    img = nib.funcs.squeeze_image(img)
    if len(img.shape) == 3:
        return img
    if len(img.shape) != 4:
        raise ValueError(
            f"Cannot collapse a {len(img.shape)}-D image to 3D for inference"
        )
    data = np.mean(img.get_fdata(), axis=-1).astype(np.float32)
    out = nib.Nifti1Image(data, img.affine, img.header)
    # Nifti1Image honours the header dtype, so an int16 source header would
    # silently truncate the float32 mean unless it is overridden here.
    out.header.set_data_dtype(np.float32)
    out.header.set_zooms(img.header.get_zooms()[:3])
    return out


def _resample_file_to_native(path: str, native_ref_img, order: int) -> None:
    """Resample a NIfTI written on the isotropic inference grid back onto the
    native grid (shape + affine of ``native_ref_img``) and overwrite it in place.

    ``resample_from_to`` returns an image that already carries the native affine and
    preserves the source dtype (the float32 the mask/prob maps were written with), so
    it can be saved directly.
    """
    img = nib.load(path)
    resampled = resample_from_to(
        img, (native_ref_img.shape[:3], native_ref_img.affine), order=order
    )
    nib.save(resampled, path)


# %%
# no_grad: inference only. Without it every forward pass built an autograd graph
# (outputs were read through .data, so results were unaffected, but memory was not).
# no_grad rather than inference_mode: the returned tensors stay ordinary tensors.
@torch.no_grad()
def predict_volumes(
    model: torch.nn.Module,
    rescale_dim: int = 256,
    num_slices: int = 3,
    num_classes: Optional[int] = 2,
    input_image: Optional[torch.Tensor] = None,
    input_label: Optional[torch.Tensor] = None,
    save_label: bool = True,
    save_prob_map: bool = True,
    output_path: Optional[str] = None,
    compute_metrics: bool = False,
    force_softmax: Optional[bool] = False,
    erosion_dilation_iterations: Optional[int] = 0,
    plot_QC_snaps: Optional[bool] = True,
    verbose: Optional[bool] = False,
    resample_anisotropic_mm: Optional[float] = 0.5,
):
    """
    Predict brain labels for input volumes using a trained model.

    Args:
        model: Trained UNet model
        rescale_dim: Dimension to rescale input to
        num_slices: Number of input slices
        input_image: Input image tensor
        input_label: Brain label input for validation (optional)
        save_label: Whether to save output labels
        save_prob_map: Whether to save probability maps
        output_path: Output path for saving files (required if save_label or save_prob_map is True)
        compute_metrics: Whether to compute Dice and IoU metrics if input_label is provided
        force_softmax: Force softmax application (True/False). If None, uses model.apply_softmax.
                      Use True if model outputs logits, False if model already outputs probabilities.
        erosion_dilation_iterations: Erosion/dilation iterations
        plot_QC_snaps: Whether to create quality control visualization plots
        verbose: Whether to print verbose output
        num_classes: Number of classes for multi-class segmentation (auto-detected from model if None, defaults to 2 for binary)
    """
    # Get the device the model is on
    device = next(model.parameters()).device

    # Ensure model is on the correct device
    if device.type == "cuda":
        # Set the CUDA device to match the model's device
        torch.cuda.set_device(device.index)

    # set model to evaluation mode
    model.eval()

    # Auto-detect number of classes from model if not provided
    if num_classes is None:
        # Try to get num_classes from model attribute first
        if hasattr(model, "num_classes"):
            num_classes = model.num_classes
        else:
            # Fallback: infer from output layer
            try:
                num_classes = model.out_layer.out_channels
            except AttributeError:
                num_classes = 2  # Default to binary
                print(
                    f"Warning: Could not auto-detect num_classes, defaulting to {num_classes}"
                )

    if verbose:
        print(f"Number of classes: {num_classes}")
        if input_label is not None:
            print("Input label provided for validation")
            print(
                f"Ground truth labels will be validated against {num_classes} classes"
            )

    # Check if input image is provided
    if input_image is None:
        print("Error: input_image must be provided")
        sys.exit(1)

    # Validate output_path requirements
    if (save_label or save_prob_map) and output_path is None:
        print(
            "Error: output_path must be specified when save_label or save_prob_map is True"
        )
        sys.exit(1)

    # Initialize metrics tracking if metrics are requested
    metrics_dict = {}
    if compute_metrics and input_label is not None:
        # Create metrics tracker with per-slice computation for 3D volumes
        create_metrics_tracker(compute_per_slice=True)

    # ------------------------------------------------------------------
    # Anisotropic input guard: the network assumes isotropic voxels, so an
    # anisotropic image is fed with a distorted aspect ratio and yields a broken
    # mask. When the input is a NIfTI path with anisotropic spacing, resample it
    # to isotropic (resample_anisotropic_mm, default 0.5 mm = training/res-05) for
    # inference, then map the prediction back to the native grid before saving so
    # the output contract (mask on the native grid) is unchanged. Isotropic or
    # tensor inputs are left untouched -> exact previous behavior.
    native_ref_img = None  # native geometry to restore predictions onto
    iso_tmp_dir = None
    if (
        resample_anisotropic_mm
        and resample_anisotropic_mm > 0
        and isinstance(input_image, (str, Path))
        # Only in pure-prediction mode: a paired label would stay on the native grid
        # and mismatch the resampled image (validation/metrics callers pass a label).
        and input_label is None
    ):
        native_ref_img = nib.load(str(input_image))
        if len(native_ref_img.shape) > 3:
            # resample_to_output below only accepts 3D. Collapsing here also makes
            # the back-mapping reference 3D. The isotropic branch is untouched: it
            # clears native_ref_img and never rebinds input_image, so a 4D file
            # still reaches VolumeDataset and is averaged there exactly as before.
            native_ref_img = _collapse_to_3d(native_ref_img)
        if _is_anisotropic(native_ref_img.header.get_zooms()):
            iso_mm = float(resample_anisotropic_mm)
            print(
                f"Anisotropic voxels {tuple(np.round(native_ref_img.header.get_zooms()[:3], 3))} mm "
                f"detected; resampling to {iso_mm} mm isotropic for inference."
            )
            iso_img = resample_to_output(
                native_ref_img, [iso_mm, iso_mm, iso_mm], order=3
            )
            iso_tmp_dir = tempfile.mkdtemp(prefix="nhp_skullstrip_iso_")
            iso_path = os.path.join(iso_tmp_dir, "input_isotropic.nii.gz")
            nib.save(iso_img, iso_path)
            input_image = iso_path
        else:
            # Isotropic at 2-decimal precision: no resampling, no back-mapping.
            native_ref_img = None

    volume_dataset = VolumeDataset(
        input_image=input_image,
        input_label=input_label,
    )
    volume_loader = DataLoader(dataset=volume_dataset, batch_size=1)

    for _, volume_data in enumerate(volume_loader):
        if len(volume_data) == 1:  # just image
            prediction_type = "prediction"
            image = volume_data
            ground_truth_label = None
            block_dataset = BlockDataset(
                image=image, label=None, num_slice=num_slices, rescale_dim=rescale_dim
            )
        elif len(volume_data) == 2:  # image & label
            prediction_type = "validation"
            image = volume_data[0]
            ground_truth_label = volume_data[1]

            # Validate that ground truth labels don't exceed expected number of classes
            if compute_metrics and ground_truth_label is not None:
                # Check if labels exceed num_classes
                max_label_value = torch.max(ground_truth_label).item()
                if max_label_value >= num_classes:
                    print(
                        f"Warning: Ground truth labels contain class {max_label_value}, but model expects {num_classes} classes"
                    )
                    print(
                        f"Adjusting num_classes to {max_label_value + 1} for metrics computation"
                    )
                    num_classes = max_label_value + 1

            block_dataset = BlockDataset(
                image=image,
                label=ground_truth_label,
                num_slice=num_slices,
                rescale_dim=rescale_dim,
            )
        else:
            print("Error: Invalid Volume Dataset!")
            sys.exit(2)

        rescale_shape = block_dataset.get_rescale_shape()
        original_shape = block_dataset.get_raw_shape()

        # Process each orientation (axial, sagittal, coronal)
        for orientation_idx in range(3):
            # Create permutation indices for this orientation
            permutation_indices = np.arange(3)
            permutation_indices = np.insert(
                np.delete(permutation_indices, 0), orientation_idx, 0
            )

            block_data, slice_list, slice_weights = block_dataset.get_one_directory(
                axis=orientation_idx
            )

            # Initialize prediction tensor based on number of classes
            if num_classes == 2:
                # For binary, store probability of foreground class
                predicted_label_2d = torch.zeros(
                    [len(slice_weights), rescale_dim, rescale_dim], device=device
                )
            else:  # multiclass
                # For multiclass, store class predictions (will be converted to class indices later)
                predicted_label_2d = torch.zeros(
                    [len(slice_weights), rescale_dim, rescale_dim, num_classes],
                    device=device,
                )

            # Process each slice in this orientation
            for slice_idx, slice_indices in enumerate(slice_list):
                if prediction_type == "prediction":
                    raw_image_block = block_data[slice_idx].to(device)
                elif prediction_type == "validation":
                    raw_image_block, label_block = block_data[slice_idx]
                    raw_image_block = raw_image_block.to(device)
                    label_block = label_block.to(device)
                else:  # bias_corrected_validation
                    raw_image_block, bias_field_block, label_block = block_data[
                        slice_idx
                    ]
                    raw_image_block = raw_image_block.to(device)
                    bias_field_block = bias_field_block.to(device)
                    label_block = label_block.to(device)

                # Get model prediction for this block
                predicted_label_block = model(
                    torch.unsqueeze(Variable(raw_image_block), 0)
                )

                # Check if model has softmax applied, if not apply it manually
                # force_softmax parameter takes precedence over model.apply_softmax
                should_apply_softmax = (
                    force_softmax
                    if force_softmax is not None
                    else (hasattr(model, "apply_softmax") and model.apply_softmax)
                )

                if not should_apply_softmax:
                    predicted_label_block = torch.softmax(predicted_label_block, dim=1)

                # Extract predictions based on number of classes
                if num_classes == 2:
                    # Extract foreground probability from multi-channel output
                    predicted_label_2d[
                        slice_indices[1], :, :
                    ] = predicted_label_block.data[0][1, :, :]
                else:  # multiclass
                    # Store all class probabilities
                    predicted_label_2d[
                        slice_indices[1], :, :, :
                    ] = predicted_label_block.data[0].permute(1, 2, 0)

            # Move to CPU for numpy operations
            predicted_label_2d = predicted_label_2d.cpu()

            # Handle tensor manipulation based on number of classes
            if num_classes == 2:
                # Permute dimensions and crop to rescale shape
                predicted_label_2d = predicted_label_2d.permute(
                    permutation_indices[0],
                    permutation_indices[1],
                    permutation_indices[2],
                )
                predicted_label_2d = predicted_label_2d[
                    : rescale_shape[0], : rescale_shape[1], : rescale_shape[2]
                ]

                # Add batch and channel dimensions for interpolation
                predicted_label_4d = torch.unsqueeze(predicted_label_2d, 0)
                predicted_label_4d = torch.unsqueeze(predicted_label_4d, 0)

                # Interpolate back to original size
                predicted_label_4d = F.interpolate(
                    predicted_label_4d,
                    size=original_shape,
                    mode="trilinear",
                    align_corners=False,
                )
                predicted_label_2d = torch.squeeze(predicted_label_4d)

            else:  # multiclass
                # For multiclass, we need to handle the class dimension
                # Permute: [slices, H, W, classes] -> [classes, slices, H, W] for interpolation
                predicted_label_2d = predicted_label_2d.permute(
                    3,
                    permutation_indices[0],
                    permutation_indices[1],
                    permutation_indices[2],
                )
                predicted_label_2d = predicted_label_2d[
                    :, : rescale_shape[0], : rescale_shape[1], : rescale_shape[2]
                ]

                # Add batch dimension for interpolation: [classes, D, H, W] -> [1, classes, D, H, W]
                predicted_label_5d = torch.unsqueeze(predicted_label_2d, 0)

                # Interpolate back to original size
                predicted_label_5d = F.interpolate(
                    predicted_label_5d,
                    size=original_shape,
                    mode="trilinear",
                    align_corners=False,
                )
                # Remove batch dimension: [1, classes, D, H, W] -> [classes, D, H, W]
                predicted_label_2d = torch.squeeze(predicted_label_5d, 0)

            # Store prediction for this orientation
            if orientation_idx == 0:
                if num_classes == 2:
                    predictions_3d = torch.unsqueeze(predicted_label_2d, 3)
                else:  # multiclass: [classes, D, H, W] -> [classes, D, H, W, orientations]
                    predictions_3d = torch.unsqueeze(predicted_label_2d, 4)
            else:
                if num_classes == 2:
                    predictions_3d = torch.cat(
                        (predictions_3d, torch.unsqueeze(predicted_label_2d, 3)), dim=3
                    )
                else:  # multiclass
                    predictions_3d = torch.cat(
                        (predictions_3d, torch.unsqueeze(predicted_label_2d, 4)), dim=4
                    )

        # Average predictions across all orientations and convert to final output
        if num_classes == 2:
            # Average across orientations (dim=3)
            final_prediction = predictions_3d.mean(dim=3)
            final_prediction_np = final_prediction.numpy()

            # Apply thresholding and morphological operations for binary case
            binary_prediction = extract_largest_component(final_prediction_np > 0.5)
            binary_prediction = fill_label_holes(binary_prediction)
            if erosion_dilation_iterations > 0:
                binary_prediction = morphological_erosion_dilation(
                    binary_prediction, iterations=erosion_dilation_iterations
                )

        else:  # multiclass
            # Average across orientations (dim=4): [classes, D, H, W, orientations] -> [classes, D, H, W]
            averaged_predictions = predictions_3d.mean(dim=4)

            # Convert to class indices using argmax: [classes, D, H, W] -> [D, H, W]
            final_prediction = torch.argmax(averaged_predictions, dim=0)
            final_prediction_np = final_prediction.numpy().astype(np.int64)

            # For probability maps, keep the averaged probabilities
            prob_map_np = averaged_predictions.numpy()

            # Apply morphological operations to each class separately
            binary_prediction = np.zeros_like(final_prediction_np)
            for class_idx in range(num_classes):
                if class_idx == 0:  # Skip background
                    continue
                class_mask = final_prediction_np == class_idx
                if class_mask.sum() > 0:  # Only process if class is present
                    processed_mask = extract_largest_component(class_mask)
                    processed_mask = fill_label_holes(processed_mask)
                    if erosion_dilation_iterations > 0:
                        processed_mask = morphological_erosion_dilation(
                            processed_mask, iterations=erosion_dilation_iterations
                        )
                    binary_prediction[processed_mask] = class_idx

        # Get input image information for output naming and QC plotting
        input_image_nifti = volume_dataset.get_current_image_nifti()

        if save_label or save_prob_map:
            input_image_path = input_image_nifti.get_filename()
            input_image_dir, input_image_file = os.path.split(input_image_path)
            if input_image_dir == "":
                input_image_dir = os.curdir
            input_image_name = os.path.splitext(input_image_file)[0]
            input_image_name = os.path.splitext(input_image_name)[0]

            if input_label is not None:
                input_label_path = (
                    volume_dataset.get_current_label_nifti().get_filename()
                )
            else:
                input_label_path = None

        # Compute metrics if requested and ground truth label is provided
        if (
            compute_metrics
            and input_label is not None
            and isinstance(ground_truth_label, torch.Tensor)
        ):
            ground_truth_np = ground_truth_label.data[0].numpy()

            # Compute input image name for metrics dictionary key
            input_image_path = input_image_nifti.get_filename()
            input_image_dir, input_image_file = os.path.split(input_image_path)
            input_image_name = os.path.splitext(input_image_file)[0]
            input_image_name = os.path.splitext(input_image_name)[0]  # Remove .nii.gz

            # Use direct probability-based metrics computation for consistency
            if num_classes == 2:
                # For binary case, compute metrics directly from probabilities and ground truth
                # Convert ground truth to binary mask (1 where foreground, 0 where background)
                gt_mask = (ground_truth_np > 0).astype(np.float32)
                pred_mask = (final_prediction_np > 0.5).astype(
                    np.float32
                )  # Binary prediction using 0.5 threshold

                # Compute Dice coefficient directly
                intersection = np.sum(pred_mask * gt_mask)
                pred_sum = np.sum(pred_mask)
                gt_sum = np.sum(gt_mask)
                dice_score = (2.0 * intersection + 1e-7) / (pred_sum + gt_sum + 1e-7)

                # Compute IoU directly
                union = pred_sum + gt_sum - intersection
                iou_score = (intersection + 1e-7) / (union + 1e-7)

                computed_metrics = {"dice": dice_score, "iou": iou_score}

            else:  # multiclass
                # For multiclass case, compute metrics directly from class predictions and ground truth
                # final_prediction_np contains class indices (argmax of probabilities)
                pred_classes = final_prediction_np.astype(np.int64)
                gt_classes = ground_truth_np.astype(np.int64)

                # Compute per-class metrics directly
                dice_scores = []
                iou_scores = []

                for class_idx in range(num_classes):
                    # Create binary masks for this class
                    pred_mask = (pred_classes == class_idx).astype(np.float32)
                    gt_mask = (gt_classes == class_idx).astype(np.float32)

                    # Compute Dice for this class
                    intersection = np.sum(pred_mask * gt_mask)
                    pred_sum = np.sum(pred_mask)
                    gt_sum = np.sum(gt_mask)
                    dice = (2.0 * intersection + 1e-7) / (pred_sum + gt_sum + 1e-7)
                    dice_scores.append(dice)

                    # Compute IoU for this class
                    union = pred_sum + gt_sum - intersection
                    iou = (intersection + 1e-7) / (union + 1e-7)
                    iou_scores.append(iou)

                # Compute aggregate metrics
                mean_dice = np.mean(dice_scores)
                mean_iou = np.mean(iou_scores)

                # Brain tissue dice (exclude background class 0)
                if num_classes > 2:
                    brain_tissue_dice = np.mean(dice_scores[1:])  # Exclude background
                else:
                    brain_tissue_dice = (
                        dice_scores[1] if len(dice_scores) > 1 else dice_scores[0]
                    )

                computed_metrics = {
                    "mean_dice": mean_dice,
                    "mean_iou": mean_iou,
                    "brain_tissue_dice": brain_tissue_dice,
                }

                # Add per-class metrics
                class_names = [
                    "background",
                    "CSF",
                    "gray_matter",
                    "subcortex",
                    "white_matter",
                ]
                for i, (dice, iou) in enumerate(zip(dice_scores, iou_scores)):
                    class_name = (
                        class_names[i] if i < len(class_names) else f"class_{i}"
                    )
                    computed_metrics[f"{class_name}_dice"] = dice
                    computed_metrics[f"{class_name}_iou"] = iou

            # Store results
            if input_image_name not in metrics_dict:
                metrics_dict[input_image_name] = {}
            metrics_dict[input_image_name].update(computed_metrics)

            if verbose:
                print(f"Metrics for {input_image_name}:")
                if num_classes == 2:
                    print(f"  Dice: {dice_score:.4f}")
                    print(f"  IoU: {iou_score:.4f}")
                else:
                    print(f"  Mean Dice: {mean_dice:.4f}")
                    print(f"  Mean IoU: {mean_iou:.4f}")
                    print(f"  Brain Tissue Dice: {brain_tissue_dice:.4f}")

        # Save outputs if requested
        if save_label or save_prob_map:
            input_nifti = volume_dataset.get_current_image_nifti()

            # Adjust shape if original was 4D but data is now 3D
            original_shape = input_nifti.shape
            if len(original_shape) == 4 and len(binary_prediction.shape) == 3:
                # Use first 3 dimensions if original was 4D
                output_shape = original_shape[:3]
            else:
                output_shape = original_shape

            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # Save binary label if requested
            if save_label:
                write_nifti(
                    np.array(binary_prediction, dtype=np.float32),
                    input_nifti.affine,
                    input_nifti.header,
                    output_path,
                    shape=output_shape,
                )
                if verbose:
                    print(f"Binary label saved to: {output_path}")

            # Save probability map if requested
            if save_prob_map:
                base_path = output_path.split(".nii")[0]

                if num_classes == 2:
                    # Save single probability map for binary case
                    probability_output_path = base_path + "_prob.nii.gz"
                    write_nifti(
                        np.array(final_prediction_np, dtype=np.float32),
                        input_nifti.affine,
                        input_nifti.header,
                        probability_output_path,
                        shape=output_shape,
                    )
                    if verbose:
                        print(f"Probability map saved to: {probability_output_path}")

                else:  # multiclass
                    # Save probability maps for each class
                    class_names = [
                        "background",
                        "CSF",
                        "gray_matter",
                        "subcortex",
                        "white_matter",
                    ]
                    for class_idx in range(num_classes):
                        class_name = (
                            class_names[class_idx]
                            if class_idx < len(class_names)
                            else f"class_{class_idx}"
                        )
                        probability_output_path = (
                            base_path + f"_prob_{class_name}.nii.gz"
                        )

                        # Extract probability map for this class
                        if len(prob_map_np.shape) == 4:  # [classes, D, H, W]
                            class_prob_map = prob_map_np[class_idx]
                        else:  # [classes, H, W]
                            class_prob_map = prob_map_np[class_idx]

                        write_nifti(
                            np.array(class_prob_map, dtype=np.float32),
                            input_nifti.affine,
                            input_nifti.header,
                            probability_output_path,
                            shape=output_shape,
                        )
                        if verbose:
                            print(
                                f"Class {class_name} probability map saved to: {probability_output_path}"
                            )

                    # Also save argmax probability map (most likely class at each voxel)
                    argmax_prob_path = base_path + "_prob_argmax.nii.gz"
                    if len(prob_map_np.shape) == 4:  # [classes, D, H, W]
                        argmax_probs = np.max(prob_map_np, axis=0)
                    else:  # [classes, H, W]
                        argmax_probs = np.max(prob_map_np, axis=0)

                    write_nifti(
                        np.array(argmax_probs, dtype=np.float32),
                        input_nifti.affine,
                        input_nifti.header,
                        argmax_prob_path,
                        shape=output_shape,
                    )
                    if verbose:
                        print(f"Argmax probability map saved to: {argmax_prob_path}")

        if plot_QC_snaps:
            # Create QC visualization with input image as underlay, probability map as overlay, and binary prediction as contour
            # Convert boolean prediction to float for proper visualization
            create_mri_image_3xN(
                underlay_data=input_image_path,
                overlay_data=binary_prediction.astype(int),
                contour_data=input_label_path if input_label_path is not None else None,
                num_cols=5,
                overlay_alpha=0.5,
                contour_alpha=0.5,
                show_legend=False,
            )
            plt.savefig(output_path.split(".nii")[0] + "_QC.png")
            if verbose:
                print(
                    f"QC snapshot saved to: {output_path.split('.nii')[0] + '_QC.png'}"
                )

    # Map saved predictions back to the native grid if we resampled the input to
    # isotropic for inference above. This restores the output contract (mask/prob on
    # the native input grid) so downstream consumers see no geometry change. The QC
    # PNG is left on the isotropic grid (it overlays iso mask on iso input, which is
    # internally consistent and diagnostic only).
    if native_ref_img is not None:
        try:
            if save_label and output_path and os.path.exists(output_path):
                # Brain mask: nearest-neighbor to keep labels crisp.
                _resample_file_to_native(output_path, native_ref_img, order=0)
            if save_prob_map and output_path:
                base_path = output_path.split(".nii")[0]
                prob_dir = os.path.dirname(base_path) or os.curdir
                prob_prefix = os.path.basename(base_path)
                for prob_f in Path(prob_dir).glob(prob_prefix + "_prob*.nii.gz"):
                    # Probability maps: trilinear.
                    _resample_file_to_native(str(prob_f), native_ref_img, order=1)
        finally:
            if iso_tmp_dir and os.path.isdir(iso_tmp_dir):
                shutil.rmtree(iso_tmp_dir, ignore_errors=True)

    # Return metrics if computed
    if metrics_dict:
        return metrics_dict
    else:
        return None


def skullstripping(
    input_image: Union[str, Path],
    modal: str,
    output_path: Union[str, Path],
    device_id: Union[int, str] = "auto",
    logger: Optional[logging.Logger] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Perform skullstripping using nhp_skullstrip_nn UNet model.

    This function provides the same interface as the old skullstripping_bak API
    but uses nhp_skullstrip_nn internally for brain mask generation.

    Args:
        input_image: Path to the input image (T1w, EPI, etc.)
        modal: 'anat' or 'func' (modality)
        output_path: Path to save the brain mask
        device_id: GPU device to use ('auto', -1 for CPU, or specific GPU index)
        logger: Logger instance (optional)
        config: Model configuration (optional)

    Returns:
        Dictionary with output file paths:
        - 'brain_mask': Path to the generated brain mask
        - 'input_image': Path to the input image

    Raises:
        FileNotFoundError: If input image doesn't exist
        RuntimeError: If skullstripping fails
        ValueError: If modality is invalid
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Starting skullstripping for {modal} modality using nhp_skullstrip_nn")

    # Validate inputs
    input_image = Path(input_image)
    if not input_image.exists():
        logger.error(f"Input image not found: {input_image}")
        raise FileNotFoundError(f"Input image not found: {input_image}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if modal not in ["anat", "func"]:
        logger.error(f"Invalid modality: {modal}. Must be 'anat' or 'func'")
        raise ValueError(f"Invalid modality: {modal}. Must be 'anat' or 'func'")

    # Determine model path based on modality
    model_mapping = {"anat": "T1w_brainmask.pth", "func": "EPI_brainmask.pth"}

    # Construct relative path from script location to pretrained_model directory
    current_dir = Path(__file__).parent
    model_path = current_dir.parent / "pretrained_model" / model_mapping[modal]

    # Validate that the model file exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found for {modal} modality: {model_path}"
        )

    logger.info(f"Using model: {model_path}")

    try:
        # Load model using nhp_skullstrip_nn's ModelLoader
        model = ModelLoader.load_model_from_file(
            model_path=str(model_path),
            device_id=device_id,
            config=config,
            logger=logger,
        )

        # Run prediction to generate brain mask
        # We only need the binary label, not probability maps
        predict_volumes(
            model=model,
            rescale_dim=config.get("rescale_dim", 256) if config else 256,
            num_slices=config.get("num_input_slices", 3) if config else 3,
            num_classes=2,  # Binary classification for brain mask
            input_image=str(input_image),
            input_label=None,
            save_label=True,  # We need the binary label
            save_prob_map=False,  # No need for probability maps
            output_path=str(output_path),
            compute_metrics=False,
            force_softmax=None,  # Let the model decide
            erosion_dilation_iterations=config.get("morph_iterations", 0)
            if config
            else 0,
            plot_QC_snaps=False,  # No QC plots for this use case
            verbose=False,
            # Resample anisotropic input to isotropic for inference (mask returned on
            # the native grid). Default 0.5 mm; set to None/0 in config to disable.
            resample_anisotropic_mm=config.get("resample_anisotropic_mm", 0.5)
            if config
            else 0.5,
        )

        logger.info("Skullstripping completed successfully")

        # Return the same format as the old API
        return {"brain_mask": str(output_path), "input_image": str(input_image)}

    except Exception as e:
        logger.error(f"Skullstripping failed: {str(e)}")
        raise RuntimeError(f"Skullstripping failed: {str(e)}") from e
