#!/usr/bin/env python3

"""
Generate HDF5 training dataset for MRI segmentation with conform() preprocessing.
Uses proportional resizing to maintain aspect ratio, then pads to exact dimensions.
"""

import argparse
import h5py
import nibabel as nib
import numpy as np
import os
import sys
import signal
import time
import threading
from pathlib import Path
from multiprocessing import Pool
import json
import gc
import psutil

# Add src/ to path for fastsurfer_nn imports (training/ -> fastsurfer_nn -> src)
_src_dir = Path(__file__).resolve().parent.parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from fastsurfer_nn.utils.config_utils import load_yaml_config, get_paths_from_config
from fastsurfer_nn.atlas.atlas_manager import get_atlas_manager
from fastsurfer_nn.data_loader.data_utils import (
    create_weight_mask,
    filter_blank_slices_thick,
    get_thick_slices,
    get_zoom_indices_for_plane,
    transform_for_plane,
)
from fastsurfer_nn.data_loader.conform import conform


# Global flag for graceful shutdown
_shutdown_requested = False
_shutdown_message_printed = False
_shutdown_count = 0
_signal_handler_lock = threading.Lock()


def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested, _shutdown_message_printed, _shutdown_count

    # Use lock to prevent concurrent execution
    with _signal_handler_lock:
        _shutdown_count += 1

        # Only print message once
        if not _shutdown_message_printed:
            signal_name = signal.Signals(signum).name
            print(
                f"\n⚠️  Received {signal_name} signal. Initiating graceful shutdown..."
            )
            print("  → Finishing current operations and closing HDF5 file properly...")
            _shutdown_message_printed = True

        _shutdown_requested = True

        # Force exit on second interrupt (Ctrl+C twice) - use os._exit for immediate termination
        if _shutdown_count >= 2:
            print("\n⚠️  Force exit requested. Terminating immediately...")
            os._exit(130)  # Use os._exit for immediate termination (bypasses cleanup)


def resize_volume_proportional(volume, target_size=256, order=1):
    from fastsurfer_nn.data_loader.data_utils import resize_to_target_size

    # Use unified resize function for consistency
    padded, scale_factor = resize_to_target_size(volume, target_size, order=order)
    return padded, scale_factor


def load_and_conform_image(image_path, preprocess_params):
    """Load and preprocess image using conform(). Returns (data, zoom) or (None, None) if invalid."""
    img = nib.load(image_path, mmap=True)
    zoom = img.header.get_zooms()[:3]

    if np.any(np.abs(np.asarray(zoom)) < 1e-6):
        print(f"  ⚠️  SKIPPING: Invalid zoom values in {image_path.name}")
        return None, None

    conformed_img = conform(
        img,
        order=preprocess_params["ORDER_IMAGE"],
        orientation=preprocess_params["ORIENTATION"].lower(),
        img_size=preprocess_params["IMG_SIZE"],
        vox_size=preprocess_params["VOX_SIZE"],
        dtype=np.dtype(preprocess_params["DTYPE_IMAGE"]),
        rescale=preprocess_params["RESCALE"],
    )

    # conform() already rescales to [0, RESCALE] range - keep that range in HDF5
    # Dataset loader will normalize to [0, 1] by dividing by RESCALE
    data = np.asarray(conformed_img.dataobj).astype(np.float32)
    # Voxel size of the CONFORMED image, in the conformed axis order. The plane indices
    # (get_zoom_indices_for_plane) refer to this order, and inference uses the conformed
    # zooms too. (Before 2026-11 the original header zooms were stored here, which is wrong
    # for anisotropic or non-LIA-stored inputs.)
    zoom = conformed_img.header.get_zooms()[:3]

    del img, conformed_img
    return data, zoom


def load_and_map_labels(
    label_path,
    plane="coronal",
    atlas_manager=None,
    preprocess_params=None,
    num_classes=None,
):
    """Load and preprocess labels using conform(). Returns dense label array."""
    img = nib.load(label_path, mmap=True)

    conformed_img = conform(
        img,
        order=preprocess_params["ORDER_LABEL"],
        orientation=preprocess_params["ORIENTATION"].lower(),
        img_size=preprocess_params["IMG_SIZE"],
        vox_size=preprocess_params["VOX_SIZE"],
        dtype=np.dtype(preprocess_params["DTYPE_LABEL"]),
        rescale=None,
    )

    sparse_labels = np.asarray(conformed_img.dataobj).astype(np.int32)
    del img, conformed_img

    if num_classes == 2:
        unique_labels = np.unique(sparse_labels)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(
                f"Binary mode (NUM_CLASSES=2) requires labels to be 0 (background) and 1 (brain).\n"
                f"Found labels: {unique_labels}\n"
                f"File: {label_path}"
            )
        return sparse_labels

    if atlas_manager is None:
        raise ValueError(
            "atlas_manager required for multi-class segmentation (NUM_CLASSES > 2)"
        )

    if plane == "sagittal":
        return atlas_manager.map_labels_to_sagittal_dense(sparse_labels)
    else:
        return atlas_manager.map_labels_to_dense(sparse_labels)


def _get_plane_transform(plane, orientation="lia"):
    """
    Get transform function and zoom extraction for given plane and orientation.

    Parameters
    ----------
    plane : str
        One of 'axial', 'coronal', 'sagittal'.
    orientation : str, default='lia'
        3-letter orientation code (e.g., 'LIA', 'RAS').

    Returns
    -------
    transform_func : callable
        Function to transform volume for the given plane.
    zoom_extract : callable
        Function to extract 2D zoom values for the in-plane dimensions.
    """

    def transform_func(vol):
        return transform_for_plane(vol, plane, orientation)

    zoom_indices = get_zoom_indices_for_plane(plane, orientation)

    def zoom_extract(z):
        return np.asarray(z)[list(zoom_indices)]

    return transform_func, zoom_extract


def _delete_samples_not_in_filter(datasets, subject_filter):
    """Delete samples from HDF5 datasets that are not in subject_filter."""
    if subject_filter is None:
        return 0

    images_ds, labels_ds, weights_ds, zooms_ds, subjects_ds = datasets

    # Read all subject names
    subjects_array = subjects_ds[:]
    subjects_list = [
        str(s.decode("utf-8") if isinstance(s, bytes) else s).strip()
        for s in subjects_array
    ]

    # Create mask for samples to keep (subjects in filter)
    keep_mask = np.array([subject in subject_filter for subject in subjects_list])
    num_to_delete = np.sum(~keep_mask)

    if num_to_delete == 0:
        return 0

    print(f"  Removing {num_to_delete} samples from subjects not in filter...")

    # Get indices to keep
    keep_indices = np.where(keep_mask)[0]

    if len(keep_indices) == 0:
        # Delete everything
        images_ds.resize(0, axis=0)
        labels_ds.resize(0, axis=0)
        weights_ds.resize(0, axis=0)
        zooms_ds.resize(0, axis=0)
        subjects_ds.resize(0, axis=0)
        print(f"  ✓ Deleted all {len(subjects_list)} samples (none in filter)")
        return num_to_delete

    # Read data for samples to keep
    images_keep = images_ds[keep_indices]
    labels_keep = labels_ds[keep_indices]
    weights_keep = weights_ds[keep_indices]
    zooms_keep = zooms_ds[keep_indices]
    subjects_keep = subjects_ds[keep_indices]

    # Resize datasets to new size
    new_size = len(keep_indices)
    images_ds.resize(new_size, axis=0)
    labels_ds.resize(new_size, axis=0)
    weights_ds.resize(new_size, axis=0)
    zooms_ds.resize(new_size, axis=0)
    subjects_ds.resize(new_size, axis=0)

    # Write back kept data
    images_ds[:] = images_keep
    labels_ds[:] = labels_keep
    weights_ds[:] = weights_keep
    zooms_ds[:] = zooms_keep
    subjects_ds[:] = subjects_keep

    print(f"  ✓ Deleted {num_to_delete} samples, kept {new_size} samples")
    return num_to_delete


def _append_to_datasets(datasets, result, subject_name):
    """Append processed subject data to HDF5 datasets."""
    images_ds, labels_ds, weights_ds, zooms_ds, subjects_ds = datasets
    num_slices = result["num_slices"]
    current_len = images_ds.shape[0]

    images = result["images"]
    if images_ds.dtype == np.uint8:
        images = np.clip(np.rint(images), 0, 255).astype(np.uint8)
    images_ds.resize(current_len + num_slices, axis=0)
    images_ds[current_len : current_len + num_slices] = images

    labels_ds.resize(current_len + num_slices, axis=0)
    labels_ds[current_len : current_len + num_slices] = result["labels"].astype(labels_ds.dtype)

    weights_ds.resize(current_len + num_slices, axis=0)
    weights_ds[current_len : current_len + num_slices] = result["weights"].astype(weights_ds.dtype)

    zooms_ds.resize(current_len + num_slices, axis=0)
    zooms_ds[current_len : current_len + num_slices] = result["zooms"]

    subjects_ds.resize(current_len + num_slices, axis=0)
    subjects_ds[current_len : current_len + num_slices] = [subject_name] * num_slices

    return num_slices


def process_subject(
    image_path,
    label_path,
    plane,
    slice_thickness=3,
    target_size=256,
    atlas_manager=None,
    preprocess_params=None,
    num_classes=None,
    verbose=False,
):
    """Process a single subject's MRI and segmentation."""
    # Validate affine match
    raw_image = nib.load(image_path)
    raw_label = nib.load(label_path)
    if not np.allclose(raw_image.affine, raw_label.affine, atol=1e-3):
        print(
            f"  ⚠️  WARNING: Image and label affines don't match for {image_path.name}!"
        )
    del raw_image, raw_label

    # Load and conform image
    image_data, zoom = load_and_conform_image(image_path, preprocess_params)
    if image_data is None or zoom is None:
        return None

    # Validate zoom values
    if not np.all((np.array(zoom) > 0.001) & (np.array(zoom) < 10)):
        return None

    # Load and map labels
    if not label_path.exists():
        print(f"  ⚠️  ERROR: Label file does not exist: {label_path}")
        return None

    try:
        label_data = load_and_map_labels(
            label_path, plane, atlas_manager, preprocess_params, num_classes
        )
    except Exception as e:
        print(f"  ⚠️  ERROR loading labels for {label_path.name}: {e}")
        return None

    # Check for all-zero labels (critical error)
    if np.sum(label_data > 0) == 0:
        print(
            f"  ⚠️  CRITICAL: All labels are zero after conforming! File: {label_path.name}"
        )
        return None

    # Cortex classes for the thin-cortex weighting. label_data holds DENSE indices, so the
    # (sparse) cortex IDs are mapped to this plane's dense indices. (Before 2026-11 the sparse
    # IDs were matched against dense labels, weighting background and 3 random classes.)
    if num_classes == 2:
        cortex_labels = None
    else:
        cortex_labels = set(atlas_manager.cortex_dense(plane))

    # Create weight mask
    weights = create_weight_mask(
        label_data,
        max_weight=5,
        max_edge_weight=5,
        max_hires_weight=5,
        gradient=False,
        cortex_labels=cortex_labels,
        verbose=False,
    )

    # Transform to requested plane (using orientation from preprocess_params)
    orientation = preprocess_params.get("ORIENTATION", "lia").lower()
    transform_func, zoom_extract = _get_plane_transform(plane, orientation)
    image_data = transform_func(image_data)
    label_data = transform_func(label_data)
    weights = transform_func(weights)
    zoom_2d = zoom_extract(zoom)

    # Resize proportionally
    image_data_resized, scale_factor = resize_volume_proportional(
        image_data, target_size, order=1
    )
    label_data_resized, _ = resize_volume_proportional(label_data, target_size, order=0)
    weights_resized, _ = resize_volume_proportional(weights, target_size, order=1)

    del image_data, label_data, weights

    # Validate scaled zoom values
    zoom_2d_scaled = zoom_2d / scale_factor
    if np.any(np.abs(zoom_2d_scaled) < 1e-6):
        print(f"  ⚠️  SKIPPING: Invalid scaled zoom values for {image_path.name}")
        del image_data_resized, label_data_resized, weights_resized
        gc.collect()
        return None

    # Create thick slices
    image_thick = get_thick_slices(image_data_resized, slice_thickness)
    del image_data_resized

    # Filter blank slices
    blank_threshold = 10
    label_sum_per_slice = np.sum(label_data_resized, axis=(0, 1))
    max_pixels = int(np.max(label_sum_per_slice))
    int(np.sum(label_sum_per_slice > blank_threshold))

    image_slices, label_slices, weights_final = filter_blank_slices_thick(
        image_thick, label_data_resized, weights_resized, threshold=blank_threshold
    )
    del image_thick, label_data_resized, weights_resized

    if image_slices.shape[2] == 0:
        if max_pixels == 0:
            print(f"  ⚠️  SKIPPING: All labels are zero for {image_path.name}")
        elif max_pixels < blank_threshold:
            print(
                f"  ⚠️  SKIPPING: Max pixels/slice ({max_pixels}) < threshold ({blank_threshold}) for {image_path.name}"
            )
        gc.collect()
        return None

    # Transpose to (slices, height, width, channels) for images
    image_slices = np.transpose(image_slices, (2, 0, 1, 3))
    label_slices = np.transpose(label_slices, (2, 0, 1))
    weights_final = np.transpose(weights_final, (2, 0, 1))

    assert image_slices.shape[1] == target_size and image_slices.shape[2] == target_size
    assert label_slices.shape[1] == target_size and label_slices.shape[2] == target_size

    result = {
        "images": image_slices,
        "labels": label_slices,
        "weights": weights_final,
        "zooms": np.tile(zoom_2d_scaled, (image_slices.shape[0], 1)),
        "num_slices": image_slices.shape[0],
        "scale_factor": scale_factor,
    }

    gc.collect()
    return result


def get_memory_usage():
    """Get current memory usage information."""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        virtual_mem = psutil.virtual_memory()
        return {
            "process_rss_gb": mem_info.rss / (1024**3),
            "system_available_gb": virtual_mem.available / (1024**3),
            "system_percent": virtual_mem.percent,
        }
    except Exception:
        return None


def check_memory_and_warn(threshold_percent=85):
    """Check system memory usage and warn if high."""
    mem = get_memory_usage()
    if mem and mem["system_percent"] > threshold_percent:
        print(
            f"\n⚠️  WARNING: High memory usage ({mem['system_percent']:.1f}%, {mem['system_available_gb']:.1f} GB available)"
        )
        print(
            f"   Process: {mem['process_rss_gb']:.2f} GB. Consider reducing num_workers.\n"
        )
        return False
    return True


HDF5_FORMAT_VERSION = "2.0"  # uint8 images/labels, float16 weights, lzf, 1 slice per chunk


def _create_resizable_dataset(hf_group, name, dtype, shape):
    """Resizable dataset, one slice per chunk, lzf (fast to decompress; little CPU)."""
    return hf_group.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        maxshape=(None,) + shape[1:],
        compression="lzf",
        chunks=(1,) + shape[1:],
    )


def _source_signature(image_file, label_dir) -> list:
    """[image path, size, mtime, label path, size, mtime] (resolved; symlinks followed)."""
    image_file = Path(image_file)
    base = image_file.name.replace(".nii.gz", "")
    labels = sorted(Path(label_dir).glob(f"{base}_*.nii.gz"))
    sig = []
    for f in [image_file] + labels[:1]:
        r = f.resolve()
        st = r.stat()
        sig += [str(r), st.st_size, int(st.st_mtime)]
    return sig


def _find_image_label_pairs(image_dir, label_dir):
    """Find matching image-label pairs."""
    all_image_files = sorted(image_dir.glob("*.nii.gz"))
    matched_pairs = []

    for image_file in all_image_files:
        base_name = image_file.name.replace(".nii.gz", "")
        label_pattern = f"{base_name}_*.nii.gz"
        label_matches = sorted(label_dir.glob(label_pattern))

        if len(label_matches) == 1:
            matched_pairs.append((image_file, label_matches[0]))
        # elif len(label_matches) == 0:
        #     print(f"Warning: No label file found for {image_file.name}")
        # else:
        #     print(f"Warning: Multiple label files found for {image_file.name}, skipping")

    return [pair[0] for pair in matched_pairs], [pair[1] for pair in matched_pairs]


def _safe_open_hdf5(file_path, mode, max_retries=5, retry_delay=1.0):
    """Safely open HDF5 file with retry logic for handling locked files.

    Args:
        file_path: Path to HDF5 file
        mode: File mode ('r', 'w', 'a', etc.)
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (exponential backoff)

    Returns:
        h5py.File object (context manager)

    Raises:
        OSError: If file cannot be opened after all retries
        KeyboardInterrupt: If shutdown is requested during retries
    """
    file_path = Path(file_path)
    last_error = None

    # Check if file is very small (likely corrupted/incomplete) - remove immediately if in write mode
    if mode == "w" and file_path.exists():
        try:
            file_size = file_path.stat().st_size
            # If file is 0 or very small (< 1KB), it's likely corrupted/incomplete - remove it
            if file_size < 1024:
                try:
                    file_path.unlink()
                    print(f"  → Removed small/corrupted file ({file_size} bytes)")
                    time.sleep(0.1)
                except (OSError, PermissionError) as e:
                    print(f"  ⚠️  Could not remove small file: {e}")
        except OSError:
            pass  # File might have been removed

    for attempt in range(max_retries):
        # Check for shutdown request
        if _shutdown_requested:
            raise KeyboardInterrupt("Shutdown requested during HDF5 file open")

        try:
            # If in write mode and file exists but is locked, try to remove it
            if mode == "w" and file_path.exists() and attempt > 0:
                try:
                    # Try to remove the locked file
                    file_path.unlink()
                    print(
                        f"  → Removed locked file (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(0.5)  # Brief pause after removal
                except (OSError, PermissionError) as e:
                    print(f"  ⚠️  Could not remove locked file: {e}")

            # Try to open the file
            return h5py.File(file_path, mode)

        except (OSError, IOError, BlockingIOError) as e:
            last_error = e
            error_msg = str(e)

            # Check if it's a locking error
            if (
                "lock" in error_msg.lower()
                or "errno 11" in error_msg
                or "Resource temporarily unavailable" in error_msg
            ):
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)  # Exponential backoff
                    print(
                        f"  ⚠️  File is locked (attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.1f}s..."
                    )

                    # Sleep in small increments to allow interruption
                    elapsed = 0.0
                    while elapsed < wait_time and not _shutdown_requested:
                        sleep_interval = min(0.5, wait_time - elapsed)
                        time.sleep(sleep_interval)
                        elapsed += sleep_interval

                    if _shutdown_requested:
                        raise KeyboardInterrupt("Shutdown requested during retry wait")

                    # Try to remove locked file before next attempt (more aggressive)
                    if mode == "w" and file_path.exists():
                        try:
                            file_path.unlink()
                            print("  → Removed locked file before retry")
                            time.sleep(0.2)
                        except (OSError, PermissionError):
                            pass  # Ignore if we can't remove it

                    continue
                else:
                    # Last attempt failed - try to remove if in write mode
                    if mode == "w" and file_path.exists():
                        try:
                            file_path.unlink()
                            print("  → Force removed locked file, retrying...")
                            time.sleep(0.5)
                            return h5py.File(file_path, mode)
                        except Exception as remove_error:
                            raise OSError(
                                f"Could not open or remove locked file after {max_retries} attempts. "
                                f"Last error: {error_msg}. Remove error: {remove_error}"
                            )
            else:
                # Not a locking error, re-raise immediately
                raise

    # If we get here, all retries failed
    raise OSError(
        f"Could not open HDF5 file after {max_retries} attempts: {last_error}"
    )


def _check_resume_capability(output_hdf5, target_size, preprocess_params):
    """Check if HDF5 file exists and can be resumed."""
    already_processed = set()
    file_mode = "w"

    if output_hdf5.exists():
        file_size = output_hdf5.stat().st_size
        print(
            f"Found existing HDF5 file: {output_hdf5.name} ({file_size / (1024**2):.1f} MB)"
        )

        # If file is very small (< 1KB), it's likely corrupted/incomplete - remove it immediately
        if file_size < 1024:
            try:
                output_hdf5.unlink()
                print(
                    f"  → Removed small/corrupted file ({file_size} bytes), will create new file"
                )
                return already_processed, file_mode
            except (OSError, PermissionError) as e:
                print(f"  ⚠️  Could not remove small file: {e}, will try to overwrite")

        try:
            with _safe_open_hdf5(output_hdf5, "r") as hf_check:
                if (
                    str(target_size) in hf_check
                    and "subject" in hf_check[str(target_size)]
                ):
                    subjects_in_hdf5 = hf_check[str(target_size)]["subject"][:]
                    already_processed = set(
                        str(s.decode("utf-8") if isinstance(s, bytes) else s).strip()
                        for s in subjects_in_hdf5
                    )
                    print(
                        f"✓ Found {len(already_processed)} subjects already processed"
                    )

                    if "preprocessing_metadata" in hf_check:
                        meta = hf_check["preprocessing_metadata"]
                        version = str(meta.attrs.get("version", "1.0"))
                        if version != HDF5_FORMAT_VERSION:
                            raise SystemExit(
                                f"ERROR: {output_hdf5} has HDF5 format {version}, this code writes "
                                f"{HDF5_FORMAT_VERSION} (uint8, conformed zooms, dense cortex "
                                f"weights). Write to a new file or move the old one away."
                            )
                        for key, expected_value in preprocess_params.items():
                            if key in meta.attrs and meta.attrs[key] != expected_value:
                                print(
                                    f"  ⚠️  Warning: Preprocessing parameter '{key}' differs!"
                                )
                                print(
                                    f"      Existing: {meta.attrs[key]}, Expected: {expected_value}"
                                )

                    file_mode = "a"
                else:
                    print(
                        f"  ⚠️  Existing file doesn't contain size {target_size}, will append/overwrite"
                    )
        except (OSError, IOError, RuntimeError) as e:
            error_msg = str(e)
            if (
                "address of object past end of allocation" in error_msg
                or "corrupt" in error_msg.lower()
            ):
                print(f"  ⚠️  HDF5 file appears corrupted: {error_msg}")
                print("  → Will overwrite with a new file")
            elif "lock" in error_msg.lower() or "errno 11" in error_msg:
                print(f"  ⚠️  File is locked ({error_msg}), will try to overwrite")
            else:
                print(
                    f"  ⚠️  Could not read existing file ({error_msg}), will overwrite"
                )
        except Exception as e:
            print(
                f"  ⚠️  Unexpected error reading file ({type(e).__name__}: {e}), will overwrite"
            )

    return already_processed, file_mode


def create_hdf5_dataset(
    data_dir,
    output_hdf5,
    plane="coronal",
    target_size=256,
    slice_thickness=3,
    atlas_manager=None,
    preprocess_params=None,
    num_classes=None,
    subject_filter=None,
    num_workers=1,
):
    """Create HDF5 dataset using conform() preprocessing.

    When plane="mixed", processes each subject for all three planes (axial, coronal, sagittal)
    and stores all slices in a single HDF5 file for plane-agnostic training.
    """
    global _shutdown_requested, _shutdown_message_printed
    image_dir = Path(data_dir) / "images"
    label_dir = Path(data_dir) / "labels"

    all_image_files, all_label_files = _find_image_label_pairs(image_dir, label_dir)

    if subject_filter is not None:
        image_files = [
            img
            for img in all_image_files
            if img.name.replace(".nii.gz", "").strip() in subject_filter
        ]
        print(
            f"Filtered {len(image_files)}/{len(all_image_files)} subjects based on data split"
        )
    else:
        image_files = all_image_files
        print(f"Found {len(image_files)} subjects")

    # Handle mixed plane mode
    if plane == "mixed":
        planes_to_process = ["axial", "coronal", "sagittal"]
        print("Processing MIXED planes (axial, coronal, sagittal)")
        print("  Each subject will be processed for all 3 planes")
        print(f"  Target size: {target_size}×{target_size}, output: {output_hdf5}")
    else:
        planes_to_process = [plane]
        print(
            f"Processing plane: {plane}, target size: {target_size}×{target_size}, output: {output_hdf5}"
        )

    if num_workers > 1:
        print(f"Using {num_workers} parallel workers")

    mem = get_memory_usage()
    if mem:
        print(
            f"Memory: {mem['system_available_gb']:.1f} GB available ({100-mem['system_percent']:.1f}% free)"
        )
        if mem["system_percent"] > 70:
            print(f"  ⚠️  Warning: System using {mem['system_percent']:.1f}% of memory")

    output_hdf5.parent.mkdir(parents=True, exist_ok=True)

    already_processed, file_mode = _check_resume_capability(
        output_hdf5, target_size, preprocess_params
    )

    # For mixed mode, track which (subject, plane) combinations are already processed
    if plane == "mixed" and already_processed:
        # Load existing subject-plane pairs from HDF5
        with _safe_open_hdf5(output_hdf5, "r") as hf_check:
            if str(target_size) in hf_check and "subject" in hf_check[str(target_size)]:
                # For mixed mode, we can't easily track which planes are done per subject
                # So we'll just check if subject exists (conservative approach)
                existing_subjects = set(
                    str(s.decode("utf-8") if isinstance(s, bytes) else s).strip()
                    for s in hf_check[str(target_size)]["subject"][:]
                )
                original_count = len(image_files)
                image_files = [
                    img
                    for img in image_files
                    if img.name.replace(".nii.gz", "").strip() not in existing_subjects
                ]
                print(
                    f"Filtered: {original_count} → {len(image_files)} subjects remaining (mixed mode)"
                )
                if len(image_files) == 0:
                    print("✅ All subjects already processed!")
                    return
    elif already_processed:
        original_count = len(image_files)
        image_files = [
            img
            for img in image_files
            if img.name.replace(".nii.gz", "").strip() not in already_processed
        ]
        print(f"Filtered: {original_count} → {len(image_files)} subjects remaining")
        if len(image_files) == 0:
            print("✅ All subjects already processed!")
            return

    total_subjects = len(image_files)
    num_samples_written = 0

    # Use safe file opening with retry logic
    with _safe_open_hdf5(output_hdf5, file_mode) as hf:
        if "preprocessing_metadata" not in hf:
            preprocess_metadata = hf.create_group("preprocessing_metadata")
            for key, value in preprocess_params.items():
                preprocess_metadata.attrs[key] = value
            preprocess_metadata.attrs["created_with_conform"] = True
            preprocess_metadata.attrs["version"] = HDF5_FORMAT_VERSION
            preprocess_metadata.attrs["plane_mode"] = plane  # Store plane mode
            print("✓ Saved preprocessing metadata")

        if str(target_size) not in hf:
            grp = hf.create_group(str(target_size))
            initial_img_shape = (0, target_size, target_size, slice_thickness * 2 + 1)
            initial_label_weight_shape = (0, target_size, target_size)
            initial_zoom_shape = (0, 2)

            label_dtype = np.uint8 if (num_classes or 0) <= 256 else np.int16
            images_ds = _create_resizable_dataset(
                grp, "orig_dataset", np.uint8, initial_img_shape
            )
            labels_ds = _create_resizable_dataset(
                grp, "aseg_dataset", label_dtype, initial_label_weight_shape
            )
            weights_ds = _create_resizable_dataset(
                grp, "weight_dataset", np.float16, initial_label_weight_shape
            )
            zooms_ds = _create_resizable_dataset(
                grp, "zoom_dataset", np.float32, initial_zoom_shape
            )
            subjects_ds = grp.create_dataset(
                "subject",
                (0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            print(f"✓ Created new datasets for size {target_size}")
        else:
            grp = hf[str(target_size)]
            images_ds = grp["orig_dataset"]
            labels_ds = grp["aseg_dataset"]
            weights_ds = grp["weight_dataset"]
            zooms_ds = grp["zoom_dataset"]
            subjects_ds = grp["subject"]
            print(
                f"✓ Appending to existing datasets (current size: {images_ds.shape[0]} slices)"
            )

        datasets = (images_ds, labels_ds, weights_ds, zooms_ds, subjects_ds)

        # Source files of each subject (path, size, mtime of image and label). On resume,
        # subjects whose sources changed are removed and processed again, so an HDF5 never
        # mixes slices from an old and a new label/image version.
        sources = json.loads(grp.attrs.get("sources", "{}"))
        stale = set()
        for image_file in all_image_files:
            name = image_file.name.replace(".nii.gz", "").strip()
            if name in sources and sources[name] != _source_signature(image_file, label_dir):
                stale.add(name)
        if stale:
            print(f"  {len(stale)} subject(s) changed since they were written; reprocessing: "
                  f"{sorted(stale)[:5]}{' ...' if len(stale) > 5 else ''}")
            keep = (subject_filter or set(sources)) - stale
            _delete_samples_not_in_filter(datasets, keep)
            image_files = image_files + [f for f in all_image_files
                                         if f.name.replace(".nii.gz", "").strip() in stale
                                         and f not in image_files]

        # Delete samples not in subject_filter if filter is provided and file exists
        if file_mode == "a" and subject_filter is not None and images_ds.shape[0] > 0:
            num_deleted = _delete_samples_not_in_filter(datasets, subject_filter)
            if num_deleted > 0:
                print(f"✓ Cleaned up {num_deleted} samples not in subject_filter")

        # Build process args: for mixed mode, process each subject for all planes
        process_args = []
        for idx, image_file in enumerate(image_files, 1):
            for p in planes_to_process:
                process_args.append(
                    (
                        idx,
                        image_file,
                        label_dir,
                        p,
                        slice_thickness,
                        target_size,
                        atlas_manager,
                        preprocess_params,
                        num_classes,
                        total_subjects,
                    )
                )

        subjects_processed = 0
        subjects_processed_set = set()  # Track unique subjects processed

        pool = None
        try:
            if num_workers > 1:
                pool = Pool(processes=num_workers)
                try:
                    iterator = pool.imap_unordered(
                        _process_single_subject_wrapper, process_args
                    )
                    for result_data, msg in iterator:
                        # Check shutdown flag - if set, terminate immediately
                        if _shutdown_requested:
                            # Flush HDF5 datasets to ensure data is saved
                            try:
                                images_ds.flush()
                                labels_ds.flush()
                                weights_ds.flush()
                                zooms_ds.flush()
                                subjects_ds.flush()
                            except Exception:
                                pass
                            # Force terminate pool immediately - don't wait
                            try:
                                pool.terminate()
                            except Exception:
                                pass
                            # Exit immediately - don't wait for anything
                            os._exit(130)
                        print(msg)
                        if result_data is not None:
                            result, subject_name = result_data
                            num_samples_written += _append_to_datasets(
                                datasets, result, subject_name
                            )
                            subjects_processed_set.add(subject_name)
                            sources[subject_name] = _source_signature(
                                image_dir / f"{subject_name}.nii.gz", label_dir)
                            grp.attrs["sources"] = json.dumps(sources)
                            if len(subjects_processed_set) % 10 == 0:
                                check_memory_and_warn()
                finally:
                    if pool is not None:
                        if _shutdown_requested:
                            # On shutdown, don't wait - just terminate and exit
                            try:
                                pool.terminate()
                            except Exception:
                                pass
                        else:
                            pool.close()
                            pool.join()
            else:
                for args in process_args:
                    if _shutdown_requested:
                        if not _shutdown_message_printed:
                            print("\n⚠️  Shutdown requested. Stopping processing...")
                        # Flush HDF5 datasets to ensure data is saved
                        try:
                            images_ds.flush()
                            labels_ds.flush()
                            weights_ds.flush()
                            zooms_ds.flush()
                            subjects_ds.flush()
                        except Exception:
                            pass
                        # Exit immediately for consistency with multiprocessing path
                        os._exit(130)
                    result_data, msg = _process_single_subject_wrapper(args)
                    print(msg)
                    if result_data is not None:
                        result, subject_name = result_data
                        num_samples_written += _append_to_datasets(
                            datasets, result, subject_name
                        )
                        subjects_processed_set.add(subject_name)
                        sources[subject_name] = _source_signature(
                            image_dir / f"{subject_name}.nii.gz", label_dir)
                        grp.attrs["sources"] = json.dumps(sources)
                        if len(subjects_processed_set) % 10 == 0:
                            check_memory_and_warn()
        except KeyboardInterrupt:
            if not _shutdown_message_printed:
                print(
                    "\n⚠️  Keyboard interrupt received. Closing HDF5 file properly..."
                )
            _shutdown_requested = True
        except Exception as e:
            if not _shutdown_requested:
                print(f"\n⚠️  Unexpected error: {e}")
                import traceback

                traceback.print_exc()
            _shutdown_requested = True
        finally:
            # Ensure pool is properly closed
            if pool is not None:
                try:
                    if _shutdown_requested:
                        # Force terminate on shutdown - don't wait
                        pool.terminate()
                        # Don't wait for join - exit immediately
                    else:
                        pool.close()
                        pool.join(timeout=10)
                except Exception:
                    pass

        subjects_processed = len(subjects_processed_set)

        if _shutdown_requested:
            print(
                f"\n⚠️  Processing interrupted. Saved {num_samples_written} slices from {subjects_processed} subjects before shutdown."
            )
            print("  → HDF5 file has been properly closed and is safe to resume later.")
            os._exit(130)  # Use os._exit for immediate termination

    # Final summary
    with _safe_open_hdf5(output_hdf5, "r") as hf:
        grp = hf[str(target_size)]
        unique_subjects = len(set(grp["subject"][:]))
        total_slices_in_file = grp["orig_dataset"].shape[0]
        skipped_count = len(image_files) - subjects_processed

        if plane == "mixed":
            print("\nMixed-plane mode summary:")
            print(
                f"  Processed {subjects_processed} subjects × {len(planes_to_process)} planes = {subjects_processed * len(planes_to_process)} subject-plane combinations"
            )
        else:
            pass

        if file_mode == "a" and num_samples_written > 0:
            print(
                f"Newly added: {num_samples_written} slices from {subjects_processed} subjects"
            )
        elif file_mode == "a" and num_samples_written == 0 and len(image_files) > 0:
            print("⚠️  Warning: No new data added! All subjects failed processing.")
        else:
            print(
                f"Successfully processed: {subjects_processed} subjects → {num_samples_written} slices"
            )

        if skipped_count > 0:
            print(
                f"Skipped: {skipped_count} subjects (invalid zoom values or no valid slices)"
            )

        print(
            f"Total in file: {total_slices_in_file} slices from {unique_subjects} subjects"
        )
        if plane == "mixed":
            print(
                "  (Mixed-plane mode: slices from axial, coronal, and sagittal planes)"
            )
        print(f"All images are exactly {target_size}×{target_size}")

        if total_slices_in_file == 0:
            print("⚠️  Warning: HDF5 file is empty!")
            return

        print(
            f"Final shapes: Images {grp['orig_dataset'].shape}, Labels {grp['aseg_dataset'].shape}"
        )

        # Validate zoom values
        zooms_array = grp["zoom_dataset"][:]
        invalid_mask = np.any(np.abs(zooms_array) < 1e-6, axis=1)
        num_invalid = np.sum(invalid_mask)

        if num_invalid > 0:
            print(
                f"\n⚠️  WARNING: Found {num_invalid} slices with invalid zoom values ({100*num_invalid/len(zooms_array):.2f}%)"
            )
            subjects_array = grp["subject"][:]
            invalid_subjects = np.unique(subjects_array[invalid_mask])
            print(
                f"   Affected subjects ({len(invalid_subjects)}): {list(invalid_subjects[:10])}"
            )
            if len(invalid_subjects) > 10:
                print(f"   ... and {len(invalid_subjects) - 10} more")
        else:
            print(
                f"\n✓ All zoom values valid (range: [{np.min(zooms_array):.4f}, {np.max(zooms_array):.4f}])"
            )

    print(f"✅ Done! HDF5 file: {output_hdf5}")


def _process_single_subject_wrapper(args):
    """Wrapper function for parallel processing."""
    (
        idx,
        image_file,
        label_dir,
        plane,
        slice_thickness,
        target_size,
        atlas_manager,
        preprocess_params,
        num_classes,
        total_subjects,
    ) = args

    base_name = image_file.name.replace(".nii.gz", "")
    label_pattern = f"{base_name}_*.nii.gz"
    label_matches = sorted(label_dir.glob(label_pattern))
    subject_name = image_file.name.replace(".nii.gz", "").strip()

    if len(label_matches) == 0:
        return None, f"Warning: No label file found for {subject_name}, skipping"
    elif len(label_matches) > 1:
        return None, f"Warning: Multiple label files found for {subject_name}, skipping"

    label_file = label_matches[0]

    try:
        result = process_subject(
            image_file,
            label_file,
            plane,
            slice_thickness,
            target_size,
            atlas_manager,
            preprocess_params,
            num_classes,
            verbose=False,
        )

        if result is None:
            return (
                None,
                f"Processing ({idx}/{total_subjects}): {subject_name} → Skipped",
            )

        status_msg = f"Processing ({idx}/{total_subjects}): {subject_name} → {result['num_slices']} slices"
        return (result, subject_name), status_msg

    except Exception as e:
        return None, f"Error processing {subject_name}: {e}"


def main():
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, _signal_handler)  # kill command

    parser = argparse.ArgumentParser(
        description="Generate HDF5 dataset with proportional resizing"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--split_type",
        type=str,
        required=True,
        choices=["train", "val"],
        help="train or val",
    )
    parser.add_argument(
        "--atlas", type=str, default=None, help="Atlas name (e.g., ARM2, ARM3)"
    )

    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    paths = get_paths_from_config(cfg)
    num_classes = cfg["MODEL"]["NUM_CLASSES"]
    is_binary = num_classes == 2

    if is_binary:
        print(f"Binary segmentation mode (NUM_CLASSES={num_classes})")
        atlas_manager = None
    else:
        if cfg["DATA"]["CLASS_OPTIONS"] is None or not cfg["DATA"]["CLASS_OPTIONS"]:
            raise ValueError(
                f"Multi-class mode (NUM_CLASSES={num_classes}) requires CLASS_OPTIONS in config."
            )

        if args.atlas:
            atlas_name = args.atlas
        elif cfg["DATA"]["CLASS_OPTIONS"] and cfg["DATA"]["CLASS_OPTIONS"][0]:
            atlas_name = cfg["DATA"]["CLASS_OPTIONS"][0]
        else:
            raise ValueError(
                "Multi-class mode requires atlas name in CLASS_OPTIONS or --atlas argument."
            )

        print(
            f"Multi-class segmentation mode (NUM_CLASSES={num_classes}), atlas: {atlas_name}"
        )
        atlas_manager = get_atlas_manager(atlas_name)

    data_dir = paths["data_dir"]
    plane = cfg["DATA"]["PLANE"]
    target_size = cfg["DATA"]["SIZES"][0]
    thickness = (cfg["MODEL"]["NUM_CHANNELS"] - 1) // 2
    num_workers = cfg["TRAIN"].get("NUM_WORKERS", 4)
    preprocess_params = cfg["DATA"]["PREPROCESSING"]

    print("\nPreprocessing parameters:")
    print(f"  Orientation: {preprocess_params['ORIENTATION']}")
    print(
        f"  Image size: {preprocess_params['IMG_SIZE']}, Voxel size: {preprocess_params['VOX_SIZE']}"
    )
    print(
        f"  Dtype (image/label): {preprocess_params['DTYPE_IMAGE']}/{preprocess_params['DTYPE_LABEL']}"
    )

    output_hdf5 = (
        paths["train_hdf5"] if args.split_type == "train" else paths["val_hdf5"]
    )

    split_json = data_dir / "data_split.json"
    if not split_json.exists():
        # Without a split, train and val HDF5 would both contain every subject (leakage).
        raise SystemExit(
            f"ERROR: {split_json} not found. Create it first (step1_split_data.py, or "
            f"monkey_unet_prep/5_balance_sites.py), so train and val subjects differ."
        )
    with open(split_json) as f:
        split = json.load(f)
    subject_filter = set(split[args.split_type])
    overlap = set(split.get("train", [])) & set(split.get("val", []))
    if overlap:
        raise SystemExit(f"ERROR: {len(overlap)} subjects are in both train and val: {sorted(overlap)[:5]}")

    print("\nConfiguration:")
    print(
        f"  Data dir: {data_dir}, Plane: {plane}, Target size: {target_size}×{target_size}"
    )
    print(f"  Thickness: {thickness}, Workers: {num_workers}, Split: {args.split_type}")
    print(f"  Output HDF5: {output_hdf5}\n")

    create_hdf5_dataset(
        data_dir=data_dir,
        output_hdf5=output_hdf5,
        plane=plane,
        target_size=target_size,
        slice_thickness=thickness,
        atlas_manager=atlas_manager,
        preprocess_params=preprocess_params,
        num_classes=num_classes,
        subject_filter=subject_filter,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    main()
