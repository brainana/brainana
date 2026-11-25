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
import numpy as np
import torch
from scipy.ndimage import (
    _ni_support,
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)
from scipy.spatial.distance import dice

from fastsurfer_nn.utils import logging

logger = logging.getLogger(__name__)


def iou_score(
    pred_cls: torch.Tensor, true_cls: torch.Tensor, nclass: int = 79
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the intersection-over-union score.

    Both inputs should be categorical (as opposed to one-hot).

    Parameters
    ----------
    pred_cls : torch.Tensor
        Network prediction (categorical).
    true_cls : torch.Tensor
        Ground truth (categorical).
    nclass : int
        Number of classes (Default value = 79).

    Returns
    -------
    np.ndarray
        An array containing the intersection for each class.
    np.ndarray
        An array containing the union for each class.
    """
    intersect_ = []
    union_ = []

    for i in range(1, nclass):
        intersect = (
            ((pred_cls == i).float() + (true_cls == i).float()).eq(2).sum().item()
        )
        union = ((pred_cls == i).float() + (true_cls == i).float()).ge(1).sum().item()
        intersect_.append(intersect)
        union_.append(union)

    return np.array(intersect_), np.array(union_)


def precision_recall(
    pred_cls: torch.Tensor, true_cls: torch.Tensor, nclass: int = 79
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate recall (TP/(TP + FN) and precision (TP/(TP+FP) per class.

    Parameters
    ----------
    pred_cls : torch.Tensor
        Network prediction (categorical).
    true_cls : torch.Tensor
        Ground truth (categorical).
    nclass : int
        Number of classes (Default value = 79).

    Returns
    -------
    np.ndarray
        An array containing the number of true positives for each class.
    np.ndarray
        An array containing the sum of true positives and false negatives for each class.
    np.ndarray
        An array containing the sum of true positives and false positives for each class.
    """
    tpos_fneg = []
    tpos_fpos = []
    tpos = []

    for i in range(1, nclass):
        all_pred = (pred_cls == i).float()
        all_gt = (true_cls == i).float()

        tpos.append((all_pred + all_gt).eq(2).sum().item())
        tpos_fpos.append(all_pred.sum().item())
        tpos_fneg.append(all_gt.sum().item())

    return np.array(tpos), np.array(tpos_fneg), np.array(tpos_fpos)


class DiceScore:
    """
    Accumulating the component of the dice coefficient i.e. the union and intersection.

    Parameters
    ----------
    op : callable
        A callable to update the accumulator. Method's signature is `(accumulator, output)`.
        For example, to compute arithmetic mean value, `op = lambda a, x: a + x`.
    output_transform : callable, optional
        A callable that is used to transform the :class:`~ignite.engine.Engine`'s `process_function`'s output into the
        form expected by the metric. This can be useful if, for example, you have a multi-output model and
        you want to compute the metric with respect to one of the outputs.
    device : str or torch.device, optional
        Device specification in case of distributed computation usage.
        In most cases, it can be defined as "cuda:local_rank" or "cuda"
        if already set `torch.cuda.set_device(local_rank)`. By default, if a distributed process group is
        initialized and available, the device is set to `cuda`.
    """

    def __init__(
        self,
        num_classes,
        class_ids=None,
        device=None,
        one_hot=False,
        output_transform=lambda y_pred, y: (y_pred.data.max(1)[1], y),
    ):
        self._device = device
        self.out_transform = output_transform
        self.class_ids = class_ids
        if self.class_ids is None:
            self.class_ids = np.arange(num_classes)
        self.n_classes = num_classes
        assert len(self.class_ids) == self.n_classes, (
            f"Number of class ids is not correct,"
            f" given {len(self.class_ids)} but {self.n_classes} is needed."
        )
        self.one_hot = one_hot
        # Create tensors on the correct device from the start
        self.union = torch.zeros(self.n_classes, self.n_classes, device=self._device)
        self.intersection = torch.zeros(
            self.n_classes, self.n_classes, device=self._device
        )

    def reset(self):
        """
        Reset the state of the object.
        """
        # Create tensors on the correct device (use stored device)
        self.union = torch.zeros(self.n_classes, self.n_classes, device=self._device)
        self.intersection = torch.zeros(
            self.n_classes, self.n_classes, device=self._device
        )

    def _check_output_type(self, output):
        """
        Check the type of the output and raise an error if it doesn't match expectations.

        Parameters
        ----------
        output : tuple
            The output to be checked, expected to be a tuple.
        """
        if not (isinstance(output, tuple)):
            raise TypeError(
                f"Output should be a tuple consisting of torch.Tensors, but given {type(output)}"
            )

    def _update_union_intersection(self, batch_output, labels_batch):
        """
        Update the union and intersection matrices based on batch predictions and labels.
        Vectorized implementation for better performance.

        Parameters
        ----------
        batch_output : torch.Tensor
            Batch predictions from the model.

        labels_batch : np.ndarray or torch.Tensor
            Batch labels from the dataset.
        """
        # Ensure tensors are on the same device
        if self.union.device != batch_output.device:
            self.union = self.union.to(batch_output.device)
            self.intersection = self.intersection.to(batch_output.device)

        # Convert labels_batch to tensor if it's numpy array
        if isinstance(labels_batch, np.ndarray):
            labels_batch = torch.from_numpy(labels_batch).to(batch_output.device)

        batch_size = batch_output.shape[0]
        num_classes = len(self.class_ids)
        device = batch_output.device

        # Flatten spatial dimensions: (batch_size, height*width)
        batch_output_flat = batch_output.view(batch_size, -1)
        labels_batch_flat = labels_batch.view(batch_size, -1)

        # Create one-hot encodings for all classes at once
        # Shape: (batch_size, height*width, num_classes)
        pred_onehot = torch.zeros(
            batch_size,
            batch_output_flat.shape[1],
            num_classes,
            device=device,
            dtype=torch.float32,
        )
        gt_onehot = torch.zeros(
            batch_size,
            labels_batch_flat.shape[1],
            num_classes,
            device=device,
            dtype=torch.float32,
        )

        for i, class_id in enumerate(self.class_ids):
            pred_onehot[:, :, i] = (batch_output_flat == class_id).float()
            gt_onehot[:, :, i] = (labels_batch_flat == class_id).float()

        # Vectorized computation using einsum for efficiency
        # Intersection: sum over batch and spatial dims for each class pair
        # Shape: (num_classes, num_classes)
        intersection = torch.einsum("bsc,bst->ct", gt_onehot, pred_onehot)

        # Union: sum of gt + sum of pred for each class pair
        # Shape: (num_classes, num_classes)
        gt_sums = torch.sum(gt_onehot, dim=(0, 1))  # (num_classes,)
        pred_sums = torch.sum(pred_onehot, dim=(0, 1))  # (num_classes,)
        union = gt_sums.unsqueeze(1) + pred_sums.unsqueeze(0)  # Broadcasting

        # Update accumulated values
        self.intersection += intersection
        self.union += union

    def update(self, output):
        """
        Update the internal state based on the output.

        Parameters
        ----------
        output : tuple of torch.Tensor
            Tuple of predictions and labels.
        """
        self._check_output_type(output)

        if self.out_transform is not None:
            y_pred, y = self.out_transform(*output)
        else:
            y_pred, y = output

        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y)

        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.from_numpy(y_pred)

        if self._device is not None:
            y = y.to(self._device)
            y_pred = y_pred.to(self._device)

        self._update_union_intersection(y_pred, y)

    def compute(self, per_class=False, class_idxs=None, exclude_background=False):
        """
        Compute the Dice score (GPU-only version).

        Parameters
        ----------
        per_class : bool
            If True, return per-class dice scores. If False, return mean dice.
        class_idxs : torch.Tensor, optional
            Class indices to include in computation.
        exclude_background : bool
            If True, exclude background class (class 0) from mean calculation.
            Default is False (include all valid classes).

        Returns
        -------
        If per_class=False: (dice_score, dice_cm_mat) tuple
        If per_class=True: (dice_score_per_class, dice_cm_mat) tuple
        """
        dice_cm_mat = self._dice_confusion_matrix_gpu(class_idxs)
        dice_score_per_class = dice_cm_mat.diagonal()

        # Only average over classes that have non-zero union (actually present in data)
        # This avoids artificially deflating the score with classes that don't exist
        dice_union = self.union
        if class_idxs is not None:
            dice_union = dice_union[class_idxs[:, None], class_idxs]

        # Determine which classes to include in mean
        if exclude_background and self.n_classes > 1:
            # Exclude background (class 0), only consider classes with non-zero union
            region_dice = dice_score_per_class[1:]  # Exclude class 0
            valid_classes = dice_union.diagonal()[1:] > 0  # Exclude class 0
        else:
            # Include all classes with non-zero union
            region_dice = dice_score_per_class
            valid_classes = dice_union.diagonal() > 0

        if valid_classes.any():
            dice_score = region_dice[valid_classes].mean()
        else:
            dice_score = torch.tensor(0.0, device=self.union.device)

        if per_class:
            return dice_score_per_class, dice_cm_mat
        else:
            return dice_score, dice_cm_mat

    def _dice_confusion_matrix_gpu(self, class_idxs):
        """
        Compute the Dice score confusion matrix (GPU-only version).
        """
        dice_intersection = self.intersection
        dice_union = self.union
        if class_idxs is not None:
            dice_union = dice_union[class_idxs[:, None], class_idxs]
            dice_intersection = dice_intersection[class_idxs[:, None], class_idxs]

        # Avoid NaN by setting dice to 0 where union is 0 (class not present in GT or prediction)
        # This is standard practice: if a class isn't present, it doesn't contribute to the mean
        dice_cnf_matrix = torch.where(
            dice_union > 0,
            2 * dice_intersection / dice_union,
            torch.tensor(0.0, device=dice_union.device),
        )
        return dice_cnf_matrix

    def _dice_confusion_matrix(self, class_idxs):
        """
        Compute the Dice score confusion matrix (legacy CPU version).
        """
        dice_intersection = self.intersection.cpu().numpy()
        dice_union = self.union.cpu().numpy()
        if class_idxs is not None:
            dice_union = dice_union[class_idxs[:, None], class_idxs]
            dice_intersection = dice_intersection[class_idxs[:, None], class_idxs]
        # if not (dice_union > 0).all():
        #     logger.info("Union of some classes are all zero")
        # Avoid NaN by setting dice to 0 where union is 0 (class not present in GT or prediction)
        # This is standard practice: if a class isn't present, it doesn't contribute to the mean
        dice_cnf_matrix = np.where(
            dice_union > 0, 2 * dice_intersection / dice_union, 0.0
        )
        return dice_cnf_matrix


def dice_score(pred, gt, validate=True):
    """
    Calculates the Dice Dissimilarity between pred and gt (best 0).

    Parameters
    ----------
    pred : np.ndarray
        Predicted image.
    gt : np.ndarray
        Ground truth image.
    validate : bool
        If True, use the scipy implementation of the Dice Similarity. If False, use the numpy implementation.

    Returns
    -------
    float
        Dice Similarity between pred and gt.
    """
    if validate:
        return dice(pred.flat, gt.flat)

    else:
        assert pred.dtype == gt.dtype == bool, "Input images must be boolean"

        ntt = (pred & gt).sum()
        nft = (~pred & gt).sum()
        ntf = (pred & ~gt).sum()
        return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))


def volume_similarity(pred, gt):
    """
    Calculate the Volume Similarity between pred and gt.
    """
    pred_vol, gt_vol = np.sum(pred), np.save(gt)
    return 1.0 - np.abs(pred_vol - gt_vol) / (pred_vol + gt_vol)


# https://github.com/amanbasu/3d-prostate-segmentation/blob/master/metric_eval.py
def hd(result, reference, voxelspacing=None, connectivity=1):
    """
    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. It is defined as the maximum surface distance between the objects.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in `result` and the
        object(s) in `reference`. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.
    hd50 : float
        The 50th percentile of the Hausdorff Distance.
    hd95 : float
        The 95th percentile of the Hausdorff Distance.

    See Also
    --------
    assd : Average Symmetric Surface Distance, computes the average symmetric surface distance.
    asd : Average Surface Distance, computes the average surface distance.

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd = max(hd1.max(), hd2.max())
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    hd50 = np.percentile(np.hstack((hd1, hd2)), 50)
    return hd, hd50, hd95


def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    Computes the 95th percentile of the Hausdorff Distance.

    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.

    Parameters
    ----------
    result : Any
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd95 : float
        The 95th percentile of the symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    See Also
    --------
    hd : Computes the symmetric Hausdorff Distance.

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError(
            "The first supplied array does not contain any binary object."
        )
    if 0 == np.count_nonzero(reference):
        raise RuntimeError(
            "The second supplied array does not contain any binary object."
        )

    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(
        reference, structure=footprint, iterations=1
    )

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


# ----------------------------------------------------------------------------- fast Dice
class DiceAccumulator:
    """Per-subject, per-class overlap counts accumulated on the GPU with ``bincount``.

    ``update(pred, target, subject)`` takes dense class maps [B, H, W] (long) and subject
    indices [B]. No one-hot tensors are built, so it costs ~3 bincounts per batch.

    * ``subject_dice()``  -> [S, C] Dice per subject and class (NaN where the class is in
      neither the label nor the prediction of that subject).
    * ``mean_subject_dice()`` -> mean over subjects of the mean foreground Dice per subject:
      the 3-D, per-subject validation score used to pick the best checkpoint.
    * ``pooled_dice()`` -> [C] Dice of all voxels pooled (the pre-2026 "val_dice" style).
    Class 0 (background) is excluded from every mean.
    """

    def __init__(self, num_classes: int, num_subjects: int = 1, device="cpu"):
        self.c = int(num_classes)
        self.s = int(num_subjects)
        shape = (self.s, self.c)
        self.inter = torch.zeros(shape, dtype=torch.float64, device=device)
        self.psum = torch.zeros(shape, dtype=torch.float64, device=device)
        self.tsum = torch.zeros(shape, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor,
               subject: torch.Tensor | None = None) -> None:
        pred = pred.long()
        target = target.long()
        n = self.s * self.c
        if subject is None:
            offset = torch.zeros_like(target)
        else:
            offset = subject.to(target.device).long().view(-1, *([1] * (target.dim() - 1))) * self.c
        t_idx = (offset + target).flatten()
        p_idx = (offset + pred).flatten()
        self.tsum += torch.bincount(t_idx, minlength=n).view(self.s, self.c)
        self.psum += torch.bincount(p_idx, minlength=n).view(self.s, self.c)
        hit = (pred == target).flatten()
        self.inter += torch.bincount(t_idx[hit], minlength=n).view(self.s, self.c)

    def subject_dice(self) -> torch.Tensor:
        denom = self.psum + self.tsum
        dice = 2 * self.inter / denom.clamp_min(1)
        dice[denom == 0] = float("nan")
        return dice

    def subject_mean(self) -> torch.Tensor:
        """[S] mean foreground Dice per subject (subjects without any voxel -> NaN)."""
        return torch.nanmean(self.subject_dice()[:, 1:], dim=1)

    def mean_subject_dice(self) -> float:
        m = self.subject_mean()
        m = m[~torch.isnan(m)]
        return float(m.mean()) if m.numel() else float("nan")

    def per_class_dice(self) -> torch.Tensor:
        """[C] mean over subjects of each class's Dice."""
        return torch.nanmean(self.subject_dice(), dim=0)

    def pooled_dice(self) -> torch.Tensor:
        i, p, t = self.inter.sum(0), self.psum.sum(0), self.tsum.sum(0)
        denom = p + t
        d = 2 * i / denom.clamp_min(1)
        d[denom == 0] = float("nan")
        return d

    def mean_pooled_dice(self) -> float:
        d = self.pooled_dice()[1:]
        d = d[~torch.isnan(d)]
        return float(d.mean()) if d.numel() else float("nan")
