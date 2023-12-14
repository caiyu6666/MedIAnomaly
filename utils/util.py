import torch
import numpy as np
from medpy import metric
from functools import partial
from multiprocessing import Pool
from skimage import measure
from statistics import mean
import pandas as pd
from numpy import ndarray
from sklearn.metrics import auc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_dice(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes the Sorensen-Dice coefficient:

    dice = 2 * TP / (2 * TP + FP + FN)

    :param preds: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    """
    preds, targets = np.array(preds), np.array(targets)

    # Check if predictions and targets are binary
    if not np.all(np.logical_or(preds == 0, preds == 1)):
        raise ValueError('Predictions must be binary')
    if not np.all(np.logical_or(targets == 0, targets == 1)):
        raise ValueError('Targets must be binary')

    # Compute Dice
    dice = 2 * np.sum(preds[targets == 1]) / \
        (np.sum(preds) + np.sum(targets))

    return dice


def compute_best_dice(preds: np.ndarray, targets: np.ndarray,
                      # n_thresh: float = 100,
                      n_thresh: float = 1000,
                      num_processes: int = 8):
    """
    Compute the best dice score for n_thresh thresholds.

    :param predictions: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    :param n_thresh: Number of thresholds to check.
    """
    preds, targets = np.array(preds), np.array(targets)

    # thresholds = np.linspace(preds.max(), preds.min(), n_thresh)
    num = preds.size
    step = num // n_thresh
    indices = np.arange(0, num, step)
    thresholds = np.sort(preds.reshape(-1))[indices]

    with Pool(num_processes) as pool:
        fn = partial(_dice_multiprocessing, preds, targets)
        scores = pool.map(fn, thresholds)

    scores = np.stack(scores, 0)
    max_dice = scores.max()
    max_thresh = thresholds[scores.argmax()]
    return max_dice, max_thresh


def _dice_multiprocessing(preds: np.ndarray, targets: np.ndarray,
                          threshold: float) -> float:
    return compute_dice(np.where(preds > threshold, 1, 0), targets)


def calculate_threshold_fpr(y_true, y_pred, target_fpr=0.01, num_iters=20):
    """
    Determine the threshold at the target fpr using dichotomy.
    :param y_true:
    :param y_pred:
    :param target_fpr:
    :param num_iters:  The larger num_iters, the higher accuracy for the threshold.
    :return:
    """
    # left, right = 0.0, 1.0
    left, right = np.min(y_pred), np.max(y_pred)
    threshold = (left + right) / 2.
    for i in range(num_iters):
        y_pred_binarized = (y_pred > threshold).astype(np.uint8)
        tn = np.sum(np.logical_and(y_true == 0, y_pred_binarized == 0))
        fp = np.sum(np.logical_and(y_true == 0, y_pred_binarized == 1))
        fpr = fp / (fp + tn)

        if fpr > target_fpr:
            left = threshold
            threshold = (left + right) / 2.
        elif fpr < target_fpr:
            right = threshold
            threshold = (left + right) / 2.
        else:
            break

    y_pred_binarized = (y_pred > threshold).astype(np.uint8)
    tn = np.sum(np.logical_and(y_true == 0, y_pred_binarized == 0))
    fp = np.sum(np.logical_and(y_true == 0, y_pred_binarized == 1))
    fpr = fp / (fp + tn)
    return fpr, threshold


def calculate_dice_thr(y_true, y_score, threshold):
    dice_l = []
    for i in range(len(y_score)):
        volume_score, volume_mask = y_score[i], y_true[i]
        volume_score = np.array(volume_score)
        volume_pred = (volume_score > threshold).astype(np.uint8)
        dice = metric.binary.dc(volume_pred, volume_mask)
        dice_l.append(dice)
    return np.mean(dice_l)


# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
import warnings
from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor


def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input: Tensor, win: Tensor) -> Tensor:
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float,
    win: Tensor,
    size_average: bool = True,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03)
) -> Tensor:
    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        data_range (float or int): value range of input images. (usually 1.0 or 255)
        win (torch.Tensor): 1-D gauss kernel
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    # ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    # cs = torch.flatten(cs_map, 2).mean(-1)
    # return ssim_per_channel, cs
    return ssim_map


def ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
) -> Tensor:
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    #if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    # ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    # if nonnegative_ssim:
    #     ssim_per_channel = torch.relu(ssim_per_channel)

    # if size_average:
    #     return ssim_per_channel.mean()
    # else:
    #     return ssim_per_channel.mean(1)
    ssim_map = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    return torch.mean(ssim_map, dim=[1, 2, 3]) if size_average else ssim_map
