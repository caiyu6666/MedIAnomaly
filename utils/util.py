import torch
import numpy as np
from medpy import metric


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


# def l2_distance(x, x_hat):
#     """
#     Compute the l2 distance for anomaly score map.
#     :param x: (N, C, H, W)
#     :param x_hat: (N, C, H, W)
#     :return: (N, C, H, W)
#     """
#     return (x - x_hat) ** 2
#
#
# def l1_distance(x, x_hat):
#     """
#     Compute the l1 distance for anomaly score map.
#     :param x: (N, C, H, W)
#     :param x_hat: (N, C, H, W)
#     :return: (N, C, H, W)
#     """
#     return torch.abs(x - x_hat)


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
