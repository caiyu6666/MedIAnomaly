from sklearn import metrics
import torch
import numpy as np
from torch.utils.data import DataLoader, sampler
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
from .plotting_utils import plot_row
from functools import partial
from multiprocessing import Pool
from scipy import ndimage


# when full_size is True, gets 256x256 images from test_dat but only feeds 224x224 into model and pads with zeros
# this is so that we can plot results for the entire image although the model is trained on 224x224 crops
def test_real_anomalies(model, test_dat, device='cuda', batch_size=16, show=False, plots=True, full_size=False, pix_metrics=False):
    model.eval()
    model = model.to(device)
    loader = DataLoader(test_dat, batch_size=batch_size, shuffle=False)
    preds = []
    sample_preds = []
    sample_labels = []
    pixel_labels = []
    inputs = []
    image_names = []
    for data, labels, masks, image_name in tqdm(loader, desc='predict'):
        inputs.append(data.cpu())
        data = data.to(device)
        if full_size:
            data = T.CenterCrop(224)(data)
        with torch.no_grad():
            pred = model.forward(data)
        if isinstance(pred, tuple):
            pred, _ = pred
        if full_size:
            pred = T.Pad(16)(pred)
        sample_preds.append(torch.mean(pred, dim=(1, 2, 3)).cpu().numpy())
        preds.append(pred.cpu().numpy())
        sample_labels.append(labels.cpu().numpy())
        pixel_labels.append(masks.cpu().numpy())
        image_names += image_name

    inputs = torch.cat(inputs)
    preds = np.concatenate(preds)  # Nx1xHxW
    sample_preds = np.concatenate(sample_preds)
    sample_labels = np.concatenate(sample_labels)
    pixel_labels = np.concatenate(pixel_labels)  # Nx1xHxW

    if plots:
        fig, ax = plt.subplots(1, 3, figsize=(20, 40), dpi=150)
        plot_row([inputs, torch.tensor(pixel_labels), torch.tensor(preds)],
                 ['input', 'ground truth', 'prediction'], ax, grid_cols=20)
    else:
        fig = None

    sample_ap = metrics.average_precision_score(sample_labels, sample_preds)
    sample_auroc = metrics.roc_auc_score(sample_labels, sample_preds)

    results = {'sample_auc': sample_auroc, 'sample_ap': sample_ap}

    if pix_metrics:
        pixel_ap = metrics.average_precision_score(pixel_labels.reshape(-1), preds.reshape(-1))

        print(preds.shape)
        print(pixel_labels.shape)
        # resize to 64 to accelerate the computation of BestDice
        img_scale = 64 * 1.0 / pixel_labels.shape[-1]
        preds = ndimage.zoom(preds, (1., 1., img_scale, img_scale), order=3)
        pixel_labels = ndimage.zoom(pixel_labels, (1., 1., img_scale, img_scale), order=0)

        best_dice, _ = compute_best_dice(preds, pixel_labels)
        results.update({'pixel_ap': pixel_ap, 'best_dice': best_dice})

    if show:
        print('sample AP: {:.5f}, AUROC: {:.5f}'.format(sample_ap, sample_auroc))
        plt.show()

    # return sample_ap, sample_auroc, fig
    return results, fig, preds, image_names


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
                      n_thresh: float = 100,
                      num_processes: int = 64):
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
