import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import argparse
import cv2
import json
import glob
import pandas as pd

from self_sup_data.chest_xray import SelfSupChestXRay
from model.resnet import resnet18_enc_dec
from experiments.training_utils import train_and_save_model

import warnings

warnings.filterwarnings("ignore")

SETTINGS = {
    ### ------------------------------------------------ NSA ------------------------------------------------ ###
    'Shift': {
        'fname': 'shift.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'self_sup_args': {'resize': True, 'shift': True, 'same': False, 'mode': cv2.NORMAL_CLONE,
                          'label_mode': 'binary'}
    },
    'Shift-Intensity': {
        'fname': 'shift_intensity.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'self_sup_args': {'resize': True, 'shift': True, 'same': False, 'mode': cv2.NORMAL_CLONE,
                          'label_mode': 'logistic-intensity'}
    },
    'Shift-Raw-Intensity': {
        'fname': 'shift_raw_intensity.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True,
        'final_activation': 'relu',
        'self_sup_args': {'resize': True, 'shift': True, 'same': False, 'mode': cv2.NORMAL_CLONE,
                          'label_mode': 'intensity'}
    },
    'Shift-M': {
        'fname': 'shift_m.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'self_sup_args': {'resize': True, 'shift': True, 'same': False, 'mode': cv2.MIXED_CLONE, 'label_mode': 'binary'}
    },
    'Shift-Raw-Intensity-M': {
        'fname': 'shift_raw_intensity_m.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True,
        'final_activation': 'relu',
        'self_sup_args': {'resize': True, 'shift': True, 'same': False, 'mode': cv2.MIXED_CLONE,
                          'label_mode': 'intensity'}
    },
    ### ------------------------------------ NSA for medical data ------------------------------------ ###
    'Shift-Intensity-M': {
        'fname': 'shift_intensity_m.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'self_sup_args': {'resize': True, 'shift': True, 'same': False, 'mode': cv2.MIXED_CLONE,
                          'label_mode': 'logistic-intensity'}
    },
    ### ---------------------------- Foreign patch poisson blending / interpolation ---------------------------- ###
    'FPI-Poisson': {
        'fname': 'fpi_poisson.pt',
        'out_dir': 'fpi/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'self_sup_args': {'resize': False, 'shift': False, 'same': False, 'mode': cv2.MIXED_CLONE,
                          'label_mode': 'continuous'}
    },
    'FPI': {
        'fname': 'fpi.pt',
        'out_dir': 'fpi/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'self_sup_args': {'resize': False, 'shift': False, 'same': False, 'mode': 'uniform', 'label_mode': 'continuous'}
    },
    ### ------------------------------------ Shifted patch pasting ------------------------------------ ###
    'CutPaste': {
        'fname': 'cut_paste.pt',
        'out_dir': 'cut_paste/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'self_sup_args': {'resize': False, 'shift': True, 'same': True, 'mode': 'swap', 'label_mode': 'binary'}
    },
}

# ((h_min, h_max), (w_min, w_max))
# note: this is half-width not width
WIDTH_BOUNDS_PCT = ((0.03, 0.4), (0.03, 0.4))

GAMMA_PARAMS = (2, 0.03, 0.05)

MIN_OVERLAP_PCT = 0.7

MIN_OBJECT_PCT = 0.7

NUM_PATCHES = 3

# k, x0 
INTENSITY_LOGISTIC_PARAMS = (1 / 2, 4)

# brightness, threshold pairs
BACKGROUND = (0, 20)


def set_seed(seed_value):
    """Set seed for reproducibility.
    """
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train(data, out_dir, setting, device, pool, preact,
          min_lr=1e-6, max_lr=1e-3, batch_size=64, seed=1982342, num_epochs=400):
    # set_seed(setting.get('seed', seed))
    train_transform = T.Compose([
        T.RandomRotation(3),
        T.CenterCrop(230),
        T.RandomCrop(224)])

    data_root = os.path.join(os.path.expanduser("~"), "MedIAnomaly-Data")
    if data in ['rsna', 'vin', 'brain', 'lag']:
        if data == 'rsna':
            path = os.path.join(data_root, 'RSNA')
        elif data == 'vin':
            path = os.path.join(data_root, "VinCXR")
        elif data == 'brain':
            path = os.path.join(data_root, "BrainTumor")
        elif data == 'lag':
            path = os.path.join(data_root, "LAG")
        else:
            raise Exception("Invalid dataset: {}".format(data))
        with open(os.path.join(path, "data.json")) as f:
            data_dict = json.load(f)
        file_list = data_dict["train"]["0"]
        file_list = [os.path.join(path, "images", e) for e in file_list]
    elif data == 'isic':
        path = os.path.join(data_root, "ISIC2018_Task3")
        data_csv = pd.read_csv(os.path.join(path, "ISIC2018_Task3_Training_GroundTruth",
                                            "ISIC2018_Task3_Training_GroundTruth.csv"))
        train_normal = list(data_csv[data_csv['NV'] == 1]['image'])
        train_normal = [e + ".jpg" for e in train_normal]
        file_list = [os.path.join(path, "ISIC2018_Task3_Training_Input", e) for e in train_normal]
    elif data == 'c16':
        path = os.path.join(data_root, "Camelyon16")
        file_list = glob.glob(os.path.join(path, "train", "good", "*.png"))
    elif data == 'brats':
        path = os.path.join(data_root, "BraTS2021")
        file_list = glob.glob(os.path.join(path, "train", "*.png"))
    else:
        raise Exception("Invalid dataset: {}".format(data))

    train_dat = SelfSupChestXRay(normal_files=file_list, is_train=True, res=256, transform=train_transform)
    # Note: resize to 256, and then crop to 224 (in train_transform).

    train_dat.configure_self_sup(self_sup_args=setting.get('self_sup_args'))
    train_dat.configure_self_sup(on=True, self_sup_args={'width_bounds_pct': WIDTH_BOUNDS_PCT,
                                                         'intensity_logistic_params': INTENSITY_LOGISTIC_PARAMS,
                                                         'num_patches': NUM_PATCHES,
                                                         'skip_background': BACKGROUND,
                                                         'min_object_pct': MIN_OBJECT_PCT,
                                                         'min_overlap_pct': MIN_OVERLAP_PCT,
                                                         'gamma_params': GAMMA_PARAMS,
                                                         'verbose': False})

    loader_train = DataLoader(train_dat, batch_size, shuffle=True, num_workers=os.cpu_count(),
                              worker_init_fn=lambda _: np.random.seed(
                                  torch.utils.data.get_worker_info().seed % 2 ** 32))

    model = resnet18_enc_dec(num_classes=1, pool=pool, preact=preact, in_channels=1,
                             final_activation=setting.get('final_activation')).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=min_lr)
    loss_func = setting.get('loss')()

    out_dir = os.path.join(out_dir, setting.get('out_dir'))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_and_save_model(model, optimizer, loss_func, loader_train, setting.get('fname'), out_dir,
                         num_epochs=num_epochs, save_freq=50, device=device, scheduler=scheduler,
                         save_intermediate_model=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, type=str)
    parser.add_argument("-g", "--gpu", required=False, type=int, default=7)
    # parser.add_argument("-e", "--epoch", required=False, type=int, default=200)
    # parser.add_argument("-l", "--file_list", required=True, type=str)
    # parser.add_argument("-o", "--out_dir", required=False, type=str)
    parser.add_argument("-s", "--setting", required=True, type=str)
    parser.add_argument("-f", '--fold', type=int, default=0, help='0-4, experiment index')
    parser.add_argument("--no_pool", required=False, action='store_true')
    parser.add_argument("--preact", required=False, action='store_true')
    args = parser.parse_args()

    epochs = {'rsna': 250, 'vin': 400, 'brain': 250, 'lag': 250, 'brats': 250, 'isic': 250, 'c16': 250}
    # epoch = epochs[args.data]
    epoch = epochs.setdefault(args.data, 250)

    out_dir = os.path.join("output", args.data, "fold_{:d}".format(args.fold))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    setting = SETTINGS.get(args.setting)

    train(args.data, out_dir, setting, device, not args.no_pool, args.preact, num_epochs=epoch)
