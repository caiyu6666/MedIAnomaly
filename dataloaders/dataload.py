import os
import time

from PIL import Image
from torch.utils import data
import json
from joblib import Parallel, delayed
import numpy as np
from torchvision import transforms
import torch
import glob
import SimpleITK as sitk
from scipy import ndimage


def parallel_load(img_dir, img_list, img_size, verbose=0):
    return Parallel(n_jobs=-1, verbose=verbose)(delayed(
        lambda file: Image.open(os.path.join(img_dir, file)).convert("L").resize(
            (img_size, img_size), resample=Image.BILINEAR))(file) for file in img_list)


class MedAD(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None, mode="train"):
        super(MedAD, self).__init__()
        assert mode in ["train", "test"]
        self.root = main_path
        self.labels = []
        self.img_id = []
        self.slices = []
        self.transform = transform if transform is not None else lambda x: x

        with open(os.path.join(main_path, "data.json")) as f:
            data_dict = json.load(f)

        print("Loading images")
        if mode == "train":
            train_normal = data_dict["train"]["0"]

            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "images"), train_normal, img_size)
            self.labels += len(train_normal) * [0]
            self.img_id += [img_name.split('.')[0] for img_name in train_normal]
            print("Loaded {} normal images, {:.3f}s".format(len(train_normal), time.time() - t0))

        else:  # test
            test_normal = data_dict["test"]["0"]
            test_abnormal = data_dict["test"]["1"]

            test_l = test_normal + test_abnormal
            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "images"), test_l, img_size)
            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_id += [img_name.split('.')[0] for img_name in test_l]
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        label = self.labels[index]
        img = self.transform(img)
        img_id = self.img_id[index]
        return {'img': img, 'label': label, 'name': img_id}

    def __len__(self):
        return len(self.slices)


class BraTSAD(data.Dataset):
    o_size = (70, 208, 208)

    def __init__(self, main_path, img_size=64, transform=None, istrain=True):
        # "/home/ycaibt/datasets/BraTS2021/BraTS_AD/"
        super(BraTSAD, self).__init__()
        self.root = main_path
        self.istrain = istrain
        self.res = img_size
        self.labels = []
        self.masks = []
        self.img_id = []
        self.slices = []  # slices for training and volumes for testing
        # self.transform = transform if transform is not None else lambda x: x
        self.transform = transform

        print("Loading images")
        if istrain:
            data_dir = os.path.join(self.root, "train")
            train_normal = glob.glob(os.path.join(data_dir, "*", "*.png"))

            t0 = time.time()
            self.slices += parallel_load("", train_normal, img_size)
            self.labels += [0] * len(train_normal)
            self.img_id += [img_name.split('/')[-1].split('.')[0] for img_name in train_normal]
            print("Loaded {} normal images, {:.3f}s".format(len(train_normal), time.time() - t0))

        else:  # test
            test_dir = os.path.join(self.root, "test")

            test_scans = sorted(glob.glob(os.path.join(test_dir, "*", "*flair.nii.gz")))
            seg_masks = sorted(glob.glob(os.path.join(test_dir, "*", "*seg.nii.gz")))

            def load_mri(img_path):
                img_scale = img_size * 1.0 / self.o_size[1]
                img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                img = ndimage.zoom(img, (1., img_scale, img_scale), order=3)
                return img

            def load_mask(img_path):
                img_scale = img_size * 1.0 / self.o_size[1]
                img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                img = (img > 0).astype(np.uint8)
                img = ndimage.zoom(img, (1., img_scale, img_scale), order=0)
                return img

            t0 = time.time()
            self.slices += Parallel(n_jobs=-1, verbose=0)(delayed(load_mri)(file) for file in test_scans)
            self.masks += Parallel(n_jobs=-1, verbose=0)(delayed(load_mask)(file) for file in seg_masks)

            self.img_id += [img_name.split('/')[-1].split('.')[0][:-6] for img_name in test_scans]
            print("Loaded {} test mri scans and {} test segmentation masks. {:.3f}s".format(
                len(test_scans), len(seg_masks), time.time() - t0))

    def __getitem__(self, index):
        img_id = self.img_id[index]
        if self.istrain:
            img = self.slices[index]
            img = self.transform(img)
            label = self.labels[index]
            return {'img': img, 'label': label, 'name': img_id}
        else:
            volume = self.slices[index]
            volume = volume.transpose(1, 2, 0)
            volume = self.transform(volume)
            mask = self.masks[index]

            return {'volume': volume, 'mask': mask, 'name': img_id}

    def __len__(self):
        return len(self.slices)
