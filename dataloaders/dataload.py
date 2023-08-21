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
    def __init__(self, main_path, img_size=64, transform=None, mode="train", context_encoding=False):
        super(MedAD, self).__init__()
        assert mode in ["train", "test"]
        self.root = main_path
        self.labels = []
        self.img_id = []
        self.slices = []
        self.transform = transform if transform is not None else lambda x: x
        if context_encoding:
            self.random_mask = transforms.RandomErasing(p=1., scale=(0.024, 0.024), ratio=(1., 1.), value=-1)
        else:
            self.random_mask = None

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

        if self.random_mask is not None:
            img_masked = self.random_mask(img)
            return {'img': img, 'label': label, 'name': img_id, 'img_masked': img_masked}
        else:
            return {'img': img, 'label': label, 'name': img_id}

    def __len__(self):
        return len(self.slices)


class BraTSAD(data.Dataset):
    o_size = (70, 208, 208)

    def __init__(self, main_path, img_size=64, transform=None, istrain=True, context_encoding=False):
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

        if context_encoding:
            self.random_mask = transforms.RandomErasing(p=1., scale=(0.024, 0.024), ratio=(1., 1.), value=-1)
        else:
            self.random_mask = None

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
            if self.random_mask is not None:
                img_masked = self.random_mask(img)
                return {'img': img, 'label': label, 'name': img_id, 'img_masked': img_masked}
            else:
                return {'img': img, 'label': label, 'name': img_id}
        else:
            volume = self.slices[index]
            volume = volume.transpose(1, 2, 0)
            volume = self.transform(volume)
            mask = self.masks[index]

            return {'volume': volume, 'mask': mask, 'name': img_id}

    def __len__(self):
        return len(self.slices)


# The following datasets, OCT2017 and Hyper-Kvasir, are not distinguishable for the evaluation of anomaly detection

class OCT2017(data.Dataset):
    def __init__(self, main_path="/home/ycaibt/datasets/OCT2017/", img_size=64, transform=None, mode="train"):
        super(OCT2017, self).__init__()
        assert mode in ["train", "test"]

        self.transform = transform
        self.root = main_path
        self.res = img_size
        self.slices = []
        self.img_ids = []

        print("Loading images")
        if mode == "train":
            train_normal = sorted(os.listdir(os.path.join(self.root, "train/NORMAL/")))

            self.labels = [0] * len(train_normal)

            self.slices += parallel_load(os.path.join(self.root, "train", "NORMAL"), train_normal, img_size)
            print("Loaded normal training images: {}".format(len(train_normal)))

            self.img_ids += [img_name.split('.')[0] for img_name in train_normal]

        else:  # mode == "test"
            test_normal = os.listdir(os.path.join(self.root, "test/NORMAL/"))
            test_CNV = os.listdir(os.path.join(self.root, "test/CNV/"))
            test_DME = os.listdir(os.path.join(self.root, "test/DME/"))
            test_DRUSEN = os.listdir(os.path.join(self.root, "test/DRUSEN/"))

            self.labels = [0] * len(test_normal) + [1] * (len(test_CNV) + len(test_DME) + len(test_DRUSEN))

            self.slices += parallel_load(os.path.join(self.root, "test", "NORMAL"), test_normal, img_size)
            self.slices += parallel_load(os.path.join(self.root, "test", "CNV"), test_CNV, img_size)
            self.slices += parallel_load(os.path.join(self.root, "test", "DME"), test_DME, img_size)
            self.slices += parallel_load(os.path.join(self.root, "test", "DRUSEN"), test_DRUSEN, img_size)

            print("Loaded normal testing images: {}".format(len(test_normal)))
            print("Loaded CNV testing images: {}".format(len(test_CNV)))
            print("Loaded DME testing images: {}".format(len(test_DME)))
            print("Loaded DRUSEN testing images: {}".format(len(test_DRUSEN)))

            self.img_ids += [img_name.split('.')[0] for img_name in test_normal+test_CNV+test_DME+test_DRUSEN]

    def __getitem__(self, index):
        img = self.slices[index]

        img = self.transform(img)
        label = self.labels[index]
        img_id = self.img_ids[index]

        return {'img': img, 'label': label, 'name': img_id}

    def __len__(self):
        return len(self.slices)


class ColonAD(data.Dataset):
    def __init__(self, main_path, img_size=224, transform=None, mode="train"):
        super(ColonAD, self).__init__()
        assert mode in ["train", "test"]

        self.root = main_path
        self.res = img_size
        self.labels = []
        self.img_ids = []
        self.slices = []
        self.transform = transform

        print("Loading images")
        if mode == "train":
            data_dir = os.path.join(self.root, "train_set")
            train_normal = os.listdir(data_dir)

            t0 = time.time()
            self.slices += parallel_load(data_dir, train_normal, img_size)
            self.labels += [0] * len(train_normal)
            self.img_ids += [img_name.split('.')[0] for img_name in train_normal]
            print("Loaded {} normal images, {:.3f}s".format(len(train_normal), time.time() - t0))

        else:  # test
            test_normal_dir = os.path.join(self.root, "test_set", "normal_test")
            test_abnormal_dir = os.path.join(self.root, "test_set", "abnormal_test")

            test_normal = os.listdir(test_normal_dir)
            test_abnormal = os.listdir(test_abnormal_dir)

            test_l = test_normal + test_abnormal
            t0 = time.time()
            self.slices += parallel_load(test_normal_dir, test_normal, img_size)
            self.slices += parallel_load(test_abnormal_dir, test_abnormal, img_size)

            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_ids += [img_name.split('.')[0] for img_name in test_l]
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        img = self.transform(img)

        label = self.labels[index]
        img_id = self.img_ids[index]

        return {'img': img, 'label': label, 'name': img_id}

    def __len__(self):
        return len(self.slices)
