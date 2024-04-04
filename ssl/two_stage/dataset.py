import json
import os
import time
from pathlib import Path

from PIL import Image
from joblib import Parallel, delayed
from torch.utils.data import Dataset
import glob
import SimpleITK as sitk
from scipy import ndimage
import numpy as np
from tqdm import tqdm
import pandas as pd


class Repeat(Dataset):
    def __init__(self, org_dataset, new_length):
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        return self.org_dataset[idx % self.org_length]


def parallel_load(img_dir, img_list, img_size, verbose=0):
    return Parallel(n_jobs=-1, verbose=verbose)(delayed(
        lambda file: Image.open(os.path.join(img_dir, file)).convert("RGB").resize(
            (img_size, img_size), resample=Image.BILINEAR))(file) for file in img_list)


class MedAD(Dataset):
    def __init__(self, root_dir, size=64, transform=None, mode="train"):
        super(MedAD, self).__init__()
        assert mode in ["train", "test"]
        self.root = root_dir
        self.mode = mode
        self.labels = []
        self.slices = []
        self.transform = transform if transform is not None else lambda x: x

        with open(os.path.join(root_dir, "data.json")) as f:
            data_dict = json.load(f)

        print("Loading images")
        if mode == "train":
            train_normal = data_dict["train"]["0"]

            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "images"), train_normal, size)
            self.labels += len(train_normal) * [0]
            print("Loaded {} normal images, {:.3f}s".format(len(train_normal), time.time() - t0))

        else:  # test
            test_normal = data_dict["test"]["0"]
            test_abnormal = data_dict["test"]["1"]

            test_l = test_normal + test_abnormal
            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "images"), test_l, size)
            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        label = self.labels[index]
        img = self.transform(img)
        if self.mode == "train":
            return img
        else:
            return img, label

    def __len__(self):
        return len(self.slices)


class BraTSAD(Dataset):
    def __init__(self, main_path, size=64, transform=None, mode="train"):
        super(BraTSAD, self).__init__()
        assert mode in ["train", "test"]

        self.mode = mode
        self.root = main_path
        self.res = size
        self.labels = []
        self.masks = []
        self.img_ids = []
        self.slices = []
        self.transform = transform

        print("Loading images")
        if mode == "train":
            data_dir = os.path.join(self.root, "train")
            train_normal = os.listdir(data_dir)

            t0 = time.time()
            self.slices += parallel_load(data_dir, train_normal, size)
            self.labels += [0] * len(train_normal)
            self.img_ids += [img_name.split('.')[0] for img_name in train_normal]
            print("Loaded {} normal images, {:.3f}s".format(len(train_normal), time.time() - t0))

        else:  # test
            test_normal_dir = os.path.join(self.root, "test", "normal")
            test_abnormal_dir = os.path.join(self.root, "test", "tumor")

            test_normal = os.listdir(test_normal_dir)
            test_abnormal = os.listdir(test_abnormal_dir)

            test_l = test_normal + test_abnormal
            t0 = time.time()
            self.slices += parallel_load(test_normal_dir, test_normal, size)
            self.slices += parallel_load(test_abnormal_dir, test_abnormal, size)

            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_ids += [img_name.split('.')[0] for img_name in test_l]
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        img = self.transform(img)

        label = self.labels[index]

        if self.mode == "train":
            return img
        else:
            return img, label

    def __len__(self):
        return len(self.slices)


class Camelyon16AD(Dataset):
    def __init__(self, main_path, size=64, transform=None, mode="train"):
        super(Camelyon16AD, self).__init__()
        assert mode in ["train", "test"]

        self.root = main_path
        self.res = size
        self.labels = []
        self.img_ids = []
        self.slices = []
        self.transform = transform
        self.mode = mode

        print("Loading images")
        if mode == "train":
            data_dir = os.path.join(self.root, "train", "good")
            train_normal = os.listdir(data_dir)

            t0 = time.time()
            self.slices += parallel_load(data_dir, train_normal, size)
            self.labels += [0] * len(train_normal)
            self.img_ids += [img_name.split('.')[0] for img_name in train_normal]
            print("Loaded {} normal images, {:.3f}s".format(len(train_normal), time.time() - t0))

        else:  # test
            test_normal_dir = os.path.join(self.root, "test", "good")
            test_abnormal_dir = os.path.join(self.root, "test", "Ungood")

            test_normal = os.listdir(test_normal_dir)
            test_abnormal = os.listdir(test_abnormal_dir)

            test_l = test_normal + test_abnormal
            t0 = time.time()
            self.slices += parallel_load(test_normal_dir, test_normal, size)
            self.slices += parallel_load(test_abnormal_dir, test_abnormal, size)

            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_ids += [img_name.split('.')[0] for img_name in test_l]
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        img = self.transform(img)

        label = self.labels[index]
        # img_id = self.img_ids[index]

        if self.mode == 'train':
            return img
        else:
            return img, label

    def __len__(self):
        return len(self.slices)


class ISIC2018(Dataset):
    def __init__(self, main_path, size=64, transform=None, mode="train"):
        super(ISIC2018, self).__init__()
        self.root = main_path
        self.res = size
        self.labels = []
        self.img_ids = []
        self.slices = []
        self.transform = transform
        self.mode = mode

        print("Loading images")
        if mode == 'train':
            data_dir = os.path.join(self.root, "ISIC2018_Task3_Training_Input")
            data_csv = pd.read_csv(os.path.join(self.root, "ISIC2018_Task3_Training_GroundTruth",
                                                           "ISIC2018_Task3_Training_GroundTruth.csv"))
            train_normal = list(data_csv[data_csv['NV'] == 1]['image'])
            train_normal = [e+".jpg" for e in train_normal]
            t0 = time.time()
            self.slices += parallel_load(data_dir, train_normal, size)
            self.labels += [0] * len(train_normal)
            self.img_ids += train_normal
            print("Loaded {} normal images, {:.3f}s".format(len(train_normal), time.time() - t0))
        else:  # test
            data_dir = os.path.join(self.root, "ISIC2018_Task3_Test_Input")
            data_csv = pd.read_csv(os.path.join(self.root, "ISIC2018_Task3_Test_GroundTruth",
                                                           "ISIC2018_Task3_Test_GroundTruth.csv"))
            test_normal = list(data_csv[data_csv['NV'] == 1]['image'])
            test_abnormal = list(data_csv[data_csv['NV'] == 0]['image'])
            test_normal = [e + ".jpg" for e in test_normal]
            test_abnormal = [e + ".jpg" for e in test_abnormal]

            t0 = time.time()
            self.slices += parallel_load(data_dir, test_normal, size)
            self.slices += parallel_load(data_dir, test_abnormal, size)
            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_ids += test_normal + test_abnormal
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        img = self.transform(img)

        label = self.labels[index]
        # img_id = self.img_ids[index]

        if self.mode == 'train':
            return img
        else:
            return img, label

    def __len__(self):
        return len(self.slices)


class zhanglab_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, defect_name, size, transform=None, mode="train"):
        """
        Args:
            root_dir (string): Directory with the MVTec AD dataset.
            defect_name (string): defect to load.
            transform: Transform to apply to data
            mode: "train" loads training samples "test" test samples default "train"
        """
        self.root_dir = Path(root_dir)
        self.defect_name = defect_name
        self.transform = transform
        self.mode = mode
        self.size = size

        # find test images
        if self.mode == "train":
            self.image_names = sorted(list((self.root_dir / "train" / "good").glob("*.jpeg")))
            print("loading images", mode, len(self.image_names))
            # during training we cache the smaller images for performance reasons (not a good coding style)
            # self.imgs = [Image.open(file).resize((size,size)).convert("RGB") for file in self.image_names]
            self.imgs = Parallel(n_jobs=10)(
                delayed(lambda file: Image.open(file).resize((size, size)).convert("RGB"))(file) for file in
                self.image_names)
            print(f"loaded {len(self.imgs)} images")
        elif self.mode == "valid":
            self.image_names = sorted(list((self.root_dir / "valid").glob(str(Path("*") / "*.jpeg"))))
            print('dataset is valid size is ', len(self.image_names))
        else:
            # test mode
            self.image_names = sorted(list((self.root_dir / "test").glob(str(Path("*") / "*.jpeg"))))
            print('dataset is test size is ', len(self.image_names))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            # img = Image.open(self.image_names[idx])
            # img = img.convert("RGB")
            img = self.imgs[idx].copy()
            if self.transform is not None:
                img = self.transform(img)
            return img
        else:
            filename = self.image_names[idx]
            label = filename.parts[-2]
            img = Image.open(filename)
            img = img.resize((self.size, self.size)).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, label != "good"


class chexpert_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, defect_name, size, transform=None, mode="train"):
        """
        Args:
            root_dir (string): Directory with the MVTec AD dataset.
            defect_name (string): defect to load.
            transform: Transform to apply to data
            mode: "train" loads training samples "test" test samples default "train"
        """
        self.root_dir = Path(root_dir)
        self.defect_name = defect_name
        self.transform = transform
        self.mode = mode
        self.size = size

        # find test images
        if self.mode == "train":
            print(self.root_dir)
            self.image_names = sorted(list((self.root_dir / "train" / "No Finding").glob("*.jpg")))
            print("loading images")
            # during training we cache the smaller images for performance reasons (not a good coding style)
            # self.imgs = [Image.open(file).resize((size,size)).convert("RGB") for file in self.image_names]
            self.imgs = Parallel(n_jobs=10)(
                delayed(lambda file: Image.open(file).resize((size, size)).convert("RGB"))(file) for file in
                self.image_names)
            # print(f"loaded {len(self.imgs)} images")

        elif self.mode == "valid":
            # test mode
            self.image_names = sorted(list((self.root_dir / "valid").glob(str(Path("*") / "*.jpg"))))
            print('dataset is valid size is ', len(self.image_names))
        else:
            # test mode
            if self.defect_name != 'chexpert':
                self.image_names = sorted(list((self.root_dir / "test").glob(str(Path("No Finding") / "*.jpg"))) + list(
                    (self.root_dir).glob(str(Path(defect_name) / "*.jpg"))))
            else:
                self.image_names = sorted(list((self.root_dir / "test").glob(str(Path("*") / "*.jpg"))))
            print('dataset is test size is ', len(self.image_names))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            # img = Image.open(self.image_names[idx])
            # img = img.convert("RGB")
            img = self.imgs[idx].copy()
            if self.transform is not None:
                img = self.transform(img)
            return img
        else:
            filename = self.image_names[idx]
            label = filename.parts[-2]
            img = Image.open(filename)
            img = img.resize((self.size, self.size)).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, label != "No Finding"


class rsna_dataset(Dataset):
    """RSNA pneumonia detection dataset."""

    def __init__(self, root_dir, defect_name, size, transform=None, mode="train"):
        """
        Args:
            root_dir (string): Directory with the MVTec AD dataset.
            defect_name (string): defect to load.
            transform: Transform to apply to data
            mode: "train" loads training samples "test" test samples default "train"
        """
        self.root_dir = Path(root_dir)
        self.defect_name = defect_name
        self.transform = transform
        self.mode = mode
        self.size = size

        # find test images
        if self.mode == "train":
            print(self.root_dir)
            self.image_names = sorted(list((self.root_dir / "train" / "normal").glob("*.png")))
            print("loading images")
            # during training we cache the smaller images for performance reasons (not a good coding style)
            # self.imgs = [Image.open(file).resize((size,size)).convert("RGB") for file in self.image_names]
            self.imgs = Parallel(n_jobs=10)(
                delayed(lambda file: Image.open(file).resize((size, size)).convert("RGB"))(file) for file in
                self.image_names)
            # print(f"loaded {len(self.imgs)} images")

        elif self.mode == "valid":
            # test mode
            self.image_names = sorted(list((self.root_dir / "valid").glob(str(Path("*") / "*.png"))))
            print('dataset is valid size is ', len(self.image_names))
        else:
            # test mode
            self.image_names = sorted(list((self.root_dir / "test").glob(str(Path("*") / "*.png"))))
            print('dataset is test size is ', len(self.image_names))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            # img = Image.open(self.image_names[idx])
            # img = img.convert("RGB")
            img = self.imgs[idx].copy()
            if self.transform is not None:
                img = self.transform(img)
            return img
        else:
            filename = self.image_names[idx]
            label = filename.parts[-2]
            img = Image.open(filename)
            img = img.resize((self.size, self.size)).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, label != "normal"
