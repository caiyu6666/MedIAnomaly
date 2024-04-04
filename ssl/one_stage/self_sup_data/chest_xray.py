from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from .self_sup_tasks import patch_ex


class SelfSupChestXRay(Dataset):
    def __init__(self, normal_files, anom_files=None, mask_files=None,
                 is_train=True, res=256, transform=None, self_sup_args={}):
        self.is_train = is_train
        self.normal_files = normal_files
        self.anom_files = anom_files
        self.mask_files = mask_files
        self.res = res

        # set transforms
        self.transform = transform
        self.final_transform = T.ToTensor()

        # load dataset
        self.x, self.y, self.mask, self.image_names = self.load_dataset_folder(res)
        self.image_names = [e.split('/')[-1].split('.')[0] for e in self.image_names]

        self.self_sup = is_train
        self.self_sup_args = self_sup_args

        self.prev_idx = np.random.randint(len(self.x))
        # if self.is_train:
        #     if self.unlabeled_files is not None:
        #         self.f = self.load_unlabeled_dataset(res)
        #     else:
        #         self.f = []
        #     self.f = self.x.copy() + self.f.copy()  # anomaly source for synthesis
        #
        #     self.foreign_indicators = np.ones(len(self.f))

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        image_name = self.image_names[idx]

        if self.transform is not None:
            x = self.transform(x)
        x = np.asarray(x)[..., None]

        if self.self_sup:
            p = self.x[self.prev_idx]
            # foreign_idx = self.sample_foreign(idx)
            # p = self.f[foreign_idx]
            if self.transform is not None:
                p = self.transform(p)
            p = np.asarray(p)[..., None]  
            x, mask = patch_ex(x, p, **self.self_sup_args)
            mask = torch.tensor(mask[None, ..., 0]).float()
            self.prev_idx = idx
        else:
            if mask is not None:
                # if self.transform is not None:
                #     mask = self.transform(mask)  # tranform should be deterministic when testing
                mask = self.final_transform(mask)
            else:
                mask = y * torch.ones((1, self.res, self.res))

        x = self.final_transform(x)
        return x, y, mask, image_name

    def __len__(self):
        return len(self.x)

    # def sample_foreign(self, idx):
    #     if np.sum(self.foreign_indicators) == 0:
    #         self.foreign_indicators = np.ones(len(self.f))
    #
    #     choices = np.where(self.foreign_indicators == 1)[0]
    #     sample_idx = np.random.choice(choices)
    #     while sample_idx == idx:
    #         sample_idx = np.random.choice(choices)
    #
    #     self.foreign_indicators[sample_idx] = 0
    #     return sample_idx

    def configure_self_sup(self, on=True, self_sup_args={}):
        self.self_sup = on 
        self.self_sup_args.update(self_sup_args)

    def load_dataset_folder(self, res):
        transform = T.Resize(res, Image.ANTIALIAS)
        xs = []
        y = []
        image_names = []
        for f in tqdm(self.normal_files, desc='read normal images'):
            # xs.append(transform(Image.open(os.path.join(self.data_dir, f)).convert('L')))
            xs.append(transform(Image.open(f).convert('L')))
            y.append(0)

        image_names += self.normal_files

        mask = [None for _ in range(len(xs))]
        if self.anom_files is not None:
            for f in tqdm(self.anom_files, desc='read anomaly images'):
                # xs.append(transform(Image.open(os.path.join(self.data_dir, f)).convert('L')))
                xs.append(transform(Image.open(f).convert('L')))
                y.append(1)
                # if self.mask_dir is not None and f[:12] in self.mask_ids:
                #     mask.append(transform(Image.open(os.path.join(self.mask_dir, f[:12] + '_bbox.png')).convert('L')))
                # else:
                #     mask.append(None)
            # mask += [None for _ in range(len(self.anom_files))]
            image_names += self.anom_files

            if self.mask_files is not None:
                mask_transform = T.Resize(res, Image.NEAREST)
                for f in tqdm(self.mask_files, desc='read anomaly masks'):
                    mask.append(mask_transform(Image.open(f)))
            else:
                mask += [None for _ in range(len(self.anom_files))]

        return list(xs), list(y), mask, image_names

    # def load_unlabeled_dataset(self, res):
    #     transform = T.Resize(res, Image.ANTIALIAS)
    #     xs = []
    #     for f in tqdm(self.unlabeled_files, desc='read unlabeled images'):
    #         xs.append(transform(Image.open(os.path.join(self.data_dir, f)).convert('L')))
    #     return xs
