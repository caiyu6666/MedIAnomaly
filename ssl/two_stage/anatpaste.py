import random
import math
from torchvision import transforms
import torch
import numpy as np
import os

from skimage.io import imread
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter

from skimage.filters import threshold_otsu
from skimage.exposure import equalize_adapthist
from skimage.color import label2rgb
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import square, closing, opening, dilation, erosion


class CutPaste(object):
    """Base class for both cutpaste variants with common operations"""

    def __init__(self, colorJitter=0.1, transform=None):
        self.transform = transform

        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(brightness=colorJitter,
                                                      contrast=colorJitter,
                                                      saturation=colorJitter,
                                                      hue=colorJitter)

    def __call__(self, org_img, img):
        # apply transforms to both images
        if self.transform:
            img = self.transform(img)
            org_img = self.transform(org_img)
        return org_img, img


def LungSegment(img, area_threshold=400):
    # Even out the contrast with CLAHE
    img = equalize_adapthist(img, kernel_size=None, clip_limit=0.01, nbins=256)

    # Make a binary threshold mask and apply it to the image 
    thresh = threshold_otsu(image=img, nbins=256)

    thresh = img > thresh
    bw = opening(img > thresh, square(3)).astype(np.uint8)

    # clean up the borders
    cleared = clear_border(bw)
    cleared = dilation(cleared, square(3)).astype(np.uint8)
    label_image = label(cleared)
    # delete areas smaller than a threshold.
    for region in regionprops(label_image):
        if region.area < area_threshold:
            minr, minc, maxr, maxc = region.bbox
            label_image[minr:maxr, minc:maxc] = 0
    return label_image > 0


class AnatPaste(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """

    def __init__(self, area_ratio=[0.02, 0.15], aspect_ratio=0.3, blur_ratio=[0, 15], mask_ratio=[150, 255], margin=10,
                 **kwags):
        super(AnatPaste, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self.blur_ratio = blur_ratio
        self.mask_ratio = mask_ratio
        self.margin = margin

    def __call__(self, img):
        # TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        h = img.size[0]
        w = img.size[1]

        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h

        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1 / self.aspect_ratio)))
        aspect = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))

        # one might also want to sample from other images. currently we only sample from the image itself
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)

        if self.colorJitter:
            patch = self.colorJitter(patch)

        lung_area = LungSegment(np.array(img.convert('L')))
        if lung_area.sum() > 0:
            index = np.random.choice(np.where(lung_area.reshape(-1, ))[0])
            coord = [index % lung_area.shape[1], index // lung_area.shape[1]]
        else:
            # if there is no lung area, select pasting area randomly
            coord = [np.random.randint(0, 255), np.random.randint(0, 255)]
            lung_area = np.array([1])

        blur_value = np.random.randint(self.blur_ratio[0], self.blur_ratio[1])
        mask_value = np.random.randint(self.mask_ratio[0], self.mask_ratio[1])

        mask = Image.new("L", (h, w))
        draw = ImageDraw.Draw(mask)

        # decide whether ellipse or rectangle is pasted to the area
        # set the margin so that the shape does not exceed the area when blurred.
        if np.random.random() < 0.5:
            draw.ellipse((coord[0] - cut_w // 2 + self.margin, coord[1] - cut_h // 2 + self.margin,
                          coord[0] + cut_w // 2 - self.margin, coord[1] + cut_h // 2 - self.margin), fill=mask_value)
        else:
            draw.rectangle((coord[0] - cut_w // 2 + self.margin, coord[1] - cut_h // 2 + self.margin,
                            coord[0] + cut_w // 2 - self.margin, coord[1] + cut_h // 2 - self.margin), fill=mask_value)

        # intersection of lung segment and　mask
        mask_blur = mask.filter(ImageFilter.GaussianBlur(blur_value))
        intersection = mask_blur * lung_area
        intersection = Image.fromarray(intersection.astype(np.uint8))

        patch_im = Image.new("L", (h, w))
        patch_im.paste(patch, (coord[0] - cut_w // 2, coord[1] - cut_h // 2))

        augmented = img.copy()
        augmented.paste(patch_im, (0, 0), intersection)

        return super().__call__(img, augmented)


def cut_paste_collate_fn(batch):
    # cutPaste return 2 tuples of tuples we convert them into a list of tuples
    img_types = list(zip(*batch))
    #     print(list(zip(*batch)))
    return [torch.stack(imgs) for imgs in img_types]


class CutPaste(object):
    """Base class for both cutpaste variants with common operations"""

    def __init__(self, colorJitter=0.1, transform=None):
        self.transform = transform

        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(brightness=colorJitter,
                                                      contrast=colorJitter,
                                                      saturation=colorJitter,
                                                      hue=colorJitter)

    def __call__(self, org_img, img):
        # apply transforms to both images
        if self.transform:
            img = self.transform(img)
            org_img = self.transform(org_img)
        return org_img, img


class CutPasteNormal(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """

    def __init__(self, area_ratio=[0.02, 0.15], aspect_ratio=0.3, **kwags):
        super(CutPasteNormal, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img):
        # TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        h = img.size[0]
        w = img.size[1]

        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h

        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1 / self.aspect_ratio)))
        aspect = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))

        # one might also want to sample from other images. currently we only sample from the image itself
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)

        if self.colorJitter:
            patch = self.colorJitter(patch)

        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))

        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
        augmented = img.copy()
        augmented.paste(patch, insert_box)

        return super().__call__(img, augmented)


class CutPasteScar(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        width (list): width to sample from. List of [min, max]
        height (list): height to sample from. List of [min, max]
        rotation (list): rotation to sample from. List of [min, max]
    """

    def __init__(self, width=[2, 16], height=[10, 25], rotation=[-45, 45], **kwags):
        super(CutPasteScar, self).__init__(**kwags)
        self.width = width
        self.height = height
        self.rotation = rotation

    def __call__(self, img):
        h = img.size[0]
        w = img.size[1]

        # cut region
        cut_w = random.uniform(*self.width)
        cut_h = random.uniform(*self.height)

        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)

        if self.colorJitter:
            patch = self.colorJitter(patch)

        # rotate
        rot_deg = random.uniform(*self.rotation)
        patch = patch.convert("RGBA").rotate(rot_deg, expand=True)

        # paste
        to_location_h = int(random.uniform(0, h - patch.size[0]))
        to_location_w = int(random.uniform(0, w - patch.size[1]))

        mask = patch.split()[-1]
        patch = patch.convert("RGB")

        augmented = img.copy()
        augmented.paste(patch, (to_location_w, to_location_h), mask=mask)

        return super().__call__(img, augmented)


class CutPasteUnion(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar = CutPasteScar(**kwags)

    def __call__(self, img):
        r = random.uniform(0, 1)
        if r < 0.5:
            return self.normal(img)
        else:
            return self.scar(img)


class CutPaste3Way(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar = CutPasteScar(**kwags)

    def __call__(self, img):
        org, cutpaste_normal = self.normal(img)
        _, cutpaste_scar = self.scar(img)

        return org, cutpaste_normal, cutpaste_scar
