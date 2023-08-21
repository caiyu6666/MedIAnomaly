import os.path

from torchvision import transforms


def get_transform(opt, phase='train'):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    return transform


def get_data_path(dataset):
    data_root = os.path.join(os.path.expanduser("~"), "Med-AD")
    if dataset == 'rsna':
        return os.path.join(data_root, "RSNA")
    elif dataset == 'vin':
        return os.path.join(data_root, "VinCXR")
    elif dataset == 'brain':
        return os.path.join(data_root, "BrainTumor")
    elif dataset == 'lag':
        return os.path.join(data_root, "LAG")
    elif dataset == 'brats':
        return os.path.join(data_root, "BraTS_AD")
    elif dataset == 'oct':
        return os.path.join(os.path.expanduser("~"), "datasets", "OCT2017")
    elif dataset == 'colon':
        return os.path.join(os.path.expanduser("~"), "datasets", "Colon_AD_public")
    else:
        raise Exception("Invalid dataset: {}".format(dataset))
