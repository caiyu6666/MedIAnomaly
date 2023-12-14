import os.path

from torchvision import transforms


def get_transform(opt, phase='train'):
    normalize = transforms.Normalize((0.5,), (0.5,)) if opt.model['in_c'] == 1 else \
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    transform = transforms.Compose([transforms.ToTensor(),
                                    normalize])
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
    elif dataset == 'c16':
        return os.path.join(data_root, "Camelyon16_AD")
    elif dataset == 'oct':
        return os.path.join(os.path.expanduser("~"), "datasets", "OCT2017")
    elif dataset == 'colon':
        return os.path.join(os.path.expanduser("~"), "datasets", "Colon_AD_public")
    elif dataset == 'isic':
        return os.path.join(data_root, "ISIC2018_Task3")
    elif dataset == 'cpchild':
        return os.path.join(data_root, "CP-CHILD/CP-CHILD-A")
    else:
        raise Exception("Invalid dataset: {}".format(dataset))
