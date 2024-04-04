from sklearn.metrics import roc_curve, auc, average_precision_score, roc_auc_score
from sklearn.manifold import TSNE
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
# from cutpaste import CutPaste
from model import ProjectionNet
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from anatpaste import CutPaste, cut_paste_collate_fn
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import numpy as np
from collections import defaultdict
from density import GaussianDensitySklearn, GaussianDensityTorch
import pandas as pd
from utils import str2bool
import os
from dataset import zhanglab_dataset, Repeat, chexpert_dataset, rsna_dataset, MedAD, BraTSAD, Camelyon16AD, ISIC2018
from thop import profile
import copy

# test_data_eval = None
# test_transform = None
# cached_type = None


def get_train_embeds(model, dataloader_normal, device):
    train_embed = []
    with torch.no_grad():
        for x in dataloader_normal:
            embed, logit = model(x.to(device))

            train_embed.append(embed.cpu())
    train_embed = torch.cat(train_embed)
    return train_embed


def eval_model(modelname, device="cpu", save_plots=False, size=256, show_training_data=True, model=None,
               train_embed=None, head_layer=1, density=GaussianDensityTorch(), mode='valid', dataloader_test=None,
               dataloader_normal=None):

    if model is None:
        print(f"loading model {modelname}")
        head_layers = [512] * head_layer + [128]
        print(head_layers)
        weights = torch.load(modelname)
        classes = weights["out.weight"].shape[0]
        model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=classes)
        model.load_state_dict(weights)

        # model = ProjectionNet(pretrained=True, head_layers=head_layers, num_classes=2)

        model.to(device)
        model.eval()

    # get embeddings for test data
    labels = []
    embeds = []
    probs = []
    with torch.no_grad():
        for x, label in dataloader_test:
            embed, logit = model(x.to(device))

            normal_prob = torch.softmax(logit, dim=1)[:, 0]
            prob = 1. - normal_prob
            # save
            embeds.append(embed.cpu())
            labels.append(label)
            probs.append(prob.cpu())

    labels = torch.cat(labels)
    embeds = torch.cat(embeds)
    probs = torch.cat(probs)

    if train_embed is None:
        train_embed = get_train_embeds(model, dataloader_normal, device)

    # norm embeds
    embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
    train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)

    print(f"using density estimation {density.__class__.__name__}")
    density.fit(train_embed)
    distances = density.predict(embeds)
    # TODO: set threshold on mahalanobis distances and use "real" probabilities

    # roc_auc = plot_roc(labels, distances, eval_dir / "roc_plot.png", modelname=modelname, save_plots=save_plots)
    roc_auc = roc_auc_score(labels, distances)
    ap = average_precision_score(labels, distances)

    cls_auc, cls_ap = roc_auc_score(labels, probs), average_precision_score(labels, probs)
    return roc_auc, ap, cls_auc, cls_ap


def plot_roc(labels, scores, filename, modelname="", save_plots=False):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # plot roc
    if save_plots:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic {modelname}')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(filename)
        plt.close()

    return roc_auc


def plot_tsne(labels, embeds, filename):
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=500)
    embeds, labels = shuffle(embeds, labels)
    tsne_results = tsne.fit_transform(embeds)
    fig, ax = plt.subplots(1)
    colormap = ["b", "r", "c", "y"]

    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], color=[colormap[l] for l in labels])
    fig.savefig(filename)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval models')
    parser.add_argument('--type', default="rsna",
                        help='which dataset to use. '
                             'In this repo, you can choose rsna, vin, brain, lag, isic, c16, brats')

    parser.add_argument('--model_dir', default="models",
                        help=' directory contating models to evaluate (default: models)')

    parser.add_argument('--cuda', default=7, type=str,
                        help='num of cuda to use')

    parser.add_argument('--variant', default="3way", choices=['normal', 'scar', '3way', 'union', 'anatpaste'],
                        help='cutpaste variant to use (dafault: "3way")')

    parser.add_argument('--head_layer', default=8, type=int,
                        help='number of layers in the projection head (default: 8)')

    parser.add_argument('--density', default="torch", choices=["torch", "sklearn"],
                        help='density implementation to use. See `density.py` for both implementations. (default: torch)')

    parser.add_argument('--save_plots', default=True, type=str2bool,
                        help='save TSNE and roc plots')

    args = parser.parse_args()

    args = parser.parse_args()
    print(args)
    # all_types = ['zhanglab', 'chexpert', 'rsna']
    # all_types = ['rsna', 'vin', 'brain', 'lag', 'isic', 'c16', 'brats']

    variant_map = {'normal': 'CutPasteNormal', 'scar': 'CutPasteScar', '3way': 'CutPaste3Way', 'union': 'CutPasteUnion',
                   'anatpaste': 'AnatPaste'}
    variant = variant_map[args.variant]

    # if args.type == "all":
    #     types = all_types
    # else:
    #     types = args.type.split(",")
    data_type = args.type

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = "cuda"

    density_mapping = {
        "torch": GaussianDensityTorch,
        "sklearn": GaussianDensitySklearn
    }
    density = density_mapping[args.density]

    # find models
    # model_names = [list(Path(args.model_dir).glob(f"model-{data_type}*"))[0] for data_type in types if
    #                len(list(Path(args.model_dir).glob(f"model-{data_type}*"))) > 0]
    # if len(model_names) < len(all_types):
    #     print("warning: not all types present in folder")
    out_dir = os.path.join(args.model_dir, args.type, variant)

    size = 224
    # Test transform
    test_transform = transforms.Compose([])
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    # test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    obj = defaultdict(list)
    # for model_name, data_type in zip(model_names, types):
    # for data_type in types:
    # model_name = "None"
    model_name = os.path.join(out_dir, "model.tch")
    print(f"evaluating {data_type}")

    data_path = os.path.join(os.path.expanduser("~"), "MedIAnomaly-Data")
    if data_type in ['rsna', 'vin', 'brain', 'lag']:
        if data_type == 'rsna':
            data_path = os.path.join(data_path, "RSNA")
        elif data_type == 'vin':
            data_path = os.path.join(data_path, "VinCXR")
        elif data_type == 'brain':
            data_path = os.path.join(data_path, "BrainTumor")
        elif data_type == 'lag':
            data_path = os.path.join(data_path, "LAG")
        else:
            raise Exception("Invalid dataset: {}".format(data_type))

        test_data_eval = MedAD(data_path, transform=test_transform, size=size, mode="test")
        normal_data = MedAD(data_path, transform=test_transform, size=size, mode="train")

    elif data_type == 'brats':
        data_path = os.path.join(data_path, "BraTS2021")
        test_data_eval = BraTSAD(data_path, transform=test_transform, size=size, mode="test")
        normal_data = BraTSAD(data_path, transform=test_transform, size=size, mode="train")

    elif data_type == 'c16':
        data_path = os.path.join(data_path, "Camelyon16")
        test_data_eval = Camelyon16AD(data_path, transform=test_transform, size=size, mode="test")
        normal_data = Camelyon16AD(data_path, transform=test_transform, size=size, mode="train")

    elif data_type == 'isic':
        data_path = os.path.join(data_path, "ISIC2018_Task3")
        test_data_eval = ISIC2018(data_path, transform=test_transform, size=size, mode="test")
        normal_data = ISIC2018(data_path, transform=test_transform, size=size, mode="train")

    else:
        raise Exception("Invalid dataset.")
    dataloader_test = DataLoader(test_data_eval, batch_size=1, shuffle=False, num_workers=0)
    dataloader_normal = DataLoader(normal_data, batch_size=64, shuffle=False, num_workers=0)
    # roc_auc = eval_model(model_name, data_type, save_plots=args.save_plots, device=device,
    #                      head_layer=args.head_layer, density=density())
    test_roc_auc, test_ap, cls_auc, cls_ap = eval_model(model_name, device=device,
                                                        save_plots=False,
                                                        size=size,
                                                        show_training_data=False,
                                                        # model=model,
                                                        mode='valid',
                                                        dataloader_test=dataloader_test,
                                                        dataloader_normal=dataloader_normal)

    results = {"AUC": test_roc_auc, "AP": test_ap, "cls_auc": cls_auc, "cls_ap": cls_ap}
    result_path = os.path.join(out_dir, "results.txt")
    # with open(result_path, "w") as f:
    for key, value in results.items():
        # f.write(str(key) + ": " + str(value) + "\n")
        print(key + ": {}".format(value))
