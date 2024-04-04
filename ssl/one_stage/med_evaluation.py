import os
import numpy as np
from torchvision import transforms as T
import cv2
import pandas as pd
import torch
from self_sup_data.chest_xray import SelfSupChestXRay
from model.resnet import resnet18_enc_dec
from train_med import SETTINGS
from experiments.chest_xray_tasks import test_real_anomalies
import matplotlib.pyplot as plt
import json
import warnings
import argparse
from thop import profile
import glob
from torchvision import transforms
import copy


# warnings.filterwarnings('ignore')


def test(test_dat, setting, model_dir, device, preact=False, pool=True, final=True, show=False, plots=False,
         pix_metrics=False):
    if final:
        fname = os.path.join(model_dir, setting.get('out_dir'), 'final_' + setting.get('fname'))
        # fname = os.path.join(model_dir, setting.get('out_dir'), setting.get('fname')[:-3]+"_249.pt")
    else:
        # fname = os.path.join(model_dir, setting.get('out_dir'), setting.get('fname'))
        fname = os.path.join(model_dir, setting.get('out_dir'), setting.get('fname')[:-3] + "_249.pt")
    print(fname)
    if not os.path.exists(fname):
        return np.nan, np.nan

    model = resnet18_enc_dec(num_classes=1, preact=preact, pool=pool, in_channels=1,
                             final_activation=setting.get('final_activation')).to(device)
    if final:
        model.load_state_dict(torch.load(fname, map_location=device))
    else:
        model.load_state_dict(torch.load(fname, map_location=device).get('model_state_dict'))
    print("Load successfully!")

    if plots:
        print("begin test")
        results, fig, preds, image_names = test_real_anomalies(model, test_dat, device=device, batch_size=16, show=show,
                                                               full_size=True, pix_metrics=pix_metrics)
        fig.savefig(os.path.join(out_dir, setting.get('fname')[:-3] + '.png'))
        plt.close(fig)
    else:
        results, _, preds, image_names = test_real_anomalies(model, test_dat, device=device, batch_size=16, show=show,
                                                             plots=plots, full_size=True, pix_metrics=pix_metrics)

    vis_dir = os.path.join(model_dir, setting.get('out_dir'), "vis")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    preds = torch.tensor(preds)
    print(preds.shape)
    # Vis
    for i in range(preds.shape[0]):
        name = image_names[i]
        # print(name)
        pred = preds[i]

        pred = transforms.ToPILImage()(pred)
        image_path = os.path.join(vis_dir, name + ".png")
        pred.save(image_path)

    example_in = torch.zeros((1, 1, 224, 224)).to(device)
    flops, params = profile(copy.deepcopy(model), inputs=(example_in,))
    flops, params = round(flops * 1e-6, 4), round(params * 1e-6, 4)  # 1e6 = M
    flops, params = str(flops) + "M", str(params) + "M"

    return results, flops, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, type=str)
    parser.add_argument("-g", "--gpu", required=False, type=int, default=7)
    parser.add_argument("-s", "--setting", required=True, type=str)
    parser.add_argument("-f", '--fold', type=int, default=0, help='0-4, experiment index')

    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    setting = SETTINGS.get(args.setting)
    data = args.data
    final = True

    data_root = os.path.join(os.path.expanduser("~"), "MedIAnomaly-Data")
    out_dir = os.path.join("output", args.data, "fold_{:d}".format(args.fold))
    model_dir = out_dir

    if data in ['rsna', 'vin', 'brain', 'lag']:
        path = None
        if data == 'rsna':
            path = os.path.join(data_root, 'RSNA')
            # 250
        elif data == 'vin':
            path = os.path.join(data_root, "VinCXR")
            # 400
        elif data == 'brain':
            path = os.path.join(data_root, "BrainTumor")
            # 400
        elif data == 'lag':
            path = os.path.join(data_root, "LAG")
            # 250
        with open(os.path.join(path, "data.json")) as f:
            data_dict = json.load(f)
        normal_test_files = data_dict["test"]["0"]
        anom_test_files = data_dict["test"]["1"]
        normal_test_files = [os.path.join(path, "images", e) for e in normal_test_files]
        anom_test_files = [os.path.join(path, "images", e) for e in anom_test_files]
        mask_files = None
    elif data == 'isic':
        path = os.path.join(data_root, "ISIC2018_Task3")
        data_csv = pd.read_csv(os.path.join(path, "ISIC2018_Task3_Test_GroundTruth",
                                            "ISIC2018_Task3_Test_GroundTruth.csv"))
        test_normal = list(data_csv[data_csv['NV'] == 1]['image'])
        test_normal = [e + ".jpg" for e in test_normal]
        test_abnormal = list(data_csv[data_csv['NV'] == 0]['image'])
        test_abnormal = [e + ".jpg" for e in test_abnormal]
        normal_test_files = [os.path.join(path, "ISIC2018_Task3_Test_Input", e) for e in test_normal]
        anom_test_files = [os.path.join(path, "ISIC2018_Task3_Test_Input", e) for e in test_abnormal]
        mask_files = None
    elif data == 'c16':
        path = os.path.join(data_root, "Camelyon16")
        normal_test_files = glob.glob(os.path.join(path, "test", "good", "*.png"))
        anom_test_files = glob.glob(os.path.join(path, "test", "Ungood", "*.png"))
        mask_files = None
    elif data == 'brats':
        path = os.path.join(data_root, "BraTS2021")
        normal_test_files = glob.glob(os.path.join(path, "test", "normal", "*.png"))
        anom_test_files = glob.glob(os.path.join(path, "test", "tumor", "*.png"))
        mask_files = [e.replace("tumor", "annotation").replace("flair", "seg") for e in anom_test_files]
    else:
        raise Exception("Invalid dataset: {}".format(data))

    pix_metrics = True if data == 'brats' else False

    test_dat = SelfSupChestXRay(normal_files=normal_test_files, anom_files=anom_test_files,
                                mask_files=mask_files, is_train=False, res=256, transform=T.CenterCrop(224))

    results, flops, params = test(test_dat, setting, model_dir, device, preact=False, pool=True, final=final,
                                  pix_metrics=pix_metrics)

    auc, ap = results['sample_auc'], results['sample_ap']
    pix_ap, best_dice = results.setdefault('pixel_ap', None), results.setdefault('best_dice', None)

    print(args.setting, data, "\tAUC", auc, "\tAP", ap, "\tPixAP", pix_ap, "\tBestDice", best_dice,
          "\tFLOPs", flops, "\tParams", params)

    performance = pd.DataFrame({"Dataset": [data], "FLOPs": [flops], "params": [params],
                                "AUC": [auc], "AP": [ap], "PixAP": [pix_ap], "BestDice": [best_dice]})
    performance.to_csv(os.path.join(out_dir, "{}.csv".format(args.setting)), index=False)
