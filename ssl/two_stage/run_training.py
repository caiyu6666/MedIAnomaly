# head dims:512,512,512,512,512,512,512,512,128
# code is basicly:https://github.com/google-research/deep_representation_one_class
from pathlib import Path
from tqdm import tqdm
import datetime
import argparse
import random
import numpy as np
import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from thop import profile
import copy

from dataset import zhanglab_dataset, Repeat, chexpert_dataset, rsna_dataset, MedAD, BraTSAD, Camelyon16AD, ISIC2018
from anatpaste import CutPasteNormal, CutPasteScar, CutPaste3Way, CutPasteUnion, AnatPaste, \
    cut_paste_collate_fn
from model import ProjectionNet
from eval import eval_model
from utils import str2bool


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_training(data_type="zhanglab",
                 model_dir="models",
                 epochs=256,
                 pretrained=True,
                 test_epochs=10,
                 freeze_resnet=20,
                 learninig_rate=0.03,
                 optim_name="SGD",
                 batch_size=64,
                 head_layer=8,
                 cutpate_type=CutPasteNormal,
                 device="cuda",
                 workers=8,
                 # size=256,
                 size=224,
                 seed=0):
    torch.multiprocessing.freeze_support()
    # TODO: use script params for hyperparameter
    # Temperature Hyperparameter currently not used
    temperature = 0.2

    weight_decay = 0.00003
    momentum = 0.9
    # TODO: use f strings also for the date LOL
    # model_name = f"model-{data_type}-{cutpate_type.__name__}-seed_{seed}" + '-{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())
    model_name = f"model-{data_type}-{cutpate_type.__name__}"

    out_dir = os.path.join(model_dir, data_type, cutpate_type.__name__)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # augmentation:
    min_scale = 1
    g = torch.Generator()
    g.manual_seed(seed)

    # create Training Dataset and Dataloader
    after_cutpaste_transform = transforms.Compose([])
    after_cutpaste_transform.transforms.append(transforms.ToTensor())
    after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.5],
                                                                    std=[0.5]))

    # Train Transform
    train_transform = transforms.Compose([])
    # train_transform.transforms.append(transforms.RandomResizedCrop(size, scale=(min_scale,1)))
    train_transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    # train_transform.transforms.append(transforms.GaussianBlur(int(size/10), sigma=(0.1,2.0)))

    # ------------------ Notice: We resize the image when loading. ------------------------- #
    # train_transform.transforms.append(transforms.Resize((size, size)))
    #

    train_transform.transforms.append(cutpate_type(transform=after_cutpaste_transform))
    # train_transform.transforms.append(transforms.ToTensor())

    # Test transform
    test_transform = transforms.Compose([])
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.Normalize(mean=[0.5], std=[0.5]))

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

        train_data = MedAD(data_path, transform=train_transform, size=size, mode="train")
        test_data_eval = MedAD(data_path, transform=test_transform, size=size, mode="test")
        normal_data = MedAD(data_path, transform=test_transform, size=size, mode="train")

    elif data_type == 'brats':
        data_path = os.path.join(data_path, "BraTS2021")
        train_data = BraTSAD(data_path, transform=train_transform, size=size, mode="train")
        test_data_eval = BraTSAD(data_path, transform=test_transform, size=size, mode="test")
        normal_data = BraTSAD(data_path, transform=test_transform, size=size, mode="train")

    elif data_type == 'c16':
        data_path = os.path.join(data_path, "Camelyon16")
        train_data = Camelyon16AD(data_path, transform=train_transform, size=size, mode="train")
        test_data_eval = Camelyon16AD(data_path, transform=test_transform, size=size, mode="test")
        normal_data = Camelyon16AD(data_path, transform=test_transform, size=size, mode="train")

    elif data_type == 'isic':
        data_path = os.path.join(data_path, "ISIC2018_Task3")
        train_data = ISIC2018(data_path, transform=train_transform, size=size, mode="train")
        test_data_eval = ISIC2018(data_path, transform=test_transform, size=size, mode="test")
        normal_data = ISIC2018(data_path, transform=test_transform, size=size, mode="train")

    else:
        raise Exception("Invalid dataset.")

    dataloader = DataLoader(Repeat(train_data, 3000), batch_size=batch_size, drop_last=True,
                            shuffle=True, num_workers=workers, collate_fn=cut_paste_collate_fn,
                            pin_memory=True, generator=g, )
    dataloader_test = DataLoader(test_data_eval, batch_size=1, shuffle=False, num_workers=0)
    dataloader_normal = DataLoader(normal_data, batch_size=64, shuffle=False, num_workers=0)

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(Path("logdirs") / model_name)

    # create Model:
    head_layers = [512] * head_layer + [128]
    num_classes = 2 if cutpate_type is not CutPaste3Way else 3
    model = ProjectionNet(pretrained=pretrained, head_layers=head_layers, num_classes=num_classes)
    model.to(device)

    if freeze_resnet > 0 and pretrained:
        model.freeze_resnet()

    loss_fn = torch.nn.CrossEntropyLoss()
    if optim_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learninig_rate, momentum=momentum, weight_decay=weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, epochs)
        # scheduler = None
    elif optim_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learninig_rate, weight_decay=weight_decay)
        scheduler = None
    else:
        print(f"ERROR unkown optimizer: {optim_name}")

    def get_data_inf():
        while True:
            for out in enumerate(dataloader):
                yield out

    dataloader_inf = get_data_inf()
    # From paper: "Note that, unlike conventional definition for an epoch,
    #              we define 256 parameter update steps as one epoch.
    best_roc, best_ap = 0, 0
    for step in tqdm(range(epochs)):
        epoch = int(step / 1)
        if epoch == freeze_resnet:
            model.unfreeze()

        for i in range(256):
            batch_idx, data = next(dataloader_inf)
            xs = [x.to(device) for x in data]

            # zero the parameter gradients
            optimizer.zero_grad()
            # print(torch.tensor(xs))
            xc = torch.cat(xs, axis=0)

            embeds, logits = model(xc)

            # calculate label
            y = torch.arange(len(xs), device=device)
            y = y.repeat_interleave(xs[0].size(0))
            loss = loss_fn(logits, y)

            # regulize weights:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step(epoch)

        writer.add_scalar('loss', loss.item(), step)
        # predicted = torch.argmax(logits, axis=1)
        #
        # accuracy = torch.true_divide(torch.sum(predicted == y), predicted.size(0))
        # writer.add_scalar('acc', accuracy, step)
        if scheduler is not None:
            writer.add_scalar('lr', scheduler.get_last_lr()[0], step)

        writer.add_scalar('epoch', epoch, step)

        # run tests
        if test_epochs > 0 and epoch % test_epochs == 0:
            # run auc calculation
            # TODO: create dataset only once.
            # TODO: train predictor here or in the model class itself. Should not be in the eval part
            # TODO: we might not want to use the training datat because of droupout etc. but it should give a indecation of the model performance???
            model.eval()
            test_roc_auc, test_ap, cls_auc, cls_ap = eval_model(model_name, device=device,
                                                                save_plots=False,
                                                                size=size,
                                                                show_training_data=False,
                                                                model=model,
                                                                mode='valid',
                                                                dataloader_test=dataloader_test,
                                                                dataloader_normal=dataloader_normal)
            model.train()
            writer.add_scalar('eval_auc', test_roc_auc, step)
            writer.add_scalar('eval_ap', test_ap, step)
            print('auc is ', test_roc_auc, 'best score is ', best_roc)
            print('ap is ', test_ap, 'best score is ', best_roc)
            print('cls_auc is ', cls_auc)
            print('cls_ap is ', cls_ap)
            if test_roc_auc > best_roc or test_ap > best_ap:
                best_roc = test_roc_auc
                best_ap = test_ap
                # print('best score! save model.')
                print('best score!')
                # torch.save(model.state_dict(), model_dir / f"{model_name}.tch")

    # torch.save(model.state_dict(), os.path.join(out_dir, "{}.tch".format(model_name)))
    torch.save(model.state_dict(), os.path.join(out_dir, "model.tch"))
    # load the best model in validation dataset.
    print('predict test dataset...')
    model.eval()
    # model.load_state_dict(torch.load(model_dir / f"{model_name}.tch"))
    # model.load_state_dict(torch.load(os.path.join(out_dir, "{}.tch".format(model_name))))
    test_roc_auc, test_ap, cls_auc, cls_ap = eval_model(model_name, device=device,
                                                        save_plots=False,
                                                        size=size,
                                                        show_training_data=False,
                                                        model=model,
                                                        mode='test',
                                                        dataloader_test=dataloader_test,
                                                        dataloader_normal=dataloader_normal)

    example_in = torch.zeros((1, 3, size, size)).cuda()
    flops, params = profile(copy.deepcopy(model), inputs=(example_in,))
    flops, params = round(flops * 1e-6, 4), round(params * 1e-6, 4)  # 1e6 = M

    flops_encoder, params_encoder = profile(copy.deepcopy(model.resnet18), inputs=(example_in,))
    flops_encoder, params_encoder = round(flops_encoder * 1e-6, 4), round(params_encoder * 1e-6, 4)  # 1e6 = M

    writer.add_scalar('final auc', test_roc_auc)
    writer.add_scalar('final ap', test_ap)
    # print("Model info. Params: {}M, FLOPs: {}M".format(params, flops))
    # print("Encoder info. Params: {}M, FLOPs: {}M".format(params_encoder, flops_encoder))
    # print('auc is ', test_roc_auc)
    # print('ap is ', test_ap)
    # print('cls_auc is ', cls_auc)
    # print('cls_ap is ', cls_ap)

    results = {"params": str(params)+"M", "FLOPs": str(flops)+"M", "params_encoder": str(params_encoder)+"M",
               "FLOPs_encoder": str(flops_encoder)+"M", "AUC": test_roc_auc, "AP": test_ap, "cls_auc": cls_auc,
               "cls_ap": cls_ap}
    result_path = os.path.join(out_dir, "results.txt")
    with open(result_path, "w") as f:
        for key, value in results.items():
            f.write(str(key) + ": " + str(value) + "\n")
            print(key + ": {}".format(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training defect detection as described in the CutPaste Paper.')
    parser.add_argument('--type', default="rsna",
                        help='which dataset to use. In this repo, you can choose zhanglab, chexpert, rsna, and both("all"). ')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of epochs to train the model , (default: 100)')

    parser.add_argument('--model_dir', default="models",
                        help='output folder of the models , (default: models)')

    parser.add_argument('--no-pretrained', dest='pretrained', default=True, action='store_false',
                        help='use pretrained values to initalize ResNet18 , (default: True)')

    parser.add_argument('--test_epochs', default=-1, type=int,
                        help='interval to calculate the auc during trainig, if -1 do not calculate test scores, (default: 10)')

    parser.add_argument('--freeze_resnet', default=20, type=int,
                        help='number of epochs to freeze resnet (default: 20)')

    parser.add_argument('--lr', default=0.03, type=float,
                        help='learning rate (default: 0.03)')

    parser.add_argument('--optim', default="sgd",
                        help='optimizing algorithm values:[sgd, adam] (dafault: "sgd")')

    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size, real batchsize is depending on cut paste config normal cutaout has effective batchsize of 2x batchsize (dafault: "64")')

    parser.add_argument('--head_layer', default=1, type=int,
                        help='number of layers in the projection head (default: 1)')

    parser.add_argument('--variant', default="3way", choices=['normal', 'scar', '3way', 'union', 'anatpaste'],
                        help='cutpaste variant to use (dafault: "3way")')

    parser.add_argument('--cuda', default=0, type=str,
                        help='num of cuda to use')
    parser.add_argument('--seed', default=0, type=int,
                        help='set random_seed')

    parser.add_argument('--workers', default=8, type=int, help="number of workers to use for data loading (default:8)")

    args = parser.parse_args()
    print(args)
    # all_types = ['zhanglab', 'chexpert', 'rsna']
    all_types = ['rsna', 'vin', 'brain', 'lag', 'brats', 'isic', 'c16', 'brats']

    # if args.type == "all":
    #     types = all_types
    # else:
    #     types = args.type.split(",")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    variant_map = {'normal': CutPasteNormal, 'scar': CutPasteScar, '3way': CutPaste3Way, 'union': CutPasteUnion,
                   'anatpaste': AnatPaste}
    variant = variant_map[args.variant]

    device = "cuda"

    # create modle dir
    Path(args.model_dir).mkdir(exist_ok=True, parents=True)
    # save config.
    # with open(Path(args.model_dir) / "run_config.txt", "w") as f:
    #     f.write(str(args))

    # seed_everything(args.seed)

    # for data_type in types:
    data_type = args.type
    print(f"training {data_type}")
    run_training(data_type,
                 model_dir=Path(args.model_dir),
                 epochs=args.epochs,
                 pretrained=args.pretrained,
                 test_epochs=args.test_epochs,
                 freeze_resnet=args.freeze_resnet,
                 learninig_rate=args.lr,
                 optim_name=args.optim,
                 batch_size=args.batch_size,
                 head_layer=args.head_layer,
                 device=device,
                 cutpate_type=variant,
                 workers=args.workers,
                 seed=args.seed)
