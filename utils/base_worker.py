import random
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn

from networks.ae import AE
from networks.mem_ae import MemAE
from networks.aeu import AEU
from networks.vae import VAE
from networks.ganomaly import Ganomaly

from dataloaders.data_utils import get_transform, get_data_path
from dataloaders.dataload import MedAD, BraTSAD
import wandb
from thop import profile
# from utils.losses import ae_loss, aeu_loss, memae_loss, vae_loss, vae_loss_grad_elbo, vae_loss_grad_kl, vae_loss_grad_rec, vae_loss_grad_combi
from utils.losses import *


class BaseWorker:
    def __init__(self, opt):
        self.logger = None
        self.opt = opt
        self.seed = None
        self.train_set = None
        self.test_set = None
        self.train_loader = None
        self.test_loader = None
        self.scheduler = None
        self.optimizer = None

        self.net = None
        self.criterion = None

    def set_gpu_device(self):
        torch.cuda.set_device(self.opt.gpu)
        print("=> Set GPU device: {}".format(self.opt.gpu))

    def set_seed(self):
        self.seed = self.opt.train['seed']
        if self.seed is None:
            self.seed = np.random.randint(1, 999999)
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def set_network_loss(self):
        if self.opt.model['name'] == 'ae':
            self.net = AE(input_size=self.opt.model['input_size'], in_planes=self.opt.model['in_c'],
                          base_width=self.opt.model['base_width'], expansion=self.opt.model['expansion'],
                          mid_num=self.opt.model['hidden_num'], latent_size=self.opt.model['ls'],
                          en_num_layers=self.opt.model["en_depth"], de_num_layers=self.opt.model["de_depth"])
            # self.criterion = l2_loss
            self.criterion = ae_loss
        elif self.opt.model['name'] == 'ae-grad':
            self.net = AE(input_size=self.opt.model['input_size'], in_planes=self.opt.model['in_c'],
                          base_width=self.opt.model['base_width'], expansion=self.opt.model['expansion'],
                          mid_num=self.opt.model['hidden_num'], latent_size=self.opt.model['ls'],
                          en_num_layers=self.opt.model["en_depth"], de_num_layers=self.opt.model["de_depth"])
            self.criterion = ae_loss_grad
        elif self.opt.model['name'] == 'memae':
            self.net = MemAE(input_size=self.opt.model['input_size'], in_planes=self.opt.model['in_c'],
                             base_width=self.opt.model['base_width'], expansion=self.opt.model['expansion'],
                             mid_num=self.opt.model['hidden_num'], latent_size=self.opt.model['ls'],
                             en_num_layers=self.opt.model["en_depth"], de_num_layers=self.opt.model["de_depth"])
            self.criterion = memae_loss
        elif self.opt.model['name'] == 'aeu':
            self.net = AEU(input_size=self.opt.model['input_size'], in_planes=self.opt.model['in_c'],
                           base_width=self.opt.model['base_width'], expansion=self.opt.model['expansion'],
                           mid_num=self.opt.model['hidden_num'], latent_size=self.opt.model['ls'],
                           en_num_layers=self.opt.model["en_depth"], de_num_layers=self.opt.model["de_depth"])
            self.criterion = aeu_loss
        # elif self.opt.model['name'] == 'vae':
        elif 'vae' in self.opt.model['name']:
            self.net = VAE(input_size=self.opt.model['input_size'], in_planes=self.opt.model['in_c'],
                           base_width=self.opt.model['base_width'], expansion=self.opt.model['expansion'],
                           mid_num=self.opt.model['hidden_num'], latent_size=self.opt.model['ls'],
                           en_num_layers=self.opt.model["en_depth"], de_num_layers=self.opt.model["de_depth"])
            if self.opt.model['name'] == 'vae':
                self.criterion = vae_loss
            elif self.opt.model['name'] == 'vae-elbo':
                self.criterion = vae_loss_grad_elbo
            elif self.opt.model['name'] == 'vae-kl':
                self.criterion = vae_loss_grad_kl
            elif self.opt.model['name'] == 'vae-rec':
                self.criterion = vae_loss_grad_rec
            elif self.opt.model['name'] == 'vae-combi':
                self.criterion = vae_loss_grad_combi
            else:
                raise Exception("Invalid VAE model: {}".format(self.opt.model['name']))
        elif self.opt.model['name'] == 'ganomaly':
            self.net = Ganomaly(input_size=self.opt.model['input_size'], in_planes=self.opt.model['in_c'],
                                base_width=self.opt.model['base_width'], expansion=self.opt.model['expansion'],
                                mid_num=self.opt.model['hidden_num'], latent_size=self.opt.model['ls'],
                                en_num_layers=self.opt.model["en_depth"], de_num_layers=self.opt.model["de_depth"])
            self.criterion = ganomaly_loss
        else:
            raise NotImplementedError("Unexpected model name: {}".format(self.opt.model['name']))
        self.net = self.net.cuda()

    # def set_loss(self):
    #     if self.opt.train['loss'] == 'l2':
    #         self.criterion = l2_loss
    #     elif self.opt.train['loss'] == 'l1':
    #         self.criterion = l1_loss
    #     elif self.opt.train['loss'] == 'aeu_loss':
    #         self.criterion = aeu_loss
    #     elif self.opt.train['loss'] == 'memae_loss':
    #         self.criterion = memae_loss
    #     else:
    #         raise NotImplementedError("Unexpected loss function: {}".format(self.opt.train['loss']))

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.opt.train['lr'],
                                          weight_decay=self.opt.train['weight_decay'])
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
        #                                                             T_max=self.opt.train['epochs'],
        #                                                             eta_min=5e-5)

    def set_dataloader(self):
        data_path = get_data_path(dataset=self.opt.dataset)
        train_transform = get_transform(self.opt, phase='train')
        test_transform = get_transform(self.opt, phase='test')
        if self.opt.dataset in ['rsna', 'vin', 'brain', 'lag']:
            self.train_set = MedAD(main_path=data_path, img_size=self.opt.model['input_size'],
                                   transform=train_transform, mode='train')
            self.test_set = MedAD(main_path=data_path, img_size=self.opt.model['input_size'], transform=test_transform,
                                  mode='test')
        elif self.opt.dataset == 'brats':
            self.train_set = BraTSAD(main_path=data_path, img_size=self.opt.model['input_size'],
                                     transform=train_transform, istrain=True)
            self.test_set = BraTSAD(main_path=data_path, img_size=self.opt.model['input_size'],
                                    transform=test_transform, istrain=False)
        else:
            raise Exception("Invalid dataset: {}".format(self.opt.dataset))

        self.train_loader = DataLoader(self.train_set, batch_size=self.opt.train['batch_size'], shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False)

    def set_logging(self, test=False):
        example_in = torch.zeros((1, self.opt.model["in_c"],
                                  self.opt.model['input_size'], self.opt.model['input_size'])).cuda()
        flops, params = profile(self.net, inputs=(example_in,))
        flops, params = round(flops * 1e-6, 4), round(params * 1e-6, 4)  # 1e6 = M
        flops, params = str(flops) + "M", str(params) + "M"

        exp_configs = {"dataset": self.opt.dataset,
                       "model": self.opt.model["name"],
                       "in_channels": self.opt.model["in_c"],
                       "input_size": self.opt.model['input_size'],

                       "base_width": self.opt.model['base_width'],
                       "expansion": self.opt.model['expansion'],
                       "mid_num": self.opt.model['hidden_num'],
                       "latent_size": self.opt.model['ls'],
                       "en_num_layers": self.opt.model["en_depth"],
                       "de_num_layers": self.opt.model["de_depth"],

                       # "loss": self.opt.train["loss"],
                       "epochs": self.opt.train["epochs"],
                       "batch_size": self.opt.train["batch_size"],
                       "lr": self.opt.train["lr"],
                       "weight_decay": self.opt.train["weight_decay"],
                       "seed": self.seed,

                       "num_params": params,
                       "FLOPs": flops}

        # self.logger = wandb.init(project="MedIAnomaly", config=exp_configs, name=self.opt.notes, tags=[self.opt.tags])
        if not test:
            self.logger = wandb.init(project=self.opt.project_name, config=exp_configs)
        print("============= Configurations =============")
        for key, values in exp_configs.items():
            # print(key + ":" + str(values), end=" | ")
            print(key + ":" + str(values))
        print()

    def close_network_grad(self):
        for param in self.net.parameters():
            param.requires_grad = False

    def enable_network_grad(self):
        for param in self.net.parameters():
            param.requires_grad = True

    def set_test_loader(self):
        """ Use for only evaluation"""
        data_path = get_data_path(dataset=self.opt.dataset)
        test_transform = get_transform(self.opt, phase='test')
        if self.opt.dataset in ['rsna', 'vin', 'brain', 'lag']:
            self.test_set = MedAD(main_path=data_path, img_size=self.opt.model['input_size'], transform=test_transform,
                                  mode='test')
        elif self.opt.dataset == 'brats':
            self.test_set = BraTSAD(main_path=data_path, img_size=self.opt.model['input_size'],
                                    transform=test_transform, istrain=False)
        else:
            raise Exception("Invalid dataset: {}".format(self.opt.dataset))
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False)
        print("=> Set test dataset: {} | Input size: {} | Batch size: {}".format(self.opt.dataset,
                                                                                 self.opt.model['input_size'], 1))

    def save_checkpoint(self):
        torch.save(self.net.state_dict(), os.path.join(self.opt.train['save_dir'], "checkpoints", "model.pt"))

    def load_checkpoint(self):
        model_path = os.path.join(self.opt.train['save_dir'], "checkpoints", "model.pt")
        self.net.load_state_dict(torch.load(model_path, map_location=torch.device("cuda:{}".format(self.opt.gpu))))
        print("=> Load model from {}".format(model_path))
