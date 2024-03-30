import time

import numpy as np
import torch
from torchvision import transforms
import os
from sklearn import metrics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from utils.base_worker import BaseWorker
from utils.util import AverageMeter, compute_best_dice


class AEWorker(BaseWorker):
    def __init__(self, opt):
        super(AEWorker, self).__init__(opt)
        self.pixel_metric = True if self.opt.dataset == "brats" else False
        self.grad_flag = True if self.opt.model['name'] in ['ae-grad', 'vae-elbo', 'vae-kl', 'vae-rec', 'vae-combi'] \
            else False

    def train_epoch(self):
        self.net.train()
        losses = AverageMeter()
        for idx_batch, data_batch in enumerate(self.train_loader):
            img = data_batch['img']
            img = img.cuda()

            net_out = self.net(img)

            loss = self.criterion(img, net_out)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), img.size(0))
        return losses.avg

    def data_rept(self):
        self.net.eval()

        train_normal = []
        test_all, test_labels = [], []
        with torch.no_grad():
            for idx_batch, data_batch in enumerate(self.train_loader):
                img = data_batch['img']
                img = img.cuda()

                net_out = self.net(img)
                z = net_out['z']
                train_normal.append(z.cpu().detach().numpy())

            for idx_batch, data_batch in enumerate(self.test_loader):
                img, label = data_batch['img'], data_batch['label']
                img = img.cuda()

                net_out = self.net(img)
                z = net_out['z']

                test_all.append(z.cpu().detach().numpy())
                test_labels.append(label.item())

            test_labels = np.array(test_labels)
            test_all = np.concatenate(test_all, axis=0)

            test_normal = test_all[np.where(test_labels == 0)]
            test_abnormal = test_all[np.where(test_labels == 1)]

        train_normal = np.concatenate(train_normal, axis=0)

        np.save(os.path.join(self.opt.test['save_dir'], '{}_train.npy'.format(self.opt.dataset)), train_normal)
        np.save(os.path.join(self.opt.test['save_dir'], '{}_test_normal.npy'.format(self.opt.dataset)), test_normal)
        np.save(os.path.join(self.opt.test['save_dir'], '{}_test_abnormal.npy'.format(self.opt.dataset)), test_abnormal)

        print("Train normal:", train_normal.shape)
        print("Test normal:", test_normal.shape)
        print("Test abnormal:", test_abnormal.shape)

    def evaluate(self):
        self.net.eval()
        self.close_network_grad()

        test_imgs, test_imgs_hat, test_scores, test_score_maps, test_names, test_labels, test_masks = \
            [], [], [], [], [], [], []
        # test_repts = []
        # with torch.no_grad():
        for idx_batch, data_batch in enumerate(self.test_loader):
            # test batch_size=1
            img, label, name = data_batch['img'], data_batch['label'], data_batch['name']
            img = img.cuda()
            img.requires_grad = self.grad_flag  # Will be True for gradient-based methods

            net_out = self.net(img)

            anomaly_score_map = self.criterion(img, net_out, anomaly_score=True, keepdim=True).detach().cpu()
            test_score_maps.append(anomaly_score_map)

            test_labels.append(label.item())
            if self.pixel_metric:
                mask = data_batch['mask']
                test_masks.append(mask)

            if self.opt.test['save_flag']:
                img_hat = net_out['x_hat']
                test_names.append(name)
                test_imgs.append(img.cpu())
                test_imgs_hat.append(img_hat.cpu())
                # z = net_out['z']
                # test_repts.append(z.cpu().detach().numpy())

        test_score_maps = torch.cat(test_score_maps, dim=0)  # Nx1xHxW
        test_scores = torch.mean(test_score_maps, dim=[1, 2, 3]).numpy()  # N

        # image-level metrics
        test_labels = np.array(test_labels)
        auc = metrics.roc_auc_score(test_labels, test_scores)
        ap = metrics.average_precision_score(test_labels, test_scores)
        results = {'AUC': auc, "AP": ap}

        # pixel-level metrics
        if self.pixel_metric:
            test_masks = torch.cat(test_masks, dim=0).unsqueeze(1)  # NxHxW -> Nx1xHxW
            pix_ap = metrics.average_precision_score(test_masks.numpy().reshape(-1),
                                                     test_score_maps.numpy().reshape(-1))
            pix_auc = metrics.roc_auc_score(test_masks.numpy().reshape(-1),
                                            test_score_maps.numpy().reshape(-1))
            best_dice, best_thresh = compute_best_dice(test_score_maps.numpy(), test_masks.numpy())
            results.update({'PixAUC': pix_auc, 'PixAP': pix_ap, 'BestDice': best_dice, 'BestThresh': best_thresh})
        else:
            test_masks = None

        # others
        test_normal_score = np.mean(test_scores[np.where(test_labels == 0)])
        test_abnormal_score = np.mean(test_scores[np.where(test_labels == 1)])
        results.update({"normal_score": test_normal_score, "abnormal_score": test_abnormal_score})

        if self.opt.test['save_flag']:
            test_imgs = torch.cat(test_imgs, dim=0)
            test_imgs_hat = torch.cat(test_imgs_hat, dim=0)
            self.visualize_2d(test_imgs, test_imgs_hat, test_score_maps, test_names, test_labels, test_masks)

            # # rept vis
            # test_repts = np.concatenate(test_repts, axis=0)  # Nxd
            # test_tsne = TSNE(n_components=2).fit_transform(test_repts)  # Nx2
            # normal_tsne = test_tsne[np.where(test_labels == 0)]
            # abnormal_tsne = test_tsne[np.where(test_labels == 1)]
            # plt.rcParams['font.family'] = 'Times New Roman'
            # plt.rcParams.update({'font.size': 14})
            # plt.scatter(normal_tsne[:, 0], normal_tsne[:, 1], color='b', label="Normal", s=2)
            # plt.scatter(abnormal_tsne[:, 0], abnormal_tsne[:, 1], color='r', label="Abnormal", s=2)
            # plt.xticks([])
            # plt.yticks([])
            # plt.legend(loc='upper left')
            # # plt.title(self.opt.data_name[self.opt.dataset] + ' | OC-SVM Perf. 0.66/0.82')
            # # plt.title('OC-SVM Perf. 0.48/0.52')
            # plt.tight_layout()
            # plt.savefig(os.path.join(self.opt.train['save_dir'], 'tsne.pdf'))
        self.enable_network_grad()
        return results

    def visualize_2d(self, imgs, imgs_hat, score_maps, names, labels, masks=None):
        imgs = (imgs + 1) / 2
        imgs_hat = (imgs_hat + 1) / 2
        imgs_hat = torch.clamp(imgs_hat, min=0, max=1)

        overall_dir = os.path.join(self.opt.test['save_dir'], 'vis', 'overall')
        separate_dir = os.path.join(self.opt.test['save_dir'], 'vis', 'separate')
        if not os.path.exists(overall_dir):
            os.makedirs(overall_dir)
        if not os.path.exists(separate_dir):
            os.makedirs(separate_dir)

        # clamp_max = torch.quantile(score_maps, 0.9999, interpolation="nearest")
        # # clamp_max = torch.quantile(score_maps, 0.999, interpolation="nearest")
        # score_maps = torch.clamp(score_maps, min=0., max=clamp_max)
        # score_maps = (score_maps - torch.min(score_maps)) / (torch.max(score_maps) - torch.min(score_maps))

        if imgs.size(1) == 3:
            score_maps = score_maps.repeat(1, 3, 1, 1)
            masks = masks.repeat(1, 3, 1, 1) if masks is not None else None

        for i in range(imgs.size(0)):
            name = names[i][0]
            img = imgs[i]
            img_hat = imgs_hat[i]
            map_norm = score_maps[i]
            label = labels[i]

            map_norm = (map_norm - torch.min(map_norm)) / (torch.max(map_norm) - torch.min(map_norm))

            if masks is not None:
                mask = masks[i]
                overview = torch.cat([img, img_hat, map_norm, mask], dim=-1)
            else:
                overview = torch.cat([img, img_hat, map_norm], dim=-1)

            overview = transforms.ToPILImage()(overview)
            img_hat = transforms.ToPILImage()(img_hat)
            pred = transforms.ToPILImage()(map_norm)

            overview_path = os.path.join(overall_dir, str(label) + "_" + name + ".png")
            rec_path = os.path.join(separate_dir, str(label) + "_" + name + "_rec" + ".png")
            score_path = os.path.join(separate_dir, str(label) + "_" + name + "_pred" + ".png")

            overview.save(overview_path)
            img_hat.save(rec_path)
            pred.save(score_path)

    def run_train(self):
        num_epochs = self.opt.train['epochs']
        print("=> Initial learning rate: {:g}".format(self.opt.train['lr']))
        t0 = time.time()
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            self.logger.log(step=epoch, data={"train/loss": train_loss})
            # self.logger.log(step=epoch, data={"train/loss": train_loss, "train/lr": self.scheduler.get_last_lr()[0]})
            # self.scheduler.step()

            if epoch == 1 or epoch % self.opt.train['eval_freq'] == 0:
                eval_results = self.evaluate()

                t = time.time() - t0
                print("Epoch[{:3d}/{:3d}]  Time:{:.1f}s  loss:{:.5f}".format(epoch, num_epochs, t, train_loss),
                      end="  |  ")

                keys = list(eval_results.keys())
                for key in keys:
                    print(key+": {:.5f}".format(eval_results[key]), end="  ")
                    eval_results["val/"+key] = eval_results.pop(key)
                print()

                self.logger.log(step=epoch, data=eval_results)
                t0 = time.time()

        self.save_checkpoint()
        self.logger.finish()
