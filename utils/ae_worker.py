import time

import numpy as np
import torch
from torchvision import transforms
import os
from sklearn import metrics

from utils.base_worker import BaseWorker
from utils.util import AverageMeter, calculate_threshold_fpr, calculate_dice_thr
from torchvision.models import resnet18


class AEWorker(BaseWorker):
    def __init__(self, opt):
        super(AEWorker, self).__init__(opt)

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

    def evaluate(self):
        if self.opt.dataset == "brats":
            return self.evaluate_3d()
        else:
            return self.evaluate_2d()

    def evaluate_2d(self):
        self.net.eval()
        self.close_network_grad()

        test_imgs, test_imgs_hat, test_scores, test_score_maps, test_names, test_labels = [], [], [], [], [], []
        # with torch.no_grad():
        for idx_batch, data_batch in enumerate(self.test_loader):
            # test batch_size=1
            img, label, name = data_batch['img'], data_batch['label'], data_batch['name']
            img = img.cuda()
            img.requires_grad = True

            net_out = self.net(img)

            anomaly_score = self.criterion(img, net_out, anomaly_score=True).detach()

            test_scores.append(anomaly_score.cpu())
            test_labels.append(label.item())

            if self.opt.test['save_flag']:
                img_hat = net_out['x_hat']
                test_names.append(name)
                test_imgs.append(img.cpu())
                test_imgs_hat.append(img_hat.cpu())

                anomaly_score_map = self.criterion(img, net_out, anomaly_score=True, keepdim=True).detach()
                test_score_maps.append(anomaly_score_map.cpu())

        if self.opt.test['save_flag']:
            test_score_maps = torch.cat(test_score_maps, dim=0)  # N x 1 x H x W
            test_imgs = torch.cat(test_imgs, dim=0)
            test_imgs_hat = torch.cat(test_imgs_hat, dim=0)
            self.visualize_2d(test_imgs, test_imgs_hat, test_score_maps, test_names, test_labels)

        test_scores = np.concatenate(test_scores)  # N
        test_labels = np.array(test_labels)

        auc = metrics.roc_auc_score(test_labels, test_scores)
        ap = metrics.average_precision_score(test_labels, test_scores)
        results = {'AUC': auc, "AP": ap}

        self.enable_network_grad()
        return results

    def evaluate_3d(self):
        self.net.eval()
        self.close_network_grad()

        test_volumes, test_volumes_hat, test_scores, test_score_maps, test_names, test_masks = [], [], [], [], [], []

        # with torch.no_grad():
        for idx_batch, data_batch in enumerate(self.test_loader):
            # test batch_size=1
            volume, mask, name = data_batch['volume'], data_batch['mask'], data_batch['name']
            # volume, mask: 1 x depth x H x W

            volume = volume.cuda()
            volume = volume.squeeze(0).unsqueeze(1)  # depth x 1 x H x W
            mask = (mask.squeeze(0).unsqueeze(1).numpy() > 0).astype(np.uint8)

            volume.requires_grad = True
            
            net_out = self.net(volume)

            anomaly_score_map = self.criterion(volume, net_out, anomaly_score=True, keepdim=True).detach()
            test_score_maps.append(anomaly_score_map.cpu())

            test_masks.append(mask)

            if self.opt.test['save_flag']:
                volume_hat = net_out['x_hat']
                test_names.append(name)
                test_volumes.append(volume.cpu())
                test_volumes_hat.append(volume_hat.cpu())

        score_concat = torch.cat(test_score_maps, dim=0)  # N x 1 x H x W

        if self.opt.test['save_flag']:
            test_volumes = torch.cat(test_volumes, dim=0)
            test_volumes_hat = torch.cat(test_volumes_hat, dim=0)

            self.visualize_3d(test_volumes, test_volumes_hat, score_concat, test_masks, test_names)

        score_concat = np.array(score_concat)  # N x 1 x H x W
        true_concat = np.concatenate(test_masks, axis=0).astype(np.uint8)

        y_score_slice = score_concat.mean(-1).mean(-1).reshape(-1)
        y_true_slice = (true_concat.sum(-1).sum(-1).reshape(-1) > 0).astype(np.uint8)

        slice_auc = metrics.roc_auc_score(y_true_slice, y_score_slice)
        slice_ap = metrics.average_precision_score(y_true_slice, y_score_slice)

        fpr01, threshold_at_fpr01 = calculate_threshold_fpr(true_concat, score_concat, target_fpr=0.001)  # 0.1%FPR
        fpr1, threshold_at_fpr1 = calculate_threshold_fpr(true_concat, score_concat, target_fpr=0.01)  # 1%FPR
        fpr5, threshold_at_fpr5 = calculate_threshold_fpr(true_concat, score_concat, target_fpr=0.05)  # 5%FPR

        dice01 = calculate_dice_thr(test_masks, test_score_maps, threshold=threshold_at_fpr01)
        dice1 = calculate_dice_thr(test_masks, test_score_maps, threshold=threshold_at_fpr1)
        dice5 = calculate_dice_thr(test_masks, test_score_maps, threshold=threshold_at_fpr5)
        # print("At FPR: {:4f}, threshold is {:.5f} || DICE: {}".format(fpr01, threshold_at_fpr01, dice01))
        # print("At FPR: {:4f}, threshold is {:.5f} || DICE: {}".format(fpr1, threshold_at_fpr1, dice1))
        # print("At FPR: {:4f}, threshold is {:.5f} || DICE: {}".format(fpr5, threshold_at_fpr5, dice5))
        # print()
        results = {"Dice_FPR0.1%": dice01, "Dice_FPR1%": dice1, "Dice_FPR5%": dice5, "Slice_AUC": slice_auc,
                   "Slice_AP": slice_ap}

        self.enable_network_grad()
        return results

    def visualize_2d(self, imgs, imgs_hat, score_maps, names, labels):
        imgs = (imgs + 1) / 2
        imgs_hat = (imgs_hat + 1) / 2
        imgs_hat = torch.clamp(imgs_hat, min=0, max=1)

        clamp_max = torch.quantile(score_maps, 0.9999, interpolation="nearest")
        score_maps = torch.clamp(score_maps, min=0., max=clamp_max)
        score_maps = (score_maps - torch.min(score_maps)) / (torch.max(score_maps) - torch.min(score_maps))

        for i in range(imgs.size(0)):
            name = names[i][0]
            img = imgs[i]
            img_hat = imgs_hat[i]
            map_norm = score_maps[i]
            label = labels[i]

            overview = torch.cat([img, img_hat, map_norm], dim=-1)
            overview = transforms.ToPILImage()(overview)

            save_path = os.path.join(self.opt.test['save_dir'], str(label) + "_" + name + ".png")
            overview.save(save_path)

    def visualize_3d(self, volumes, volumes_hat, score_maps, masks, names):
        """

        :param volumes:
        :param volumes_hat:
        :param score_maps:
        :param masks: list of ndarray, [(Depth, 1, H, W)]
        :param names:
        :return:
        """
        volumes = (volumes + 1) / 2
        volumes_hat = (volumes_hat + 1) / 2
        volumes_hat = torch.clamp(volumes_hat, min=0, max=1)

        # clamp_max = torch.quantile(score_maps, 0.99, interpolation="nearest")
        # # torch.quantile() cannot handle a so huge tensor
        clamp_max = np.quantile(score_maps.numpy(), 0.9999, interpolation="nearest")
        score_maps = torch.clamp(score_maps, min=0., max=clamp_max)
        score_maps = (score_maps - torch.min(score_maps)) / (torch.max(score_maps) - torch.min(score_maps))

        n = 0
        for i in range(len(masks)):
            name = names[i][0]
            mask = masks[i]
            mask = torch.tensor(mask)

            for j in range(mask.size(0)):
                mask_sli = mask[j]
                sli = volumes[n]
                sli_hat = volumes_hat[n]
                map_norm = score_maps[n]

                overview = torch.cat([sli, sli_hat, map_norm, mask_sli], dim=-1)
                overview = transforms.ToPILImage()(overview)
                save_path = os.path.join(self.opt.test['save_dir'], name + "_" + str(j) + ".png")
                overview.save(save_path)
                n += 1

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
                    print(key+": {:.4f}".format(eval_results[key]), end="  ")
                    eval_results["val/"+key] = eval_results.pop(key)
                print()

                self.logger.log(step=epoch, data=eval_results)
                t0 = time.time()

        self.save_checkpoint()
        self.logger.finish()

    def run_eval(self):
        results = self.evaluate()
        metrics_save_path = os.path.join(self.opt.train['save_dir'], "metrics.txt")
        with open(metrics_save_path, "w") as f:
            for key, value in results.items():
                f.write(str(key) + ": " + str(value) + "\n")
                print(key + ": {:.4f}".format(value))
