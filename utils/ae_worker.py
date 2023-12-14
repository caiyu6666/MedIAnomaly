import time

import numpy as np
import torch
from torchvision import transforms
import os
from sklearn import metrics

from utils.base_worker import BaseWorker
from utils.util import AverageMeter, calculate_threshold_fpr, calculate_dice_thr, compute_best_dice


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

    def eval_func(self, pixel_metric=False):
        self.net.eval()
        self.close_network_grad()

        test_imgs, test_imgs_hat, test_scores, test_score_maps, test_names, test_labels, test_masks = \
            [], [], [], [], [], [], []
        # with torch.no_grad():
        for idx_batch, data_batch in enumerate(self.test_loader):
            # test batch_size=1
            img, label, name = data_batch['img'], data_batch['label'], data_batch['name']
            img = img.cuda()
            img.requires_grad = True

            net_out = self.net(img)

            anomaly_score_map = self.criterion(img, net_out, anomaly_score=True, keepdim=True).detach().cpu()
            test_score_maps.append(anomaly_score_map)

            test_labels.append(label.item())
            if pixel_metric:
                mask = data_batch['mask']
                test_masks.append(mask)

            if self.opt.test['save_flag']:
                img_hat = net_out['x_hat']
                test_names.append(name)
                test_imgs.append(img.cpu())
                test_imgs_hat.append(img_hat.cpu())

        test_score_maps = torch.cat(test_score_maps, dim=0)  # Nx1xHxW
        test_scores = torch.mean(test_score_maps, dim=[1, 2, 3]).numpy()  # N

        # image-level metrics
        test_labels = np.array(test_labels)
        auc = metrics.roc_auc_score(test_labels, test_scores)
        ap = metrics.average_precision_score(test_labels, test_scores)
        results = {'AUC': auc, "AP": ap}

        # pixel-level metrics
        if pixel_metric:
            test_masks = torch.cat(test_masks, dim=0).unsqueeze(1)  # NxHxW -> Nx1xHxW
            pix_ap = metrics.average_precision_score(test_masks.numpy().reshape(-1),
                                                     test_score_maps.numpy().reshape(-1))
            best_dice, best_thresh = compute_best_dice(test_score_maps.numpy(), test_masks.numpy())
            results.update({'PixAP': pix_ap, 'BestDice': best_dice, 'BestThresh': best_thresh})
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

        self.enable_network_grad()
        return results

    def visualize_2d(self, imgs, imgs_hat, score_maps, names, labels, masks=None):
        imgs = (imgs + 1) / 2
        imgs_hat = (imgs_hat + 1) / 2
        imgs_hat = torch.clamp(imgs_hat, min=0, max=1)

        clamp_max = torch.quantile(score_maps, 0.9999, interpolation="nearest")
        # clamp_max = torch.quantile(score_maps, 0.999, interpolation="nearest")
        score_maps = torch.clamp(score_maps, min=0., max=clamp_max)
        score_maps = (score_maps - torch.min(score_maps)) / (torch.max(score_maps) - torch.min(score_maps))

        if imgs.size(1) == 3:
            score_maps = score_maps.repeat(1, 3, 1, 1)
            masks = masks.repeat(1, 3, 1, 1) if masks is not None else None

        for i in range(imgs.size(0)):
            name = names[i][0]
            img = imgs[i]
            img_hat = imgs_hat[i]
            map_norm = score_maps[i]
            label = labels[i]

            if masks is not None:
                mask = masks[i]
                overview = torch.cat([img, img_hat, map_norm, mask], dim=-1)
            else:
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
                    print(key+": {:.5f}".format(eval_results[key]), end="  ")
                    eval_results["val/"+key] = eval_results.pop(key)
                print()

                self.logger.log(step=epoch, data=eval_results)
                t0 = time.time()

        self.save_checkpoint()
        self.logger.finish()
