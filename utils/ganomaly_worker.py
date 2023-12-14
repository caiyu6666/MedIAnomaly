import time
import torch
import numpy as np
from sklearn import metrics

from utils.base_worker import BaseWorker
from utils.util import AverageMeter, calculate_threshold_fpr, calculate_dice_thr


class GanomalyWorker(BaseWorker):
    def __init__(self, opt):
        super(GanomalyWorker, self).__init__(opt)

    def set_optimizer(self):
        self.optimizer_g = torch.optim.Adam(self.net.netg.parameters(), self.opt.train['lr'],
                                            weight_decay=self.opt.train['weight_decay'])
        self.optimizer_d = torch.optim.Adam(self.net.netd.parameters(), self.opt.train['lr'],
                                            weight_decay=self.opt.train['weight_decay'])

    def train_epoch(self):
        self.net.train()

        g_losses, adv_losses, rec_losses, enc_losses, d_losses = \
            AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        for idx_batch, data_batch in enumerate(self.train_loader):
            img = data_batch['img']
            img = img.cuda()

            net_out = self.net(img, test=False)

            loss_g, loss_adv, loss_rec, loss_enc = self.criterion(img, net_out, mode='g')
            self.optimizer_g.zero_grad()
            loss_g.backward(retain_graph=True)
            self.optimizer_g.step()

            loss_d = self.criterion(img, net_out, mode='d')
            self.optimizer_d.zero_grad()
            loss_d.backward()
            self.optimizer_d.step()

            bs = img.size(0)
            g_losses.update(loss_g.item(), bs)
            adv_losses.update(loss_adv, bs)
            rec_losses.update(loss_rec, bs)
            enc_losses.update(loss_enc, bs)
            d_losses.update(loss_d.item(), bs)
        return g_losses.avg, adv_losses.avg, rec_losses.avg, enc_losses.avg, d_losses.avg

    def run_train(self):
        num_epochs = self.opt.train['epochs']
        print("=> Initial learning rate: {:g}".format(self.opt.train['lr']))
        t0 = time.time()
        for epoch in range(1, num_epochs + 1):
            g_loss, adv_loss, recon_loss, enc_loss, d_loss = self.train_epoch()
            self.logger.log(step=epoch, data={"train/g_loss": g_loss,
                                              "train/adv_loss": adv_loss,
                                              "train/recon_loss": recon_loss,
                                              "train/enc_loss": enc_loss,
                                              "train/d_loss": d_loss})

            if d_loss < 1e-5:
                self.net.reinit_d()
                print('=> Reloading net d in Epoch {}.'.format(epoch))

            if epoch == 1 or epoch % self.opt.train['eval_freq'] == 0:
                eval_results = self.evaluate()

                t = time.time() - t0
                print("Epoch[{:3d}/{:3d}]  Time:{:.1f}s  g_loss:{:.5f}  adv_loss:{:.5f}  recon_loss:{:.5f}  "
                      "enc_loss:{:.5f}  d_loss:{:.5f}".format(epoch, num_epochs, t, g_loss, adv_loss, recon_loss,
                                                              enc_loss, d_loss), end="  |  ")

                keys = list(eval_results.keys())
                for key in keys:
                    print(key+": {:.4f}".format(eval_results[key]), end="  ")
                    eval_results["val/"+key] = eval_results.pop(key)
                print()

                self.logger.log(step=epoch, data=eval_results)
                t0 = time.time()

        self.save_checkpoint()
        self.logger.finish()

    def evaluate_img(self):
        # Notice! GANomaly uses the train mode for evaluation, so that the BN layers use the mean and variance of the
        # testing sample.

        test_scores, test_labels = [], []
        with torch.no_grad():
            for idx_batch, data_batch in enumerate(self.test_loader):
                # test batch_size=1
                img, label, name = data_batch['img'], data_batch['label'], data_batch['name']
                img = img.cuda()

                net_out = self.net(img)

                anomaly_score = self.criterion(img, net_out, anomaly_score=True)

                test_scores.append(anomaly_score.cpu())
                test_labels.append(label.item())

        test_scores = torch.cat(test_scores, dim=0).numpy()  # N
        test_labels = np.array(test_labels)
        auc = metrics.roc_auc_score(test_labels, test_scores)
        ap = metrics.average_precision_score(test_labels, test_scores)
        results = {'AUC': auc, "AP": ap}

        return results

    def evaluate_pix(self):
        # Notice! GANomaly uses the train mode for evaluation, so that the BN layers use the mean and variance of the
        # testing sample.

        # GANomaly does not support pixel-level anomaly score map.
        self.evaluate_img()

        # test_scores, test_names, test_labels = [], [], []
        #
        # with torch.no_grad():
        #     for idx_batch, data_batch in enumerate(self.test_loader):
        #         # test batch_size=1
        #         volume, mask, name = data_batch['volume'], data_batch['mask'], data_batch['name']
        #         # volume, mask: 1 x depth x H x W
        #
        #         volume = volume.cuda()
        #         volume = volume.squeeze(0).unsqueeze(1)  # depth x 1 x H x W
        #         label = (mask.squeeze(0).sum(-1).sum(-1).numpy() > 0).astype(np.uint8)
        #
        #         net_out = self.net(volume)
        #
        #         anomaly_score = self.criterion(volume, net_out, anomaly_score=True)
        #         test_scores.append(anomaly_score.cpu())
        #
        #         test_labels.append(label)
        #
        # test_scores = np.concatenate(test_scores)
        # test_labels = np.concatenate(test_labels)
        #
        # slice_auc = metrics.roc_auc_score(test_labels, test_scores)
        # slice_ap = metrics.average_precision_score(test_labels, test_scores)
        # results = {'Slice_AUC': slice_auc, "Slice_AP": slice_ap}
        #
        # return results
