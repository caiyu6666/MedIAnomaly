import time
import torch
import numpy as np
from torchvision import transforms
import os
from sklearn import metrics

from utils.base_worker import BaseWorker
from utils.util import AverageMeter


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

    def evaluate(self):
        # Notice! GANomaly uses the train mode for evaluation, so that the BN layers use the mean and variance of the
        # testing sample.
        self.net.train()

        test_imgs, test_imgs_hat, test_scores, test_score_maps, test_names, test_labels, test_masks = \
            [], [], [], [], [], [], []
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

                if self.opt.test['save_flag']:
                    img_hat = net_out['x_hat']
                    test_names.append(name)
                    test_imgs.append(img.cpu())
                    test_imgs_hat.append(img_hat.cpu())

        test_scores = torch.cat(test_scores, dim=0).numpy()  # N
        test_labels = np.array(test_labels)
        auc = metrics.roc_auc_score(test_labels, test_scores)
        ap = metrics.average_precision_score(test_labels, test_scores)
        results = {'AUC': auc, "AP": ap}

        if self.opt.test['save_flag']:
            # test_imgs = torch.cat(test_imgs, dim=0)
            test_imgs_hat = torch.cat(test_imgs_hat, dim=0)
            test_imgs_hat = (test_imgs_hat + 1) / 2
            test_imgs_hat = torch.clamp(test_imgs_hat, min=0, max=1)

            # overall_dir = os.path.join(self.opt.test['save_dir'], 'vis', 'overall')
            separate_dir = os.path.join(self.opt.test['save_dir'], 'vis', 'separate')
            # if not os.path.exists(overall_dir):
            #     os.makedirs(overall_dir)
            if not os.path.exists(separate_dir):
                os.makedirs(separate_dir)

            for i in range(test_imgs_hat.size(0)):
                name = test_names[i][0]
                img_hat = test_imgs_hat[i]
                label = test_labels[i]

                img_hat = transforms.ToPILImage()(img_hat)

                rec_path = os.path.join(separate_dir, str(label) + "_" + name + "_rec" + ".png")

                img_hat.save(rec_path)

        return results
