import time

import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image
import os
from sklearn import metrics

from utils.ae_worker import AEWorker
from utils.util import AverageMeter, calculate_threshold_fpr, calculate_dice_thr
import torch.autograd as autograd


class FanoGANWorker(AEWorker):
    def __init__(self, opt):
        super(FanoGANWorker, self).__init__(opt)
        self.n_critic = 5
        self.lambda_gp = 10

        self.wgan_epochs = 3000
        self.encoder_epochs = 300

    def set_optimizer(self):
        self.optimizer_d = torch.optim.Adam(self.net.discriminator.parameters(), self.opt.train['lr'],
                                            weight_decay=self.opt.train['weight_decay'])
        self.optimizer_g = torch.optim.Adam(self.net.generator.parameters(), self.opt.train['lr'],
                                            weight_decay=self.opt.train['weight_decay'])
        self.optimizer_e = torch.optim.Adam(self.net.encoder.parameters(), self.opt.train['lr'],
                                            weight_decay=self.opt.train['weight_decay'])

    def run_train(self):
        self.run_train_wgan()
        # self.load_checkpoint()
        # self.net.initialize_encoder()

        self.run_train_encoder()
        self.logger.finish()

    def run_train_wgan(self):
        # num_epochs = self.opt.train['epochs']
        print("=> Initial learning rate: {:g}".format(self.opt.train['lr']))
        t0 = time.time()
        for epoch in range(1, self.wgan_epochs + 1):
            d_loss, d_real, d_fake, g_loss = self.train_wgan_epoch()
            self.logger.log(step=epoch, data={"train/d_loss": d_loss})
            self.logger.log(step=epoch, data={"train/d_real": d_real})
            self.logger.log(step=epoch, data={"train/d_fake": d_fake})
            self.logger.log(step=epoch, data={"train/g_loss": g_loss})

            if epoch == 1 or epoch % self.opt.train['eval_freq'] == 0:
                if epoch % 100 == 0:
                    self.visualize_generation(epoch)
                t = time.time() - t0
                print("Epoch[{:3d}/{:3d}]  Time:{:.1f}s  d_loss:{:.5f}  d_real:{:.5f}  d_fake:{:.5f}  "
                      "g_loss:{:.5f}".format(epoch, self.wgan_epochs, t, d_loss, d_real, d_fake, g_loss))
                t0 = time.time()

        self.save_checkpoint()
        # self.logger.finish()

    def run_train_encoder(self):
        # num_epochs = self.opt.train['epochs']
        print("=> Initial learning rate: {:g}".format(self.opt.train['lr']))
        t0 = time.time()
        for epoch in range(1, self.encoder_epochs + 1):
            loss, img_loss, feat_loss = self.train_encoder_epoch()
            self.logger.log(step=epoch, data={"train/loss": loss})
            self.logger.log(step=epoch, data={"train/img_loss": img_loss})
            self.logger.log(step=epoch, data={"train/feat_loss": feat_loss})

            # if epoch == 1 or epoch % self.opt.train['eval_freq'] == 0:
            eval_results = self.evaluate()

            t = time.time() - t0
            print("Epoch[{:3d}/{:3d}]  Time:{:.1f}s  loss:{:.5f}  img_loss:{:.5f}  feat_loss:{:.5f}"
                  "".format(epoch, self.encoder_epochs, t, loss, img_loss, feat_loss), end="  |  ")

            keys = list(eval_results.keys())
            for key in keys:
                print(key + ": {:.4f}".format(eval_results[key]), end="  ")
                eval_results["val/" + key] = eval_results.pop(key)
            print()

            self.logger.log(step=epoch, data=eval_results)
            t0 = time.time()

        self.save_checkpoint()

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(*real_samples.shape[:2], 1, 1).cuda()
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples.detach() + (1 - alpha) * fake_samples.detach())
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        d_interpolates = self.net.discriminator(interpolates)
        fake = torch.ones(*d_interpolates.shape).cuda()
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                  grad_outputs=fake, create_graph=True,
                                  retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.shape[0], -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_wgan_epoch(self):
        self.net.train()
        d_losses, d_reals, d_fakes, g_losses = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        for idx_batch, data_batch in enumerate(self.train_loader):
            img = data_batch['img']
            img = img.cuda()
            bs = img.shape[0]

            z = torch.randn(bs, self.opt.model['ls']).cuda()  # bs x latent_size
            fake_img = self.net.generator(z)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            self.optimizer_d.zero_grad()

            d_real = torch.mean(self.net.discriminator(img))
            d_fake = torch.mean(self.net.discriminator(fake_img.detach()))

            gradient_penalty = self.compute_gradient_penalty(img.data, fake_img.data)

            d_loss = -d_real + d_fake + self.lambda_gp * gradient_penalty

            d_loss.backward()
            self.optimizer_d.step()

            d_losses.update(d_loss.item(), bs)
            d_reals.update(d_real.item(), bs)
            d_fakes.update(d_fake.item(), bs)

            if (idx_batch + 1) % self.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------
                self.optimizer_g.zero_grad()

                z = torch.randn(bs, self.opt.model['ls']).cuda()  # bs x latent_size
                fake_img = self.net.generator(z)

                d_fake = self.net.discriminator(fake_img)
                g_loss = -torch.mean(d_fake)

                g_loss.backward()
                self.optimizer_g.step()

                g_losses.update(g_loss.item(), bs)

        return d_losses.avg, d_reals.avg, d_fakes.avg, g_losses.avg

    def train_encoder_epoch(self):
        self.net.generator.eval()
        self.net.discriminator.eval()
        self.net.encoder.train()

        for p in self.net.generator.parameters():
            p.requires_grad = False
        for p in self.net.discriminator.parameters():
            p.requires_grad = False

        for p in self.net.encoder.parameters():
            p.requires_grad = True

        losses, img_losses, feat_losses = AverageMeter(), AverageMeter(), AverageMeter()
        for idx_batch, data_batch in enumerate(self.train_loader):
            img = data_batch['img']
            img = img.cuda()
            bs = img.shape[0]

            z = self.net.encoder(img)
            fake_img = self.net.generator(z)

            d_input = torch.cat([img, fake_img], dim=0)
            real_features, fake_features = self.net.discriminator.forward_features(d_input).chunk(2, 0)
            # # Real features
            # real_features = self.net.discriminator.forward_features(img)
            # # Fake features
            # fake_features = self.net.discriminator.forward_features(fake_img)

            img_loss = torch.mean((img - fake_img) ** 2)
            feat_loss = torch.mean((real_features - fake_features) ** 2)

            loss = img_loss + feat_loss

            self.optimizer_e.zero_grad()
            loss.backward()
            self.optimizer_e.step()

            losses.update(loss.item(), bs)
            img_losses.update(img_loss.item(), bs)
            feat_losses.update(feat_loss.item(), bs)

        return losses.avg, img_losses.avg, feat_losses.avg

    def visualize_generation(self, epoch):
        z = torch.randn(self.opt.train['batch_size'], self.opt.model['ls']).cuda()  # bs x latent_size
        fake_img = self.net.generator(z)
        save_dir = os.path.join(self.opt.test['save_dir'], 'generation')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, str(epoch) + ".png")
        save_image(fake_img * 0.5 + 0.5, save_path)

    def evaluate_2d(self):
        self.net.eval()

        test_imgs, test_imgs_hat, test_scores, test_score_maps, test_names, test_labels = [], [], [], [], [], []
        with torch.no_grad():
            for idx_batch, data_batch in enumerate(self.test_loader):
                # test batch_size=1
                img, label, name = data_batch['img'], data_batch['label'], data_batch['name']
                img = img.cuda()

                img_fake, feat_real, feat_fake = self.net(img)

                img_dis_map = (img - img_fake) ** 2
                img_dis = torch.mean(img_dis_map, dim=[1, 2, 3])
                feat_dis = torch.mean((feat_real - feat_fake) ** 2, dim=1)

                anomaly_score = img_dis + feat_dis
                test_scores.append(anomaly_score.cpu())
                test_labels.append(label.item())

                if self.opt.test['save_flag']:
                    test_names.append(name)
                    test_imgs.append(img.cpu())
                    test_imgs_hat.append(img_fake.cpu())

                    test_score_maps.append(img_dis_map.cpu())

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

        return results

    def evaluate_3d(self):
        self.net.eval()

        test_volumes, test_volumes_hat, test_scores, test_score_maps, test_names, test_masks = [], [], [], [], [], []
        test_labels = []
        # with torch.no_grad():
        for idx_batch, data_batch in enumerate(self.test_loader):
            # test batch_size=1
            volume, mask, name = data_batch['volume'], data_batch['mask'], data_batch['name']
            # volume, mask: 1 x depth x H x W

            volume = volume.cuda()
            volume = volume.squeeze(0).unsqueeze(1)  # depth x 1 x H x W
            mask = (mask.squeeze(0).unsqueeze(1).numpy() > 0).astype(np.uint8)
            label = (mask.squeeze(1).sum(-1).sum(-1) > 0).astype(np.uint8)

            volume_fake, feat_real, feat_fake = self.net(volume)

            img_dis_map = (volume - volume_fake) ** 2
            img_dis = torch.mean(img_dis_map, dim=[1, 2, 3])
            feat_dis = torch.mean((feat_real - feat_fake) ** 2, dim=1)

            anomaly_score = img_dis + feat_dis

            test_scores.append(anomaly_score.cpu())
            test_score_maps.append(img_dis_map.cpu())

            test_labels.append(label.item())
            test_masks.append(mask)

            if self.opt.test['save_flag']:
                test_names.append(name)
                test_volumes.append(volume.cpu())
                test_volumes_hat.append(volume_fake.cpu())

        score_concat = torch.cat(test_score_maps, dim=0)  # N x 1 x H x W

        if self.opt.test['save_flag']:
            test_volumes = torch.cat(test_volumes, dim=0)
            test_volumes_hat = torch.cat(test_volumes_hat, dim=0)
            self.visualize_3d(test_volumes, test_volumes_hat, score_concat, test_masks, test_names)

        test_scores = np.concatenate(test_scores)
        test_labels = np.concatenate(test_labels)
        # score_concat = np.array(score_concat)  # N x 1 x H x W
        true_concat = np.concatenate(test_masks, axis=0).astype(np.uint8)

        slice_auc = metrics.roc_auc_score(test_labels, test_scores)
        slice_ap = metrics.average_precision_score(test_labels, test_scores)

        fpr01, threshold_at_fpr01 = calculate_threshold_fpr(true_concat, score_concat, target_fpr=0.001)  # 0.1%FPR
        fpr1, threshold_at_fpr1 = calculate_threshold_fpr(true_concat, score_concat, target_fpr=0.01)  # 1%FPR
        fpr5, threshold_at_fpr5 = calculate_threshold_fpr(true_concat, score_concat, target_fpr=0.05)  # 5%FPR

        dice01 = calculate_dice_thr(test_masks, test_score_maps, threshold=threshold_at_fpr01)
        dice1 = calculate_dice_thr(test_masks, test_score_maps, threshold=threshold_at_fpr1)
        dice5 = calculate_dice_thr(test_masks, test_score_maps, threshold=threshold_at_fpr5)
        results = {"Dice_FPR0.1%": dice01, "Dice_FPR1%": dice1, "Dice_FPR5%": dice5, "Slice_AUC": slice_auc,
                   "Slice_AP": slice_ap}

        return results
