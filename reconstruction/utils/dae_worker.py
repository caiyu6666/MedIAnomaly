import time
import torch
import random
import torch.nn.functional as F

from utils.ae_worker import AEWorker
from utils.util import AverageMeter


class DAEWorker(AEWorker):
    def __init__(self, opt):
        super(DAEWorker, self).__init__(opt)
        self.noise_res = 16
        self.noise_std = 0.2

    def train_epoch(self):
        self.net.train()
        losses = AverageMeter()
        for idx_batch, data_batch in enumerate(self.train_loader):
            img = data_batch['img']
            img = img.cuda()
            noisy_img, noise_tensor = self.add_noise(img)

            net_out = self.net(noisy_img)

            loss = self.criterion(img, net_out)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), img.size(0))
        return losses.avg

    def add_noise(self, x):
        """
        Generate and apply randomly translated noise to batch x
        """

        # input N x C x H x W
        # to apply it in for rgb maybe not take diff noise for each channel? (input.shape[1] should be 1)
        ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], self.noise_res, self.noise_res),
                          std=self.noise_std).cuda()

        ns = F.interpolate(ns, size=self.opt.model['input_size'], mode='bilinear', align_corners=True)

        # Roll to randomly translate the generated noise.
        roll_x = random.choice(range(self.opt.model['input_size']))
        roll_y = random.choice(range(self.opt.model['input_size']))
        ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])

        # Use foreground mask for MRI, to only apply noise in the foreground.
        if self.opt.dataset in ['brain', 'brats']:
            mask = (x > x.min())
            ns *= mask
        # if config.center:
        ns = (ns - 0.5) * 2
        res = x + ns

        return res, ns
