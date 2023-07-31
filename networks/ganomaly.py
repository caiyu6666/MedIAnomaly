import torch.nn as nn
from networks.base_units.blocks import BasicBlock, BottleNeck


class Encoder(nn.Module):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
                 block_depth=1):
        super(Encoder, self).__init__()
        fm = input_size // 16  # down-sample for 4 times. 2^4=16

        self.en_block1 = BasicBlock(in_planes, 1 * base_width * expansion, block_depth, downsample=True)

        self.en_block2 = BasicBlock(1 * base_width * expansion, 2 * base_width * expansion, block_depth,
                                    downsample=True)
        self.en_block3 = BasicBlock(2 * base_width * expansion, 4 * base_width * expansion, block_depth,
                                    downsample=True)
        self.en_block4 = BasicBlock(4 * base_width * expansion, 4 * base_width * expansion, block_depth,
                                    downsample=True)

        self.linear_enc = nn.Sequential(
            nn.Linear((4 * base_width * expansion) * fm * fm, mid_num),
            nn.BatchNorm1d(mid_num),
            nn.ReLU(True),
            nn.Linear(mid_num, latent_size))

    def forward(self, x):
        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        en4 = self.en_block4(en3)

        en4 = en4.view(en4.size(0), -1)
        z = self.linear_enc(en4)

        return z


class Decoder(nn.Module):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
                 block_depth=1):
        super(Decoder, self).__init__()
        self.fm = input_size // 16  # down-sample for 4 times. 2^4=16

        self.channels = 4 * base_width * expansion
        self.linear_dec = nn.Sequential(
            nn.Linear(latent_size, mid_num),
            nn.BatchNorm1d(mid_num),
            nn.ReLU(True),
            nn.Linear(mid_num, (4 * base_width * expansion) * self.fm * self.fm))

        self.de_block1 = BasicBlock(4 * base_width * expansion, 4 * base_width * expansion, block_depth,
                                    upsample=True)
        self.de_block2 = BasicBlock(4 * base_width * expansion, 2 * base_width * expansion, block_depth,
                                    upsample=True)
        self.de_block3 = BasicBlock(2 * base_width * expansion, 1 * base_width * expansion, block_depth,
                                    upsample=True)
        self.de_block4 = BasicBlock(1 * base_width * expansion, in_planes, block_depth, upsample=True,
                                    last_layer=True)

    def forward(self, x):
        de4 = self.linear_dec(x)
        de4 = de4.view(x.size(0), self.channels, self.fm, self.fm)

        de3 = self.de_block1(de4)
        de2 = self.de_block2(de3)
        de1 = self.de_block3(de2)
        x_hat = self.de_block4(de1)

        return x_hat


class Generator(nn.Module):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
                 en_num_layers=1, de_num_layers=1):
        super(Generator, self).__init__()
        self.encoder1 = Encoder(input_size=input_size, in_planes=in_planes, base_width=base_width, expansion=expansion,
                                mid_num=mid_num, latent_size=latent_size, block_depth=en_num_layers)
        self.decoder = Decoder(input_size=input_size, in_planes=in_planes, base_width=base_width, expansion=expansion,
                               mid_num=mid_num, latent_size=latent_size, block_depth=de_num_layers)
        self.encoder2 = Encoder(input_size=input_size, in_planes=in_planes, base_width=base_width, expansion=expansion,
                                mid_num=mid_num, latent_size=latent_size, block_depth=en_num_layers)

    def forward(self, x):
        z = self.encoder1(x)
        x_hat = self.decoder(z)
        z_hat = self.encoder2(x_hat)
        return x_hat, z, z_hat


class Discriminator(nn.Module):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, out_size=1,
                 block_depth=1):
        super(Discriminator, self).__init__()

        fm = input_size // 16  # down-sample for 4 times. 2^4=16

        self.en_block1 = BasicBlock(in_planes, 1 * base_width * expansion, block_depth, downsample=True)

        self.en_block2 = BasicBlock(1 * base_width * expansion, 2 * base_width * expansion, block_depth,
                                    downsample=True)
        self.en_block3 = BasicBlock(2 * base_width * expansion, 4 * base_width * expansion, block_depth,
                                    downsample=True)
        self.en_block4 = BasicBlock(4 * base_width * expansion, 4 * base_width * expansion, block_depth,
                                    downsample=True)

        self.features = nn.Sequential(
            nn.Linear((4 * base_width * expansion) * fm * fm, mid_num),
            nn.BatchNorm1d(mid_num),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(mid_num, out_size),
            nn.Sigmoid()  # probability
        )

    def forward(self, x):
        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        en4 = self.en_block4(en3)

        en4 = en4.view(en4.size(0), -1)

        features = self.features(en4)
        pred = self.classifier(features).squeeze(1)  # probability

        return pred, features


def weights_init(m):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Ganomaly(nn.Module):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
                 en_num_layers=1, de_num_layers=1):
        super(Ganomaly, self).__init__()

        self.fm = input_size // 16  # down-sample for 4 times. 2^4=16

        self.netg = Generator(input_size=input_size, in_planes=in_planes, base_width=base_width, expansion=expansion,
                              mid_num=mid_num, latent_size=latent_size, en_num_layers=en_num_layers,
                              de_num_layers=de_num_layers)
        self.netd = Discriminator(input_size=input_size, in_planes=in_planes, base_width=base_width,
                                  expansion=expansion, mid_num=mid_num, out_size=1, block_depth=de_num_layers)
        # netd output the prob.

    def forward(self, x):
        x_hat, z, z_hat = self.netg(x)

        pred_real, feat_real = self.netd(x)
        pred_fake, feat_fake = self.netd(x_hat)
        pred_fake_detach, feat_fake_detach = self.netd(x_hat.detach())
        return {'x_hat': x_hat, 'z': z, 'z_hat': z_hat, 'pred_real': pred_real, 'feat_real': feat_real,
                'pred_fake': pred_fake, 'feat_fake': feat_fake,
                'pred_fake_detach': pred_fake_detach, 'feat_fake_detach': feat_fake_detach}

    def reinit_d(self):
        """ Re-initialize the weights of netD
        """
        self.netd.apply(weights_init)
        # print('   Reloading net d')
