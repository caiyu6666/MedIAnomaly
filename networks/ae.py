import torch.nn as nn
from networks.base_units.blocks import BasicBlock, BottleNeck, SpatialBottleNeck


class AE(nn.Module):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
                 en_num_layers=1, de_num_layers=1, spatial=False):
        super(AE, self).__init__()

        bottleneck = SpatialBottleNeck if spatial else BottleNeck

        self.fm = input_size // 16  # down-sample for 4 times. 2^4=16

        self.en_block1 = BasicBlock(in_planes, 1 * base_width * expansion, en_num_layers, downsample=True)

        self.en_block2 = BasicBlock(1 * base_width * expansion, 2 * base_width * expansion, en_num_layers,
                                    downsample=True)
        self.en_block3 = BasicBlock(2 * base_width * expansion, 4 * base_width * expansion, en_num_layers,
                                    downsample=True)
        self.en_block4 = BasicBlock(4 * base_width * expansion, 4 * base_width * expansion, en_num_layers,
                                    downsample=True)

        # self.bottle_neck = BottleNeck(4 * base_width * expansion, feature_size=self.fm, mid_num=mid_num,
        #                               latent_size=latent_size)
        self.bottle_neck = bottleneck(4 * base_width * expansion, feature_size=self.fm, mid_num=mid_num,
                                      latent_size=latent_size)

        self.de_block1 = BasicBlock(4 * base_width * expansion, 4 * base_width * expansion, de_num_layers,
                                    upsample=True)
        self.de_block2 = BasicBlock(4 * base_width * expansion, 2 * base_width * expansion, de_num_layers,
                                    upsample=True)
        self.de_block3 = BasicBlock(2 * base_width * expansion, 1 * base_width * expansion, de_num_layers,
                                    upsample=True)
        self.de_block4 = BasicBlock(1 * base_width * expansion, in_planes, de_num_layers, upsample=True,
                                    last_layer=True)

    def forward(self, x):
        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        en4 = self.en_block4(en3)

        bottle_out = self.bottle_neck(en4)
        z, de4 = bottle_out['z'], bottle_out['out']

        de3 = self.de_block1(de4)
        de2 = self.de_block2(de3)
        de1 = self.de_block3(de2)
        x_hat = self.de_block4(de1)

        return {'x_hat': x_hat, 'z': z, 'en_features': [en1, en2, en3], 'de_features': [de1, de2, de3]}
