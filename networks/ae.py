import torch.nn as nn
from networks.base_units.blocks import BasicBlock, BottleNeck


class AE(nn.Module):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
                 en_num_layers=None, de_num_layers=None):
        super(AE, self).__init__()

        self.fm = input_size // 16  # down-sample for 4 times. 2^4=16

        if en_num_layers is None:
            en_num_layers = 1
        if de_num_layers is None:
            de_num_layers = 1

        self.en_block1 = BasicBlock(in_planes, 1 * base_width * expansion, en_num_layers, downsample=True)

        self.en_block2 = BasicBlock(1 * base_width * expansion, 2 * base_width * expansion, en_num_layers,
                                    downsample=True)
        self.en_block3 = BasicBlock(2 * base_width * expansion, 4 * base_width * expansion, en_num_layers,
                                    downsample=True)
        self.en_block4 = BasicBlock(4 * base_width * expansion, 4 * base_width * expansion, en_num_layers,
                                    downsample=True)

        self.bottle_neck = BottleNeck(4 * base_width * expansion, feature_size=self.fm, mid_num=mid_num,
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


# class AE(nn.Module):
#     def __init__(self, latent_size=16, multiplier=1, img_size=64, **kwargs):
#         super(AE, self).__init__()
#         self.fm = img_size // 16
#         self.mp = multiplier
#
#         self.en_layer1 = nn.Sequential(
#             nn.Conv2d(1, int(16 * multiplier), 4, 2, 1, bias=False),
#             nn.BatchNorm2d(int(16 * multiplier)),
#             nn.ReLU(True),
#
#             # nn.Conv2d(int(16 * multiplier), int(16 * multiplier), 3, 1, 1, bias=False),
#             # nn.BatchNorm2d(int(16 * multiplier)),
#             # nn.ReLU(True),
#         )
#
#         self.en_layer2 = nn.Sequential(
#             nn.Conv2d(int(16 * multiplier), int(32 * multiplier), 4, 2, 1, bias=False),
#             nn.BatchNorm2d(int(32 * multiplier)),
#             nn.ReLU(True),
#
#             # nn.Conv2d(int(32 * multiplier), int(32 * multiplier), 3, 1, 1, bias=False),
#             # nn.BatchNorm2d(int(32 * multiplier)),
#             # nn.ReLU(True),
#         )
#
#         self.en_layer3 = nn.Sequential(
#             nn.Conv2d(int(32 * multiplier), int(64 * multiplier), 4, 2, 1, bias=False),
#             nn.BatchNorm2d(int(64 * multiplier)),
#             nn.ReLU(True),
#
#             # nn.Conv2d(int(64 * multiplier), int(64 * multiplier), 3, 1, 1, bias=False),
#             # nn.BatchNorm2d(int(64 * multiplier)),
#             # nn.ReLU(True),
#         )
#
#         self.en_layer4 = nn.Sequential(
#             nn.Conv2d(int(64 * multiplier), int(64 * multiplier), 4, 2, 1, bias=False),
#             nn.BatchNorm2d(int(64 * multiplier)),
#             nn.ReLU(True),
#
#             # nn.Conv2d(int(64 * multiplier), int(64 * multiplier), 3, 1, 1, bias=False),
#             # nn.BatchNorm2d(int(64 * multiplier)),
#             # nn.ReLU(True),
#         )
#
#         self.linear_enc = nn.Sequential(
#             nn.Linear(int(64 * multiplier) * self.fm * self.fm, 2048),
#             nn.BatchNorm1d(2048),
#             nn.ReLU(True),
#             nn.Linear(2048, latent_size))
#
#         # self.bottle_neck = BottleNeck(multiplier=multiplier, ls=latent_size)
#
#         # ------------------------------------------- Decoder ------------------------------------------- #
#         self.linear_dec = nn.Sequential(
#             nn.Linear(latent_size, 2048),
#             nn.BatchNorm1d(2048),
#             nn.ReLU(True),
#             nn.Linear(2048, int(64 * multiplier) * self.fm * self.fm))
#
#         self.de_layer1 = nn.Sequential(
#             nn.ConvTranspose2d(int(64 * multiplier), int(64 * multiplier), 4, 2, 1, bias=False),
#             nn.BatchNorm2d(int(64 * multiplier)),
#             nn.ReLU(True),
#
#             # nn.Conv2d(int(64 * multiplier), int(64 * multiplier), 3, 1, 1, bias=False),
#             # nn.BatchNorm2d(int(64 * multiplier)),
#             # nn.ReLU(True),
#         )
#         self.de_layer2 = nn.Sequential(
#             nn.ConvTranspose2d(int(64 * multiplier), int(32 * multiplier), 4, 2, 1, bias=False),
#             nn.BatchNorm2d(int(32 * multiplier)),
#             nn.ReLU(True),
#
#             # nn.Conv2d(int(32 * multiplier), int(32 * multiplier), 3, 1, 1, bias=False),
#             # nn.BatchNorm2d(int(32 * multiplier)),
#             # nn.ReLU(True),
#         )
#         self.de_layer3 = nn.Sequential(
#             nn.ConvTranspose2d(int(32 * multiplier), int(16 * multiplier), 4, 2, 1, bias=False),
#             nn.BatchNorm2d(int(16 * multiplier)),
#             nn.ReLU(True),
#
#             # nn.Conv2d(int(16 * multiplier), int(16 * multiplier), 3, 1, 1, bias=False),
#             # nn.BatchNorm2d(int(16 * multiplier)),
#             # nn.ReLU(True),
#         )
#
#         self.de_layer4 = nn.ConvTranspose2d(int(16 * multiplier), 1, 4, 2, 1, bias=False)
#
#     def forward(self, x, feature_map=False, bottle_neck=False):
#         lat_rep, en_features = self.feature(x, feature_map=feature_map)
#         decoder_in = lat_rep
#         # out, de_features = self.decode(lat_rep, feature_map=feature_map)
#         out, de_features = self.decode(decoder_in, feature_map=feature_map)
#         return {'x_hat': out}
#         # return lat_rep, decoder_in, out, en_features, de_features
#
#     def feature(self, x, feature_map=False):
#         # lat_rep = self.encoder(x)
#         e1 = self.en_layer1(x)  # 16
#         e2 = self.en_layer2(e1)  # 32
#         e3 = self.en_layer3(e2)  # 64
#         e4 = self.en_layer4(e3)  # 64
#
#         # lat_rep = lat_rep.view(lat_rep.size(0), -1)
#         lat_rep = e4.view(e4.size(0), -1)
#         lat_rep = self.linear_enc(lat_rep)
#
#         feat = [e1, e2, e3] if feature_map else None
#         return lat_rep, feat
#
#     def decode(self, x, feature_map=False):
#         d4 = self.linear_dec(x)
#         d4 = d4.view(d4.size(0), int(64 * self.mp), self.fm, self.fm)  # 64
#         d3 = self.de_layer1(d4)  # 64
#         d2 = self.de_layer2(d3)  # 32
#         d1 = self.de_layer3(d2)  # 16
#         out = self.de_layer4(d1)  # 1
#
#         feat = [d1, d2, d3] if feature_map else None
#         return out, feat
