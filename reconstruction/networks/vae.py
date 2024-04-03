from networks.ae import AE
from networks.base_units.blocks import VaeBottleNeck


class VAE(AE):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
                 en_num_layers=None, de_num_layers=None):
        super(VAE, self).__init__(input_size, in_planes, base_width, expansion, mid_num, latent_size, en_num_layers,
                                  de_num_layers)
        self.bottle_neck = VaeBottleNeck(4 * base_width * expansion, feature_size=self.fm, mid_num=mid_num,
                                         latent_size=latent_size)

    def forward(self, x):
        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        en4 = self.en_block4(en3)

        bottle_out = self.bottle_neck(en4)
        de4, mu, log_var = bottle_out['out'], bottle_out['mu'], bottle_out["log_var"]

        de3 = self.de_block1(de4)
        de2 = self.de_block2(de3)
        de1 = self.de_block3(de2)
        x_hat = self.de_block4(de1)

        return {'x_hat': x_hat, 'log_var': log_var, 'mu': mu,
                'en_features': [en1, en2, en3], 'de_features': [de1, de2, de3]}
