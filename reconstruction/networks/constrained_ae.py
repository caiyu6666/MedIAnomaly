from networks.ae import AE


class ConstrainedAE(AE):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
                 en_num_layers=None, de_num_layers=None):
        super(ConstrainedAE, self).__init__(input_size, in_planes, base_width, expansion, mid_num, latent_size,
                                            en_num_layers, de_num_layers)

    def forward(self, x, istrain=False):
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

        if istrain:
            # Latent code of x_hat
            en1_rec = self.en_block1(x_hat)
            en2_rec = self.en_block2(en1_rec)
            en3_rec = self.en_block3(en2_rec)
            en4_rec = self.en_block4(en3_rec)

            bottle_out_rec = self.bottle_neck(en4_rec)
            z_rec = bottle_out_rec['z']
        else:
            z_rec = None

        return {'x_hat': x_hat, 'z': z, 'z_rec': z_rec,
                'en_features': [en1, en2, en3], 'de_features': [de1, de2, de3]}
