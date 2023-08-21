import torch.nn as nn
from networks.base_units.blocks import BasicBlock
import torch

# class Encoder(nn.Module):
#     def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
#                  block_depth=1):
#         super(Encoder, self).__init__()
#         fm = input_size // 16  # down-sample for 4 times. 2^4=16
#
#         self.en_block1 = BasicBlock(in_planes, 1 * base_width * expansion, block_depth, downsample=True)
#
#         self.en_block2 = BasicBlock(1 * base_width * expansion, 2 * base_width * expansion, block_depth,
#                                     downsample=True)
#         self.en_block3 = BasicBlock(2 * base_width * expansion, 4 * base_width * expansion, block_depth,
#                                     downsample=True)
#         self.en_block4 = BasicBlock(4 * base_width * expansion, 4 * base_width * expansion, block_depth,
#                                     downsample=True)
#
#         self.linear_enc = nn.Sequential(
#             nn.Linear((4 * base_width * expansion) * fm * fm, mid_num),
#             nn.BatchNorm1d(mid_num),
#             nn.ReLU(True),
#             nn.Linear(mid_num, latent_size)
#         )
#
#     def forward(self, x):
#         en1 = self.en_block1(x)
#         en2 = self.en_block2(en1)
#         en3 = self.en_block3(en2)
#         en4 = self.en_block4(en3)
#
#         en4 = en4.view(en4.size(0), -1)
#         return self.linear_enc(en4)
#
#
# class Generator(nn.Module):
#     def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
#                  block_depth=1):
#         super(Generator, self).__init__()
#         self.fm = input_size // 16  # down-sample for 4 times. 2^4=16
#
#         self.channels = 4 * base_width * expansion
#         self.linear_dec = nn.Sequential(
#             nn.Linear(latent_size, mid_num),
#             nn.BatchNorm1d(mid_num),
#             nn.ReLU(True),
#             nn.Linear(mid_num, (4 * base_width * expansion) * self.fm * self.fm))
#
#         self.de_block1 = BasicBlock(4 * base_width * expansion, 4 * base_width * expansion, block_depth,
#                                     upsample=True)
#         self.de_block2 = BasicBlock(4 * base_width * expansion, 2 * base_width * expansion, block_depth,
#                                     upsample=True)
#         self.de_block3 = BasicBlock(2 * base_width * expansion, 1 * base_width * expansion, block_depth,
#                                     upsample=True)
#         self.de_block4 = BasicBlock(1 * base_width * expansion, in_planes, block_depth, upsample=True,
#                                     last_layer=True)
#
#     def forward(self, x):
#         de4 = self.linear_dec(x)
#         de4 = de4.view(x.size(0), self.channels, self.fm, self.fm)
#
#         de3 = self.de_block1(de4)
#         de2 = self.de_block2(de3)
#         de1 = self.de_block3(de2)
#         x_hat = self.de_block4(de1)
#
#         return x_hat
#
#
# class Discriminator(nn.Module):
#     def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
#                  block_depth=1):
#         super(Discriminator, self).__init__()
#         fm = input_size // 16  # down-sample for 4 times. 2^4=16
#
#         self.en_block1 = BasicBlock(in_planes, 1 * base_width * expansion, block_depth, downsample=True)
#
#         self.en_block2 = BasicBlock(1 * base_width * expansion, 2 * base_width * expansion, block_depth,
#                                     downsample=True)
#         self.en_block3 = BasicBlock(2 * base_width * expansion, 4 * base_width * expansion, block_depth,
#                                     downsample=True)
#         self.en_block4 = BasicBlock(4 * base_width * expansion, 4 * base_width * expansion, block_depth,
#                                     downsample=True)
#
#         self.feature = nn.Sequential(
#             nn.Linear((4 * base_width * expansion) * fm * fm, mid_num),
#             nn.BatchNorm1d(mid_num),
#             nn.ReLU(True)
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Linear(mid_num, 1)
#         )
#
#     def forward_features(self, x):
#         en1 = self.en_block1(x)
#         en2 = self.en_block2(en1)
#         en3 = self.en_block3(en2)
#         en4 = self.en_block4(en3)
#
#         en4 = en4.view(en4.size(0), -1)
#         return self.feature(en4)
#
#     def forward(self, x):
#         feature = self.forward_features(x)
#         pred = self.classifier(feature)
#
#         return pred

DIM = 64
OUTPUT_DIM = 64*64*1


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height,
                             input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size, input_height,
                              output_width, output_depth) for t_t in spl]
        output = torch.stack(stacks, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(
            batch_size, output_height, output_width, output_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class UpSampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True, bias=True):
        super(UpSampleConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim,
                              kernel_size, he_init=self.he_init, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output


class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True, stride=1, bias=True):
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1)/2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size,
                              stride=1, padding=self.padding, bias=bias)

    def forward(self, input):
        output = self.conv(input)
        return output


class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True):
        super(ConvMeanPool, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim,
                              kernel_size, he_init=self.he_init)

    def forward(self, input):
        output = self.conv(input)
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] +
                  output[:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]) / 4
        return output


class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True):
        super(MeanPoolConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim,
                              kernel_size, he_init=self.he_init)

    def forward(self, input):
        output = input
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] +
                  output[:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]) / 4
        output = self.conv(output)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, resample=None, hw=DIM):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if resample == 'down':
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        elif resample == 'up':
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
        elif resample == None:
            # TODO: ????
            self.bn1 = nn.BatchNorm2d(output_dim)
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        else:
            raise Exception('invalid resample value')

        if resample == 'down':
            self.conv_shortcut = MeanPoolConv(
                input_dim, output_dim, kernel_size=1, he_init=False)
            self.conv_1 = MyConvo2d(
                input_dim, input_dim, kernel_size=kernel_size, bias=False)
            self.conv_2 = ConvMeanPool(
                input_dim, output_dim, kernel_size=kernel_size)
        elif resample == 'up':
            self.conv_shortcut = UpSampleConv(
                input_dim, output_dim, kernel_size=1, he_init=False)
            self.conv_1 = UpSampleConv(
                input_dim, output_dim, kernel_size=kernel_size, bias=False)
            self.conv_2 = MyConvo2d(
                output_dim, output_dim, kernel_size=kernel_size)
        elif resample == None:
            self.conv_shortcut = MyConvo2d(
                input_dim, output_dim, kernel_size=1, he_init=False)
            self.conv_1 = MyConvo2d(
                input_dim, input_dim, kernel_size=kernel_size, bias=False)
            self.conv_2 = MyConvo2d(
                input_dim, output_dim, kernel_size=kernel_size)
        else:
            raise Exception('invalid resample value')

    def forward(self, input):
        if self.input_dim == self.output_dim and self.resample == None:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)

        output = input
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv_1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output


class Generator(nn.Module):
    def __init__(self, dim=DIM, output_dim=OUTPUT_DIM):
        super(Generator, self).__init__()

        self.dim = dim

        self.ln1 = nn.Linear(128, 4*4*8*self.dim)
        self.rb1 = ResidualBlock(8*self.dim, 8*self.dim, 3, resample='up')
        self.rb2 = ResidualBlock(8*self.dim, 4*self.dim, 3, resample='up')
        self.rb3 = ResidualBlock(4*self.dim, 2*self.dim, 3, resample='up')
        self.rb4 = ResidualBlock(2*self.dim, 1*self.dim, 3, resample='up')
        self.bn = nn.BatchNorm2d(self.dim)

        self.conv1 = MyConvo2d(1*self.dim, 1, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        output = self.ln1(input.contiguous())
        output = output.view(-1, 8*self.dim, 4, 4)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)

        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.tanh(output)
        # output = output.view(-1, OUTPUT_DIM)
        return output


class Discriminator(nn.Module):
    def __init__(self, dim=DIM):
        super(Discriminator, self).__init__()

        self.dim = dim

        # self.conv1 = MyConvo2d(3, self.dim, 3, he_init=False)
        self.conv1 = MyConvo2d(1, self.dim, 3, he_init=False)
        self.rb1 = ResidualBlock(self.dim, 2*self.dim,
                                 3, resample='down', hw=DIM)
        self.rb2 = ResidualBlock(
            2*self.dim, 4*self.dim, 3, resample='down', hw=int(DIM/2))
        self.rb3 = ResidualBlock(
            4*self.dim, 8*self.dim, 3, resample='down', hw=int(DIM/4))
        self.rb4 = ResidualBlock(
            8*self.dim, 8*self.dim, 3, resample='down', hw=int(DIM/8))
        self.ln1 = nn.Linear(4*4*8*self.dim, 1)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, input):
        output = input.contiguous()
        output = output.view(-1, 1, DIM, DIM)
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = output.view(-1, 4*4*8*self.dim)
        return output

    def forward(self, input):
        output = self.forward_features(input)
        output = self.ln1(output)
        output = output.view(-1)
        return output


class Encoder(nn.Module):
    def __init__(self, dim, output_dim, drop_rate=0.0):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.conv_in = nn.Conv2d(1, dim, 3, 1, padding=1)
        self.res1 = ResidualBlock(dim, dim*2, 3, 'down', 64)
        self.res2 = ResidualBlock(dim*2, dim*4, 3, 'down', 32)
        self.res3 = ResidualBlock(dim*4, dim*8, 3, 'down', 16)
        self.res4 = ResidualBlock(dim*8, dim*8, 3, 'down', 8)
        self.fc = nn.Linear(4*4*8*dim, output_dim)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv_in(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.tanh(x)


class FAnoGAN(nn.Module):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
                 en_num_layers=1, de_num_layers=1):
        super(FAnoGAN, self).__init__()
        # self.encoder = Encoder(input_size=input_size, in_planes=in_planes, base_width=base_width, expansion=expansion,
        #                        mid_num=mid_num, latent_size=latent_size, block_depth=en_num_layers)
        #
        # self.generator = Generator(input_size=input_size, in_planes=in_planes, base_width=base_width,
        #                            expansion=expansion, mid_num=mid_num, latent_size=latent_size,
        #                            block_depth=de_num_layers)
        #
        # self.discriminator = Discriminator(input_size=input_size, in_planes=in_planes, base_width=base_width,
        #                                    expansion=expansion, mid_num=mid_num, block_depth=de_num_layers)
        self.encoder = Encoder(dim=64, output_dim=latent_size)
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def initialize_encoder(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = self.encoder(x)
        x_fake = self.generator(z)

        feat_real = self.discriminator.forward_features(x)
        feat_fake = self.discriminator.forward_features(x_fake)

        return x_fake, feat_real, feat_fake


