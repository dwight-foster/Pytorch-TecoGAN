from dataloader import *
import torch.nn.functional as F


def down_block(inputs, output_channel=64, stride=1):
    net = nn.Sequential(conv2(inputs, 3, output_channel, stride, use_bias=True), lrelu(0.2),
                        conv2(output_channel, 3, output_channel, stride, use_bias=True)
                        , lrelu(0.2), maxpool())

    return net


def up_block(inputs, output_channel=64, stride=1):
    net = nn.Sequential(conv2(inputs, 3, output_channel, stride, use_bias=True), lrelu(0.2),
                        conv2(output_channel, 3, output_channel, stride, use_bias=True)
                        , lrelu(0.2), nn.Upsample(scale_factor=2, mode="bilinear"))

    return net


class f_net(nn.Module):
    def __init__(self):
        super(f_net, self).__init__()
        self.down1 = down_block(6, 32)
        self.down2 = down_block(32, 64)
        self.down3 = down_block(64, 128)

        self.up1 = up_block(128, 256)
        self.up2 = up_block(256, 128)
        self.up3 = up_block(128, 64)

        self.output_block = nn.Sequential(conv2(64, 3, 32, 1), lrelu(0.2), conv2(32, 3, 2, 1))

    def forward(self, x):
        net = self.down1(x)
        net = self.down2(net)
        net = self.down3(net)
        net = self.up1(net)
        net = self.up2(net)
        net = self.up3(net)

        net = self.output_block(net)
        net = torch.tanh(net) * 24.0

        return net


def residual_block(inputs, output_channel=64, stride=1):
    net = nn.Sequential(conv2(inputs, 3, output_channel, stride, use_bias=True), nn.ReLU(),
                        conv2(output_channel, 3, output_channel, stride, use_bias=False))

    return net


class generator(nn.Module):
    def __init__(self, gen_output_channels, FLAGS=None):
        super(generator, self).__init__()

        if FLAGS is None:
            raise ValueError("No FLAGS is provided for generator")

        self.conv = nn.Sequential(conv2(51, 3, 64, 1), nn.ReLU())
        self.num = FLAGS.num_resblock
        self.resid = residual_block(64, 64, 1)
        self.conv_trans = nn.Sequential(conv2_tran(64, 3, 64, stride=2, output_padding=1), nn.ReLU()
                                        ,conv2_tran(64, 3, 64, stride=2, output_padding=1), nn.ReLU())
        self.output = conv2(64, 3, gen_output_channels, 1)

    def forward(self, x):
        net = self.conv(x)

        for i in range(1, self.num + 1, 1):
            net = self.resid(net)
        net = self.conv_trans(net)
        net = self.output(net)

        low_res_in = x[:, 0:3, :, :]
        bicubic_hi = bicubic_four(low_res_in)
        net = net + bicubic_hi
        net = preprocess(net)
        return net
