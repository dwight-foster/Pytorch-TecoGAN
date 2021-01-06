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
                        , lrelu(0.2))

    return net


class f_net(nn.Module):
    def __init__(self):
        super(f_net, self).__init__()
        self.down1 = down_block(3, 32)
        self.down2 = down_block(64, 64)
        self.down3 = down_block(128, 128)

        self.up1 = up_block(128, 256)
        self.up2 = up_block(256, 128)
        self.up3 = up_block(128, 64)

        self.output_block = nn.Sequential(conv2(64, 3, 32, 1), lrelu(0.2), conv2(32, 3, 2, 1))

    def forward(self, x):
        net = self.down1(x)
        net = self.down2(net)
        net = self.down3(net)
        net = self.up1(net)
        net = net.view(net.shape[1:-1] * 2)
        net = self.up2(net)
        net = net.view(net.shape[1:-1] * 2)
        net = self.up3(net)
        net = net.view(net.shape[1:-1] * 2)
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

        self.conv = nn.Sequential(conv2(12, 3, 64, 1), nn.ReLU())
        self.num = FLAGS.num_resblock
        self.resid = {}
        for i in range(1, FLAGS.num_resblock + 1, 1):
            if i == 1:
                net = nn.Sequential(residual_block(64, 64, 1), conv2_tran(64, 3, 64, 2), nn.ReLU(),
                                    conv2_tran(64, 3, 64, 2), nn.ReLU(), conv2(64, 3, gen_output_channels, 1))
            else:
                net = nn.Sequential(residual_block(gen_output_channels, 64, 1), conv2_tran(64, 3, 64, 2), nn.ReLU(),
                                    conv2_tran(64, 3, 64, 2), nn.ReLU(), conv2(64, 3, gen_output_channels, 1))

            self.resid[str(i)] = net

    def forward(self, x):
        net = self.conv(x)
        for i in range(1, self.num + 1, 1):
            resid = self.resid[str(i)]
            net = resid(net)
            low_res_in = x[:, 0:3, :, :]
            bicubic_hi = bicubic_four(low_res_in)
            net = net + bicubic_hi
            net = preprocess(net)
        return net
