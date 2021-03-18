from dataloader import *
import torch.nn.functional as F


# Defining the fnet model for image warping
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
        self.down1 = down_block(3, 32)
        self.down2 = down_block(32, 64)
        self.down3 = down_block(64, 128)
        self.down4 = down_block(128, 256)

        self.up1 = up_block(256, 512)
        self.up2 = up_block(512, 256)
        self.up3 = up_block(256, 128)
        self.up4 = up_block(128, 64)

        self.output_block = nn.Sequential(conv2(64, 3, 32, 1), lrelu(0.2), conv2(32, 3, 2, 1))

    def forward(self, x):
        net = self.down1(x)
        net = self.down2(net)
        net = self.down3(net)
        net = self.down4(net)
        net = self.up1(net)
        net = self.up2(net)
        net = self.up3(net)
        net = self.up4(net)

        net = self.output_block(net)
        net = torch.tanh(net) * 24.0

        return net


# Defining the generator to upscale images
def residual_block(inputs, output_channel=64, stride=1):
    net = nn.Sequential(conv2(inputs, 3, output_channel, stride, use_bias=True), nn.ReLU(),
                        conv2(output_channel, 3, output_channel, stride, use_bias=False))

    return net


class generator(nn.Module):
    def __init__(self, gen_output_channels, args=None):
        super(generator, self).__init__()

        if args is None:
            raise ValueError("No args is provided for generator")

        self.conv = nn.Sequential(conv2(51, 3, 64, 1), nn.ReLU())
        self.num = args.num_resblock
        self.resids = nn.ModuleList([residual_block(64, 64, 1) for i in range(int(self.num))])

        self.conv_trans = nn.Sequential(conv2_tran(64, 3, 64, stride=2, output_padding=1), nn.ReLU()
                                        , residual_block(64, 64, 1), residual_block(64, 128, 1),
                                        conv2_tran(128, 3, 128, stride=2, output_padding=1), nn.ReLU(),
                                        conv2(128, 3, 64, 1), nn.ReLU())
        self.output = conv2(64, 3, gen_output_channels, 1)

    def forward(self, x):
        net = self.conv(x)

        for block in self.resids:
            net = block(net) + net
        net = self.conv_trans(net)
        net = self.output(net)

        return torch.sigmoid(net)


# Defining the discriminator for adversarial part
def discriminator_block(inputs, output_channel, kernel_size, stride):
    net = nn.Sequential(conv2(inputs, kernel_size, output_channel, stride, use_bias=False),
                        batchnorm(output_channel, is_training=True),
                        lrelu(0.2))
    return net


class discriminator(nn.Module):
    def __init__(self, args=None):
        super(discriminator, self).__init__()
        if args is None:
            raise ValueError("No args is provided for discriminator")
        self.conv = nn.Sequential(conv2(27, 3, 64, 1), lrelu(0.2))
        # block1
        self.block1 = discriminator_block(64, 64, 4, 2)
        self.resids1 = nn.ModuleList(
            [nn.Sequential(residual_block(64, 64, 1), batchnorm(64, True)) for i in range(int(args.discrim_resblocks))])

        # block2
        self.block2 = discriminator_block(64, args.discrim_channels, 4, 2)
        self.resids2 = nn.ModuleList([nn.Sequential(residual_block(args.discrim_channels, args.discrim_channels, 1),
                                                    batchnorm(args.discrim_channels, True)) for i in range(int(args.discrim_resblocks))])

        # block3
        self.block3 = discriminator_block(args.discrim_channels, args.discrim_channels, 4, 2)
        self.resids3 = nn.ModuleList([nn.Sequential(residual_block(args.discrim_channels, args.discrim_channels, 1),
                                                    batchnorm(args.discrim_channels, True)) for i in
                                      range(int(args.discrim_resblocks))])

        self.block4 = discriminator_block(args.discrim_channels, 64, 4, 2)

        self.block5 = discriminator_block(64, 3, 4, 2)

        self.fc = denselayer(48, 1)

    def forward(self, x):
        layer_list = []
        net = self.conv(x)
        net = self.block1(net)
        for block in self.resids1:
            net = block(net) + net
        layer_list.append(net)
        net = self.block2(net)
        for block in self.resids2:
            net = block(net) + net
        layer_list.append(net)
        net = self.block3(net)
        for block in self.resids3:
            net = block(net) + net
        layer_list.append(net)
        net = self.block4(net)
        layer_list.append(net)
        net = self.block5(net)
        net = net.view(net.shape[0], -1)
        net = self.fc(net)
        net = torch.sigmoid(net)
        return net, layer_list
