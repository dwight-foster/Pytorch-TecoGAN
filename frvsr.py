from dataloader import *
import torch.nn.functional as F


def fnet(fnet_input, reuse=False):
    def down_block(inputs, output_channel=64, stride=1):
        net = conv2(inputs, 3, output_channel, stride, use_bias=True)
        net = lrelu(net, 0.2)
        net = conv2(net, 3, output_channel, stride, use_bias=True)
        net = lrelu(net, 0.2)
        net = maxpool(net)

        return net

    def up_block(inputs, output_channel=64, stride=1):
        net = conv2(inputs, 3, output_channel, stride, use_bias=True)
        net = lrelu(net, 0.2)
        net = conv2(net, 3, output_channel, stride, use_bias=True)
        net = lrelu(net, 0.2)
        net = net.view(net.shape[1:-1] * 2)
        return net

    net = down_block(fnet_input, 32)
    net = down_block(net, 64)
    net = down_block(net, 128)

    net = up_block(net, 256)
    net = up_block(net, 128)
    net1 = up_block(net, 64)

    net = conv2(net1, 3, 32, 1)
    net = lrelu(net, 0.2)
    net2 = conv2(net, 3, 2, 1)
    net = torch.tanh(net2) * 24.0

    return net


def generator_F(gen_inputs, gen_output_channels, reuse=False, FLAGS=None):
    if FLAGS is None:
        raise ValueError("No FLAGS is provided for generator")

    def residual_block(inputs, output_channel=64, stride=1):
        net = conv2(inputs, 3, output_channel, stride, use_bias=True)
        net = F.relu(net)
        net = conv2(net, 3, output_channel, stride, use_bias=False)
        net = net + inputs
        return net

    net = conv2(gen_inputs, 3, 64, 1)
    stage1_output = F.relu(net)
    net = stage1_output

    for i in range(1, FLAGS.num_resblock + 1, 1):
        net = residual_block(net, 64, 1)
        net = conv2_tran(net, 3, 64, 2)
        net = F.relu(net)
        net = conv2_tran(net, 3, 64, 2)
        net = F.relu(net)
        net = conv2(net, 3, gen_output_channels, 1)
        low_res_in = gen_inputs[:, 0:3, :, :]
        bicubic_hi = bicubic_four(low_res_in)
        net = net + bicubic_hi
        net = preprocess(net)
    return net
