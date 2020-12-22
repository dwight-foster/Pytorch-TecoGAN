from frvsr import *

VGG_MEAN = [123.68, 116.78, 103.94]
identity = torch.nn.Identity()


def VGG19_slim(input, reuse, deep_list=None, norm_flag=True):
    input_img = deprocess(input)
    input_img_ab = input_img * 255.0 - torch.tensor(VGG_MEAN)
    model = VGG19()
    _, output = model(input_img_ab)

    results = {}
    for key in output:
        if deep_list is None or key in deep_list:
            orig_deep_feature = output[key]
            if norm_flag:
                orig_len = torch.sqrt(torch.min(torch.square(orig_deep_feature), dim=1, keepdim=True) + 1e-12)
                results[key] = orig_deep_feature / orig_len
            else:
                results[key] = orig_deep_feature
    return results


def discriminator_F(dis_inputs, FLAGS=None):
    if FLAGS is None:
        raise ValueError("No FLAGS is provided for generator")

    def discriminator_block(inputs, output_channel, kernel_size, stride):
        net = conv2(inputs, kernel_size, stride, use_bias=False)
        net = batchnorm(net, is_training=True)
        net = lrelu(net, 0.2)
        return net

    layer_list = []

    net = conv2(dis_inputs, 3, 64, 1)
    net = lrelu(net, 0.2)

    # block1
    net = discriminator_block(net, 64, 4, 2)
    layer_list += [net]

    # block2
    net = discriminator_block(net, 64, 4, 2)
    layer_list += [net]

    # block3
    net = discriminator_block(net, 128, 4, 2)
    layer_list += [net]

    # block4
    net = discriminator_block(net, 256, 4, 2)
    layer_list += [net]

    net = denselayer(net, 1)
    net = torch.sigmoid(net)

    return net, layer_list


def TecoGAN(r_inputs, r_targets, FLAGS, GAN_FLAG=True):
    inputimages = FLAGS.RNN_N
    if FLAGS.pingpang:
        r_inputs_rev_input = r_inputs[:, -2::-1, :, :, :]
        r_targets_rev_input = r_targets[:, -2::-1, :, :, :]
        r_inputs = torch.cat([r_inputs, r_inputs_rev_input], axis=1)
        r_targets = torch.cat([r_targets, r_targets_rev_input], axis=1)
        inputimages = FLAGS.RNN_N * 2 - 1

    output_channel = list(r_targets.shape())[-1]

    gen_outputs, gen_warppre = [], []
    learning_rate = FLAGS.learning_rate
    Frame_t_pre = r_inputs[:, 0:-1, :, :, :]
    Frame_t = r_inputs[:, 1:, :, :, :]

    fnet_input = torch.cat((Frame_t_pre, Frame_t), dim=2)
    fnet_input = torch.reshape(fnet_input, (
        FLAGS.batch_size * (inputimages - 1), 2 * output_channel, FLAGS.crop_size, FLAGS.crop_size))
    gen_flow_lr = fnet(fnet_input)
    gen_flow = upscale_four(gen_flow_lr * 4.)
    gen_flow = torch.reshape(gen_flow,
                             (FLAGS.batch_size, (inputimages - 1), output_channel, FLAGS.crop_size, FLAGS.crop_size))

    s_input_warp = F.grid_sample(torch.reshape(Frame_t_pre, (
        FLAGS.batch_size * (inputimages - 1), output_channel, FLAGS.crop_size, FLAGS.crop_size)), gen_flow_lr)

    input0 = torch.cat(
        (r_inputs[:, 0, :, :, :], torch.zeros(size=(FLAGS.batch_size, 3 * 4 * 4, FLAGS.crop_size, FLAGS.crop_size),
                                              dtype=torch.float32)), dim=1)
    gen_pre_output = generator_F(input0, output_channel, FLAGS=FLAGS)

    gen_pre_output = gen_pre_output.view(FLAGS.batch_size, 3, FLAGS.crop_size * 4, FLAGS.crop_size * 4)
    gen_outputs.append(gen_pre_output)

    for frame_i in range(inputimages - 1):
        cur_flow = gen_flow[:, frame_i, :, :, :]
        cur_flow = cur_flow.view(FLAGS.batch_size, 2, FLAGS.crop_size * 4, FLAGS.crop_size * 4)
        gen_pre_output_warp = F.grid_sample(gen_pre_output, cur_flow)
        gen_warppre.append(gen_pre_output_warp)
        gen_pre_output_warp = preprocessLr(deprocess(gen_pre_output_warp))

        gen_pre_output_reshape = gen_pre_output_warp.view(FLAGS.batch_size, 3, FLAGS.crop_size, 4, FLAGS.crop_size, 4)
        gen_pre_output_reshape = gen_pre_output_reshape.permute(0, 1, 3, 5, 2, 4)

        gen_pre_output_reshape = torch.reshape(gen_pre_output_reshape,
                                               (FLAGS.batch_size, 3 * 4 * 4, FLAGS.crop_size, FLAGS.crop_size))
        inputs = torch.cat((r_inputs[:, frame_i + 1, :, :, :], gen_pre_output_reshape), dim=1)

        gen_output = generator_F(inputs, output_channel, FLAGS=FLAGS)
        gen_outputs.append(gen_output)
        gen_pre_output = gen_output
        gen_pre_ouput = gen_pre_ouput.view(FLAGS.batch_size, 3, FLAGS.crop_size * 4, FLAGS.crop_size * 4)

    gen_outputs = torch.stack(gen_outputs, dim=1)

    gen_outputs = gen_outputs.view(FLAGS.batch_size, inputimages, 3, FLAGS.crop_size * 4, FLAGS.crop_size * 4)

    gen_warppre = torch.stack(gen_warppre, dim=1)

    gen_warppre = gen_warppre.view(FLAGS.batch_size, inputimages, 3, FLAGS.crop_size * 4, FLAGS.crop_size * 4)

    s_gen_output = torch.reshape(gen_outputs,
                                 (FLAGS.batch_size * inputimages, 3, FLAGS.crop_size * 4, FLAGS.crop_size * 4))
    s_targets = torch.reshape(r_targets, (FLAGS.batch_size * inputimages, 3, FLAGS.crop_size * 4, FLAGS.crop_size * 4))

    update_list = []
    update_list_name = []

    if FLAGS.vgg_scaling > 0.0:
        vgg_layer_labels = ['vgg_19/conv2_2', 'vgg_19/conv3_4', 'vgg_19/conv4_4']
        gen_vgg = VGG19_slim(s_gen_output, deep_list=vgg_layer_labels)
        target_vgg = VGG19_slim(s_targets, deep_list=vgg_layer_labels)

    if (GAN_FLAG):
        t_size = int(3 * (inputimages // 3))
        t_gen_output = torch.reshape(gen_outputs[:, :t_size, :, :, :],
                                     (FLAGS.batch_size * t_size, 3, FLAGS.crop_size * 4, FLAGS.crop_size * 4))
        t_targets = torch.reshape(r_targets[:, :t_size, :, :, :],
                                  (FLAGS.batch_size * t_size, 3, FLAGS.crop_size * 4, FLAGS.crop_size * 4))
        t_batch = FLAGS.batch_size * t_size // 3

        if not FLAGS.pingpang:
            fnet_input_back = torch.cat((r_inputs[:, 2:t_size:3, :, :, :], r_inputs[:, 1:t_size:3, :, :, :]), dim=1)
            fnet_input_back = torch.reshape(fnet_input_back,
                                            (t_batch, 2 * output_channel, FLAGS.crop_size, FLAGS.crop_size))

            gen_flow_back_lr = fnet(fnet_input_back)

            gen_flow_back = upscale_four(gen_flow_back_lr * 4.0)

            gen_flow_back = torch.reshape(gen_flow_back,
                                          (FLAGS.batch_size, t_size // 3, 2, FLAGS.crop_size * 4, FLAGS.crop_size * 4))

            T_inputs_VPre_batch = identity(gen_flow[:, 0:t_size:3, :, :, :])
            T_inputs_V_batch = torch.zeros_like(T_inputs_VPre_batch)
            T_inputs_VNxt_batch = T_inputs_V_batch

        else:
            T_inputs_VPre_batch = identity(gen_flow[:, 0:t_size:3, :, :, :])
            T_inputs_V_batch = torch.zeros_like(T_inputs_VPre_batch)
            T_inputs_VNxt_batch = gen_flow[:, -2:-1 - t_size:-3, :, :, :]

        T_vel = torch.stack([T_inputs_VPre_batch, T_inputs_V_batch, T_inputs_VNxt_batch], axis=2)
        T_vel = torch.reshape(T_vel, (FLAGS.batch_size * t_size, 2, FLAGS.crop_size * 4, FLAGS.crop_size * 4))
        T_vel = T_vel.detach()

    if (FLAGS.crop_dt < 1.0):
        crop_size_dt = int(FLAGS.crop_size * 4 * FLAGS.crop_dt)
        offset_dt = (FLAGS.crop_size * 4 - crop_size_dt) // 2
        crop_size_dt = FLAGS.crop_size * 4 - offset_dt * 2
        paddings = torch.tensor([[0, 0], [offset_dt, offset_dt], [offset_dt, offset_dt], [0, 0]])
    real_warp0 = F.grid_sample(t_targets, T_vel)

    real_warp = torch.reshape(real_warp0, (t_batch, 3, 3, FLAGS.crop_size*4, FLAGS.crop_size*4))