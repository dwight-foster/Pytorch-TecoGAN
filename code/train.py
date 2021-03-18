from models import *
from torchvision.transforms import functional
from torch.cuda.amp import GradScaler, autocast
import collections

VGG_MEAN = [123.68, 116.78, 103.94]
identity = torch.nn.Identity()

scaler = GradScaler()


# Computing EMA
class EMA(nn.Module):
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def forward(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


# VGG function for layer outputs
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


# Main train function
def TecoGAN(r_inputs, r_targets, discriminator_F, generator_F, args, Global_step, counter1, counter2, optimizer_g
            , optimizer_d,
            GAN_FLAG=True):
    Global_step += 1

    inputimages = args.RNN_N
    # Getting inputs for Fnet using pingpang loss for forward and reverse videos
    if args.pingpang:
        r_inputs_rev_input = torch.flip(r_inputs, dims=[1])[:, 1:, :, :, :]

        r_targets_rev_input = torch.flip(r_targets, dims=[1])[:, 1:, :, :, :]
        r_inputs = torch.cat([r_inputs, r_inputs_rev_input], axis=1)
        r_targets = torch.cat([r_targets, r_targets_rev_input], axis=1)
        inputimages = args.RNN_N * 2 - 1
    output_channel = r_targets.shape[2]
    gen_outputs = []
    gen_warppre = []
    learning_rate = args.learning_rate
    Frame_t_pre = r_inputs[:, 0:-1, :, :, :]
    Frame_t = r_inputs[:, 1:, :, :, :]
    # Reshaping the fnet input and passing it to the model
    with autocast():
        fnet_input = torch.reshape(Frame_t_pre, (
            Frame_t_pre.shape[0] * (inputimages - 1), output_channel, args.crop_size, args.crop_size))
        # Preparing generator input
        gen_flow = upscale_four(fnet_input * 4.)

        gen_flow = torch.reshape(gen_flow[:, 0:2],
                                 (Frame_t.shape[0], (inputimages - 1), 2, args.crop_size * 4, args.crop_size * 4))
        input_frames = torch.reshape(Frame_t,
                                     (Frame_t.shape[0] * (inputimages - 1), output_channel, args.crop_size,
                                      args.crop_size))
        s_input_warp = F.grid_sample(torch.reshape(Frame_t_pre, (
            Frame_t_pre.shape[0] * (inputimages - 1), output_channel, args.crop_size, args.crop_size)),
                                     torch.reshape(Frame_t[:, :, 0:2],
                                                   (Frame_t.shape[0] * (inputimages - 1), args.crop_size, args.crop_size, 2)))

        input0 = torch.cat(
            (r_inputs[:, 0, :, :, :], torch.zeros(size=(r_inputs.shape[0], 3 * 4 * 4, args.crop_size, args.crop_size),
                                                  dtype=torch.float32).cuda()), dim=1)
        # Passing inputs into model and reshaping output
        gen_pre_output = generator_F(input0.detach())
        gen_pre_output = gen_pre_output.view(gen_pre_output.shape[0], 3, args.crop_size * 4, args.crop_size * 4)
        gen_outputs.append(gen_pre_output)
        # Getting outputs of generator for each frame
        for frame_i in range(inputimages - 1):
            cur_flow = gen_flow[:, frame_i, :, :, :]
            cur_flow = cur_flow.view(cur_flow.shape[0], args.crop_size * 4, args.crop_size * 4, 2)

            gen_pre_output_warp = F.grid_sample(gen_pre_output, cur_flow.half())
            gen_warppre.append(gen_pre_output_warp)

            gen_pre_output_warp = preprocessLr(deprocess(gen_pre_output_warp))
            gen_pre_output_reshape = gen_pre_output_warp.view(gen_pre_output_warp.shape[0], 3, args.crop_size, 4, args.crop_size, 4)
            gen_pre_output_reshape = gen_pre_output_reshape.permute(0, 1, 3, 5, 2, 4)

            gen_pre_output_reshape = torch.reshape(gen_pre_output_reshape,
                                                   (gen_pre_output_reshape.shape[0], 3 * 4 * 4, args.crop_size, args.crop_size))
            inputs = torch.cat((r_inputs[:, frame_i + 1, :, :, :], gen_pre_output_reshape), dim=1)
            gen_output = generator_F(inputs.detach())
            gen_outputs.append(gen_output)
            gen_pre_output = gen_output
            gen_pre_output = gen_pre_output.view(gen_pre_output.shape[0], 3, args.crop_size * 4, args.crop_size * 4)
        # Converting list of gen outputs and reshaping
        gen_outputs = torch.stack(gen_outputs, dim=1)
        gen_outputs = gen_outputs.view(gen_outputs.shape[0], inputimages, 3, args.crop_size * 4, args.crop_size * 4)

        s_gen_output = torch.reshape(gen_outputs,
                                     (gen_outputs.shape[0] * inputimages, 3, args.crop_size * 4, args.crop_size * 4))
        s_targets = torch.reshape(r_targets, (r_targets.shape[0] * inputimages, 3, args.crop_size * 4, args.crop_size * 4))

        update_list = []
        update_list_name = []

        # Preparing vgg layers
        if args.vgg_scaling > 0.0:
            vgg_layer_labels = ['vgg_19/conv2_2', 'vgg_19/conv3_4', 'vgg_19/conv4_4']
            gen_vgg = VGG19_slim(s_gen_output, deep_list=vgg_layer_labels)
            target_vgg = VGG19_slim(s_targets, deep_list=vgg_layer_labels)

        if (GAN_FLAG):
            t_size = int(3 * (inputimages // 3))
            t_gen_output = torch.reshape(gen_outputs[:, :t_size, :, :, :],
                                         (gen_outputs.shape[0] * t_size, 3, args.crop_size * 4, args.crop_size * 4))
            t_targets = torch.reshape(r_targets[:, :t_size, :, :, :],
                                      (r_targets.shape[0] * t_size, 3, args.crop_size * 4, args.crop_size * 4))
            t_batch = r_targets.shape[0] * t_size // 3

            # Preparing inputs for discriminator
            if not args.pingpang:
                fnet_input_back = torch.cat((r_inputs[:, 2:t_size:3, :, :, :], r_inputs[:, 1:t_size:3, :, :, :]), dim=1)
                fnet_input_back = torch.reshape(fnet_input_back,
                                                (t_batch, 2 * output_channel, args.crop_size, args.crop_size))

                gen_flow_back = upscale_four(fnet_input_back[:, 0:2] * 4.0)

                gen_flow_back = torch.reshape(gen_flow_back,
                                              (gen_flow_back.shape[0], t_size // 3, 2, args.crop_size * 4, args.crop_size * 4))

                T_inputs_VPre_batch = identity(gen_flow[:, 0:t_size:3, :, :, :])
                T_inputs_V_batch = torch.zeros_like(T_inputs_VPre_batch)
                T_inputs_VNxt_batch = preprocess(gen_flow_back)

            else:
                T_inputs_VPre_batch = identity(gen_flow[:, 0:t_size:3, :, :, :])
                T_inputs_V_batch = torch.zeros_like(T_inputs_VPre_batch)
                T_inputs_VNxt_batch = torch.flip(gen_flow, dims=[1])[:, 1: t_size:3, :, :, :]

            T_vel = torch.stack([T_inputs_VPre_batch, T_inputs_V_batch, T_inputs_VNxt_batch], axis=2)
            T_vel = torch.reshape(T_vel, (T_vel.shape[0] * t_size, args.crop_size * 4, args.crop_size * 4, 2))
            T_vel = T_vel.detach()

            if args.crop_dt < 1.0:
                crop_size_dt = int(args.crop_size * 4 * args.crop_dt)
                offset_dt = (args.crop_size * 4 - crop_size_dt) // 2
                crop_size_dt = args.crop_size * 4 - offset_dt * 2
                paddings = (offset_dt, offset_dt, offset_dt, offset_dt)
            real_warp0 = F.grid_sample(t_targets, T_vel)

            real_warp = torch.reshape(real_warp0, (t_batch, 9, args.crop_size * 4, args.crop_size * 4))
            if (args.crop_dt < 1.0):
                real_warp = functional.resized_crop(real_warp, offset_dt, offset_dt, crop_size_dt, crop_size_dt,
                                                    [crop_size_dt, crop_size_dt])
            # Passing real inputs to discriminator
            if (args.Dt_mergeDs):
                if (args.crop_dt < 1.0):
                    real_warp = F.pad(real_warp, paddings, "constant")
                before_warp = torch.reshape(t_targets, (t_batch, 9, args.crop_size * 4, args.crop_size * 4))
                t_input = torch.reshape(r_inputs[:, :t_size, :, :, :],
                                        (t_batch, 9, args.crop_size, args.crop_size))
                input_hi = functional.resize(t_input, [args.crop_size * 4, args.crop_size * 4])
                real_warp = torch.cat((before_warp, real_warp, input_hi), dim=1)

                tdiscrim_real_output, real_layers = discriminator_F(real_warp)

            else:
                tdiscrim_real_output = discriminator_F(real_warp.cuda())

            # Reshaping Generator output to pass to Discriminator
            fake_warp0 = F.grid_sample(t_gen_output, T_vel.half())

            fake_warp = torch.reshape(fake_warp0, (t_batch, 9, args.crop_size * 4, args.crop_size * 4))
            if (args.crop_dt < 1.0):
                fake_warp = functional.resized_crop(fake_warp, offset_dt, offset_dt, crop_size_dt, crop_size_dt,
                                                    size=[crop_size_dt, crop_size_dt])
            # Passing generated images to discriminator
            if (args.Dt_mergeDs):
                if (args.crop_dt < 1.0):
                    fake_warp = F.pad(fake_warp, paddings, "constant")
                before_warp = torch.reshape(before_warp, (t_batch, 9, args.crop_size * 4, args.crop_size * 4))
                fake_warp = torch.cat((before_warp, fake_warp, input_hi), dim=1)
                tdiscrim_fake_output, fake_layers = discriminator_F(fake_warp.cuda().detach())

            else:
                tdiscrim_fake_output = discriminator_F(fake_warp.cuda().detach())

            # Computing the layer loss using the VGG network and discriminator outputs
            if (args.D_LAYERLOSS):
                Fix_Range = 0.02
                Fix_margin = 0.0

                sum_layer_loss = 0
                d_layer_loss = 0

                layer_loss_list = []
                layer_n = len(real_layers)
                layer_norm = [12.0, 14.0, 24.0, 100.0]
                for layer_i in range(layer_n):
                    real_layer = real_layers[layer_i]
                    false_layer = fake_layers[layer_i]

                    layer_diff = real_layer.detach() - false_layer.detach()
                    layer_loss = torch.mean(torch.sum(torch.abs(layer_diff), dim=[3]))

                    layer_loss_list += [layer_loss]

                    scaled_layer_loss = Fix_Range * layer_loss / layer_norm[layer_i]

                    sum_layer_loss += scaled_layer_loss
                    if Fix_margin > 0.0:
                        d_layer_loss += torch.max(0.0, torch.tensor(Fix_margin - scaled_layer_loss))

                update_list += layer_loss_list
                update_list_name += [("D_layer_%d_loss" % _) for _ in range(layer_n)]
                update_list += [sum_layer_loss]
                update_list_name += ["D_layer_loss_sum"]

                if Fix_margin > 0.0:
                    update_list += [d_layer_loss]
                    update_list_name += ["D_layer_loss_for_D_sum"]
        # Computing the generator and fnet losses
        diff1_mse = s_gen_output - s_targets

        content_loss = torch.mean(torch.sum(torch.square(diff1_mse), dim=[3]))
        update_list += [content_loss]
        update_list_name += ["l2_content_loss"]
        gen_loss = content_loss
        fnet_loss = content_loss

        diff2_mse = input_frames - s_input_warp

        warp_loss = torch.mean(torch.sum(torch.square(diff2_mse), dim=[3]))
        update_list += [warp_loss]
        update_list_name += ["l2_warp_loss"]

        vgg_loss = None
        vgg_loss_list = []
        if args.vgg_scaling > 0.0:
            vgg_wei_list = [1.0, 1.0, 1.0, 1.0]
            vgg_loss = 0
            vgg_layer_n = len(vgg_layer_labels)

            for layer_i in range(vgg_layer_n):
                curvgg_diff = torch.sum(gen_vgg[vgg_layer_labels[layer_i]] * target_vgg[vgg_layer_labels[layer_i]],
                                        dim=[3])
                scaled_layer_loss = vgg_wei_list[layer_i] * curvgg_diff
                vgg_loss_list += [curvgg_diff]
                vgg_loss += scaled_layer_loss

            gen_loss += args.vgg_scaling * vgg_loss
            fnet_loss += args.vgg_scaling * vgg_loss.detach()
            vgg_loss_list += [vgg_loss]

            update_list += vgg_loss_list
            update_list_name += ["vgg_loss_%d" % (_ + 2) for _ in range(len(vgg_loss_list) - 1)]
            update_list_name += ["vgg_all"]

        if args.pingpang:
            gen_out_first = gen_outputs[:, 0:args.RNN_N - 1, :, :, :]
            gen_out_last_rev = torch.flip(gen_outputs, dims=[1])[:, :args.RNN_N - 1:1, :, :, :]

            pploss = torch.mean(torch.abs(gen_out_first - gen_out_last_rev))

            if args.pp_scaling > 0:
                gen_loss += pploss.cpu() * args.pp_scaling
                fnet_loss += pploss.cpu() * args.pp_scaling
            update_list += [pploss]
            update_list_name += ["PingPang"]

        if (GAN_FLAG):
            t_adversarial_loss = torch.mean(-torch.log(tdiscrim_fake_output.detach() + args.EPS))
            d_adversarial_loss = torch.mean(-torch.log(tdiscrim_fake_output + args.EPS))
            dt_ratio = torch.min(torch.tensor(args.Dt_ratio_max),
                                 args.Dt_ratio_0 + args.Dt_ratio_add * torch.tensor(Global_step, dtype=torch.float32))

        gen_loss += args.ratio * t_adversarial_loss.cpu()
        fnet_loss += args.ratio * t_adversarial_loss.cpu()
        update_list += [t_adversarial_loss]
        update_list_name += ["t_adversarial_loss"]
        # Computing gradients for Generator and updating weights
        if (args.D_LAYERLOSS):
            gen_loss += sum_layer_loss.cpu() * dt_ratio
        gen_loss = gen_loss.cuda()

        # Computing discriminator loss
        if (GAN_FLAG):
            t_discrim_fake_loss = torch.log(1 - tdiscrim_fake_output + args.EPS)
            t_discrim_real_loss = torch.log(tdiscrim_real_output + args.EPS)

            t_discrim_loss = torch.mean(-(t_discrim_fake_loss + t_discrim_real_loss))
            t_balance = torch.mean(t_discrim_real_loss) + d_adversarial_loss

            update_list += [t_discrim_loss]
            update_list_name += ["t_discrim_loss"]

            update_list += [torch.mean(tdiscrim_real_output), torch.mean(tdiscrim_fake_output)]
            update_list_name += ["t_discrim_real_output", "t_discrim_fake_output"]

            if (args.D_LAYERLOSS and Fix_margin > 0.0):
                discrim_loss = t_discrim_loss + d_layer_loss * dt_ratio

            # Computing gradients for Discriminator and updating weights
            else:
                discrim_loss = t_discrim_loss
            discrim_loss = discrim_loss.cuda()

            tb_exp_averager = EMA(0.99)
            init_average = torch.zeros_like(t_balance)
            tb_exp_averager.register("TB_average", init_average)
            tb = tb_exp_averager.forward("TB_average", t_balance)

            update_list += [gen_loss]
            update_list_name += ["All_loss_Gen"]

            tb_exp_averager.register("Loss_average", init_average)
            update_list_avg = [tb_exp_averager.forward("Loss_average", _) for _ in update_list]
        # Computing gradients for fnet and updating weights
        optimizer_g.zero_grad()
        scaler.scale(gen_loss).backward()
        scaler.step(optimizer_g)
        scaler.update()
        optimizer_d.zero_grad()
        scaler.scale(discrim_loss).backward()
        scaler.step(optimizer_d)
        scaler.update()
        # fnet_loss = fnet_loss.cuda()
        # fnet_optimizer.zero_grad()
        # fnet_loss.backward()
        # fnet_optimizer.step()
        update_list_avg += [tb, dt_ratio]
        update_list_name += ["t_balance", "Dst_ratio"]

        update_list_avg += [counter1, counter2]
        update_list_name += ["withD_counter", "w_o_D_counter"]
    max_outputs = min(4, r_targets.shape[0])
    # Returning output tuple
    Network = collections.namedtuple('Network', 'gen_output, learning_rate, update_list, '
                                                'update_list_name, update_list_avg, global_step, d_loss, gen_loss, '
                                                'fnet_loss ,tb, target')
    return Network(
        gen_output=gen_outputs,
        learning_rate=learning_rate,
        update_list=update_list,
        update_list_name=update_list_name,
        update_list_avg=update_list_avg,
        global_step=Global_step,
        d_loss=discrim_loss,
        gen_loss=gen_loss,
        fnet_loss=fnet_loss,
        tb=tb,
        target=real_warp

    )


# Defining train function
def FRVSR_Train(r_inputs, r_targets, args, discriminator_F, generator_F, step, counter1, counter2, optimizer_g,
                optimizer_d):
    return TecoGAN(r_inputs, r_targets, discriminator_F, generator_F, args, step, counter1, counter2,
                   optimizer_g, optimizer_d)
