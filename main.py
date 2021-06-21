import argparse
import os
import sys
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import time
import cv2
from torch.cuda.amp import autocast

sys.path.insert(1, './code')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


from train import FRVSR_Train
from dataloader import train_dataset, inference_dataset
from models import generator, f_net, discriminator
from tqdm import tqdm
from ops import *

# All arguments. These are the same arguments as in the original TecoGan repo. I might prune them at a later date.

parser = argparse.ArgumentParser()
parser.add_argument('--rand_seed', default=1, type=int, help='random seed')

# Directories
parser.add_argument('--input_dir_LR', default='', nargs="?",
                    help='The directory of the input resolution input data, for inference mode')
parser.add_argument('--input_dir_len', default=-1, type=int,
                    help='length of the input for inference mode, -1 means all')
parser.add_argument('--input_dir_HR', default='', nargs="?",
                    help='The directory of the input resolution input data, for inference mode')
parser.add_argument('--mode', default='train', nargs="?", help='train, or inference')
parser.add_argument('--output_dir', default="output", help='The output directory of the checkpoint')
parser.add_argument('--output_pre', default='', nargs="?", help='The name of the subfolder for the images')
parser.add_argument('--output_name', default='output', nargs="?", help='The pre name of the outputs')
parser.add_argument('--output_ext', default='jpg', nargs="?", help='The format of the output when evaluating')
parser.add_argument('--summary_dir', default="summary", nargs="?", help='The dirctory to output the summary')
parser.add_argument('--videotype', default=".mp4", type=str, help="Video type for inference output")
parser.add_argument('--inferencetype', default="dataset", type=str, help="The type of input to the inference loop. "
                                                                         "Either video or dataset folder.")

# Models
parser.add_argument('--g_checkpoint', default=None,
                    help='If provided, the generator will be restored from the provided checkpoint')
parser.add_argument('--d_checkpoint', default=None, nargs="?",
                    help='If provided, the discriminator will be restored from the provided checkpoint')
# parser.add_argument('--f_checkpoint', default=None, nargs="?",
#                  help='If provided, the fnet will be restored from the provided checkpoint')
parser.add_argument('--num_resblock', type=int, default=16, help='How many residual blocks are there in the generator')
parser.add_argument('--discrim_resblocks', type=int, default=4, help='Number of resblocks in each resnet layer in the '
                                                                     'discriminator')
parser.add_argument('--discrim_channels', type=int, default=128, help='How many channels to use in the last two '
                                                                      'resnet blocks in the discriminator')
# Models for training
parser.add_argument('--pre_trained_model', type=str2bool, default=False,
                    help='If True, the weight of generator will be loaded as an initial point'
                         'If False, continue the training')
parser.add_argument('--vgg_ckpt', default=None, help='path to checkpoint file for the vgg19')

# Machine resources
parser.add_argument('--cudaID', default='0', help='CUDA devices')
parser.add_argument('--queue_thread', default=8, type=int,
                    help='The threads of the queue (More threads can speedup the training process.')

# Training details
# The data preparing operation

parser.add_argument('--RNN_N', default=10, nargs="?", help='The number of the rnn recurrent length')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size of the input batch')
parser.add_argument('--flip', default=True, type=str2bool, help='Whether random flip data augmentation is applied')
parser.add_argument('--random_crop', default=True, type=str2bool, help='Whether perform the random crop')
parser.add_argument('--movingFirstFrame', default=True, type=str2bool,
                    help='Whether use constant moving first frame randomly.')
parser.add_argument('--crop_size', default=32, type=int, help='The crop size of the training image')
# Training data settings
parser.add_argument('--input_video_dir', type=str, default="../TrainingDataPath",
                    help='The directory of the video input data, for training')
parser.add_argument('--input_video_pre', default='scene', type=str,
                    help='The pre of the directory of the video input data')
parser.add_argument('--str_dir', default=1000, type=int, help='The starting index of the video directory')
parser.add_argument('--end_dir', default=1400, type=int, help='The ending index of the video directory')
parser.add_argument('--end_dir_val', default=2050, type=int,
                    help='The ending index for validation of the video directory')
parser.add_argument('--max_frm', default=119, type=int, help='The ending index of the video directory')
# The loss parameters

parser.add_argument('--vgg_scaling', default=-0.002,
                    type=float, help='The scaling factor for the VGG perceptual loss, disable with negative value')
parser.add_argument('--warp_scaling', default=1.0, type=float, help='The scaling factor for the warp')
parser.add_argument('--pingpang', default=False, type=str2bool, help='use bi-directional recurrent or not')
parser.add_argument('--pp_scaling', default=1.0, type=float,
                    help='factor of pingpang term, only works when pingpang is True')
# Training parameters
parser.add_argument('--EPS', default=1e-12, type=float, help='The eps added to prevent nan')
parser.add_argument('--learning_rate', default=0.0001, type=float, help='The learning rate for the network')
parser.add_argument('--decay_step', default=250, type=int, help='The steps needed to decay the learning rate')
parser.add_argument('--decay_rate', default=0.8, type=float, help='The decay rate of each decay step')
parser.add_argument('--stair', default=False, type=str2bool,
                    help='Whether perform staircase decay. True => decay in discrete interval.')
parser.add_argument('--beta', default=0.9, type=float, help='The beta1 parameter for the Adam optimizer')
parser.add_argument('--adameps', default=1e-8, type=float, help='The eps parameter for the Adam optimizer')
parser.add_argument('--max_epochs', default=10000000, type=int, help='The max epoch for the training')

# Dst parameters
parser.add_argument('--ratio', default=0.01, type=float, help='The ratio between content loss and adversarial loss')
parser.add_argument('--Dt_mergeDs', default=True, type=str2bool, help='Whether only use a merged Discriminator.')
parser.add_argument('--Dt_ratio_0', default=1.0, type=float,
                    help='The starting ratio for the temporal adversarial loss')
parser.add_argument('--Dt_ratio_add', default=0.0, type=float,
                    help='The increasing ratio for the temporal adversarial loss')
parser.add_argument('--Dt_ratio_max', default=1.0, type=float, help='The max ratio for the temporal adversarial loss')
parser.add_argument('--Dbalance', default=0.4, type=float, help='An adaptive balancing for Discriminators')
parser.add_argument('--crop_dt', default=0.75, type=float,
                    help='factor of dt crop')  # dt input size = crop_size*crop_dt
parser.add_argument('--D_LAYERLOSS', default=True, type=str2bool, help='Whether use layer loss from D')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cudaID

##Checking to make sure necessary args are filled.
if args.output_dir is None:
    raise ValueError("The output directory is needed")
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

if not os.path.exists(args.summary_dir):
    os.mkdir(args.summary_dir)

# an inference mode that I will complete soon
if args.mode == "inference":
    if args.inferencetype == "dataset":
        dataset = inference_dataset(args)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
    elif args.inferencetype == "video":
        cap = cv2.VideoCapture(args.input_dir_LR)
        frames = []
        for k in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                continue
            # Our operations on the frame come here

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (args.crop_size, args.crop_size), interpolation=cv2.INTER_AREA)
            next_img = torchvision.transforms.functional.to_tensor(frame)

            frames.append(next_img)
        cap.release()
        loader = torch.stack(frames, dim=0).unsqueeze(0).unsqueeze(0)

    else:
        raise ValueError("Invalid data type entered. Please use either video or dataset.")
    if args.g_checkpoint is None:
        raise ValueError("The checkpoint file is needed to perform the test")
    generator_F = generator(3, args=args).cuda()

    g_checkpoint = torch.load(args.g_checkpoint)
    generator_F.load_state_dict(g_checkpoint["model_state_dict"])
    with torch.no_grad():
        with autocast():
            for batch_idx, r_inputs in enumerate(loader):

                output_channel = r_inputs.shape[2]
                inputimages = r_inputs.shape[1]
                gen_outputs = []
                gen_warppre = []
                learning_rate = args.learning_rate

                Frame_t_pre = r_inputs[:, 0:-1, :, :, :]
                # Reshaping the fnet input and passing it to the model
                fnet_input = torch.reshape(Frame_t_pre, (
                    1 * (inputimages - 1), output_channel, args.crop_size, args.crop_size))
                # Preparing generator input
                gen_flow = upscale_four(fnet_input * 4.)

                gen_flow = torch.reshape(gen_flow[:, 0:2],
                                         (1, (inputimages - 1), 2, args.crop_size * 4, args.crop_size * 4))

                input0 = torch.cat(
                    (r_inputs[:, 0, :, :, :], torch.zeros(size=(1, 3 * 4 * 4, args.crop_size, args.crop_size),
                                                          dtype=torch.float32)), dim=1)
                # Passing inputs into model and reshaping output
                gen_pre_output = generator_F(input0.cuda()).cpu()
                gen_pre_output = gen_pre_output.view(1, 3, args.crop_size * 4, args.crop_size * 4)
                gen_outputs.append(gen_pre_output)
                # Getting outputs of generator for each frame
                for frame_i in range(inputimages - 1):
                    cur_flow = gen_flow[:, frame_i, :, :, :]
                    cur_flow = cur_flow.view(1, args.crop_size * 4, args.crop_size * 4, 2)

                    gen_pre_output_warp = F.grid_sample(gen_pre_output.cuda(), cur_flow.cuda().half()).cpu()
                    gen_warppre.append(gen_pre_output_warp)

                    gen_pre_output_warp = preprocessLr(deprocess(gen_pre_output_warp))
                    gen_pre_output_reshape = gen_pre_output_warp.view(1, 3, args.crop_size, 4, args.crop_size,
                                                                      4)
                    gen_pre_output_reshape = gen_pre_output_reshape.permute(0, 1, 3, 5, 2, 4)

                    gen_pre_output_reshape = torch.reshape(gen_pre_output_reshape,
                                                           (1, 3 * 4 * 4, args.crop_size, args.crop_size))
                    inputs = torch.cat((r_inputs[:, frame_i + 1, :, :, :], gen_pre_output_reshape), dim=1)
                    gen_output = generator_F(inputs.cuda()).cpu()
                    gen_outputs.append(gen_output)
                    gen_pre_output = gen_output
                # Converting list of gen outputs and reshaping
                gen_outputs = torch.stack(gen_outputs, dim=1)
                gen_outputs = gen_outputs.cpu().squeeze(0)
            save_as_gif(gen_outputs, f"./output/output{batch_idx}{args.videotype}")
# My training loop for TecoGan

elif args.mode == "train":
    # Defining dataset and dataloader
    dataset = train_dataset(args)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Defining the models as well as the optimizers and lr schedulers
    generator_F = generator(3, args=args).cuda()
    # fnet = f_net().cuda()
    discriminator_F = discriminator(args=args).cuda()
    counter1 = 0.
    counter2 = 0.
    min_gen_loss = np.inf
    tdis_learning_rate = args.learning_rate
    if not args.Dt_mergeDs:
        tdis_learning_rate = tdis_learning_rate * 0.3
    tdiscrim_optimizer = torch.optim.Adam(discriminator_F.parameters(), tdis_learning_rate,
                                          betas=(args.beta, 0.999),
                                          eps=args.adameps)
    gen_optimizer = torch.optim.Adam(generator_F.parameters(), args.learning_rate, betas=(args.beta, 0.999),
                                     eps=args.adameps)
    # fnet_optimizer = torch.optim.Adam(fnet.parameters(), args.learning_rate, betas=(args.beta, 0.999),
    # eps=args.adameps)
    GAN_FLAG = True
    d_scheduler = torch.optim.lr_scheduler.StepLR(tdiscrim_optimizer, args.decay_step, args.decay_rate)
    g_scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, args.decay_step, args.decay_rate)
    # f_scheduler = torch.optim.lr_scheduler.StepLR(fnet_optimizer, args.decay_step, args.decay_rate)
    # Loading pretrained models and optimizers
    if args.pre_trained_model:
        g_checkpoint = torch.load(args.g_checkpoint)
        generator_F.load_state_dict(g_checkpoint["model_state_dict"])
        gen_optimizer.load_state_dict(g_checkpoint["optimizer_state_dict"])
        current_epoch = g_checkpoint["epoch"]
        d_checkpoint = torch.load(args.d_checkpoint)
        discriminator_F.load_state_dict(d_checkpoint["model_state_dict"])
        tdiscrim_optimizer.load_state_dict(d_checkpoint["optimizer_state_dict"])
        # f_checkpoint = torch.load(args.f_checkpoint)
        # fnet.load_state_dict(f_checkpoint["model_state_dict"])
        # fnet_optimizer.load_state_dict(f_checkpoint["optimizer_state_dict"])
    else:
        current_epoch = 0
    # Starting epoch loop
    since = time.time()
    for e in tqdm(range(current_epoch, args.max_epochs)):
        d_loss = 0.
        g_loss = 0.
        f_loss = 0.
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            # Passing targets and inputs to the train function
            output = FRVSR_Train(inputs, targets, args, discriminator_F, generator_F, batch_idx, counter1,
                                 counter2, gen_optimizer, tdiscrim_optimizer)

            # Computing epoch losses
            # f_loss = f_loss + ((1 / (batch_idx + 1)) * (output.fnet_loss.data - f_loss))

            g_loss = g_loss + ((1 / (batch_idx + 1)) * (output.gen_loss.data - g_loss))

            d_loss = d_loss + ((1 / (batch_idx + 1)) * (output.d_loss.data - d_loss))
        # Saving outputs as gifs and images
        index = np.random.randint(0, targets.shape[0])
        save_as_gif(output.gen_output[index][:args.RNN_N].cpu().data, "gan.gif")
        save_as_gif(targets[index].cpu().data, "real.gif")
        save_as_gif(inputs[index].cpu().data, "original.gif")
        torchvision.utils.save_image(
            output.gen_output.view(output.gen_output.shape[0] * output.gen_output.shape[1], 3, args.crop_size * 4, args.crop_size * 4),
            fp="Gan_examples.jpg")
        torchvision.utils.save_image(
            targets.view(targets.shape[0] * args.RNN_N, 3, args.crop_size * 4, args.crop_size * 4), fp="real_image.jpg")
        torchvision.utils.save_image(inputs.view(targets.shape[0] * args.RNN_N, 3, args.crop_size, args.crop_size),
                                     fp="original_image.jpg")
        # Updating the lr schedulers
        d_scheduler.step()
        g_scheduler.step()
        # Printing out metrics
        print("Epoch: {}".format(e + 1))
        print("\nGenerator loss is: {} \nDiscriminator loss is: {}".format(g_loss, d_loss))
        for param_group in gen_optimizer.param_groups:
            gen_lr = param_group["lr"]
        for param_group in tdiscrim_optimizer.param_groups:
            disc_lr = param_group["lr"]
        print(f"\nGenerator lr is: {gen_lr}, Discriminator lr is: {disc_lr}")
        print("\nSaving model...")
        # Saving the models
        torch.save({
            'epoch': e,
            'model_state_dict': generator_F.state_dict(),
            'optimizer_state_dict': gen_optimizer.state_dict(),
        }, "generator.pt")

        torch.save({
            'model_state_dict': discriminator_F.state_dict(),
            'optimizer_state_dict': tdiscrim_optimizer.state_dict(),
        }, "discrim.pt")
        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
