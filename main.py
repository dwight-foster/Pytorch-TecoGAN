import os
import argparse
import os
import subprocess
import sys

import torchvision
from torch.utils.data import DataLoader

from lib.Teco import FRVSR, discriminator
from lib.dataloader import train_dataset
from lib.frvsr import generator, f_net
from lib.ops import *

parser = argparse.ArgumentParser()
parser.add_argument('--rand_seed', default=1, nargs="?", help='random seed')

# Directories
parser.add_argument('--input_dir_LR', default='', nargs="?",
                    help='The directory of the input resolution input data, for inference mode')
parser.add_argument('--input_dir_len', default=-1, nargs="?",
                    help='length of the input for inference mode, -1 means all')
parser.add_argument('--input_dir_HR', default='', nargs="?",
                    help='The directory of the input resolution input data, for inference mode')
parser.add_argument('--mode', default='train', nargs="?", help='train, or inference')
parser.add_argument('--output_dir', default="output", help='The output directory of the checkpoint')
parser.add_argument('--output_pre', default='', nargs="?", help='The name of the subfolder for the images')
parser.add_argument('--output_name', default='output', nargs="?", help='The pre name of the outputs')
parser.add_argument('--output_ext', default='jpg', nargs="?", help='The format of the output when evaluating')
parser.add_argument('--summary_dir', default="summary", nargs="?", help='The dirctory to output the summary')

# Models
parser.add_argument('--g_checkpoint', default=None,
                    help='If provided, the generator will be restored from the provided checkpoint')
parser.add_argument('--d_checkpoint', default=None, nargs="?",
                    help='If provided, the discriminator will be restored from the provided checkpoint')
parser.add_argument('--f_checkpoint', default=None, nargs="?",
                    help='If provided, the fnet will be restored from the provided checkpoint')
parser.add_argument('--num_resblock', nargs="?", default=16, help='How many residual blocks are there in the generator')
# Models for training
parser.add_argument('--pre_trained_model', nargs="?", default=False,
                    help='If True, the weight of generator will be loaded as an initial point'
                         'If False, continue the training')
parser.add_argument('--vgg_ckpt', default=None, nargs="?", help='path to checkpoint file for the vgg19')

# Machine resources
parser.add_argument('--cudaID', default='0', nargs="?", help='CUDA devices')
parser.add_argument('--queue_thread', default=6, nargs="?",
                    help='The threads of the queue (More threads can speedup the training process.')
parser.add_argument('--name_video_queue_capacity', default=512, nargs="?",
                    help='The capacity of the filename queue (suggest large to ensure'
                         'enough random shuffle.')
parser.add_argument('--video_queue_capacity', default=256, nargs="?",
                    help='The capacity of the video queue (suggest large to ensure'
                         'enough random shuffle')
parser.add_argument('--video_queue_batch', default=2, nargs="?", help='shuffle_batch queue capacity')

# Training details
# The data preparing operation
parser.add_argument('--RNN_N', default=10, nargs="?", help='The number of the rnn recurrent length')
parser.add_argument('--batch_size', default=4, nargs="?", help='Batch size of the input batch')
parser.add_argument('--flip', default=True, nargs="?", help='Whether random flip data augmentation is applied')
parser.add_argument('--random_crop', default=True, nargs="?", help='Whether perform the random crop')
parser.add_argument('--movingFirstFrame', default=True, nargs="?",
                    help='Whether use constant moving first frame randomly.')
parser.add_argument('--crop_size', default=32, nargs="?", help='The crop size of the training image')
# Training data settings
parser.add_argument('--input_video_dir', type=str, default="../TrainingDataPath",
                    help='The directory of the video input data, for training')
parser.add_argument('--input_video_pre', default='scene', nargs="?",
                    help='The pre of the directory of the video input data')
parser.add_argument('--str_dir', default=2000, nargs="?", help='The starting index of the video directory')
parser.add_argument('--end_dir', default=2400, nargs="?", help='The ending index of the video directory')
parser.add_argument('--end_dir_val', default=2050, nargs="?",
                    help='The ending index for validation of the video directory')
parser.add_argument('--max_frm', default=119, nargs="?", help='The ending index of the video directory')
# The loss parameters

parser.add_argument('--vgg_scaling', default=-0.002,
                    nargs="?", help='The scaling factor for the VGG perceptual loss, disable with negative value')
parser.add_argument('--warp_scaling', default=1.0, nargs="?", help='The scaling factor for the warp')
parser.add_argument('--pingpang', default=True, nargs="?", help='use bi-directional recurrent or not')
parser.add_argument('--pp_scaling', default=1.0, nargs="?",
                    help='factor of pingpang term, only works when pingpang is True')
# Training parameters
parser.add_argument('--EPS', default=1e-12, nargs="?", help='The eps added to prevent nan')
parser.add_argument('--learning_rate', default=0.0003, nargs="?", help='The learning rate for the network')
parser.add_argument('--decay_step', default=500000, nargs="?", help='The steps needed to decay the learning rate')
parser.add_argument('--decay_rate', default=0.5, nargs="?", help='The decay rate of each decay step')
parser.add_argument('--stair', default=False, nargs="?",
                    help='Whether perform staircase decay. True => decay in discrete interval.')
parser.add_argument('--beta', default=0.9, nargs="?", help='The beta1 parameter for the Adam optimizer')
parser.add_argument('--adameps', default=1e-8, nargs="?", help='The eps parameter for the Adam optimizer')
parser.add_argument('--max_epoch', default=10000, nargs="?", help='The max epoch for the training')
parser.add_argument('--max_iter', default=1000000, nargs="?", help='The max iteration of the training')
parser.add_argument('--display_freq', default=20, nargs="?", help='The diplay frequency of the training process')
parser.add_argument('--summary_freq', default=100, nargs="?", help='The frequency of writing summary')
parser.add_argument('--save_freq', default=10000, nargs="?", help='The frequency of saving images')
# Dst parameters
parser.add_argument('--ratio', default=0.01, nargs="?", help='The ratio between content loss and adversarial loss')
parser.add_argument('--Dt_mergeDs', default=True, nargs="?", help='Whether only use a merged Discriminator.')
parser.add_argument('--Dt_ratio_0', default=1.0, nargs="?", help='The starting ratio for the temporal adversarial loss')
parser.add_argument('--Dt_ratio_add', default=0.0, nargs="?",
                    help='The increasing ratio for the temporal adversarial loss')
parser.add_argument('--Dt_ratio_max', default=1.0, nargs="?", help='The max ratio for the temporal adversarial loss')
parser.add_argument('--Dbalance', default=0.4, nargs="?", help='An adaptive balancing for Discriminators')
parser.add_argument('--crop_dt', default=0.75, nargs="?", help='factor of dt crop')  # dt input size = crop_size*crop_dt
parser.add_argument('--D_LAYERLOSS', default=True, nargs="?", help='Whether use layer loss from D')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cudaID

if args.output_dir is None:
    raise ValueError("The output directory is needed")
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

if not os.path.exists(args.summary_dir):
    os.mkdir(args.summary_dir)


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(args.summary_dir + "logfile.txt", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()


sys.stdout = Logger()


def preexec():  # Don't forward signals.
    os.setpgrp()


def testWhileTrain(args, testno=0):
    '''
        this function is called during training, Hard-Coded!!
        to try the "inference" mode when a new model is saved.
        The code has to be updated from machine to machine...
        depending on python, and your training settings
    '''
    desstr = os.path.join(args.output_dir, 'train/')  # saving in the ./train/ directory
    cmd1 = ["python3", "main.py",  # never tested with python2...
            "--output_dir", desstr,
            "--summary_dir", desstr,
            "--mode", "inference",
            "--num_resblock", "%d" % args.num_resblock,
            "--checkpoint", os.path.join(args.output_dir, 'model-%d' % testno),
            "--cudaID", args.cudaID]
    # a folder for short test
    cmd1 += ["--input_dir_LR", "./LR/calendar/",  # update the testing sequence
             "--output_pre", "",  # saving in train folder directly
             "--output_name", "%09d" % testno,  # name
             "--input_dir_len", "10", ]
    print('[testWhileTrain] step %d:' % testno)
    print(' '.join(cmd1))
    # ignore signals
    return subprocess.Popen(cmd1, preexec_fn=preexec)


if args.mode == "inference":
    if args.checkpoint is None:
        raise ValueError("The checkpoint file is needed to perform the test")

elif args.mode == "train":
    dataset = train_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    generator_F = generator(3, FLAGS=args).cuda()
    fnet = f_net().cuda()
    discriminator_F = discriminator(FLAGS=args).cuda()
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
    fnet_optimizer = torch.optim.Adam(fnet.parameters(), args.learning_rate, betas=(args.beta, 0.999), eps=args.adameps)
    GAN_FLAG = True
    if args.pre_trained_model:
        g_checkpoint = torch.load(args.g_checkpoint)
        generator_F.load_state_dict(g_checkpoint["model_state_dict"])
        gen_optimizer.load_state_dict(g_checkpoint["optimizer_state_dict"])
        current_epoch = g_checkpoint["epoch"]
        d_checkpoint = torch.load(args.d_checkpoint)
        discriminator_F.load_state_dict(d_checkpoint["model_state_dict"])
        tdiscrim_optimizer.load_state_dict(d_checkpoint["optimizer_state_dict"])
        f_checkpoint = torch.load(args.f_checkpoint)
        fnet.load_state_dict(f_checkpoint["model_state_dict"])
        fnet_optimizer.load_state_dict(f_checkpoint["optimizer_state_dict"])
    else:
        current_epoch = 0

    for e in range(current_epoch, args.max_epoch):
        d_loss = 0.
        g_loss = 0.
        f_loss = 0.
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            output = FRVSR(inputs, targets, args, discriminator_F, fnet, generator_F, batch_idx, counter1, counter2,
                           gen_optimizer, tdiscrim_optimizer, fnet_optimizer)

            fnet_optimizer.zero_grad()
            fnet_optimizer.step()
            f_loss = f_loss + ((1 / (batch_idx + 1)) * (output.fnet_loss.data - f_loss))

            gen_optimizer.zero_grad()
            gen_optimizer.step()
            g_loss = g_loss + ((1 / (batch_idx + 1)) * (output.gen_loss.data - g_loss))

            tdiscrim_optimizer.zero_grad()
            d_loss = d_loss + ((1 / (batch_idx + 1)) * (output.d_loss.data - d_loss))

            if (not GAN_FLAG):
                print("Not training Discriminator")
            else:

                if output.tb < args.Dbalance:
                    tdiscrim_optimizer.step()
                    counter1 += 1
                else:
                    counter2 += 1

        print("Epoch: {}".format(e + 1))
        print("\nGenerator loss is: {} \nDiscriminator loss is: {} \nFnet loss is: {}".format(d_loss, g_loss, f_loss))
        torchvision.utils.save_image(output.gen_output, fp="Gan_examples.jpg")
        torchvision.utils.save_image(
            targets.view(args.batch_size * args.RNN_N, 3, args.crop_size * 4, args.crop_size * 4), fp="real_image.jpg")
        torchvision.utils.save_image(inputs.view(args.batch_size * args.RNN_N, 3, args.crop_size, args.crop_size),
                                     fp="original_image.jpg")

        print("\nSaving model...")
        torch.save({
                'epoch': e,
                'model_state_dict': generator_F.state_dict(),
                'optimizer_state_dict': gen_optimizer.state_dict(),
                'loss': g_loss,
        }, "generator.pt")
        torch.save({
                'model_state_dict': fnet.state_dict(),
                'optimizer_state_dict': fnet_optimizer.state_dict(),
                'loss': f_loss,
        }, "fnet.pt")
        torch.save({
                'model_state_dict': discriminator_F.state_dict(),
                'optimizer_state_dict': tdiscrim_optimizer.state_dict(),
                'loss': d_loss,
        }, "discrim.pt")

