import numpy as np
import os, math, time, collections, numpy as np

''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
Disable Logs for now '''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
from torch.utils.data import Dataloader
import random as rn

# fix all randomness, except for multi-treading or GPU process
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
torch.set_random_seed(1234)

import sys, shutil, subprocess

from ops import *
from dataloader import Inference_Dataset, train_dataset
from frvsr import generator, f_net
from Teco import FRVSR, TecoGAN, discriminator

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('rand_seed', default=1, 'random seed')

# Directories
parser.add_argument('input_dir_LR',  'The directory of the input resolution input data, for inference mode')
parser.add_argument('input_dir_len', default=-1, 'length of the input for inference mode, -1 means all')
parser.add_argument('input_dir_HR', 'The directory of the input resolution input data, for inference mode')
parser.add_argument('mode', default='inference', 'train, or inference')
parser.add_argument('output_dir', 'The output directory of the checkpoint')
parser.add_argument('output_pre', default='', 'The name of the subfolder for the images')
parser.add_argument('output_name', default='output', 'The pre name of the outputs')
parser.add_argument('output_ext', default='jpg', 'The format of the output when evaluating')
parser.add_argument('summary_dir', 'The dirctory to output the summary')

# Models
parser.add_argument('checkpoint', 'If provided, the weight will be restored from the provided checkpoint')
parser.add_argument('num_resblock', default=16, 'How many residual blocks are there in the generator')
# Models for training
parser.add_argument('pre_trained_model', default=False, 'If True, the weight of generator will be loaded as an initial point'
                                                 'If False, continue the training')
parser.add_argument('vgg_ckpt', 'path to checkpoint file for the vgg19')

# Machine resources
parser.add_argument('cudaID', default='0', 'CUDA devices')
parser.add_argument('queue_thread', default=6, 'The threads of the queue (More threads can speedup the training process.')
parser.add_argument('name_video_queue_capacity', default=512, 'The capacity of the filename queue (suggest large to ensure'
                                                       'enough random shuffle.')
parser.add_argument('video_queue_capacity', default=256, 'The capacity of the video queue (suggest large to ensure'
                                                  'enough random shuffle')
parser.add_argument('video_queue_batch', default=2, 'shuffle_batch queue capacity')

# Training details
# The data preparing operation
parser.add_argument('RNN_N', default=10, 'The number of the rnn recurrent length')
parser.add_argument('batch_size', default=4, 'Batch size of the input batch')
parser.add_argument('flip', default=True, 'Whether random flip data augmentation is applied')
parser.add_argument('random_crop', default=True, 'Whether perform the random crop')
parser.add_argument('movingFirstFrame',default= True, 'Whether use constant moving first frame randomly.')
parser.add_argument('crop_size', default=32, 'The crop size of the training image')
# Training data settings
parser.add_argument('input_video_dir',default= '', 'The directory of the video input data, for training')
parser.add_argument('input_video_pre', default='scene', 'The pre of the directory of the video input data')
parser.add_argument('str_dir',default= 1000, 'The starting index of the video directory')
parser.add_argument('end_dir',default= 2000, 'The ending index of the video directory')
parser.add_argument('end_dir_val',default= 2050, 'The ending index for validation of the video directory')
parser.add_argument('max_frm', default=119, 'The ending index of the video directory')
# The loss parameters

parser.add_argument('vgg_scaling', default=-0.002, 'The scaling factor for the VGG perceptual loss, disable with negative value')
parser.add_argument('warp_scaling',default= 1.0, 'The scaling factor for the warp')
parser.add_argument('pingpang', default=False, 'use bi-directional recurrent or not')
parser.add_argument('pp_scaling', default=1.0, 'factor of pingpang term, only works when pingpang is True')
# Training parameters
parser.add_argument('EPS',default= 1e-12, 'The eps added to prevent nan')
parser.add_argument('learning_rate',default= 0.0001, 'The learning rate for the network')
parser.add_argument('decay_step', default=500000, 'The steps needed to decay the learning rate')
parser.add_argument('decay_rate', default=0.5, 'The decay rate of each decay step')
parser.add_argument('stair', default=False, 'Whether perform staircase decay. True => decay in discrete interval.')
parser.add_argument('beta', default=0.9, 'The beta1 parameter for the Adam optimizer')
parser.add_argument('adameps', default=1e-8, 'The eps parameter for the Adam optimizer')
parser.add_argument('max_epoch', default=10, 'The max epoch for the training')
parser.add_argument('max_iter',default= 1000000, 'The max iteration of the training')
parser.add_argument('display_freq', default=20, 'The diplay frequency of the training process')
parser.add_argument('summary_freq', default=100, 'The frequency of writing summary')
parser.add_argument('save_freq', default=10000, 'The frequency of saving images')
# Dst parameters
parser.add_argument('ratio', default=0.01, 'The ratio between content loss and adversarial loss')
parser.add_argument('Dt_mergeDs', default=True, 'Whether only use a merged Discriminator.')
parser.add_argument('Dt_ratio_0', default=1.0, 'The starting ratio for the temporal adversarial loss')
parser.add_argument('Dt_ratio_add', default=0.0, 'The increasing ratio for the temporal adversarial loss')
parser.add_argument('Dt_ratio_max', default=1.0, 'The max ratio for the temporal adversarial loss')
parser.add_argument('Dbalance', default=0.4, 'An adaptive balancing for Discriminators')
parser.add_argument('crop_dt', default=0.75, 'factor of dt crop')  # dt input size = crop_size*crop_dt
parser.add_argument('D_LAYERLOSS', default=True, 'Whether use layer loss from D')

FLAGS = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cudaID
my_seed = FLAGS.rand_seed
rn.seed(my_seed)
np.random.seed(my_seed)
torch.set_random_seed(my_seed)

if FLAGS.output_dir is None:
    raise ValueError("The output directory is needed")
if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

if not os.path.exists(FLAGS.summary_dir):
    os.mkdir(FLAGS.summary_dir)

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(FLAGS.summary_dir + "logfile.txt", "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.log.flush()

sys.stdout = Logger()
def preexec(): # Don't forward signals.
    os.setpgrp()

def testWhileTrain(FLAGS, testno = 0):
    '''
        this function is called during training, Hard-Coded!!
        to try the "inference" mode when a new model is saved.
        The code has to be updated from machine to machine...
        depending on python, and your training settings
    '''
    desstr = os.path.join(FLAGS.output_dir, 'train/') # saving in the ./train/ directory
    cmd1 = ["python3", "main.py", # never tested with python2...
        "--output_dir", desstr,
        "--summary_dir", desstr,
        "--mode","inference",
        "--num_resblock", "%d"%FLAGS.num_resblock,
        "--checkpoint", os.path.join(FLAGS.output_dir, 'model-%d'%testno),
        "--cudaID", FLAGS.cudaID]
    # a folder for short test
    cmd1 += ["--input_dir_LR", "./LR/calendar/", # update the testing sequence
             "--output_pre", "", # saving in train folder directly
             "--output_name", "%09d"%testno, # name
             "--input_dir_len", "10",]
    print('[testWhileTrain] step %d:'%testno)
    print(' '.join(cmd1))
    # ignore signals
    return subprocess.Popen(cmd1, preexec_fn = preexec)

if FLAGS.mode == "inference":
    if FLAGS.checkpoint is None:
        raise ValueError("The checkpoint file is needed to perform the test")

elif FLAGS.mode == "train":
    dataset = train_dataset(FLAGS)
    dataloader = Dataloader(dataset, batch_size=16, shuffle=True)
    generator_F = generator()
    fnet = f_net()
    discriminator_F = discriminator()
    counter1 = 0
    counter2 = 0
    min_gen_loss = np.inf

    for e in range(FLAGS.max_epoch):
        d_loss = 0.
        g_loss = 0.
        f_loss = 0.
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            output = FRVSR(inputs,targets, FLAGS, discriminator_F, fnet, generator_F, batch_idx, counter1, counter2)
            d_loss = d_loss + ((1 / (batch_idx + 1)) * (output["d_loss"].data - d_loss))
            g_loss = g_loss + ((1 / (batch_idx + 1)) * (output["gen_loss"].data - g_loss))
            f_loss = f_loss + ((1 / (batch_idx + 1)) * (output["fnet_loss"].data - f_loss))

        print("Epoch: {}".format(e+1))
        print("\nGenerator loss is: {} \n Discriminator loss is: {} \n Fnet loss is: {}".format(d_loss,g_loss,f_loss))
        if g_loss < min_gen_loss:
            print("\nSaving model...")
            torch.save(generator_F.state_dict(), "generator.pt")
            torch.save(fnet.state_dict(), "fnet.pt")


