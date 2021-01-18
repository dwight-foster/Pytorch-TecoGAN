from ops import *
import torchvision
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

import cv2 as cv
import collections, os, math
import numpy as np
from scipy import signal
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_video_dir", nargs="?", type=str, default="../TrainingDataPath")
parser.add_argument("str_dir", nargs="?", type=int, default=2000)
parser.add_argument("end_dir", nargs="?", type=int, default=2400)
parser.add_argument("max_frm", nargs="?", default=119)
parser.add_argument("input_video_pre", nargs="?", default="scene")
parser.add_argument("crop_size", nargs="?", default=32)
args = parser.parse_args()


class Inference_Dataset(Dataset):
    def __init__(self, FLAGS):
        filedir = FLAGS.input_dir_LR
        self.downSP = False
        if (FLAGS.input_dir_LR is None) or (not os.path.exists(FLAGS.input_dir_LR)):
            if (FLAGS.input_dir_HR is None) or (not os.path.exists(FLAGS.input_dir_HR)):
                raise ValueError('Input directory not found')
            filedir = FLAGS.input_dir_HR
            self.downSP = True

        image_list_LR_temp = os.listdir(filedir)
        image_list_LR_temp = [_ for _ in image_list_LR_temp if _.endswith(".png")]
        image_list_LR_temp = sorted(image_list_LR_temp)  # first sort according to abc, then sort according to 123
        image_list_LR_temp.sort(key=lambda f: int(''.join(list(filter(str.isdigit, f))) or -1))
        if FLAGS.input_dir_len > 0:
            image_list_LR_temp = image_list_LR_temp[:FLAGS.input_dir_len]

        self.image_list_LR = [os.path.join(filedir, _) for _ in image_list_LR_temp]

        # Read in and preprocess the images

    def __len__(self):
        return len(self.image_list_LR)

    def __getitem__(self, idx):
        def preprocess_test(name):
            im = cv.imread(name, 3).astype(np.float32)[:, :, ::-1]

            if self.downSP:
                icol_blur = cv.GaussianBlur(im, (0, 0), sigmaX=1.5)
                im = icol_blur[::4, ::4, ::]
            im = im / 255.0  # np.max(im)
            return im

        image = preprocess_test(self.image_list_LR[idx])
        if True:  # a hard-coded symmetric padding
            self.image_list_LR = self.image_list_LR[5:0:-1] + self.image_list_LR
            image_LR = image[5:0:-1] + image
        return torch.from_numpy(image_LR).float()


class train_dataset(Dataset):
    def __init__(self, FLAGS):
        if FLAGS.input_video_dir == '':
            raise ValueError('Video input directory input_video_dir is not provided')
        if not os.path.exists(FLAGS.input_video_dir):
            raise ValueError('Video input directory not found')
        self.image_list_len = []
        image_set_lists = []
        for dir_i in range(FLAGS.str_dir, FLAGS.end_dir + 1):
            inputDir = os.path.join(FLAGS.input_video_dir, '%s_%04d' % (FLAGS.input_video_pre, dir_i))
            if os.path.exists(inputDir):  # the following names are hard coded: col_high_
                if not os.path.exists(os.path.join(inputDir, 'col_high_%04d.png' % FLAGS.max_frm)):
                    print("Skip %s, since folder doesn't contain enough frames!" % inputDir)
                    continue

                image_list = [os.path.join(inputDir, 'col_high_%04d.png' % frame_i)
                              for frame_i in range(FLAGS.max_frm + 1)]

                self.image_list_len.append(os.path.join(inputDir, 'col_high_%04d.png' % frame_i)
                                           for frame_i in range(FLAGS.max_frm + 1))

                for i in range(110):
                    rnn_list = image_list[i:i + 10]
                    image_set_lists.append(rnn_list)
        self.image_set_lists = image_set_lists
        self.lr_first = transforms.RandomCrop(FLAGS.crop_size)
        self.hr_first = transforms.RandomCrop(FLAGS.crop_size * 4)
        self.hr_transforms = transforms.Compose(
            [transforms.Resize((FLAGS.crop_size * 4, FLAGS.crop_size * 4)), transforms.ToTensor()])
        self.lr_transforms = transforms.Compose(
            [transforms.Resize((FLAGS.crop_size, FLAGS.crop_size)), transforms.ToTensor()])

    def __len__(self):
        return len(self.image_list_len)

    def __getitem__(self, idx):
        rnn_images = self.image_set_lists[idx]
        hr_images = []
        lr_images = []
        for i in range(len(rnn_images)):
            hr_image = Image.open(rnn_images[i])
            lr_image = hr_image
            if i == 0:
                hr_image = self.hr_first(hr_image)
                lr_image = self.lr_first(lr_image)
            hr_image = self.hr_transforms(hr_image)
            lr_image = self.lr_transforms(lr_image)
            lr_images.append(lr_image.unsqueeze(0))
            hr_images.append(hr_image.unsqueeze(0))
        hr_images = torch.cat(hr_images, dim=0)
        lr_images = torch.cat(lr_images, dim=0)
        return [lr_images.float(), hr_images.float()]

dataset = train_dataset(args)
dataloader = DataLoader(dataset,batch_size=4)
lr, hr = next(iter(dataloader))