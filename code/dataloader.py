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


class inference_dataset(Dataset):
    def __init__(self, args):
        filedir = args.input_dir_LR
        self.args = args
        self.downSP = False
        if (args.input_dir_LR is None) or (not os.path.exists(args.input_dir_LR)):
            if (args.input_dir_HR is None) or (not os.path.exists(args.input_dir_HR)):
                raise ValueError('Input directory not found')
            filedir = args.input_dir_HR
            self.downSP = True
        self.filedir = filedir
        self.image_list_LR = os.listdir(filedir)

        # Read in and preprocess the images

    def __len__(self):
        return len(self.image_list_LR)

    def __getitem__(self, idx):
        path = self.image_list_LR[idx]
        imgs = []
        for img in os.listdir(self.filedir + "/" + path):
            image = Image.open(self.filedir + "/" + path + "/" + img)
            image = transforms.functional.resize(image, size=(self.args.crop_size, self.args.crop_size))
            image = transforms.functional.to_tensor(image)
            imgs.append(image)
        images = torch.stack(imgs, dim=0)
        return images


class train_dataset(Dataset):
    def __init__(self, args):
        if args.input_video_dir == '':
            raise ValueError('Video input directory input_video_dir is not provided')
        if not os.path.exists(args.input_video_dir):
            raise ValueError('Video input directory not found')
        self.image_list_len = []
        image_set_lists = []
        for dir_i in range(args.str_dir, args.end_dir + 1):
            inputDir = os.path.join(args.input_video_dir, '%s_%04d' % (args.input_video_pre, dir_i))
            if os.path.exists(inputDir):  # the following names are hard coded: col_high_
                if not os.path.exists(os.path.join(inputDir, 'col_high_%04d.png' % args.max_frm)):
                    print("Skip %s, since folder doesn't contain enough frames!" % inputDir)
                    continue

                image_list = [os.path.join(inputDir, 'col_high_%04d.png' % frame_i)
                              for frame_i in range(args.max_frm + 1)]

                self.image_list_len.append(os.path.join(inputDir, 'col_high_%04d.png' % frame_i)
                                           for frame_i in range(args.max_frm + 1))

                for i in range(110):
                    rnn_list = image_list[i:i + 10]
                    image_set_lists.append(rnn_list)
        self.image_set_lists = image_set_lists
        self.lr_first = transforms.RandomResizedCrop(args.crop_size)
        self.hr_first = transforms.RandomResizedCrop(args.crop_size * 4)
        self.hr_transforms = transforms.Compose(
            [transforms.Resize((args.crop_size * 4, args.crop_size * 4)), transforms.ToTensor()])
        self.lr_transforms = transforms.Compose(
            [transforms.Resize((args.crop_size, args.crop_size)), transforms.ToTensor()])

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
