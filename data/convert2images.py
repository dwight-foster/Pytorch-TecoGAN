import torchvision
import torch
from torchvision import io, datasets, transforms, utils
import cv2
import PIL
import os
import argparse
from converter import Converter
from tqdm import tqdm

conv = Converter()

parser = argparse.ArgumentParser()
parser.add_argument("--VideoDir", type=str, help="Path to videos")
parser.add_argument("--OutputDir", default="../TrainingDataPath", type=str, help="Director for output images")
parser.add_argument("--keep_video", default=True, type=bool, help="Decides on whether to keep the input video")
parser.add_argument("--keepshortvideos", default=True, type=bool, help="Keep images from videos under max frames")
parser.add_argument("--numframes", default=120, type=int, help="Should be one more than max_frm in main.py")
args = parser.parse_args()

videos = os.listdir(args.VideoDir)
num_videos = len(videos)


def convert():
    for i in tqdm(range(num_videos)):

        path = args.VideoDir + videos[i]
        output = ".." + path.split(".")[2] + ".mp4"
        if os.path.exists(output):
            continue
        print(output)

        convert = convert = conv.convert(path, output, {
            'format': 'mp4',
            'audio': {
                'codec': 'aac',
                'samplerate': 11025,
                'channels': 2
            },
            'video': {
                'codec': 'hevc',
                'width': 720,
                'height': 400,
                'fps': 25
            }})
        for timecode in convert:
            print(f"\rConverting ({timecode:.2f})...")


# convert()

for i in tqdm(range(num_videos)):
    if not os.path.isdir(args.VideoDir):
        print("Input folder does not exist!")
        break
    if not os.path.isdir(args.OutputDir):
        os.mkdir(args.OutputDir)
    path = args.VideoDir + videos[i]

    path = args.VideoDir + videos[i]
    images = io.read_video(path)[0]
    if images.shape[0] < args.numframes:
        for n in range(images.shape[0]):
            folder_path = args.OutputDir + "scene_" + str(1000 + i)
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
            image = images[n]
            image = image.permute(2, 0, 1)
            image_path = folder_path + "/" + "col_high_" + str(n).zfill(4) + ".png"
            utils.save_image(image.float(), fp=image_path)
    else:
        for n in range(args.numframes):
            folder_path = args.OutputDir + "scene_" + str(1000 + i)
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
            image = images[n]
            image = image.permute(2, 0, 1)
            image_path = folder_path + "/" + "col_high_" + str(n).zfill(4) + '.png'
            utils.save_image(image.float(), image_path)
    if not args.keep_video:
        os.remove(path)
