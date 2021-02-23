import torchvision
import torch
from torchvision import io, datasets, transforms, utils
import PIL
import os
import argparse
from tqdm import tqdm
import cv2
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser()
parser.add_argument("--VideoDir", type=str, help="Path to videos")
parser.add_argument("--OutputDir", default="../TrainingDataPath", type=str, help="Director for output images")
parser.add_argument("--keep_video", default=True, type=str2bool, help="Decides on whether to keep the input video")
parser.add_argument("--keepshortvideos", default=True, type=str2bool,  help="Keep images from videos under max frames")
parser.add_argument("--numframes", default=120, type=int, help="Should be one more than max_frm in main.py")
parser.add_argument("--max_scenes", default=2000, type=int, help="The max number of scenes to save")
args = parser.parse_args()
z = 0
folders = os.listdir(args.VideoDir)
num_folders = len(folders)
for n in tqdm(range(num_folders)):
    videos = os.listdir(args.VideoDir + folders[n])
    num_videos = len(videos)
    for i in range(num_videos):

        if not os.path.isdir(args.VideoDir):
            print("Input folder does not exist!")
            break
        if not os.path.isdir(args.OutputDir):
            os.mkdir(args.OutputDir)

        path = args.VideoDir + folders[n] + "/" + videos[i]
        cap = cv2.VideoCapture(path)
        images = []
        for k in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Display the resulting frame
            images.append(frame)

        # When everything done, release the capture
        cap.release()
        if len(images) == 0:
            print("here")
            continue

        elif len(images) < args.numframes:
            if bool(args.keepshortvideos):
                for x in range(len(images)):
                    folder_path = args.OutputDir + "scene_" + str(1000 + z)
                    if not os.path.isdir(folder_path):
                       os.mkdir(folder_path)
                    image = images[x]
                    if image is None:
                        pass
                    else:
                        image_path = folder_path + "/" + "col_high_" + str(x).zfill(4) + ".png"
                        cv2.imwrite(image_path, image)

                z += 1
            else:
                continue
        else:
            for x in range(args.numframes):
                folder_path = args.OutputDir + "scene_" + str(1000 + z)
                if not os.path.isdir(folder_path):
                    os.mkdir(folder_path)
                image = images[x]
                image_path = folder_path + "/" + "col_high_" + str(x).zfill(4) + ".png"
                if image is None:
                    pass
                else:
                    cv2.imwrite(image_path, image)
            z += 1
        if not args.keep_video:
            os.remove(path)

        if z >= args.max_scenes:
            break
    if z >= args.max_scenes: 
        break
