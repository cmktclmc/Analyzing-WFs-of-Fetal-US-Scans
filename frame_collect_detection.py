import os
import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

os.chdir('/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos')

# Import SonoNet
import torch 
from torch.autograd import Variable
from PIL import Image

# Importing necessary libraries
import time
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
from moviepy.editor import VideoFileClip
from scipy.spatial.transform import Rotation as R
import csv
import copy
from skimage.metrics import structural_similarity as ssim

from utils import (device)

import SonoNet.sononet as sononet
from utils_sononet import *

# Setup directories for input videos and outputs.
video_folder = r'data/videos/'
output_folder = r'output/frames_posereg/masks/'
os.makedirs(output_folder, exist_ok=True)

# SonoNet configuration
weights_path = 'sononet_finetuned.pth'

#sononet_model = sononet.SonoNet('SN64', weights=True).to(device)
sononet_model = sononet.SonoNet('SN64', weights=weights_path).to(device)
sononet_model.eval()
input_size_sononet = [224, 288]

label_names = ['3VV',
               '4CH',
               'Abdominal',
               'Background',
               'Brain (Cb.)',
               'Brain (Tv.)',
               'Femur',
               'Kidneys',
               'Lips',
               'LVOT',
               'Profile',
               'RVOT',
               'Spine (cor.)',
               'Spine (sag.)']



# Function to extract frames from video files at a specific rate
def extract_video_frames(video_file, video_folder, bb_x1, bb_y1, bb_w1, bb_h1, bb_x2, bb_y2, bb_w2, bb_h2):

    """
    Generator to extract frames from a specified video file at 10 fps.
    Yields each frame along with its index number.
    """

    video_path = os.path.join(video_folder, video_file)
    video_clip = VideoFileClip(video_path)
    previous_image = None
    black_pixel_threshold = 0.29
    for idx, frame in enumerate(video_clip.iter_frames(fps=10, dtype="uint8")):
        
        image = frame[bb_y1:bb_y1+bb_h1, bb_x1:bb_x1+bb_w1]

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_pixel_count = np.sum(red_mask > 0)

        if red_pixel_count > 10000:
            print('DOPPLER', red_pixel_count)
            
            image = frame[bb_y2:bb_y2+bb_h2, bb_x2:bb_x2+bb_w2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            previous_image = copy.copy(gray)
            yield frame, idx, 1, 0, 1
            continue

        image = frame[bb_y2:bb_y2+bb_h2, bb_x2:bb_x2+bb_w2]
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        black_pixels = np.sum(gray <= 0)
        total_pixels = image.size
        black_pixel_proportion = black_pixels / total_pixels
        print('BPP:', black_pixel_proportion)
        
        if black_pixel_proportion > black_pixel_threshold:
            print(f"Frame {idx}: Too many black pixels ({black_pixel_proportion:.2%}), skipping update.")
            yield frame, idx, 0, 1, 0
            continue
        
        if previous_image is None:
            previous_image = copy.copy(gray)
            yield frame, idx, 0, 1, 0
        else:
            ssim_index, _ = ssim(gray, previous_image, full=True)
            
            print(f"SSIM: {ssim_index}")
            
            # If the difference is all zeros, the images are the same
            if ssim_index > 0.98:
                print("Images are the same")
                same = 0
            else:
                print("Images are different")
                same = 1
            
            previous_image = copy.copy(gray)

            yield frame, idx, same, ssim_index, 0 


#---------------------------------------------------------------------

case_dictionary = {
    '0': ['Operator_17_Expert', '23w_train_out', 'Operator17', 420, 0, 1313, 1080, 0, 400, 360, 1080],
    '1': ['Operator_16_Expert', '23w_train_out', 'Operator16', 320, 50, 1036, 777, 777, 0, 300, 280, 669],
    #'2': ['Operator_15_Novice', '23w_train_out', 'Operator15', 320, 50, 1036, 777, 777, 0, 300, 280, 669],
    #'3': ['Operator_14_Expert', '23w_train_out', 'Operator14', 320, 50, 1036, 777, 777, 0, 300, 280, 669],
    #'4': ['Operator_13_Novice', '23w_train_out', 'Operator13', 320, 50, 1036, 777, 0, 300, 280, 669],
    #'5': ['Operator_12_Novice', '23w_train_out', 'Operator12', 320, 50, 1036, 777, 0, 300, 280, 669],
    #'6': ['Operator_11_Novice', '23w_train_out', 'Operator11', 320, 50, 1036, 777, 0, 300, 280, 669],
    #'7': ['Operator_10_Novice', '23w_train_out', 'Operator10', 320, 50, 1036, 777, 0, 300, 280, 669],
    #'8': ['Operator_9_Novice', '23w_train_out', 'Operator9', 320, 50, 1036, 777, 0, 300, 280, 669],
    #'9': ['Operator_8_Novice', '23w_train_out', 'Operator8', 320, 50, 1036, 777, 0, 300, 280, 669],
    #'10': ['Operator_7_Novice', '23w_train_out', 'Operator7', 320, 50, 1036, 777, 0, 300, 280, 669],
    #'11': ['Operator_6_Novice', '23w_train_out', 'Operator6', 320, 50, 1036, 777, 0, 300, 280, 669],
    #'12': ['Operator_5_Novice', '23w_train_out', 'Operator5', 205, 0, 545, 374, 0, 100, 190, 355],
    #'13': ['Operator_4_Novice', '23w_train_out', 'Operator4', 320, 50, 1036, 777, 0, 300, 280, 669],
    #'14': ['Operator_3_Novice', '23w_train_out', 'Operator3', 320, 50, 1036, 777, 0, 300, 280, 669],
    # '15': ['Operator_2_Novice', '23w_train_out', 'Operator2', 180, 0, 576, 408, 0, 100, 160, 374],
    #'16': ['Operator_1_Novice', '23w_train_out', 'Operator1', 320, 50, 1036, 777, 0, 300, 280, 669]
}

#---------------------------------------------------------------------

# Loop to process only specific videos defined in case_dictionary, rpocess them and runs inference
for exp, op in case_dictionary.items():
    video_filename = f'{op[0]}.mp4'
    print(f'Processing video: {video_filename}')
    video_file_path = os.path.join(video_folder, video_filename)
    
    if os.path.exists(video_file_path):
        frames = []
        sononet = []

        bb_x2 = op[7]
        bb_y2 = op[8]
        bb_w2 = op[9]
        bb_h2 = op[10]

        bb_x1 = op[3]
        bb_y1 = op[4]
        bb_w1 = op[5]
        bb_h1 = op[6]

        for frame, idx, same, ssim_index, doppler in extract_video_frames(video_filename, video_folder, bb_x1, bb_y1, bb_w1, bb_h1, bb_x2, bb_y2, bb_w2, bb_h2):
            frames.append(frame)

            sononet.append((idx, same, ssim_index, doppler))
            print(f"row: {idx} {same} {ssim_index} {doppler}")

        sononet_path = f'output/frame_collect'
        os.makedirs(sononet_path, exist_ok=True)

        with open(os.path.join(sononet_path, f'frame_collect_{op[0]}.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Index', 'Collected', 'SSIM', 'Doppler'])
            writer.writerows(sononet)
    else:
        print(f"Video file {video_filename} not found in directory.")

print("Completed")