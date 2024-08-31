import os
import warnings

# Suppressing unnecessary warnings to make the output cleaner
warnings.filterwarnings("ignore")

# Setting the GPU device to be used
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Setting the working directory to the specific path for debbuging purposes
# Comment out if not needed
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

# Importing functions and classes from other files in the project

from utils import (device)

import SonoNet.sononet as sononet
from utils_sononet import *

# Setup directories for input videos and outputs.
video_folder = r'data/videos/'
output_folder = r'output/frames_posereg/masks/'
os.makedirs(output_folder, exist_ok=True)

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



# ---------------------------------------------------------------------
# STEP 1: EXTRACT VIDEO FRAMES
# ---------------------------------------------------------------------

def imcrop(image, bbox_x, bbox_y, bbox_width, bbox_height):
    """ Crop an image to a crop range """
    crop_range = [(bbox_y, bbox_y+bbox_height), (bbox_x, bbox_x+bbox_width)]  # [(top, bottom), (left, right)]

    return image[crop_range[0][0]:crop_range[0][1],
                 crop_range[1][0]:crop_range[1][1], ...]

def prepare_image_1(frame, bb_x, bb_y, bb_w, bb_h):
    # Convert frame to grayscale and resize
    image = imcrop(frame, bb_x, bb_y, bb_w, bb_h)
    image = np.array(Image.fromarray(np.uint8(image * 255.0)).resize(input_size_sononet, resample=Image.BICUBIC))
    image = np.mean(image, axis=2)
    return image

def prepare_image_2(image):
    # convert to 4D tensor of type float32
    image_data = np.float32(np.reshape(image, (1, 1, image.shape[0], image.shape[1])))

    # normalise images by substracting mean and dividing by standard dev.
    mean = image_data.mean()
    std = image_data.std()
    image_data = np.array(255.0 * np.divide(image_data - mean, std), dtype=np.float32)
    return image_data

# Function to extract frames from video files at a specific rate
def extract_video_frames(video_file, video_folder, bb_x, bb_y, bb_w, bb_h):

    """
    Generator to extract frames from a specified video file at 10 fps.
    Yields each frame along with its index number.
    """

    # SonoNet configuration
    weights_path = '/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/final_models/sononet_finetuned/sononet_finetuned_gt_labels_w_orig_2.pth'

    #sononet_model = sononet.SonoNet('SN64', weights=True).to(device)
    net = sononet.SonoNet('SN64', weights=weights_path).to(device)

    print('Testing with newly tuned model')
    net.eval()  # Switch to evaluation mode


    video_path = os.path.join(video_folder, video_file)
    video_clip = VideoFileClip(video_path)
    
    for idx, frame in enumerate(video_clip.iter_frames(fps=10, dtype="uint8")):
        
        image = prepare_image_1(frame, bb_x, bb_y, bb_w, bb_h)
        image_data = prepare_image_2(image)

        # Classify with SonoNet
        with torch.no_grad():
            x = Variable(torch.from_numpy(image_data).to(device))
            outputs = net(x)

            confidence, prediction = torch.max(outputs.data, 1)
            predicted_label = label_names[prediction[0].item()]
            if predicted_label != 'Background':
                print(f"{idx} Model Outputs: {outputs}")
                print(f"Predicted Label: {predicted_label}, Confidence: {confidence[0].item()}")


        yield frame, idx, confidence, prediction # Yield frame, index, and predicted class


#---------------------------------------------------------------------

# case_dictionary = {
#     # '0': ['Operator_17_Expert', '23w_train_out', 'Operator17', 420, 0, 1313, 1080],
#     #'1': ['Operator_16_Expert', '23w_train_out', 'Operator16', 320, 50, 1036, 777],
#     # '2': ['Operator_15_Novice', '23w_train_out', 'Operator15', 320, 50, 1036, 777],
#     # '3': ['Operator_14_Expert', '23w_train_out', 'Operator14', 320, 50, 1036, 777],
#     # '4': ['Operator_13_Novice', '23w_train_out', 'Operator13', 320, 50, 1036, 777],
#     # '5': ['Operator_12_Novice', '23w_train_out', 'Operator12', 320, 50, 1036, 777],
#     # '6': ['Operator_11_Novice', '23w_train_out', 'Operator11', 320, 50, 1036, 777],
#     # '7': ['Operator_10_Novice', '23w_train_out', 'Operator10', 320, 50, 1036, 777],
#     #'8': ['Operator_9_Novice', '23w_train_out', 'Operator9', 320, 50, 1036, 777],
#     # '9': ['Operator_8_Novice', '23w_train_out', 'Operator8', 320, 50, 1036, 777],
#     #'10': ['Operator_7_Novice', '23w_train_out', 'Operator7', 320, 50, 1036, 777],
#     # '11': ['Operator_6_Novice', '23w_train_out', 'Operator6', 320, 50, 1036, 777],
#     # '12': ['Operator_5_Novice', '23w_train_out', 'Operator5', 205, 0, 545, 374],
#     # '13': ['Operator_4_Novice', '23w_train_out', 'Operator4', 320, 50, 1036, 777],
#     # '14': ['Operator_3_Novice', '23w_train_out', 'Operator3', 320, 50, 1036, 777],
#     # '15': ['Operator_2_Novice', '23w_train_out', 'Operator2', 180, 0, 576, 408],
#     '16': ['Operator_1_Novice', '23w_train_out', 'Operator1', 320, 50, 1036, 777]
# }

case_dictionary = {'2': ['Operator_15_Novice', '23w_train_out', 'Operator15', 320, 50, 1036, 777, 'Breech'], 
                        #'4': ['Operator_13_Novice', '23w_train_out', 'Operator13', 320, 50, 1036, 777, 'Breech'], 
                        '16': ['Operator_1_Novice', '23w_train_out', 'Operator1', 320, 50, 1036, 777, 'Cephalic'], 
                        '1': ['Operator_16_Expert', '23w_train_out', 'Operator16', 320, 50, 1036, 777, 'Cephalic']}

#---------------------------------------------------------------------

# Loop to process only specific videos defined in case_dictionary, rpocess them and runs inference
for exp, op in case_dictionary.items():
#for video_file in os.listdir(video_folder):
    #print(f'Processing video: {video_file}')
    #if video_file.endswith(".mp4"):
        #video_filename = video_file
    video_filename = f'{op[0]}.mp4'
    print(f'Processing video: {video_filename}')
    video_file_path = os.path.join(video_folder, video_filename)
    
    if os.path.exists(video_file_path):
        frames = []

        bb_x = op[3]
        bb_y = op[4]
        bb_w = op[5]
        bb_h = op[6]
        sononet_path = f'output/sononet_redo_orig'
        os.makedirs(sononet_path, exist_ok=True)
        # Save results to CSV file
        with open(os.path.join(sononet_path, f'sononet_prediction_{op[0]}.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Index', 'Predicted Label', 'Confidence'])
            for frame, idx, conf, pred in extract_video_frames(video_filename, video_folder, bb_x, bb_y, bb_w, bb_h):
                frames.append(frame)
                writer.writerow([idx, label_names[pred[0]], conf[0].item()])
                print(f"row: {idx} {label_names[pred[0]]} {conf[0].item()}")
        
            
            
    else:
        print(f"Video file {video_filename} not found in directory.")

# Notify when processing is complete for all specified setups.
print("Completed")