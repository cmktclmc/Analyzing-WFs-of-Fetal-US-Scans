import os
import warnings
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from collections import Counter
import pandas as pd
import cv2
import torch
from moviepy.editor import VideoFileClip

import csv

from collections import defaultdict

from utils import (device)

from sklearn.metrics import precision_score, recall_score, f1_score

import SonoNet.sononet as sononet
from utils_sononet import *

import logging
logging.getLogger('matplotlib.font_manager').disabled = True

import copy

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

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

# Suppressing unnecessary warnings to make the output cleaner
warnings.filterwarnings("ignore")

# Setting the GPU device to be used
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Setting the working directory to the specific path for debbuging purposes
# Comment out if not needed
os.chdir('/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos')


# Configuration
network_name = 'SN64'   # 'SN16', 'SN32' or 'SN64'
GPU_NR = 4     # Choose the device number of your GPU


# Setup directories for input videos and outputs.
video_folder = r'data/videos/'
gt_folder = r'output/cleaned_frame_collect_labels/'
output_folder = r'output/frames_posereg/masks/'
os.makedirs(output_folder, exist_ok=True)

input_size_sononet = [224, 288]

sononet_label_list = [
               'Heart including',
               'Brain with Skull Head and Neck',
               'Abdomen',
               'Background',
               'Spine',
               'Maternal Anatomy including Doppler',
               'Kidneys',
               'Arms',
               'Nose and Lips',
               'Profile',
               'Legs',
               'Feet',
               'Hands',
               'Umbilical Cord Insertion',
               'Femur']


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
def extract_video_frames(net, video_filename, video_folder, gt_filename, conv_df, bb_x, bb_y, bb_w, bb_h):
    """
    Generator to extract frames from a specified video file at 10 fps.
    Yields each frame along with its index number.
    """

    gt_path = os.path.join(gt_folder, gt_filename)
    labels_df = pd.read_csv(gt_path)

    video_path = os.path.join(video_folder, video_filename)
    video_clip = VideoFileClip(video_path)

    for idx, frame in enumerate(video_clip.iter_frames(fps=10, dtype="uint8")):
        label = labels_df.loc[idx, 'New Label']
        if pd.isna(label):
            label = 3
        elif label not in conv_df['Item'].values:
            label = 3
        else:
            label = conv_df.loc[conv_df['Item'] == label, 'Category'].iloc[0]
            if 'Other' in label:
                label = 3
            else:
                label = sononet_label_list.index(label)
        
        image = prepare_image_1(frame, bb_x, bb_y, bb_w, bb_h)
        image_data = prepare_image_2(image)

        # Classify with SonoNet
        with torch.no_grad():
            x = Variable(torch.from_numpy(image_data).to(device))
            outputs = net(x)

            confidence, prediction = torch.max(outputs.data, 1)
            #predicted_label = sononet_label_list[prediction[0].item()]
            #if predicted_label != 'Background':
                #print(f"{idx} Model Outputs: {outputs}")
                #print(f"Predicted Label: {predicted_label}, Confidence: {confidence[0].item()}")

            yield frame, idx, label, confidence.item(), prediction.item() # Yield frame, index, and predicted class



#---------------------------------------------------------------------
case_dictionary = {'2': ['Operator_15_Novice', '23w_train_out', 'Operator15', 320, 50, 1036, 777, 'Breech'], 
                        '4': ['Operator_13_Novice', '23w_train_out', 'Operator13', 320, 50, 1036, 777, 'Breech'], 
                        '16': ['Operator_1_Novice', '23w_train_out', 'Operator1', 320, 50, 1036, 777, 'Cephalic'], 
                        '1': ['Operator_16_Expert', '23w_train_out', 'Operator16', 320, 50, 1036, 777, 'Cephalic']}

class ModifiedSonoNet(nn.Module):
    def __init__(self, original_model, num_classes):
        super(ModifiedSonoNet, self).__init__()
        # Copy the original features and adaptation layers
        self.features = original_model.features
        self.adaptation = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(num_classes)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # GAP layer

    def forward(self, x):
        x = self.features(x)
        x = self.adaptation(x)
        x = self.gap(x)  # Apply GAP
        x = torch.flatten(x, 1)  # Flatten to shape [N, C]
        return x



# Loop to process only specific videos defined in case_dictionary, rpocess them and runs inference
# Initialize lists to hold true labels and predicted labels
true_labels = []
pred_labels = []

 # # SonoNet configuration
weights_path = '/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/final_models/sononet_fine_block/sononet_finetuned_block_new.pth'

orig_net = sononet.SonoNet(network_name, weights=True)
net = ModifiedSonoNet(orig_net, num_classes=15)

# Load the model's state dictionary
net.load_state_dict(torch.load(weights_path))

# Move to GPU
print('Moving to GPU:')
torch.cuda.device(GPU_NR)
print(torch.cuda.get_device_name(torch.cuda.current_device()))
net.cuda()

print('Testing with newly tuned model')
net.eval()  # Switch to evaluation mode


# correct_predictions = 0
# valid_labels = 0
# label_correct_counts = {i: 0 for i in range(len(sononet_label_list))}
# label_total_counts = {i: 0 for i in range(len(sononet_label_list))}
# misclassifications = {}

for exp, op in case_dictionary.items():
    video_filename = f'{op[0]}.mp4'
    gt_filename = f'clean_frame_collect_{op[0]}.csv'
    print(f'Processing video: {video_filename}')
    video_file_path = os.path.join(video_folder, video_filename)
    conv_df = pd.read_csv('/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/sononet_label_categories_expanded.csv')

    if os.path.exists(video_file_path):
        sononet_list = []

        bb_x = op[3]
        bb_y = op[4]
        bb_w = op[5]
        bb_h = op[6]
        sononet_path = f'output/sononet_block_final'
        os.makedirs(sononet_path, exist_ok=True)
        # Save results to CSV file
        with open(os.path.join(sononet_path, f'sononet_prediction_{op[0]}.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Index', 'Predicted Label', 'Confidence'])
            for frame, idx, true_label_int, conf, pred_label_int in extract_video_frames(net, video_filename, video_folder, gt_filename, conv_df, bb_x, bb_y, bb_w, bb_h):
                pred_label = sononet_label_list[pred_label_int]
                true_label = sononet_label_list[true_label_int]

                true_labels.append(true_label)
                pred_labels.append(pred_label)

                writer.writerow([idx, pred_label, conf])
                print(f"row: {idx} {pred_label} {conf}")


            # if true_label not in label_correct_counts:
            #     label_correct_counts[true_label] = 0
            #     label_total_counts[true_label] = 0
            # label_total_counts[true_label] += 1
            
            # if true_label == pred_label:
            #     label_correct_counts[true_label] += 1
            # else:
            #     if (true_label, pred_label) not in misclassifications:
            #         misclassifications[(true_label, pred_label)] = 0
            #     misclassifications[(true_label, pred_label)] += 1

        
        
            
    else:
        print(f"Video file {video_filename} not found in directory.")

print(len(true_labels))
print(len(pred_labels))

# Calculate and print the metrics
accuracy = accuracy_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels, average='macro')  # Use 'micro', 'macro' or 'weighted' based on your requirement
precision = precision_score(true_labels, pred_labels, average='macro')
f1 = f1_score(true_labels, pred_labels, average='macro')

print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'F1 Score: {f1}')


# Print accuracy for each label
# print("Accuracy for each label:")
# for label_id in sorted(label_correct_counts.keys()):
#     if label_total_counts[label_id] != 0:
#         accuracy = label_correct_counts[label_id] / label_total_counts[label_id]
#         print(f"Label {sononet_label_list[label_id]}: {accuracy * 100:.9f}%")

# # Overall accuracy including label 3
# overall_accuracy_incl_3 = sum(label_correct_counts.values()) / sum(label_total_counts.values())
# print(f'Overall Accuracy including label 3: {overall_accuracy_incl_3 * 100:.9f}%')

# # Overall accuracy excluding label 3
# correct_counts_excl_3 = sum(count for label, count in label_correct_counts.items() if label != 3)
# total_counts_excl_3 = sum(count for label, count in label_total_counts.items() if label != 3)
# overall_accuracy_excl_3 = correct_counts_excl_3 / total_counts_excl_3 if total_counts_excl_3 != 0 else 0
# print(f'Overall Accuracy excluding label 3: {overall_accuracy_excl_3 * 100:.9f}%')

# # Find and print the most frequent misclassification
# if misclassifications:
#     most_frequent_misclassification = max(misclassifications, key=misclassifications.get)
#     most_frequent_label, most_frequent_prediction = most_frequent_misclassification
#     print(f"Most frequent misclassification: Label {sononet_label_list[most_frequent_label]} "
#         f"misclassified as {sononet_label_list[most_frequent_prediction]} "
#         f"{misclassifications[most_frequent_misclassification]} times.")
# else:
#     print("No misclassifications found.")

# Notify when processing is complete for all specified setups.
print("Completed")