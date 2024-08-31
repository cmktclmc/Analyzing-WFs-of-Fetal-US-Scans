import os
import warnings

# Suppressing unnecessary warnings to make the output cleaner
warnings.filterwarnings("ignore")

os.chdir('/Users/Caitlin/Documents/GitHub/proximity-to-sp-us-videos')

# Import SonoNet
import torch 
from PIL import Image, ImageEnhance

# Importing necessary libraries
import time
import numpy as np
import pandas as pd
import cv2
import torch
from moviepy.editor import VideoFileClip
from scipy.spatial.transform import Rotation as R
import csv
import re
from fuzzywuzzy import fuzz

import pytesseract
from pytesseract import Output
import difflib

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device is available and will be used.")
else:
    device = torch.device("cpu")
    print("MPS device is not available. Using CPU instead.")

import SonoNet.sononet as sononet
from utils_sononet import *

# Setup directories for input videos and outputs.
video_folder = r'data/videos/'
output_folder = r'output/frames_posereg/masks/'
os.makedirs(output_folder, exist_ok=True)

input_size_sononet = [224, 288]

sononet_label_names = ['3VV',
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

pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"
    
# Preprocessing functions
def convert_to_grayscale(image):
    pil_image = Image.fromarray(image)
    grayscale_image = pil_image.convert('L')
    return grayscale_image

def enhance_contrast(image):
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(2.0)
    return enhanced_image

def binarize_image(image, threshold=128):
    return image.point(lambda p: p > threshold and 255)

def remove_noise(image):
    open_cv_image = np.array(image)
    open_cv_image = cv2.fastNlMeansDenoising(open_cv_image, None, 30, 7, 21)
    return Image.fromarray(open_cv_image)

    
# Function to extract the text from frames from video files at a specific rate
def extract_text_from_frames(video_file, video_folder):

    """
    Generator to extract frames from a specified video file at 10 fps.
    Yields each frame along with its index number.
    """

    video_path = os.path.join(video_folder, video_file)
    video_clip = VideoFileClip(video_path)
    
    for idx, frame in enumerate(video_clip.iter_frames(fps=10, dtype="uint8")):
        height, width = frame.shape[:2]

        bottom_25_percent = int(0.3 * height)  # Bottom 25% of the image
        bottom_5_percent = int(0 * height)  # Bottom 5% of the image
        
        cropped_image = frame[height - bottom_25_percent:, :]
        cropped_image = cropped_image[:bottom_25_percent - bottom_5_percent, :]

        # Convert back to BGR
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        altered_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # Apply OCR
        altered_frame = np.array(altered_frame)

        results = pytesseract.image_to_data(altered_frame, output_type=Output.DICT, lang='eng')
        
        # Process OCR results
        all_text = ''
        high_conf_text = []
        for i in range(len(results['text'])):
            text = results['text'][i]
            conf = int(results['conf'][i])
            if conf > 60:  # Filter out low-confidence detections
                high_conf_text.append(text.strip())
        
        # Clean out all non-letters/numbers
        high_conf_text = re.sub(r'[^a-zA-Z0-9\s]', '', str(high_conf_text).lower())
        # Clean out multiple spaces
        high_conf_text = re.sub(r'\s+', ' ', high_conf_text)         

        yield frame, idx, high_conf_text, all_text, altered_frame # Yield frame, index, and best label


#---------------------------------------------------------------------

# Case dictionary
case_dictionary = {
    #'0': ['Operator_1_Novice', '23w_train_out', 'Operator1'],
    #'1': ['Operator_2_Novice', '23w_train_out', 'Operator2'],
    #'2': ['Operator_3_Novice', '23w_train_out', 'Operator3'],
    #'3': ['Operator_4_Novice', '23w_train_out', 'Operator4'],
    #'4': ['Operator_5_Novice', '23w_train_out', 'Operator5'],
    #'5': ['Operator_6_Novice', '23w_train_out', 'Operator6'],
    #'6': ['Operator_7_Novice', '23w_train_out', 'Operator7'],
    # '7': ['Operator_8_Novice', '23w_train_out', 'Operator8'],
    # '8': ['Operator_9_Novice', '23w_train_out', 'Operator9'],
    # '9': ['Operator_10_Novice', '23w_train_out', 'Operator10'],
    # '10': ['Operator_11_Novice', '23w_train_out', 'Operator11'],
    # '11': ['Operator_12_Novice', '23w_train_out', 'Operator12'],
    # '12': ['Operator_13_Novice', '23w_train_out', 'Operator13'],
    # '13': ['Operator_14_Expert', '23w_train_out', 'Operator14'],
    # '14': ['Operator_15_Novice', '23w_train_out', 'Operator15'],
    # '15': ['Operator_16_Expert', '23w_train_out', 'Operator16'],
    '16': ['Operator_17_Expert', '23w_train_out', 'Operator17']
}

#---------------------------------------------------------------------

# Loop to process only specific videos defined in case_dictionary, rpocess them and runs inference
for exp, op in case_dictionary.items():
    video_filename = f'{op[0]}.mp4' 
    print(f'Processing video: {video_filename}')
    video_file_path = os.path.join(video_folder, video_filename)
    
    if os.path.exists(video_file_path):
        frames = []
        ground_truth = []
        
        for frame, idx, true_label, extracted, altered_frame in extract_text_from_frames(video_filename, video_folder):
            frames.append(frame)

            # Ensure frame is writable
            frame = cv2.UMat(frame)

            # Display the best match text in red over the dilated image
            cv2.putText(frame, true_label, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Frame', frame)
            cv2.imshow('Altered For Label Image', altered_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
            ground_truth.append((idx, true_label, extracted))
            print(f"{idx} {true_label} : '{extracted}'")

        cv2.destroyAllWindows()

        output_path = f'output/extracted_gt'
        os.makedirs(output_path, exist_ok=True)

        # Save results to CSV file
        with open(os.path.join(output_path, f'ground_truth_{op[0]}.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Index', 'True Label', 'Extracted'])
            writer.writerows(ground_truth)
    else:
        print(f"Video file {video_filename} not found in directory.")

# Notify when processing is complete for all specified setups.
print("Completed")







