import cv2
import pandas as pd
import numpy as np
from PIL import Image
import os

# Define the paths
video_path = '/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/data/videos/Operator_5_Novice.mp4'
csv_path = '/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/output/cleaned_gt/ground_truth_Operator_5_Novice.csv'
output_folder = '/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/SonoNet/tiff_frames'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the CSV file
df = pd.read_csv(csv_path)

# Drop rows where 'True Label' is NaN and find the first instance for each unique label
df = df.dropna(subset=['True Label'])
unique_labels = df['True Label'].unique()

# Dictionary to store the first occurrence index for each unique label
first_occurrence = {label: df[df['True Label'] == label]['Index'].iloc[0] for label in unique_labels}

# Open the video
cap = cv2.VideoCapture(video_path)

# Function to extract and save frame as TIFF
def save_frame_as_tiff(frame, label, output_folder):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image.save(os.path.join(output_folder, f'{label}.tiff'), 'TIFF')

# Extract frames and save them
for label, index in first_occurrence.items():
    # Set the video to the specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    if ret:
        save_frame_as_tiff(frame, label, output_folder)

# Release the video capture object
cap.release()