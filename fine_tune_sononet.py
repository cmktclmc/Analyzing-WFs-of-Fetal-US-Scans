import glob
import os
import time
import math
import random
import csv
import pickle
import logging
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from PIL import Image
import cv2
from skimage.metrics import mean_squared_error, structural_similarity as ssim
from scipy.spatial.transform import Rotation as R
from moviepy.editor import VideoFileClip
from imgaug import augmenters as iaa

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold

from utils import device
from utils_sononet import *
import SonoNet.sononet as sononet

# Configuration
network_name = 'SN64'   # 'SN16', 'SN32' or 'SN64'
GPU_NR = 4     # Choose the device number of your GPU

video_folder = r'/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/data/videos/'
gt_folder = r'/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/output/cleaned_gt/'
idxs_folder = r'/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/SonoNet/fine_tune_index_csvs'

# Training parameters
num_epochs = 10
learning_rate = 0.0001
batch_size = 16

input_size_sononet = [224, 288]

weights = True

# The input images will be resized to this size
input_size = [224, 288]

image_path = './imgs/tmrb_dsr/copycoords/*.png'
label_path = './labels/tmrb_dsr/copycoords/*.txt'  # Assuming labels are in text files

augment_rotate = iaa.Affine(rotate=(-30, 30))
augment_shear = iaa.Affine(shear=(-20, 20))
augment_flip = iaa.Fliplr(0.5)
augment_noise = iaa.AdditiveGaussianNoise(scale=0.01*255)

sononet_label_list = ['3VV',
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

augmentations = [
    augment_rotate.augment_image,
    augment_shear.augment_image,
    augment_flip.augment_image,
    augment_noise.augment_image
]

def aug_prepare_image_2(image, include_orig):
    """ Given an image augment it and then preprocess it to fit sononet before returning as list of arrays """
    image_list = []
    image_data_list = []

    if include_orig:
        image_list.append(image)

    image = image.astype(np.uint8)

    for aug in augmentations:
        image_list.append(aug(image))
    
    for im in image_list:
        image_data = prepare_image_2(im)
        image_data_list.append(image_data)
    
    return image_data_list



# Function to extract frames from video files at a specific rate
def extract_anatomy_video_frames(video_file, gt_filename, conv_df, bb_x, bb_y, bb_w, bb_h):
    """
    Extract frames only the anatomy frames from a specified video file at 10 fps.
    Yields each frame along with its index number.
    """
    print('Extracting frames')
    count = 0
    gt_path = os.path.join(gt_folder, gt_filename)
    labels_df = pd.read_csv(gt_path)

    video_path = os.path.join(video_folder, video_file)
    video_clip = VideoFileClip(video_path)

    for idx, frame in enumerate(video_clip.iter_frames(fps=10, dtype="uint8")):
        label = labels_df.loc[idx, 'True Label']

        if pd.isna(label) or label not in conv_df['Item'].values:
            label = 3
            image = prepare_image_1(frame, bb_x, bb_y, bb_w, bb_h)

            yield image, label
        else:
            label = conv_df.loc[conv_df['Item'] == label, 'Category'].iloc[0]
            if 'Other' in label:
                label = 3
                image = prepare_image_1(frame, bb_x, bb_y, bb_w, bb_h)

                yield image, label
            else:
                label = sononet_label_list.index(label)
                
                image = prepare_image_1(frame, bb_x, bb_y, bb_w, bb_h)

                yield image, label

                            
                            
def get_image_list(conv_df):
    """
    Returns the list of image data as arrays including augmentation from specified videos.
    """
    image_list = []
    label_image_dict = {label: [] for label in range(len(sononet_label_list))}  
    for exp, op in train_case_dictionary.items():
        video_filename = f'{op[0]}.mp4'
        gt_filename = f'ground_truth_{op[0]}.csv'
        video_file_path = os.path.join(video_folder, video_filename)

        if os.path.exists(video_file_path):
            print(f'Processing video: {video_filename}')
            bb_x, bb_y, bb_w, bb_h = op[3], op[4], op[5], op[6]

            for image, label in extract_anatomy_video_frames(video_filename, gt_filename, conv_df, bb_x, bb_y, bb_w, bb_h):
                label_image_dict[label].append(image)
                image_list.append((image, label))
                
    print('Counging labels')
    label_counts = Counter(label for _, label in image_list if label != 3)
    max_label_count = max(label_counts.values())
    
    images_with_3 = [(image, label) for image, label in image_list if label == 3]
    images_with_other_labels = [(image, label) for image, label in image_list if label != 3]
    
    print('Down sampling')
    downsampled_images_with_3 = random.sample(images_with_3, max_label_count)
    print(label_counts)

    print('Oversampling to ', max_label_count)
    oversampled_images_with_other_labels = []
    
    for label, count in label_counts.items():
        images_with_label = [(image, lbl) for image, lbl in images_with_other_labels if lbl == label]
        if count < max_label_count:
            oversampled_images = random.choices(images_with_label, k=max_label_count)
        else:
            oversampled_images = images_with_label
        oversampled_images_with_other_labels.extend(oversampled_images)

    balanced_image_list = oversampled_images_with_other_labels + downsampled_images_with_3
    
    label_counts = Counter(label for _, label in balanced_image_list)
    print('Balanced:', label_counts)

    augmented_images_list = []
    for im, im_label in balanced_image_list:
        augmented_images = aug_prepare_image_2(im, True)
        augmented_images_list.append((augmented_images, im_label))
    
    label_counts = Counter(label for _, label in augmented_images_list)
    print('Augmented', label_counts)

    with open('augmented_images_list.pkl', 'wb') as f:
        pickle.dump(augmented_images_list, f)

    with open('augmented_images_list.pkl', 'rb') as f:
        augmented_images_list = pickle.load(f)

    return augmented_images_list

def cross_validation_split(conv_df, num_splits):
    if os.path.exists('augmented_images_list.pkl'):
        with open('augmented_images_list.pkl', 'rb') as f:
            print("Loading images.")
            image_list = pickle.load(f)
            print("File loaded successfully.")
    else:
        image_list = get_image_list(conv_df)

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    
    X = [image_data for image_data, _ in image_list]
    y = [label for _, label in image_list]
    
    for train_index, val_index in skf.split(X, y):
        train_data_2 = [(image, label) for i in train_index for image in image_list[i][0] for label in [image_list[i][1]]]
        val_data_2 = [(image, label) for i in val_index for image in image_list[i][0] for label in [image_list[i][1]]]

        yield train_data_2, val_data_2

def prepare_batch(batch):
    batch_images = [torch.from_numpy(image_data).to(device) for image_data, _ in batch]
    batch_labels = [torch.tensor(label, dtype=torch.long).to(device) for _, label in batch]

    # Stack images and labels into batches
    image_data_batch = torch.cat(batch_images, dim=0).to(device)
    label_batch = torch.tensor(batch_labels, dtype=torch.long).to(device)
    return image_data_batch, label_batch

def load_checkpoint(checkpoint_path, net, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    fold_index = checkpoint['fold_index']
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    epoch_losses = checkpoint['epoch_losses']
    val_losses = checkpoint['val_losses']

    print(f"Checkpoint loaded: resuming from fold {fold_index}, epoch {epoch+1} with best validation loss of {best_val_loss:.4f}")
    return fold_index, epoch, best_val_loss, epoch_losses, val_losses

def fine_tune_model(net, criterion, optimizer, scheduler, conv_df, num_splits=5, early_stopping_rounds=3, resume=False):
    """
    Fine tune the model using a list of images and their labels.
    """
    checkpoint_dir = "/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/checkpoints_orig_ft"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    epoch_losses = []
    val_losses = []
    total_batches = len(conv_df) // batch_size
    print('Getting list')


    fold_index = 0
    start_epoch = 0

    if resume:
        # Look for the most recent checkpoint
        latest_checkpoint = None
        for i in range(1, num_splits + 1):
            checkpoint_path = os.path.join(checkpoint_dir, f'best_model_fold_{i}.pth')
            if os.path.exists(checkpoint_path):
                latest_checkpoint = checkpoint_path

        if latest_checkpoint:
            fold_index, start_epoch, best_val_loss, epoch_losses, val_losses = load_checkpoint(latest_checkpoint, net, optimizer, scheduler)
        else:
            print("No checkpoint found, starting from scratch.")
            fold_index = 0
            best_val_loss = float('inf')
    else:
        best_val_loss = float('inf')


    for train_data, val_data in cross_validation_split(conv_df, num_splits):
        fold_index += 1
        print(f"Fold {fold_index}/{num_splits}")

        if fold_index < start_epoch // num_epochs + 1:
            continue

        # Initialize early stopping
        best_val_loss = float('inf')
        early_stopping_counter = 0
        start_epoch = 0
    
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            random.shuffle(train_data)
            epoch_loss = 0.0
            total_batches = len(train_data) // batch_size + (1 if len(train_data) % batch_size != 0 else 0)
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(train_data))
                batch = train_data[start_idx:end_idx]

                # Prepare batch tensors
                image_data_batch, label_batch = prepare_batch(batch)

                # Forward pass
                outputs = net(image_data_batch)

                loss = criterion(outputs, label_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                if (batch_idx+1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{total_batches}], Loss: {loss.item():.4f}')

            epoch_loss /= total_batches
            print(f'Total loss after Epoch [{epoch+1}/{num_epochs}]: {epoch_loss:.4f}')
            epoch_losses.append(epoch_loss)

            # Step the scheduler
            scheduler.step()
            
            # Validation after each epoch
            with torch.no_grad():
                val_loss = 0.0
                val_total_batches = math.ceil(len(val_data) / batch_size)
                net.eval()
                val_idx = 0
                for val_batch in val_data:
                    val_idx += 1
                    
                    image_data_batch, label_batch = prepare_batch([val_batch])

                    outputs = net(image_data_batch)
                    loss = criterion(outputs, label_batch)
                    val_loss += loss.item()

                    if (val_idx+1) % 100 == 0:
                        avg_loss = val_loss / val_idx
                        print(f'Val Batch [{val_idx}/{val_total_batches}], Running Avg Loss: {avg_loss:.4f}')

                val_loss /= len(val_data)
                print(f'Validation Loss: {val_loss:.4f}')
                val_losses.append(val_loss)
                # Reset model to training mode
                net.train()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0

                    # Save the best model checkpoint
                    checkpoint_path = os.path.join(checkpoint_dir, f'best_model_fold_{fold_index}.pth')
                    torch.save({
                        'fold_index': fold_index,
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'epoch_losses': epoch_losses,
                        'val_losses': val_losses
                    }, checkpoint_path)
                    print(f'Checkpoint saved: {checkpoint_path}')

                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_rounds:
                        print(f'Early stopping triggered after epoch {epoch+1}')
                        break

        # Plotting the loss after each fold
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Losses - Fold {fold_index}/{num_splits}')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(f'training_validation_loss_plot_fold_{fold_index}_2.png')

    print('Finished Training')

    torch.save(net.state_dict(), '/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/sononet_finetuned_gt_labels_w_orig_2.pth')
    return
    



train_case_dictionary = {
                        '7': ['Operator_10_Novice', '23w_train_out', 'Operator10', 320, 50, 1036, 777, 'Breech'], 
                         '11': ['Operator_6_Novice', '23w_train_out', 'Operator6', 320, 50, 1036, 777, 'Breech'], 
                         '12': ['Operator_5_Novice', '23w_train_out', 'Operator5', 205, 0, 545, 374, 'Breech'], 
                          '9': ['Operator_8_Novice', '23w_train_out', 'Operator8', 320, 50, 1036, 777, 'Breech'], 
                          '14': ['Operator_3_Novice', '23w_train_out', 'Operator3', 320, 50, 1036, 777, 'Breech'], 
                          '6': ['Operator_11_Novice', '23w_train_out', 'Operator11', 320, 50, 1036, 777, 'Breech'], 
                          '10': ['Operator_7_Novice', '23w_train_out', 'Operator7', 320, 50, 1036, 777, 'Breech'], 
                          '15': ['Operator_2_Novice', '23w_train_out', 'Operator2', 180, 0, 576, 408, 'Cephalic'], 
                          '8': ['Operator_9_Novice', '23w_train_out', 'Operator9', 320, 50, 1036, 777, 'Cephalic'], 
                          '13': ['Operator_4_Novice', '23w_train_out', 'Operator4', 320, 50, 1036, 777, 'Cephalic'], 
                          '5': ['Operator_12_Novice', '23w_train_out', 'Operator12', 320, 50, 1036, 777, 'Cephalic'], 
                          '0': ['Operator_17_Expert', '23w_train_out', 'Operator17', 420, 0, 1313, 1080, 'Breech'], 
                          '3': ['Operator_14_Expert', '23w_train_out', 'Operator14', 320, 50, 1036, 777, 'Cephalic']
                         }

test_case_dictionary = {'2': ['Operator_15_Novice', '23w_train_out', 'Operator15', 320, 50, 1036, 777, 'Breech'], 
                        '4': ['Operator_13_Novice', '23w_train_out', 'Operator13', 320, 50, 1036, 777, 'Breech'], 
                        '16': ['Operator_1_Novice', '23w_train_out', 'Operator1', 320, 50, 1036, 777, 'Cephalic'], 
                        '1': ['Operator_16_Expert', '23w_train_out', 'Operator16', 320, 50, 1036, 777, 'Cephalic']}

def main():
    # Load network
    print('Loading network')
    net = sononet.SonoNet(network_name, weights=weights)
    net.train()

    # Move to GPU
    print('Moving to GPU:')
    torch.cuda.device(GPU_NR)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    net.cuda()

    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    conv_df = pd.read_csv('/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/sononet_label_categories.csv')

    fine_tune_model(net, criterion, optimizer, scheduler, conv_df)


if __name__ == '__main__':
    main()