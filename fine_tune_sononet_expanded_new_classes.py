import os
import time
import glob
import csv
import pickle
import random
import copy
import hashlib
import logging
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from PIL import Image
import cv2

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision

from moviepy.editor import VideoFileClip
from scipy.spatial.transform import Rotation as R
from skimage.metrics import mean_squared_error, structural_similarity as ssim
from imgaug import augmenters as iaa

import SonoNet.sononet as sononet
from utils import device
from utils_sononet import *

# Disable specific logging
logging.getLogger('matplotlib.font_manager').disabled = True

# Configuration
network_name = 'SN64'   # 'SN16', 'SN32' or 'SN64'
GPU_NR = 4     # Choose the device number of your GPU

video_folder = r'/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/data/videos/'
gt_folder = r'/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/output/cleaned_frame_collect_labels/'
idxs_folder = r'/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/SonoNet/fine_tune_index_csvs'
# Training parameters
num_epochs = 100
learning_rate = 0.0001

input_size_sononet = [224, 288]

weights = True

# The input images will be resized to this size
input_size = [224, 288]

augment_rotate = iaa.Affine(rotate=(-30, 30))
augment_shear = iaa.Affine(shear=(-20, 20))
augment_flip = iaa.Fliplr(0.5)
augment_noise = iaa.AdditiveGaussianNoise(scale=0.01*255)


train_case_dictionary = {
                         '11': ['Operator_6_Novice', '23w_train_out', 'Operator6', 320, 50, 1036, 777, 'Breech'], 
                         '12': ['Operator_5_Novice', '23w_train_out', 'Operator5', 205, 0, 545, 374, 'Breech'], 
                         '9': ['Operator_8_Novice', '23w_train_out', 'Operator8', 320, 50, 1036, 777, 'Breech'], 
                         '6': ['Operator_11_Novice', '23w_train_out', 'Operator11', 320, 50, 1036, 777, 'Breech'], 
                         '10': ['Operator_7_Novice', '23w_train_out', 'Operator7', 320, 50, 1036, 777, 'Breech'], 
                         '15': ['Operator_2_Novice', '23w_train_out', 'Operator2', 180, 0, 576, 408, 'Cephalic'], 
                         '8': ['Operator_9_Novice', '23w_train_out', 'Operator9', 320, 50, 1036, 777, 'Cephalic'], 
                         '13': ['Operator_4_Novice', '23w_train_out', 'Operator4', 320, 50, 1036, 777, 'Cephalic'],   
                         '0': ['Operator_17_Expert', '23w_train_out', 'Operator17', 420, 0, 1313, 1080, 'Breech'], 
                         '3': ['Operator_14_Expert', '23w_train_out', 'Operator14', 320, 50, 1036, 777, 'Cephalic'],
                         '5': ['Operator_12_Novice', '23w_train_out', 'Operator12', 320, 50, 1036, 777, 'Cephalic'],
                         '14': ['Operator_3_Novice', '23w_train_out', 'Operator3', 320, 50, 1036, 777, 'Breech'], 
                         '7': ['Operator_10_Novice', '23w_train_out', 'Operator10', 320, 50, 1036, 777, 'Breech'], 
                    }

test_case_dictionary = {'2': ['Operator_15_Novice', '23w_train_out', 'Operator15', 320, 50, 1036, 777, 'Breech'], 
                        '4': ['Operator_13_Novice', '23w_train_out', 'Operator13', 320, 50, 1036, 777, 'Breech'], 
                        '16': ['Operator_1_Novice', '23w_train_out', 'Operator1', 320, 50, 1036, 777, 'Cephalic'], 
                        '1': ['Operator_16_Expert', '23w_train_out', 'Operator16', 320, 50, 1036, 777, 'Cephalic']}

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
    return


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
        if idx % 5 != 0:
            continue

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

def prepare_image_2(image):
    # convert to 4D tensor of type float32
    image_data = np.float32(np.reshape(image, (1, 1, image.shape[0], image.shape[1])))

    # normalise images by substracting mean and dividing by standard dev.
    mean = image_data.mean()
    std = image_data.std()
    image_data = np.array(255.0 * np.divide(image_data - mean, std), dtype=np.float32)
    return image_data


def hash_image(im):
    """
    Create a hashable representation of a numpy array image.
    """
    image_bytes = im.tobytes()
    return hashlib.sha256(image_bytes).hexdigest()

def get_image_list(conv_df, train_case_dictionary, val=False):
    """
    Returns the list of image data as arrays including augmentation from specified videos.
    """
    image_set = set()
    image_list = []
    label_image_dict = {label: [] for label in range(len(sononet_label_list))}  
    for exp, op in train_case_dictionary.items():
        video_filename = f'{op[0]}.mp4'
        gt_filename = f'clean_frame_collect_{op[0]}.csv'
        video_file_path = os.path.join(video_folder, video_filename)

        if os.path.exists(video_file_path):
            print(f'Processing video: {video_filename}')
            bb_x, bb_y, bb_w, bb_h = op[3], op[4], op[5], op[6]

            for image, label in extract_anatomy_video_frames(video_filename, gt_filename, conv_df, bb_x, bb_y, bb_w, bb_h):
                im_key = hash_image(image)
                if im_key not in image_set:
                    label_image_dict[label].append(image)
                    image_set.add(im_key)
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

    if os.path.exists('images_list_blocks.pkl'):
        os.remove('images_list_blocks.pkl')

    with open('images_list_blocks.pkl', 'wb') as f:
        pickle.dump(balanced_image_list, f)

    with open('images_list_blocks.pkl', 'rb') as f:
        balanced_image_list = pickle.load(f)

    return balanced_image_list



class SononetDataset(Dataset):
    def __init__(self, balanced_image_list, augmentations, include_orig=True):
        self.balanced_image_list = balanced_image_list
        self.augmentations = augmentations
        self.include_orig = include_orig

    def __len__(self):
        return len(self.balanced_image_list)

    def __getitem__(self, idx):
        image, label = self.balanced_image_list[idx]
        augmented_images = aug_prepare_image_2(image, True)

        augmented_images = [(img, label) for img in augmented_images]

        return augmented_images

def create_data_loader(balanced_image_list, augmentations, batch_size=32, shuffle=True, num_workers=4):
    dataset = SononetDataset(balanced_image_list, augmentations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=prepare_batch)
    return dataloader

def prepare_batch(batch):
    flattened_batch = [item for sublist in batch for item in sublist]

    batch_images = [torch.from_numpy(image_data).float().to(device) for image_data, _ in flattened_batch]
    batch_labels = [torch.tensor(label, dtype=torch.long).to(device) for _, label in flattened_batch]

    image_data_batch = torch.stack(batch_images, dim=0).to(device)

    if image_data_batch.dim() == 5:
        image_data_batch = image_data_batch.squeeze(1)

    label_batch = torch.tensor(batch_labels, dtype=torch.long).to(device)

    return image_data_batch, label_batch


def load_checkpoint(checkpoint_path, net, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    epoch_losses = checkpoint['epoch_losses']

    print(f"Checkpoint loaded: resuming from epoch {epoch+1}")
    return epoch, epoch_losses

def fine_tune_model(net, criterion, optimizer, scheduler, conv_df, num_splits=5, early_stopping_rounds=3, resume=False):
    """
    Fine-tune the model using a list of images and their labels.
    """
    checkpoint_dir = "/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/checkpoints_sono_new_block"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    epoch_losses = []
    val_losses = []

    if os.path.exists('images_list_blocks.pkl'):
        # Load the file
        with open('images_list_blocks.pkl', 'rb') as f:
            print("Loading images.")
            image_list = pickle.load(f)
            print("File loaded successfully.")
    else:
        image_list = get_image_list(conv_df, train_case_dictionary)
    
    label_counts = Counter(label for _, label in image_list)
    print('image_list:', label_counts)
    
    train_loader = create_data_loader(image_list, augmentations, batch_size=32)

    start_epoch = 0

    if resume:
        # Look for the most recent checkpoint
        latest_checkpoint = None
        for i in range(1, num_splits + 1):
            checkpoint_path = os.path.join(checkpoint_dir, f'best_model_fold_{i}.pth')
            if os.path.exists(checkpoint_path):
                latest_checkpoint = checkpoint_path

        if latest_checkpoint:
            epoch, epoch_losses = load_checkpoint(latest_checkpoint, net, optimizer, scheduler)
        else:
            print("No checkpoint found, starting from scratch.")

   
    # Training loop for each epoch
    early_stopping_patience = 5
    min_loss = float('inf')
    no_improvement_epochs = 0

    # Training loop for each epoch
    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0.0
        
        for batch_idx, (image_data_batch, label_batch) in enumerate(train_loader):
            # Forward pass
            outputs = net(image_data_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        epoch_loss /= len(train_loader)
        print(f'Total loss after Epoch [{epoch + 1}/{num_epochs}]: {epoch_loss:.4f}')
        epoch_losses.append(epoch_loss)
        
        scheduler.step()

         # Early Stopping
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
        
        if no_improvement_epochs >= early_stopping_patience:
            print(f'Stopping early at epoch {epoch + 1} due to no improvement in loss.')
            break

        if epoch % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch_losses': epoch_losses
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f'training_validation_loss_plot_big_block_new.png')
    
    print('Finished Training')
    
    torch.save(net.state_dict(), '/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/sononet_finetuned_block_new.pth')

   
    return image_list


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
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.adaptation(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return x


    
def main():
    # Load network
    print('Loading network')
    net = sononet.SonoNet(network_name, weights=True)

    conv_df = pd.read_csv('/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/sononet_label_categories_expanded.csv')

    modified_net = ModifiedSonoNet(net, 15)
    modified_net.features.load_state_dict(net.features.state_dict())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(modified_net.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Move to GPU
    print('Moving to GPU:')
    torch.cuda.device(GPU_NR)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    modified_net.cuda()

    fine_tune_model(modified_net, criterion, optimizer, scheduler, conv_df)



if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    main()