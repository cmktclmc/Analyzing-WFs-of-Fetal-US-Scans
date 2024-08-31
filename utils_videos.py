import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss
import segmentation_models_pytorch as sample_data

import matplotlib.image as img
import torchvision.models as models

from tools import compute_rotation_matrix_from_ortho6d, compute_geodesic_distance_from_two_matrices

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.font_manager as font_manager

img_size = (320, 320)
encoder_name = 'timm-efficientnet-b0'
out_params = 9

# Set the 'Lato' font as the default for all text elements
font_dirs = '/home/chiara/workspace/Lato' 
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

plt.rcParams['font.family'] = 'Lato'
plt.rcParams['font.size'] = 14


class SegmentationDatasetVideoFrames(Dataset):
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 320))
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.Tensor(image) / 255.0
        return image


class SegmentationRegressionDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames
        self.transform = T.Compose([T.ToTensor(), T.Resize(img_size)])

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image
    
class FetalDatasetFrames(Dataset):
    def __init__(self, labels, frames, transform=None):
        super().__init__()
        self.frames = frames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        image = self.frames[idx]
        label_string = self.labels.iloc[idx]
        label = np.array([float(num) for num in label_string[0].strip('[]').split(',')])
        if self.transform is not None:
            image = self.transform(image)
            if torch.isnan(image).any() is True:
              print("Nan values in image")
              print(torch.isnan(image).any())
        return image, label

class SegmentationModel(nn.Module):

  def __init__(self):
    super(SegmentationModel, self).__init__()

    # Initialize the UNet architecture using sample_data.Unet
    self.arc=sample_data.Unet(
      encoder_name = encoder_name,
      encoder_weights = 'imagenet',
      in_channels = 3, 
      classes = 1,
      activation = None
    )

  def forward(self, images=None, masks=None, unlabeled_images1=None, unlabeled_images2=None):
  
  # def forward(self, labeled_images=None, labeled_masks=None, unlabeled_images1=None, unlabeled_images2=None, unlabeled_images3=None, unlabeled_images4=None, unlabeled_images5=None):
    
    if images is not None:
      # Pass labeled images through the UNet architecture
      logits = self.arc(images)
      if masks is not None:
        # Calculate Dice Loss and Binary Cross Entropy (BCE) Loss for labeled images
        labeled_loss1 = DiceLoss(mode='binary')(logits, masks)
        labeled_loss2 = nn.BCEWithLogitsLoss()(logits, masks)
        labeled_loss = (labeled_loss1 + labeled_loss2) / 2
      else:
        labeled_loss = None
    else:
      logits = None
      labeled_loss = None
    
    if unlabeled_images1 is not None and unlabeled_images2 is not None:
      # Pass unlabeled images through the UNet architecture
      unlabeled_logits1 = self.arc(unlabeled_images1)
      unlabeled_logits2 = self.arc(unlabeled_images2)
      # Apply sigmoid activation to obtain probabilities for unlabeled images
      unlabeled_probs1 = torch.sigmoid(unlabeled_logits1)
      unlabeled_probs2 = torch.sigmoid(unlabeled_logits2)
      #print(unlabeled_probs1.shape)
      # Calculate Mean Squared Error (MSE) Loss for unlabeled images
      unlabeled_loss = nn.MSELoss()(unlabeled_probs1, unlabeled_probs2)
    else:
      unlabeled_loss = None

    return logits, labeled_loss, unlabeled_loss

class ClassSegmentationModel(nn.Module):

  def __init__(self):
    super(ClassSegmentationModel, self).__init__()

    aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    activation='sigmoid',      # activation function, default is None
    classes=1,                 # define number of output labels
    )
    
    self.arc=sample_data.Unet(
      encoder_name = encoder_name,
      encoder_weights = 'imagenet',
      in_channels = 3, 
      classes = 1,
      activation = None,
      aux_params = aux_params
    )

  def forward(self, images=None, masks=None, unlabeled_images1=None, unlabeled_images2=None, unlabeled_images3=None, labels=None):
    
    if images is not None:

      # Pass labeled images through the UNet architecture
      logits, _ = self.arc(images)

      if masks is not None:
        # Calculate Dice Loss and Binary Cross Entropy (BCE) Loss for labeled images
        labeled_loss1 = DiceLoss(mode='binary')(logits, masks)
        labeled_loss2 = nn.BCEWithLogitsLoss()(logits, masks)
        labeled_loss = (labeled_loss1 + labeled_loss2) / 2
      else:
        labeled_loss = None
    else:
      logits = None
      labeled_loss = None
    
    if unlabeled_images1 is not None and unlabeled_images2 is not None and unlabeled_images3 is not None:
      # Pass unlabeled images through the UNet architecture
      unlabeled_logits1, _ = self.arc(unlabeled_images1)
      unlabeled_logits2, _ = self.arc(unlabeled_images2)
      unlabeled_logits3, _ = self.arc(unlabeled_images3)
      # Apply sigmoid activation to obtain probabilities for unlabeled images
      unlabeled_probs1 = torch.sigmoid(unlabeled_logits1)
      unlabeled_probs2 = torch.sigmoid(unlabeled_logits2)
      unlabeled_probs3 = torch.sigmoid(unlabeled_logits3)
      #print(unlabeled_probs1.shape)
      # Calculate Mean Squared Error (MSE) Loss for unlabeled images
      unlabeled_loss1 = nn.MSELoss()(unlabeled_probs1, unlabeled_probs2)
      unlabeled_loss2 = nn.MSELoss()(unlabeled_probs1, unlabeled_probs3)
      unlabeled_loss3 = nn.MSELoss()(unlabeled_probs2, unlabeled_probs3)

      unlabeled_loss = (unlabeled_loss1 + unlabeled_loss2 + unlabeled_loss3) / 3

    elif unlabeled_images1 is not None and unlabeled_images2 is not None and unlabeled_images3 is None:
      # Pass unlabeled images through the UNet architecture
      unlabeled_logits1, _ = self.arc(unlabeled_images1)
      unlabeled_logits2, _ = self.arc(unlabeled_images2)
      # Apply sigmoid activation to obtain probabilities for unlabeled images
      unlabeled_probs1 = torch.sigmoid(unlabeled_logits1)
      unlabeled_probs2 = torch.sigmoid(unlabeled_logits2)
      # Calculate Mean Squared Error (MSE) Loss for unlabeled images
      unlabeled_loss = nn.MSELoss()(unlabeled_probs1, unlabeled_probs2)
    else:
      unlabeled_loss = None

    # Classification
    if images is not None:
      _, class_probs = self.arc(images)
      if labels is not None:
        class_probs = class_probs.squeeze(dim=1)
        classification_loss = nn.BCELoss()(class_probs, labels)
      else:
        classification_loss = None
    else:
      class_probs = None
      classification_loss = None
    
    return logits, labeled_loss, unlabeled_loss, class_probs, classification_loss
  
class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        full_model =  models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*(list(full_model.children())[:-1]))
        self.regression_layer = torch.nn.Sequential(torch.nn.Linear(512, out_params))
        # self.regression_layer = torch.nn.Sequential(torch.nn.Linear(512, 128), torch.nn.Linear(128, out_params))

    def forward(self, x):
        batch_size, _, _, _ = x.shape #taking out batch_size from input image
        x = self.model(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x,1).reshape(batch_size,-1) # then reshaping the batch_size
        x = self.regression_layer(x) 
        x_transl = x[:, -3:]
        x_rot = compute_rotation_matrix_from_ortho6d(x[:, :6].view(batch_size, -1))

        return x_transl, x_rot

    def compute_rotation_matrix_l2_loss(self, gt_rotation_matrix, predict_rotation_matrix):
        loss_function = torch.nn.MSELoss()
        loss = loss_function(predict_rotation_matrix, gt_rotation_matrix)

        return loss

    def compute_rotation_matrix_geodesic_loss(self, gt_rotation_matrix, predict_rotation_matrix):
        theta = compute_geodesic_distance_from_two_matrices(gt_rotation_matrix, predict_rotation_matrix)
        error = theta.mean()

        return error
