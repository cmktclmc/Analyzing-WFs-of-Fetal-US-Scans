'''
test.py:    Modified version of the original example.py file.
            This file runs classification on the examples images.
'''

import glob
import re
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import numpy as np
from PIL import Image
import sononet
import torch
from torch.autograd import Variable


disclaimer = '''
This is a PyTorch implementation of SonoNet:

Baumgartner et al., "Real-Time Detection and Localisation of Fetal Standard
Scan Planes in 2D Freehand Ultrasound", arXiv preprint:1612.05601 (2016)

This repository is based on
  https://github.com/baumgach/SonoNet-weights
which provides a theano+lasagne implementation.
'''
print(disclaimer)

# Configuration
network_name = 'SN64'   # 'SN16', 'SN32' pr 'SN64'
display_images = False  # Whether or not to show the images during inference
GPU_NR = 4     # Choose the device number of your GPU

# If you provide the original lasagne parameter file, it will be converted to a
# pytorch state_dict and saved as *.pth.
# In this repository, the converted parameters are already provided.
weights = True
# weights = ('/local/ball4916/dphil/SonoNet/SonoNet-weights/SonoNet{}.npz'
#                .format(network_name[2:]))


# Other parameters
# crop_range = [(115, 734), (81, 874)]  # [(top, bottom), (left, right)]

# Crop range used to get rid of the vendor info etc around the images
# crop_range = [(100, 400), (50, 350)]  # [(top, bottom), (left, right)]

# The input images will be resized to this size
input_size = [224, 288]
# image_path = './example_images/*.tiff'
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

operators = {
    # '1': './video_frames/Operator_1_Novice/*.png',
    # '2': './video_frames/Operator_2_Novice/*.png',
    # '3': './video_frames/Operator_3_Novice/*.png',
    # '4': './video_frames/Operator_4_Novice/*.png',
    # '5': './video_frames/Operator_5_Novice/*.png',
    # '6': './video_frames/Operator_6_Novice/*.png',
    # '7': './video_frames/Operator_7_Novice/*.png',
    '8': './video_frames/Operator_8_Novice/*.png',
    # '9': './video_frames/Operator_9_Novice/*.png',
    # '10': './video_frames/Operator_10_Novice/*.png',
    # '11': './video_frames/Operator_11_Novice/*.png',
    # '12': './video_frames/Operator_12_Novice/*.png',
    # '13': './video_frames/Operator_13_Novice/*.png',
    # '14': './video_frames/Operator_14_Expert/*.png',
    # '15': './video_frames/Operator_15_Novice/*.png',
    # '16': './video_frames/Operator_16_Expert/*.png',
    # '17': './video_frames/Operator_17_Expert/*.png',
 }

# image_path = './imgs/tmrb2/*.png'
# image_path = './imgs/tmrb_dsr/annotations/*.png'
# image_path = './imgs/tmrb_dsr/copycoords/*.png'
# image_path = './imgs/tmrb_dsr/annotations2/*.png'
# image_path = './imgs/tmrb_pre_reg/*.png'


# Display the images during the prediction
# display_images = True

def imcrop(image, crop_range):
    """ Crop an image to a crop range """
    return image[crop_range[0][0]:crop_range[0][1],
                 crop_range[1][0]:crop_range[1][1], ...]


def prepare_inputs(operator_name):
    
    input_list = []
    image_path_pattern = operators[operator_name]  # Select path based on operator

    # Retrieve all matching filenames for the operator
    filenames = glob.glob(image_path_pattern)

    # Sort the filenames based on their numerical part
    sorted_filenames = sorted(filenames, key=extract_number)

    for filename in sorted_filenames:

        # prepare images
        image = imread(filename)  # read
        # image = imcrop(image, crop_range)  # crop
        image = np.array(Image.fromarray(np.uint8(image*255.0)).resize(input_size, resample=Image.BICUBIC))
        # print(np.amax(image))
        image = np.mean(image, axis=2)  # convert to gray scale

        # convert to 4D tensor of type float32
        image_data = np.float32(np.reshape(image,
                                           (1, 1, image.shape[0],
                                            image.shape[1])))

        # normalise images by substracting mean and dividing by standard dev.
        # print(image_data)
        mean = image_data.mean()
        # print(mean)
        std = image_data.std()
        # print(std)
        image_data = np.array(255.0 * np.divide(image_data - mean, std), dtype=np.float32)
    
        # Note that the 255.0 scale factor is arbitrary
        # it is necessary because the network was trained
        # like this, but the same results would have been
        # achieved without this factor for training.

        input_list.append(image_data)

    return input_list

def extract_number(filename):
    """
    Extracts the number from a filename.
    """
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

def main():

    print('Loading network')
    net = sononet.SonoNet(network_name, weights=weights)
    net.eval()

    print('Moving to GPU:')
    torch.cuda.device(GPU_NR)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    net.cuda()

    # Open a file to save the labels and predictions
    with open('predictions.csv', 'w') as file:

        file.write("Operator,True Label,Predicted Label,Confidence\n")  # Header

        for operator_name in operators.keys():

            print("\nPredictions using {}:".format(network_name))
            print(f"Operator: {operator_name}")

            predictions = []

            input_list = prepare_inputs(operator_name)
            sorted_filenames = sorted(glob.glob(operators[operator_name]), key=extract_number)  

            for image, file_name in zip(input_list, sorted_filenames):

                x = Variable(torch.from_numpy(image).cuda())
                outputs = net(x)
                confidence, prediction = torch.max(outputs.data, 1)

                # True labels are obtained from file name.
                # true_label = file_name.split('/')[-1][0:-5]
                # true_label = file_name.split('/')[-1][0:-4]
                true_label = os.path.splitext(os.path.basename(file_name))[0]
                predicted_label = label_names[prediction[0]]
                conf = confidence[0].item()

                # Collect each prediction
                predictions.append((true_label, predicted_label, conf))

                if display_images:
                    plt.imshow(np.squeeze(image), cmap='gray')
                    plt.show()

            # Sort predictions by true label numerically
            predictions.sort(key=lambda x: int(x[0]))

            # Process sorted predictions
            for true_label, predicted_label, conf in predictions:
                print(f" - {predicted_label} (conf: {conf:.2f}, true label: {true_label})")
                file.write(f"{operator_name},{true_label},{predicted_label},{conf:.2f}\n")
                
    print('\nCompleted predictions. Results saved to predictions.csv\n')


if __name__ == '__main__':
    main()
