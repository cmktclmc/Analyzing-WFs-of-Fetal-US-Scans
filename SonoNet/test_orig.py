'''
test.py:    Modified version of the original example.py file.
            This file runs classification on the examples images.
'''

import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import numpy as np
from PIL import Image
import sononet
import torch
from torch.autograd import Variable
import os 

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
GPU_NR = 0     # Choose the device number of your GPU

# If you provide the original lasagne parameter file, it will be converted to a
# pytorch state_dict and saved as *.pth.
# In this repository, the converted parameters are already provided.
weights = True
# weights = ('/local/ball4916/dphil/SonoNet/SonoNet-weights/SonoNet{}.npz'
#                .format(network_name[2:]))


# Other parameters
#crop_range = [(115, 734), (81, 874)]  # [(top, bottom), (left, right)]
bbox_x, bbox_y, bbox_width, bbox_height = 205, 0, 545, 374
crop_range = [(bbox_y, bbox_y+bbox_height), (bbox_x, bbox_x+bbox_width)]  # [(top, bottom), (left, right)]

input_size = [224, 288]
# image_path = '/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/SonoNet/example_images/*.tiff'
image_path = '/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/SonoNet/tiff_frames/*.tiff'
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
               'Spine (sag.) ']


def imcrop(image, crop_range):
    """ Crop an image to a crop range """
    return image[crop_range[0][0]:crop_range[0][1],
                 crop_range[1][0]:crop_range[1][1], ...]


def prepare_inputs():
    input_list = []
    for filename in glob.glob(image_path):

        # prepare images
        image = imread(filename)  # read

        image = imcrop(image, crop_range)  # crop
        image = np.array(Image.fromarray(image).resize(input_size, resample=Image.BICUBIC))
        image = np.mean(image, axis=2)  # convert to gray scale

        # convert to 4D tensor of type float32
        image_data = np.float32(np.reshape(image,
                                           (1, 1, image.shape[0],
                                            image.shape[1])))

        # normalise images by substracting mean and dividing by standard dev.
        mean = image_data.mean()
        std = image_data.std()
        image_data = np.array(255.0 * np.divide(image_data - mean, std), dtype=np.float32)
        # Note that the 255.0 scale factor is arbitrary
        # it is necessary because the network was trained
        # like this, but the same results would have been
        # achieved without this factor for training.

        input_list.append(image_data)

    return input_list


def convert_to_png():
    input_dir = '/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/SonoNet/tiff_frames'
    output_dir = '/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/SonoNet/tiff_frames'

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".tiff") or filename.endswith(".tif"):
            # Open TIFF file
            with Image.open(os.path.join(input_dir, filename)) as img:
                # Construct output filename
                output_filename = os.path.splitext(filename)[0] + ".png"
                output_path = os.path.join(output_dir, output_filename)
                
                # Convert and save as PNG
                img.save(output_path, "PNG")
                #print(f"Converted {filename} to PNG")

def main():
    #Convert example images from tiff images to png
    convert_to_png()

    print('Loading network')
    net = sononet.SonoNet(network_name, weights=weights)
    net.eval()

    print('Moving to GPU:')
    torch.cuda.device(GPU_NR)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    net.cuda()

    print("\nPredictions using {}:".format(network_name))
    input_list = prepare_inputs()
    for image, file_name in zip(input_list, glob.glob(image_path)):

        x = Variable(torch.from_numpy(image).cuda())
        outputs = net(x)
        confidence, prediction = torch.max(outputs.data, 1)

        # True labels are obtained from file name.
        true_label = file_name.split('/')[-1][0:-5]
        print(" - {} (conf: {:.2f}, true label: {})"
              .format(label_names[prediction[0]],
                      confidence[0], true_label))

        if display_images:
            plt.imshow(np.squeeze(image), cmap='gray')
            plt.show()


if __name__ == '__main__':
    main()
