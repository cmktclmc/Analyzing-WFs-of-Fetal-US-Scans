import os
import sys
from tkinter.messagebox import NO
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib.font_manager import findfont, FontProperties
import matplotlib.patches as mpatches
from matplotlib.ticker import NullFormatter
import pickle as pkl
# from pylab import *

'''
params is a dictionary that contains the font size, label size, tick size, etc. of the plot. It is later updated using plt.rcParams.update(params).
mpl.rcParams['font.sans-serif'] and mpl.rcParams['font.family'] set the font family to 'Lato' and 'sans-serif', respectively.
font = findfont(FontProperties(family=['sans-serif'])) finds the specified font.
The code then imports various packages, including PyTorch and its related libraries, NumPy, and Seaborn.
'''
params = {
    'legend.fontsize': 'x-large',
    'legend.fancybox': 'True',
    # 'figure.figsize': (10, 6),
    'axes.labelsize': 'xx-large',
    'axes.titlesize': 'xx-large',
    'axes.labelpad': '5.0',
    'xtick.labelsize': 'xx-large',
    'ytick.labelsize': 'xx-large',
    'figure.titlesize': 'xx-large',
    'figure.constrained_layout.use': 'True',
    # 'savefig.dpi': '1200',
    'savefig.dpi': '600'
}

mpl.rcParams['font.sans-serif'] = 'Lato'
mpl.rcParams['font.family'] = 'sans-serif'

font = findfont(FontProperties(family=['sans-serif']))
plt.rcParams.update(params)

import torch
from torchvision import transforms as T
from torchvision import *
from torch.utils.data import Dataset, DataLoader, random_split, Subset, RandomSampler
import random
from scipy import stats
import seaborn as sns
from matplotlib.patches import Polygon
from sklearn.metrics import mean_squared_error, r2_score

from tools import *

# -----------------------------------------------------------------------------
'''
train_on_gpu is a boolean flag that checks if a GPU is available for training. The device for training is set to 'cuda' if a GPU is available, or 'cpu' otherwise.
'''
train_on_gpu = torch.cuda.is_available()
device = 'cuda' if train_on_gpu else 'cpu'

print(f"\n>>> Python: {sys.version}")
print(f">>> Pytorch: {torch.__version__}")
print(f">>> Device in use: {device}\n")

'''
img_size, crop_size, num_channels, and other parameters are set for use later in the code.
lambda_val is a regularization term.
out_params is the number of parameters to be outputted.
space_dim is the dimension of space.
vol_dims_phantom and vol_dims_real are the dimensions of phantom and real volumes, respectively.
vol_unity_phantom and vol_unity_real are the scaling factors for the phantom and real volumes, respectively.
div is a parameter for use later in the code.
color_train, color_valid, color_test, color_test_sp, and color_train_loocv are colors used for different purposes in plotting.
'''
img_size = (128, 128)
# img_size = (10, 10)
# crop_size = 82
crop_size = 92
# crop_size = 64
num_channels = 1
lambda_val = 0.01
batch_size = 64
# num_epochs = 50
# num_epochs = 100
num_epochs = 200
learning_rate = 1e-04
out_params = 9
space_dim = 2
# vol_dims_phantom = [249, 199, 160]
# vol_unity_phantom = [1, 0.79, 0.64]
vol_dims_phantom = [249, 199, 160]
vol_unity_phantom = [1, 0.799, 0.638]
# vol_dims_real = [249, 174, 155]
# vol_unity_real = [1, 0.699, 0.62]
# vol_dims_real = [211, 232, 153]
# vol_unity_real = [0.909, 1, 0.655]
vol_dims_real = [211, 232, 153]
vol_unity_real = [0.9094828, 1, 0.6553735]

div = 15
color_train = '#00627E'   # blue
# color_valid = '#A2C4C9' # light blue
color_valid = '#9B0A0E'   # red
# color_test = '#FFBD59'    # yellow
color_test = '#A2C4C9'    # light blue
# color_test_sp = '#FFE0B2' # light yellow
color_test_sp = '#DDE6E8' # light light blue
color_train_loocv = '#e6e6e6' # light grey

# -----------------------------------------------------------------------------

'''
This code sets a seed for different random number generators in order to have deterministic behavior.
torch.manual_seed() sets the seed for PyTorch's random number generator.
torch.cuda.manual_seed() sets the seed for PyTorch's GPU random number generator.
np.random.seed() sets the seed for Numpy's random number generator.
random.seed() sets the seed for Python's random number generator.
torch.backends.cudnn.benchmark = False disables CUDNN benchmark.
torch.backends.cudnn.deterministic = True sets CUDNN to deterministic mode.
'''
seed = 2

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def _init_fn(worker_id):
    '''
    _init_fn sets the seed for each worker thread in a multi-thread environment.
    The seed is set to the value of the "seed" variable casted as an integer.
    '''
    np.random.seed(int(seed))

# -----------------------------------------------------------------------------
# # Fetus
# data_mean = [0.3405, 0.3556, 0.3443]
# data_std = [0.2519, 0.2411, 0.2390]

# Brain
data_mean_brainUS = [0.1584, 0.1816, 0.1703]
data_std_brainUS = [0.1000, 0.0908, 0.0916]

# Normalization values for ImageNet
data_mean_imagenet = [0.485, 0.456, 0.406]
data_std_imagenet = [0.229, 0.224, 0.225]

# Normalization for real data
data_mean = [0.14728942, 0.14728798, 0.14729099]
data_std = [0.15950748, 0.15950628, 0.15950804]

# -----------------------------------------------------------------------------

'''
This code defines four different data transforms for images.
Each data transform consists of a sequence of operations to be performed on the images.
The first data transform, 'phantom', applies the following operations:
1. Convert the image from a PIL Image to a PyTorch Tensor.
2. Resize the image to the specified size (img_size).
3. Apply color jitter with brightness, contrast, and saturation set to 1.
4. Normalize the image by subtracting the mean and dividing by the standard deviation (data_mean, data_std).
The second data transform, 'real', applies the following operations:
1. Convert the image from a PIL Image to a PyTorch Tensor.
2. Resize the image to the specified size (img_size).
3. Apply color jitter with brightness, contrast, and saturation set to 1.
4. Crop the image to the specified size (crop_size) using center cropping.
5. Normalize the image by subtracting the mean and dividing by the standard deviation (data_mean, data_std).
The third data transform, 'real_no_da', applies the following operations:
1. Convert the image from a PIL Image to a PyTorch Tensor.
2. Resize the image to the specified size (img_size).
3. Crop the image to the specified size (crop_size) using center cropping.
4. Normalize the image by subtracting the mean and dividing by the standard deviation (data_mean, data_std).
The fourth data transform, 'real_test', applies the following operations:
1. Convert the image from a PIL Image to a PyTorch Tensor.
2. Resize the image to the specified size (img_size).
3. Crop the image to the specified size (crop_size) using center cropping.
4. Normalize the image by subtracting the mean and dividing by the standard deviation (data_mean, data_std).
'''
data_transforms = {
    'phantom': T.Compose([
        T.ToTensor(),
        T.Resize(img_size),
        T.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9),
        T.CenterCrop(crop_size),
        T.Normalize(data_mean, data_std)
    ]),
    'real': T.Compose([
        T.ToTensor(),
        T.Resize(img_size),
        T.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9),
        # T.CenterCrop(crop_size),
        T.Normalize(data_mean, data_std)
    ]),
    'real_test': T.Compose([
        T.ToTensor(),
        T.Resize(img_size),
        # T.CenterCrop(crop_size),
        T.Normalize(data_mean, data_std)
    ]),
    'phantom_test': T.Compose([
        T.ToTensor(),
        T.Resize(img_size),
        T.CenterCrop(crop_size),
        T.Normalize(data_mean, data_std)
    ])
}
# -----------------------------------------------------------------------------

def get_mean_std(dataset):

    '''
    This code is a function to calculate the mean and standard deviation of a dataset. The means and standard deviation are computed for each of the 3 color channels (red, green, blue) in the images in the dataset. The means and std arrays are initialized to be zero, and then the mean and standard deviation are computed for each channel by iterating over the dataset, summing the mean and standard deviation of each image and channel, and then dividing by the number of images in the dataset. Finally, the computed means and standard deviation are returned from the function.
    '''
    
    means = [0,0,0]
    std = [0,0,0]
    num_imgs = len(dataset)
    for _, sampled_batch in enumerate(dataset):
        img = sampled_batch['image']
        for i in range(3):
            # compute mean and std for each channel
            means[i] += img[:, i, :, :].mean()
            std[i] += img[:, i, :, :].std()
    means = np.asarray(means) / num_imgs
    std = np.asarray(std) / num_imgs
    print("normMean = {}".format(means))
    print("normstd = {}".format(std))
    
    return means, std

# -----------------------------------------------------------------------------

def from_lh_to_rh_sys(df):
    '''
    This function changes the coordinate system of the input data frame 'df' from a left-handed system to a right-handed system.
    '''
    
    df['pos_z'] = df['pos_z'].mul(-1)
    df['rot_x'] = (df['rot_x'].mul(-1)) * np.pi / 180
    df['rot_y'] = (df['rot_y'].mul(-1)) * np.pi / 180
    df['rot_z'] = df['rot_z'] * np.pi / 180

    # # negate the pos_z coordinate
    # df['pos_z'] = df['pos_z'].mul(-1)
    # # convert rot_x to radians and negate it
    # df['rot_x'] = (df['rot_x'].mul(-1)) * np.pi / 180
    # # convert rot_y to radians and negate it
    # df['rot_y'] = (df['rot_y'].mul(-1)) * np.pi / 180
    # # convert rot_z to radians
    # df['rot_z'] = df['rot_z'] * np.pi / 180

def from_rh_to_lh_sys(df):
    '''
    This function changes the coordinate system of the input data frame 'df' from a right-handed system to a left-handed system.
    '''

    df['pos_z'] = df['pos_z'].mul(-1)
    df['rot_x'] = df['rot_x'].mul(-1)
    df['rot_y'] = df['rot_y'].mul(-1)

    # # negate the pos_z coordinate
    # df['pos_z'] = df['pos_z'].mul(-1)
    # # negate the rot_z coordinate
    # df['rot_z'] = df['rot_x'].mul(-1)

# -----------------------------------------------------------------------------

'''
This code implements a function for preparing label data for a machine learning model. The following functions are defined:
'''

def compute_scaler(in_path, out_path):
    '''
    compute_scaler: reads a CSV file specified by in_path into a pandas DataFrame, calls an external function from_lh_to_rh_sys on the DataFrame, normalizes the position values using a MinMaxScaler, adds the normalized positions and rotations as a list in a new column called "pose", drops the original position and rotation columns, and writes the resulting DataFrame back to a CSV file specified by out_path. The function also returns the resulting DataFrame and the MinMaxScaler object.
    '''

    df = pd.read_csv(in_path, sep=',')

    from_lh_to_rh_sys(df)

    # Normalize the position values using a MinMaxScaler object. The MinMaxScaler object scales the values to the range of [-1, 1].
    scaler = MinMaxScaler(feature_range=(-1, 1), copy = True)
    pos = df[['pos_x', 'pos_y', 'pos_z']]
    new_pos = scaler.fit_transform(pos)
    df[['pos_x', 'pos_y', 'pos_z']] = new_pos

    # sLength = len(df['pos_x'])
    # data = list(range(1, sLength + 1))
    # df.insert(0, 'id', data, True)
    # df['id'] = df['id'].astype(str)
    # df['id'] = 'plane' + df['id'].astype(str) + '.png'

    # Add the normalized positions and rotations as a list in a new column called "pose".
    pose = df[['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z']].values.tolist()
    df.insert(1, 'pose', pose, True)
    # Drop the original position and rotation columns.
    df.drop(['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z'], axis=1, inplace=True)
    # Write the resulting DataFrame back to a CSV file specified by out_path.
    df.to_csv(out_path, index=False)

    labels = pd.read_csv(out_path, sep=',')

    # Return the resulting DataFrame and the MinMaxScaler object.
    return labels, scaler 

def apply_scaler(in_path, out_path, scaler):

    '''
    apply_scaler: reads a CSV file specified by in_path into a pandas DataFrame, calls an external function from_lh_to_rh_sys on the DataFrame, normalizes the position values using a scaler specified by scaler, adds the normalized positions and rotations as a list in a new column called "pose", drops the original position and rotation columns, and writes the resulting DataFrame back to a CSV file specified by out_path. The function returns the resulting DataFrame.
    '''

    df = pd.read_csv(in_path, sep=',')

    from_lh_to_rh_sys(df)

    pos = df[['pos_x', 'pos_y', 'pos_z']]
    new_pos = scaler.transform(pos)
    df[['pos_x', 'pos_y', 'pos_z']] = new_pos

    # sLength = len(df['pos_x'])
    # data = list(range(1, sLength + 1))
    # df.insert(0, 'id', data, True)
    # df['id'] = df['id'].astype(str)
    # df['id'] = 'plane' + df['id'].astype(str) + '.png'

    pose = df[['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z']].values.tolist()
    df.insert(1, 'pose', pose, True)
    df.drop(['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z'], axis=1, inplace=True)
    df.to_csv(out_path, index=False)

    labels = pd.read_csv(out_path, sep=',')

    return labels


def prepare_labels_train(in_path, out_path):

    '''
    prepare_labels_train: calls compute_scaler on the input path and output path, and returns the resulting DataFrame and scaler.
    '''

    labels, scaler = compute_scaler(in_path, out_path)

    return labels, scaler

def prepare_labels_test(in_path, out_path, scaler):

    '''
    prepare_labels_test: calls apply_scaler on the input path, output path, and scaler, and returns the resulting DataFrame.
    '''

    labels = apply_scaler(in_path, out_path, scaler)

    return labels

def prepare_labels(in_path_train, out_path_train, in_path_test, out_path_test):

    '''
    prepare_labels: calls prepare_labels_train on the train input path and output path, then calls prepare_labels_test on the test input path and output path and scaler obtained from the previous step, and returns the resulting train and test DataFrames and scaler.
    '''

    labels_train, scaler = prepare_labels_train(in_path_train, out_path_train)
    labels_test = prepare_labels_test(in_path_test, out_path_test, scaler)

    return labels_train, labels_test, scaler

# -----------------------------------------------------------------------------
class FetalDataset(Dataset):
    def __init__(self, data, path, transform=None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label = self.data[index]
        img_path = os.path.join(self.path, img_name)
        # print(img_path)
        image = img.imread(img_path)
        label = np.fromstring(label[2:-2], sep=',')
        label = np.array([label])
        label = label.astype('float').reshape(-1)
        # x = ast.literal_eval(x)
        # np.array(x, dtype=float)
        if self.transform is not None:
            image = self.transform(image)
            # image = T.functional.rotate(image, 180)
            if torch.isnan(image).any() is True:
                print(torch.isnan(image).any())
        return image, label
        # return self.image[index], self.label[index]
        
class FetalDatasetFrames(Dataset):
    def __init__(self, data, path, transform=None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label = self.data[index]
        img_path = os.path.join(self.path, img_name)
        # print(img_path)
        image = img.imread(img_path)
        label = np.fromstring(label[2:-2], sep=',')
        label = np.array([label])
        label = label.astype('float').reshape(-1)
        # x = ast.literal_eval(x)
        # np.array(x, dtype=float)
        if self.transform is not None:
            image = self.transform(image)
            # image = T.functional.rotate(image, 180)
            if torch.isnan(image).any() is True:
                print(torch.isnan(image).any())
        return img_name, image, label
        # return self.image[index], self.label[index]

# -----------------------------------------------------------------------------
def split(dataset_train, dataset_test):

    data_size = len(dataset_train)
    valid_split = int(np.floor(data_size * 0.2))
    train_split = data_size - valid_split
    train_dataset, valid_dataset = random_split(dataset_train,[train_split, valid_split])
    test_dataset = dataset_test

    return train_dataset, valid_dataset, test_dataset

# -----------------------------------------------------------------------------

def data_loaders(dataset_train, dataset_valid, dataset_test, subset):

    if subset is not None:
        dataset_train = Subset(dataset_train, list(range(0, len(dataset_train), subset)))
        dataset_valid = Subset(dataset_valid, list(range(0, len(dataset_valid), subset)))
        dataset_test = Subset(dataset_test, list(range(0, len(dataset_test), subset)))

    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=1)
    print(f"Number of batches in train loader: {len(train_loader)}")
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True, num_workers=1)
    print(f"Number of batches in test loader: {len(test_loader)}")

    if dataset_valid is not None:
        valid_loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=False, num_workers=1)
        print(f"Number of batches in valid loader: {len(valid_loader)}")

        return train_loader, valid_loader, test_loader
    else:
        return train_loader, test_loader

# -----------------------------------------------------------------------------

def imshow(image, ax=None, title=None, normalize=True):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array(data_mean)
        std = np.array(data_std)
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels(' ')
    ax.set_yticklabels(' ')

    return ax

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def compute_metrics(labels_list, predictions_list, scaler, data_type):

    mse = mean_squared_error(predictions_list, labels_list)
    r_square = r2_score(predictions_list, labels_list)

    # err = np.sqrt(((predictions_list-labels_list)**2).sum(axis = 1))
    # err_perc = (err / space_dim) * 100
    # print('Error [%]: ', err_perc[0])

    err = compute_err_mm(labels_list, predictions_list, scaler, data_type)
    # print('Error [mm]: ', err_mm[0])

    return mse, r_square, err

# -----------------------------------------------------------------------------

def compute_err_mm(labels_list, predictions_list, scaler, data_type):

    # labels_list_denorm = scaler.inverse_transform(labels_list)
    # predictions_list_denorm = scaler.inverse_transform(predictions_list)

    # err_x = np.sqrt(((predictions_list_denorm[:, 0] - labels_list_denorm[:, 0])**2))
    # err_y = np.sqrt(((predictions_list_denorm[:, 1] - labels_list_denorm[:, 1])**2))
    # err_z = np.sqrt(((predictions_list_denorm[:, 2] - labels_list_denorm[:, 2])**2))

    # if data_type == 'phantom':
    #     dim_x, dim_y, dim_z = vol_dims_phantom[0], vol_dims_phantom[1], vol_dims_phantom[2]
    #     unity_x, unity_y, unity_z = vol_unity_phantom[0], vol_unity_phantom[1], vol_unity_phantom[2]
    # else:
    #     dim_x, dim_y, dim_z = vol_dims_real[0], vol_dims_real[1], vol_dims_real[2]
    #     unity_x, unity_y, unity_z = vol_unity_real[0], vol_unity_real[1], vol_unity_real[2]

    # mm_x = (dim_x * err_x) / unity_x
    # mm_y = (dim_y * err_y) / unity_y
    # mm_z = (dim_z * err_z) / unity_z

    # err = np.sqrt(mm_x**2 + mm_y**2 + mm_z**2)
    
    # return err
    
    labels_list_denorm = scaler.inverse_transform(labels_list)
    predictions_list_denorm = scaler.inverse_transform(predictions_list)

    if data_type == 'phantom':
        dim_x, dim_y, dim_z = vol_dims_phantom[0], vol_dims_phantom[1], vol_dims_phantom[2]
        unity_x, unity_y, unity_z = vol_unity_phantom[0], vol_unity_phantom[1], vol_unity_phantom[2]
    else:
        dim_x, dim_y, dim_z = vol_dims_real[0], vol_dims_real[1], vol_dims_real[2]
        unity_x, unity_y, unity_z = vol_unity_real[0], vol_unity_real[1], vol_unity_real[2]

    predictions_list_denorm[:, 0] = (dim_x * predictions_list_denorm[:, 0]) / unity_x
    predictions_list_denorm[:, 1] = (dim_y * predictions_list_denorm[:, 1]) / unity_y
    predictions_list_denorm[:, 2] = (dim_z * predictions_list_denorm[:, 2]) / unity_z

    labels_list_denorm[:, 0] = (dim_x * labels_list_denorm[:, 0]) / unity_x
    labels_list_denorm[:, 1] = (dim_y * labels_list_denorm[:, 1]) / unity_y
    labels_list_denorm[:, 2] = (dim_z * labels_list_denorm[:, 2]) / unity_z
    
    err_x = ((predictions_list_denorm[:, 0] - labels_list_denorm[:, 0])**2)
    err_y = ((predictions_list_denorm[:, 1] - labels_list_denorm[:, 1])**2)
    err_z = ((predictions_list_denorm[:, 2] - labels_list_denorm[:, 2])**2) 
    
    err = np.sqrt(err_x + err_y + err_z)

    return err

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def visualize_losses(loss_train, loss_valid, label_y, figure_path, figtitle, figname):

    plt.figure()
    plt.plot(loss_train, color=color_train, linewidth=2, label='Train')
    plt.plot(loss_valid, color=color_valid, linewidth=2, label='Validation')
    plt.ylabel(label_y)
    plt.xlabel('Epochs')
    plt.legend(frameon=True, loc='best')
    plt.grid(color='lightgrey', alpha=0.5)
    plt.title(figtitle)
    # plt.savefig(os.path.join(figure_path, figname))
    plt.savefig(os.path.join(figure_path, figname), format='png')
    plt.clf()

# -----------------------------------------------------------------------------

def visualize_losses_comparison(loss1_train, loss2_train, loss1_valid, loss2_valid, figure_path, figname):
    
    plt.figure()
    plt.plot(loss1_train, color=color_train, linewidth=2, label='L2 Transl - Train')
    plt.plot(loss1_valid, color=color_valid, linewidth=2, label='L2 Transl - Validation')
    plt.plot(loss2_train, color=color_train, linewidth=2, linestyle='dashed', label='L2 Rot - Train')
    plt.plot(loss2_valid, color=color_valid, linewidth=2, linestyle='dashed', label='L2 Rot - Validation')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epochs')
    plt.legend(frameon=True, loc='best')
    plt.grid(color='lightgrey', alpha=0.5)
    plt.title('Losses comparison')
    plt.savefig(os.path.join(figure_path, figname))
    # plt.savefig(os.path.join(figure_path, figname), format='png')
    plt.clf()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def visualize_errors_percentiles_rot(error_lst_1, error_lst_2, label_y, figure_path, figname):

    percentile_lst = []
    score_lst = []
    percentile_lst_test = []
    score_lst_test = []
    i_lst = np.linspace(0, 100, num=400)
    
    for i in i_lst:
        percentile = i
        score = stats.scoreatpercentile(error_lst_1, percentile)
        score_lst = score_lst + [score]
        percentile_lst = percentile_lst + [percentile]

        score_test = stats.scoreatpercentile(error_lst_2, percentile)
        score_lst_test = score_lst_test + [score_test]
        percentile_lst_test = percentile_lst_test + [percentile]

    percentile_lst = np.array(percentile_lst)


    score_lst = np.array(score_lst)
    percentile_lst_test = np.array(percentile_lst_test)
    score_lst_test = np.array(score_lst_test)
    shape = ""
    plt.figure()
    plt.plot(percentile_lst, score_lst, color_train, linewidth=2, label='Train')
    plt.plot(percentile_lst_test, score_lst_test, color_test, linewidth=2, label='Test')
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_func_y_degree))
    plt.gca().xaxis.set_minor_formatter(NullFormatter())
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func_x_percentile))
    plt.legend(frameon=True, loc='best')
    plt.ylabel(label_y)
    plt.xlabel('Percentiles')
    plt.grid(color='lightgrey', alpha=0.5)
    plt.title('Percentiles of errors')
    plt.savefig(os.path.join(figure_path, figname))
    # plt.savefig(os.path.join(figure_path, figname), format='png')
    plt.clf()

# -----------------------------------------------------------------------------

def visualize_errors_percentiles_transl(error_lst_1, error_lst_2, label_y, figure_path, figname):

    percentile_lst = []
    score_lst = []
    percentile_lst_test = []
    score_lst_test = []
    i_lst = np.linspace(0, 100, num=400)
    
    for i in i_lst:
        percentile = i
        score = stats.scoreatpercentile(error_lst_1, percentile)
        score_lst = score_lst + [score]
        percentile_lst = percentile_lst + [percentile]

        score_test = stats.scoreatpercentile(error_lst_2, percentile)
        score_lst_test = score_lst_test + [score_test]
        percentile_lst_test = percentile_lst_test + [percentile]

    percentile_lst = np.array(percentile_lst)

    score_lst = np.array(score_lst)
    percentile_lst_test = np.array(percentile_lst_test)
    score_lst_test = np.array(score_lst_test)
    shape = ""
    plt.figure()
    plt.plot(percentile_lst, score_lst, color_train, linewidth=2, label='Train')
    plt.plot(percentile_lst_test, score_lst_test, color_test, linewidth=2, label='Test')
    plt.gca().xaxis.set_minor_formatter(NullFormatter())
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func_x_percentile))
    plt.legend(frameon=True, loc='best')
    plt.ylabel(label_y)
    plt.xlabel('Percentiles')
    plt.grid(color='lightgrey', alpha=0.5)
    plt.title('Percentiles of errors')
    plt.savefig(os.path.join(figure_path, figname))
    # plt.savefig(os.path.join(figure_path, figname), format='png')
    plt.clf()

# -----------------------------------------------------------------------------

def visualize_errors_distribution_rot(error_lst_1, error_lst_2, label_x, figure_path, figname):

    error_groups = [[0, 30], [30, 60], [60, 90], [90, 120], [120, 150], [150, 180]]
    error_groups = []

    for i in range(18):
        error_groups = error_groups + [[10 * i, 10 * i + 10]]
        percents = []
        percents_test = []
        axis = []
        for group in error_groups:
            num = 0
            num_test = 0

            for error in error_lst_1:
                if ((error < group[1]) and (error >= group[0])):
                    num = num + 1
            percent = num / error_lst_1.shape[0]
            if (percent == 1.0):
                percent = percent - 1e-5
            elif (percent == 0.0):
                percent = percent + 1e-5
            percents = percents + [percent]

            for error_2 in error_lst_2:
                if ((error_2 < group[1]) and (error_2 >= group[0])):
                    num_test = num_test + 1
            percent_test = num_test / error_lst_2.shape[0]
            if (percent_test == 1.0):
                percent_test = percent_test - 1e-5
            elif (percent_test == 0.0):
                percent_test = percent_test + 1e-5
            percents_test = percents_test + [percent_test]

            axis = axis + [group[1] - 5]

        percents = np.array(percents)
        percents_test = np.array(percents_test)
        axis = np.array(axis)

        plt.figure()
        plt.plot(axis, percents, color_train, linewidth=2, label='Train')
        plt.plot(axis, percents_test, color_test, linewidth=2, label='Test')

    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_func_y))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func_x))
    plt.legend(frameon=True, loc='best')
    plt.xlabel(label_x)
    plt.ylabel(' ')
    plt.title('Distribution of errors - Rotation')
    plt.grid(color='lightgrey', alpha=0.5)
    plt.savefig(os.path.join(figure_path, figname))
    # plt.savefig(os.path.join(figure_path, figname), format='png')
    plt.clf()
# -----------------------------------------------------------------------------

def visualize_errors_hist(error_lst_1, error_lst_2, figname, figtitle, label_x, figure_path, plot_type):

    nbins = 50
    plt.figure()
    ax = plt.hist(error_lst_1, density=False, bins=nbins, log=False, color=color_train, alpha=0.5, edgecolor='k', label='Train')
    ax = plt.hist(error_lst_2, density=False, bins=nbins, log=False, color=color_test, alpha=0.5, edgecolor='k', label='Test')
    ax = sns.distplot(error_lst_1, hist=False, rug=True, color=color_train, kde=False)
    ax = sns.distplot(error_lst_2,
                      hist=False,
                      rug=True,
                      color=color_test,
                      kde=False)
    plt.legend(frameon=True, loc='best')
    plt.xlabel(label_x)
    plt.ylabel('Number of samples')

    if plot_type == 'rotation':
        plt.gca().xaxis.set_minor_formatter(NullFormatter())
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func_x))

    plt.title(figtitle)
    ax.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_axisbelow(True)
    # plt.savefig(os.path.join(figure_path, figname))
    plt.savefig(os.path.join(figure_path, figname), format='png')
    plt.clf()

# -----------------------------------------------------------------------------
def boxplot_weekly(arr_train, list_test_rp, list_test_sp, figname, figpath, type):

    data_to_plot = [arr_train] + list_test_rp + list_test_sp
    plt.boxplot(data_to_plot)

    if type == 'rotation':
        plt.gca().yaxis.set_minor_formatter(NullFormatter())
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_func_y_degree2))

    if figpath is not None:
        # plt.savefig(os.path.join(figpath, figname))
        plt.savefig(os.path.join(figpath, figname), format='png')
    else:
        plt.show()
    plt.close()

def boxplot(data_to_plot_list, boxColors_list, labels_list, label_y, figtitle, figname, figpath, type):

    data_to_plot = data_to_plot_list
    labels = labels_list
    boxColors = boxColors_list

    numBoxes = len(data_to_plot_list)

    fig, ax = plt.subplots()
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xticklabels(labels)

    bp = plt.boxplot(
        data_to_plot,
        # labels=labels,
        notch=False,
        patch_artist=True,
        # flierprops=custom_fliers,
        vert=1,
        whis=1.5,
        showfliers=True,
        showmeans=True,
        meanprops={'marker': 'o', 'markeredgecolor': 'black','markerfacecolor': 'white'})

    plt.setp(bp['boxes'], color='black', linewidth=1)
    plt.setp(bp['whiskers'], color='black', linestyle='-', linewidth=1)
    plt.setp(bp['means'], color='black', linewidth=1)
    plt.setp(bp['medians'], color=color_valid, linewidth=1)
    plt.setp(bp['caps'], color='black', linewidth=1)
    plt.setp(bp['fliers'], color='black', marker='+')

    for patch, color in zip(bp['boxes'], boxColors):
        patch.set_facecolor(color)

    for mean in bp['means']:
        mean.set(marker ='o', color ='white')

    pos = np.arange(numBoxes) + 1
    for line, tick in zip(bp['medians'], range(numBoxes)):
        x, y = line.get_xydata()[1]
        ax.annotate(str(np.round(y, 3)), xy=(pos[tick],-0.1), horizontalalignment='center', size='large', weight='bold', color=color_valid, xycoords=ax.get_xaxis_transform())

    if type == 'rotation':
        plt.gca().yaxis.set_minor_formatter(NullFormatter())
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_func_y_degree2))

    train_patch = mpatches.Patch(color=boxColors[0], label='Train')
    test_patch = mpatches.Patch(color=boxColors[1], label='Test')
    plt.legend([bp['medians'][0], bp['means'][0], train_patch, test_patch], ['Median', 'Mean', 'Train', 'Test'])

    ax.set_ylabel(label_y)
    ax.set_xlabel(' ')
    ax.set_title(figtitle)

    if figpath is not None:
        # plt.savefig(os.path.join(figpath, figname))
        plt.savefig(os.path.join(figpath, figname), format='png')
    else:
        plt.show()
    plt.close()


def visualize_errors_boxplot(error_lst_1, error_lst_2, figname, figtitle, label_y, figure_path, type):

    data_to_plot = [error_lst_1, error_lst_2]
    labels = ['Train', 'Test']
    fig, ax = plt.subplots()
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_axisbelow(True)
    # custom_fliers = dict(markerfacecolor='#A2C4C9', marker='o')
    # custom_fliers = dict(markerfacecolor='red', color='k', marker='o')
    # custom_fliers = dict(markerfacecolor='black', color='red', marker='o')

    bp = plt.boxplot(data_to_plot, notch=False, vert=1, whis=1.5,showfliers=True)
    plt.setp(bp['boxes'], color='black', linewidth=1)
    plt.setp(bp['whiskers'], color='black', linestyle='--', linewidth=1)
    plt.setp(bp['means'], color='black', linewidth=1)
    plt.setp(bp['medians'], color='red', linewidth=1)
    plt.setp(bp['caps'], color='black', linewidth=1)
    plt.setp(bp['fliers'], color='black', marker='+')

    boxColors = [color_train, color_test]
    numBoxes = 2
    medians = list(range(numBoxes))
    for i in range(numBoxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        # Alternate between Dark Khaki and Royal Blue
        k = i % 2
        boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
        ax.add_patch(boxPolygon)
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            plt.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        plt.plot([np.average(med.get_xdata())], [np.average(data_to_plot[i])], color='w', marker='o', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax.set_xlim(0.5, numBoxes + 0.5)
    # ax.set_ylim(-0.5, None)       rotation
    # ax.set_ylim(-0.1, None)       translation
    # ax.set_xticklabels([labels[0], labels[1]])
    ax.set(xticklabels=[labels[0], labels[1]])

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(numBoxes) + 1
    upperLabels = [str(np.round(s, 2)) for s in medians]
    weights = ['bold', 'semibold']
    # for tick, label in zip(range(numBoxes), ax.get_xticklabels()):
    #     k = tick % 2
    #     ax.text(pos[tick], -20.00, upperLabels[tick], horizontalalignment='center', size='x-small', backgroundcolor=boxColors[k], weight=weights[k], color='black')

    if type == 'rotation':
        plt.gca().yaxis.set_minor_formatter(NullFormatter())
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_func_y_degree2))

    plt.figtext(0.87, 0.35, 'Median:', color='black', weight='roman', size='medium')
    plt.figtext(0.88, 0.30, upperLabels[0], size='medium', backgroundcolor=boxColors[0], weight=weights[0], color='black')
    plt.figtext(0.88, 0.23, upperLabels[1], size='medium', backgroundcolor=boxColors[1], weight=weights[0], color='black')
    plt.figtext(0.87, 0.17, 'o', color='black', weight='roman', size='medium')
    plt.figtext(0.89, 0.17, 'Mean', color='black', weight='roman', size='medium')

    ax.set_ylabel(label_y)
    ax.set_xlabel(' ')
    ax.set_title(figtitle)
    if figure_path is not None:
        # plt.savefig(os.path.join(figure_path, figname))
        plt.savefig(os.path.join(figure_path, figname), format='png')
    else:
        plt.show()
    plt.close(fig)

# -----------------------------------------------------------------------------

def visualize_errors_boxplot_2(error_lst_1, error_lst_2, error_lst_3, figname, figtitle, label_y, figure_path, type):

    data_to_plot = [error_lst_1, error_lst_2, error_lst_3]
    labels = ['Train', 'Test RP', 'Test SP']
    fig, ax = plt.subplots()
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_axisbelow(True)
    # custom_fliers = dict(markerfacecolor='#A2C4C9', marker='o')
    # custom_fliers = dict(markerfacecolor='red', color='k', marker='o')
    # custom_fliers = dict(markerfacecolor='black', color='red', marker='o')

    bp = plt.boxplot(data_to_plot, notch=False, vert=1, whis=1.5,showfliers=True)
    plt.setp(bp['boxes'], color='black', linewidth=1)
    plt.setp(bp['whiskers'], color='black', linestyle='--', linewidth=1)
    plt.setp(bp['means'], color='black', linewidth=1)
    plt.setp(bp['medians'], color='red', linewidth=1)
    plt.setp(bp['caps'], color='black', linewidth=1)
    plt.setp(bp['fliers'], color='black', marker='+')

    boxColors = [color_train, color_test, color_test_sp]
    numBoxes = 3
    medians = list(range(numBoxes))
    for i in range(numBoxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        # Alternate between Dark Khaki and Royal Blue
        k = i % 3
        boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
        ax.add_patch(boxPolygon)
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            plt.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        plt.plot([np.average(med.get_xdata())], [np.average(data_to_plot[i])], color='w', marker='o', markeredgecolor='k')

    # Set the axes ranges and axes labels
    # ax.set_xlim(0.5, numBoxes + 0.5)
    ax.set_xlim(0.5, numBoxes + 1.0)
    # ax.set_ylim(-0.5, None)
    ax.set_xticklabels([labels[0], labels[1], labels[2]])

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(numBoxes) + 1
    upperLabels = [str(np.round(s, 2)) for s in medians]
    weights = ['bold', 'semibold']
    # for tick, label in zip(range(numBoxes), ax.get_xticklabels()):
    #     k = tick % 2
    #     ax.text(pos[tick], -20.00, upperLabels[tick], horizontalalignment='center', size='x-small', backgroundcolor=boxColors[k], weight=weights[k], color='black')

    if type == 'rotation':
        plt.gca().yaxis.set_minor_formatter(NullFormatter())
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_func_y_degree2))

    plt.figtext(0.87, 0.45, 'Median:', color='black', weight='roman', size='medium')
    plt.figtext(0.88, 0.40, upperLabels[0], size='medium', backgroundcolor=boxColors[0], weight=weights[0], color='black')
    plt.figtext(0.88, 0.33, upperLabels[1], size='medium', backgroundcolor=boxColors[1], weight=weights[0], color='black')
    plt.figtext(0.88, 0.26, upperLabels[2], size='medium', backgroundcolor=boxColors[2], weight=weights[0], color='black')
    plt.figtext(0.87, 0.17, 'o', color='black', weight='roman', size='medium')
    plt.figtext(0.89, 0.17, 'Mean', color='black', weight='roman', size='medium')

    ax.set_ylabel(label_y)
    ax.set_xlabel(' ')
    ax.set_title(figtitle)
    # plt.savefig(os.path.join(figure_path, figname))
    plt.savefig(os.path.join(figure_path, figname), format='png')
    plt.close(fig)

# -----------------------------------------------------------------------------

def visualize_errors_distplot(error_lst_1, col_1, label_1, error_lst_2, col_2, label_2, figname, figtitle, label_x, label_y, figure_path):
    
    nbins = 50
    plt.figure()
    ax1 = sns.distplot(error_lst_1, norm_hist=False, bins=nbins, hist_kws=dict(edgecolor='k', color=col_1), rug=False, kde=False, label=label_1)
    ax2 = sns.distplot(error_lst_2, norm_hist=False, bins=nbins, hist_kws=dict(edgecolor='k', color=col_2), rug=False, kde=False, label=label_2)
    # sns.kdeplot(error_lst_1, color=col_1, shade=True, alpha=0.5,linewidth=2,label=label_1)
    sns.distplot(error_lst_1, ax=ax1, hist=False, rug=True, color=col_1, kde=False)
    sns.distplot(error_lst_2, ax=ax2, hist=False, rug=True, color=col_2, kde=False)
    # sns.kdeplot(error_lst_1, color=col_1, linewidth=2)
    # sns.kdeplot(error_lst_2, color=col_2, linewidth=2)
    plt.legend(frameon=True, loc='best')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.gca().xaxis.set_minor_formatter(NullFormatter())
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func_x))
    plt.xlim(0, None)
    plt.title(figtitle)
    plt.grid(color='lightgrey', alpha=0.5)
    # plt.savefig(os.path.join(figure_path, figname))
    plt.savefig(os.path.join(figure_path, figname), format='png')
    plt.clf()

# -----------------------------------------------------------------------------

def visualize_evolution_boxplot(error_lst, col, label, figname, figtitle, label_x, label_y, figure_path):

    data_to_plot = error_lst
    n = div
    labels = [label]
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.075, right=0.92, top=0.9, bottom=0.25)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_axisbelow(True)

    bp = plt.boxplot(data_to_plot, notch=False, vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black', linewidth=1)
    plt.setp(bp['whiskers'], color='black', linestyle='--', linewidth=1)
    plt.setp(bp['means'], color='black', linewidth=1)
    plt.setp(bp['medians'], color='red', linewidth=1)
    plt.setp(bp['caps'], color='black', linewidth=1)
    plt.setp(bp['fliers'], color='black', marker='+')

    boxColors = col
    numBoxes = len(data_to_plot)
    medians = list(range(numBoxes))
    for i in range(numBoxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        # Alternate between Dark Khaki and Royal Blue
        k = i % 2
        boxPolygon = Polygon(boxCoords, facecolor=boxColors)
        ax.add_patch(boxPolygon)
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            plt.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        plt.plot([np.average(med.get_xdata())], [np.average(data_to_plot[i])], color='w', marker='o', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax.set_xlim(0.5, numBoxes + 0.5)
    top = 190
    bottom = -1
    ax.set_ylim(bottom, top)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(numBoxes) + 1
    upperLabels = [str(np.round(s, 2)) for s in medians]
    weights = ['bold', 'semibold']
    for tick in range(numBoxes):
        k = tick % 2
        ax.text(pos[tick], -42.5, upperLabels[tick],horizontalalignment='center', size='x-small', weight=weights[k],color='k',  rotation=90)

    if type == 'rotation':
        plt.gca().yaxis.set_minor_formatter(NullFormatter())
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_func_y_degree2))

    plt.figtext(0.79, 0.06, 'o', color='black', weight='roman', size='medium')
    plt.figtext(0.82, 0.055, 'Average', color='black', weight='roman', size='small')
    plt.figtext(0.82, 0.03, 'Value', color='black', weight='roman', size='small')

    ax.set_ylabel(label_y)
    ax.set_xlabel(label_x)
    ax.set_title(figtitle)
    # plt.savefig(os.path.join(figure_path, figname))
    plt.savefig(os.path.join(figure_path, figname), format='png')
    plt.close(fig)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def save_results(transl_errors_list, geodesic_errors_list, res_file):

    res_file.write(
        'Translation' + '\n' + 'median: ' + '\n' +
        str((np.round(np.median(transl_errors_list), 6))) + ' mm' +
        '\n' + 'min: ' + '\n' +
        str((np.round(transl_errors_list.min(), 6))) + ' mm' +
        '\n' + 'max: ' + '\n' +
        str((np.round(transl_errors_list.max(), 6))) + ' mm' +
        '\n' + 'max index: ' + '\n' +
        str(transl_errors_list.argmax()) + '\n' + 'mean: ' +
        '\n' + str((np.round(transl_errors_list.mean(), 6))) +
        ' mm' + '\n' + 'std: ' + '\n' +
        str((np.round(transl_errors_list.std(), 6))) + '\n' +
        '\n' + 'Rotation' + '\n' + 'median: ' + '\n' +
        str((np.round(np.median(geodesic_errors_list), 6))) + ' deg' +
        '\n' + 'min: ' + '\n' +
        str((np.round(geodesic_errors_list.min(), 6))) + ' deg' +
        '\n' + 'max: ' + '\n' +
        str((np.round(geodesic_errors_list.max(), 6))) + ' deg' +
        '\n' + 'max index: ' + '\n' +
        str(geodesic_errors_list.argmax()) + '\n' + 'mean: ' + '\n' +
        str((np.round(geodesic_errors_list.mean(), 6))) + ' deg' +'\n' +
        'std: ' + '\n' +
        str((np.round(geodesic_errors_list.std(), 6))) + ' deg')
    res_file.close()

# -----------------------------------------------------------------------------


def save_files(labels_list_transl, predictions_list_transl, scaler, labels_list_rot, predictions_list_rot, gt_filename, pred_filename, file_path):

    labels_list_transl = [a.squeeze().tolist() for a in labels_list_transl]
    # np.save('labels_list.npy', np.array(labels_list_transl), allow_pickle=True)
    labels_list_transl = np.concatenate(labels_list_transl, axis=0)
    predictions_list_transl = [a.squeeze().tolist() for a in predictions_list_transl]
    predictions_list_transl = np.concatenate(predictions_list_transl, axis=0)


    labels_list_rot = [a.squeeze().tolist() for a in labels_list_rot]
    labels_list_rot = np.concatenate(labels_list_rot, axis=0)
    predictions_list_rot = [a.squeeze().tolist() for a in predictions_list_rot]
    predictions_list_rot = np.concatenate(predictions_list_rot, axis=0)

    labels_list_rot = labels_list_rot * 180 / np.pi
    predictions_list_rot = predictions_list_rot * 180 / np.pi

    gt_transl = pd.DataFrame(np.array(labels_list_transl), columns=['pos_x', 'pos_y', 'pos_z'])
    pred_transl = pd.DataFrame(np.array(predictions_list_transl), columns=['pos_x', 'pos_y', 'pos_z'])
    gt_rot = pd.DataFrame(np.array(labels_list_rot), columns=['rot_x', 'rot_y', 'rot_z'])
    pred_rot = pd.DataFrame(np.array(predictions_list_rot), columns=['rot_x', 'rot_y', 'rot_z'])

    gt = pd.concat([gt_transl, gt_rot], axis=1)
    pred = pd.concat([pred_transl, pred_rot], axis=1)

    # gt.to_csv(os.path.join(file_path, 'gt_train_norm.csv'), index=False, header=True)
    # pred.to_csv(os.path.join(file_path, 'pred_train_norm.csv'), index=False, header=True)

    gt[['pos_x', 'pos_y', 'pos_z']] = scaler.inverse_transform(gt[['pos_x', 'pos_y', 'pos_z']])
    pred[['pos_x', 'pos_y', 'pos_z']] = scaler.inverse_transform(pred[['pos_x', 'pos_y', 'pos_z']])

    # gt_order = gt['rot_x'].argsort()
    # gt_t_x, gt_t_y, gt_t_z = gt['pos_x'][gt_order], gt['pos_y'][gt_order], gt['pos_z'][gt_order]
    # gt_r_x, gt_r_y, gt_r_z = gt['rot_x'][gt_order], gt['rot_y'][gt_order], gt['rot_z'][gt_order]
    # gt = pd.concat([gt_t_x, gt_t_y, gt_t_z, gt_r_x, gt_r_y, gt_r_z], axis=1, keys=['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z'])
    # pred_t_x, pred_t_y, pred_t_z = pred['pos_x'][gt_order], pred['pos_y'][gt_order], pred['pos_z'][gt_order]
    # pred_r_x, pred_r_y, pred_r_z = pred['rot_x'][gt_order], pred['rot_y'][gt_order], pred['rot_z'][gt_order]
    # pred = pd.concat([pred_t_x, pred_t_y, pred_t_z, pred_r_x, pred_r_y, pred_r_z], axis=1, keys=['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z'])

    from_rh_to_lh_sys(gt)
    from_rh_to_lh_sys(pred)

    gt.to_csv(os.path.join(file_path, gt_filename), index=False, header=True)
    pred.to_csv(os.path.join(file_path, pred_filename), index=False, header=True)
    
    
def save_files_frames(image_names, labels_list_transl, predictions_list_transl, scaler, labels_list_rot, predictions_list_rot, gt_filename, pred_filename, file_path):

    labels_list_transl = [a.squeeze().tolist() for a in labels_list_transl]
    # np.save('labels_list.npy', np.array(labels_list_transl), allow_pickle=True)
    labels_list_transl = np.concatenate(labels_list_transl, axis=0)
    predictions_list_transl = [a.squeeze().tolist() for a in predictions_list_transl]
    predictions_list_transl = np.concatenate(predictions_list_transl, axis=0)


    labels_list_rot = [a.squeeze().tolist() for a in labels_list_rot]
    labels_list_rot = np.concatenate(labels_list_rot, axis=0)
    predictions_list_rot = [a.squeeze().tolist() for a in predictions_list_rot]
    predictions_list_rot = np.concatenate(predictions_list_rot, axis=0)

    labels_list_rot = labels_list_rot * 180 / np.pi
    predictions_list_rot = predictions_list_rot * 180 / np.pi

    gt_transl = pd.DataFrame(np.array(labels_list_transl).reshape(-1, 3), columns=['pos_x', 'pos_y', 'pos_z'])
    pred_transl = pd.DataFrame(np.array(predictions_list_transl), columns=['pos_x', 'pos_y', 'pos_z'])
    gt_rot = pd.DataFrame(np.array(labels_list_transl).reshape(-1, 3), columns=['rot_x', 'rot_y', 'rot_z'])
    pred_rot = pd.DataFrame(np.array(predictions_list_rot), columns=['rot_x', 'rot_y', 'rot_z'])

    gt = pd.concat([gt_transl, gt_rot], axis=1)
    pred = pd.concat([pred_transl, pred_rot], axis=1)
    
    image_names_with_extension = [str(name) + '.png' for name in image_names]
    
    gt.insert(0, 'img_name', image_names_with_extension)
    pred.insert(0, 'img_name', image_names_with_extension)

    # gt.to_csv(os.path.join(file_path, 'gt_train_norm.csv'), index=False, header=True)
    # pred.to_csv(os.path.join(file_path, 'pred_train_norm.csv'), index=False, header=True)

    gt[['pos_x', 'pos_y', 'pos_z']] = scaler.inverse_transform(gt[['pos_x', 'pos_y', 'pos_z']])
    pred[['pos_x', 'pos_y', 'pos_z']] = scaler.inverse_transform(pred[['pos_x', 'pos_y', 'pos_z']])

    from_rh_to_lh_sys(gt)
    from_rh_to_lh_sys(pred)

    gt.to_csv(os.path.join(file_path, gt_filename), index=False, header=True)
    pred.to_csv(os.path.join(file_path, pred_filename), index=False, header=True)

# -----------------------------------------------------------------------------

def save_files_generalization(predictions_list_transl, predictions_list_rot, filename, file_path):

    predictions_list_transl = [a.squeeze().tolist() for a in predictions_list_transl]
    predictions_list_transl = np.concatenate(predictions_list_transl, axis=0)

    predictions_list_rot = [a.squeeze().tolist() for a in predictions_list_rot]
    predictions_list_rot = np.concatenate(predictions_list_rot, axis=0)

    predictions_list_rot = predictions_list_rot * 180 / np.pi

    pred_transl = pd.DataFrame(np.array(predictions_list_transl), columns=['pos_x', 'pos_y', 'pos_z'])
    pred_rot = pd.DataFrame(np.array(predictions_list_rot), columns=['rot_x', 'rot_y', 'rot_z'])

    # pred_transl[['pos_x', 'pos_y', 'pos_z']] = scaler.inverse_transform(pred_transl[['pos_x', 'pos_y', 'pos_z']])

    pred = pd.concat([pred_transl, pred_rot], axis=1)

    from_rh_to_lh_sys(pred)

    pred.to_csv(os.path.join(file_path, filename), index=False, header=True)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def save_errors(transl_errors_list_mm, rot_errors_list, transl_filename, rot_filename, file_path):

    transl_errors_list_mm = [a.squeeze().tolist() for a in transl_errors_list_mm]
    rot_errors_list = [a.squeeze().tolist() for a in rot_errors_list]

    transl_err_mm = pd.DataFrame(np.array(transl_errors_list_mm))
    rot_err = pd.DataFrame(np.array(rot_errors_list))

    transl_err_mm.to_csv(os.path.join(file_path, transl_filename), index=False, header=False)
    rot_err.to_csv(os.path.join(file_path, rot_filename), index=False, header=False)
    
def save_errors_frames(image_names, transl_errors_list_mm, rot_errors_list, transl_filename, rot_filename, file_path):

    image_names_with_extension = [str(name) + '.png' for name in image_names]
    
    transl_errors_list_mm = [a.squeeze().tolist() for a in transl_errors_list_mm]
    rot_errors_list = [a.squeeze().tolist() for a in rot_errors_list]

    transl_err_mm = pd.DataFrame(np.array(transl_errors_list_mm))
    transl_err_mm.insert(0, 'Frames', image_names_with_extension)  # Insert image names as the first column
    rot_err = pd.DataFrame(np.array(rot_errors_list))
    rot_err.insert(0, 'Frames', image_names_with_extension)  # Insert image names as the first column

    transl_err_mm.to_csv(os.path.join(file_path, transl_filename), index=False, header=False)
    rot_err.to_csv(os.path.join(file_path, rot_filename), index=False, header=False)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def visualize_errors_boxplot_weekly(error_train, error_weekly_list_rp, error_weekly_list_sp, figname, figtitle, label_y, figure_path, type):

    data_to_plot = [error_train] + error_weekly_list_rp + error_weekly_list_sp

    labels = ['23w', '20w', '21w', '22w', '23w', '24w', '25w', '26w', '20w', '21w', '22w', '23w', '24w', '25w', '26w']
    fig, ax = plt.subplots()
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_axisbelow(True)

    bp = plt.boxplot(data_to_plot, notch=False, vert=1, whis=1.5,showfliers=True)
    plt.setp(bp['boxes'], color='black', linewidth=1)
    plt.setp(bp['whiskers'], color='black', linestyle='--', linewidth=1)
    plt.setp(bp['means'], color='black', linewidth=1)
    plt.setp(bp['medians'], color='red', linewidth=1)
    plt.setp(bp['caps'], color='black', linewidth=1)
    plt.setp(bp['fliers'], color='black', marker='+')

    boxColors = [color_train, color_test, color_test, color_test, color_test, color_test, color_test, color_test, color_test_sp, color_test_sp, color_test_sp, color_test_sp, color_test_sp, color_test_sp, color_test_sp]
    numBoxes = len(labels)

    medians = list(range(numBoxes))
    for i in range(numBoxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        # Alternate between Dark Khaki and Royal Blue
        k = i % numBoxes
        boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
        ax.add_patch(boxPolygon)
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            plt.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        plt.plot([np.average(med.get_xdata())], [np.average(data_to_plot[i])], color='w', marker='o', markeredgecolor='k')

    # Set the axes ranges and axes labels
    # ax.set_xlim(0.5, numBoxes + 0.5)
    ax.set_xlim(0.5, numBoxes + 0.5)
    # ax.set_ylim(-0.1, None)
    ax.set_xticklabels([labels[0], labels[1], labels[2], labels[3], labels[4], labels[5], labels[6], labels[7], labels[8], labels[9], labels[10], labels[11], labels[12], labels[13], labels[14]], rotation=90)

    pos = np.arange(numBoxes) + 1
    upperLabels = [str(np.round(s, 2)) for s in medians]
    weights = ['bold', 'semibold']

    plt.figtext(0.92, 0.93, 'Median:', size='small', color='black', weight='roman')
    plt.figtext(0.93, 0.915, upperLabels[0], size='small', backgroundcolor=boxColors[0], weight=weights[0], color='black') # 23
    plt.figtext(0.93, 0.865, upperLabels[1], size='small', backgroundcolor=boxColors[1], weight=weights[0], color='black') # 20
    plt.figtext(0.93, 0.815, upperLabels[2], size='small', backgroundcolor=boxColors[2], weight=weights[0], color='black') # 21
    plt.figtext(0.93, 0.765, upperLabels[3], size='small', backgroundcolor=boxColors[3], weight=weights[0], color='black') # 22
    plt.figtext(0.93, 0.715, upperLabels[4], size='small', backgroundcolor=boxColors[4], weight=weights[0], color='black') # 23
    plt.figtext(0.93, 0.665, upperLabels[5], size='small', backgroundcolor=boxColors[5], weight=weights[0], color='black') # 24
    plt.figtext(0.93, 0.615, upperLabels[6], size='small', backgroundcolor=boxColors[6], weight=weights[0], color='black') # 25
    plt.figtext(0.93, 0.565, upperLabels[7], size='small', backgroundcolor=boxColors[7], weight=weights[0], color='black') # 26
    plt.figtext(0.93, 0.515, upperLabels[8], size='small', backgroundcolor=boxColors[8], weight=weights[0], color='black') # 20
    plt.figtext(0.93, 0.465, upperLabels[9], size='small', backgroundcolor=boxColors[9], weight=weights[0], color='black') # 21
    plt.figtext(0.93, 0.415, upperLabels[10], size='small', backgroundcolor=boxColors[10], weight=weights[0], color='black') # 22
    plt.figtext(0.93, 0.365, upperLabels[11], size='small', backgroundcolor=boxColors[11], weight=weights[0], color='black') # 23
    plt.figtext(0.93, 0.315, upperLabels[12], size='small', backgroundcolor=boxColors[12], weight=weights[0], color='black') # 24
    plt.figtext(0.93, 0.265, upperLabels[13], size='small', backgroundcolor=boxColors[13], weight=weights[0], color='black') # 25
    plt.figtext(0.93, 0.22, upperLabels[14], size='small', backgroundcolor=boxColors[14], weight=weights[0], color='black') # 26
    plt.figtext(0.92, 0.18, 'o', color='black', weight='roman', size='medium')
    plt.figtext(0.94, 0.18, 'Mean', color='black', weight='roman', size='small')

    if type == 'rotation':
        # ax.set_ylim(-0.1, 30)
        plt.gca().yaxis.set_minor_formatter(NullFormatter())
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_func_y_degree2))
        pd.DataFrame(data_to_plot).to_csv(os.path.join(figure_path, 'data_plot_rot.csv'), index=False, header=False)
    else:
        pd.DataFrame(data_to_plot).to_csv(os.path.join(figure_path, 'data_plot_transl.csv'), index=False, header=False)

    ax.set_ylabel(label_y)
    ax.set_xlabel(' ')
    ax.set_title(figtitle)
    # plt.savefig(os.path.join(figure_path, figname))
    plt.savefig(os.path.join(figure_path, figname), format='png')
    plt.close(fig)


# -----------------------------------------------------------------------------

def visualize_errors_boxplot_weekly_new(error_train, error_weekly_list_rp, error_weekly_list_sp, figname, figtitle, label_y, figure_path, type):

    data_to_plot = [error_train] + error_weekly_list_rp + error_weekly_list_sp
    labels = ['23w', '20w', '21w', '22w', '23w', '24w', '25w', '26w', '20w', '21w', '22w', '23w', '24w', '25w', '26w']
    fig, ax = plt.subplots()
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_axisbelow(True)

    bp = plt.boxplot(data_to_plot, notch=False, vert=1, whis=1.5, showfliers=True, showmeans=True, patch_artist=True)
    plt.setp(bp['boxes'], color='black', linewidth=1)
    plt.setp(bp['whiskers'], color='black', linestyle='--', linewidth=1)
    plt.setp(bp['means'], color='black', linewidth=1)
    plt.setp(bp['medians'], color='red', linewidth=1)
    plt.setp(bp['caps'], color='black', linewidth=1)
    plt.setp(bp['fliers'], color='black', marker='+')

    boxColors = [color_train, color_test, color_test, color_test, color_test, color_test, color_test, color_test, color_test_sp, color_test_sp, color_test_sp, color_test_sp, color_test_sp, color_test_sp, color_test_sp]
    numBoxes = len(labels)

    # medians = list(range(numBoxes))
    # for i in range(numBoxes):
    #     box = bp['boxes'][i]
    #     boxX = []
    #     boxY = []
    #     for j in range(5):
    #         boxX.append(box.get_xdata()[j])
    #         boxY.append(box.get_ydata()[j])
    #     boxCoords = list(zip(boxX, boxY))
    #     # Alternate between Dark Khaki and Royal Blue
    #     k = i % numBoxes
    #     boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
    #     ax.add_patch(boxPolygon)
    # # Now draw the median lines back over what we just filled in
    # med = bp['medians'][i]
    # medianX = []
    # medianY = []
    # for j in range(2):
    #     medianX.append(med.get_xdata()[j])
    #     medianY.append(med.get_ydata()[j])
    #     plt.plot(medianX, medianY, 'k')
    # medians[i] = medianY[0]
    # # Finally, overplot the sample averages, with horizontal alignment
    # # in the center of each box
    # plt.plot([np.average(med.get_xdata())], [np.average(data_to_plot[i])],
    #          color='w',
    #          marker='o',
    #          markeredgecolor='k')

    # Set the axes ranges and axes labels
    # ax.set_xlim(0.5, numBoxes + 0.5)
    ax.set_xlim(0.5, numBoxes + 0.5)
    # ax.set_ylim(-0.1, None)
    ax.set_xticklabels([labels[0], labels[1], labels[2], labels[3], labels[4], labels[5], labels[6], labels[7], labels[8], labels[9], labels[10], labels[11], labels[12], labels[13], labels[14]], rotation=90)

    # pos = np.arange(numBoxes) + 1
    # upperLabels = [str(np.round(s, 2)) for s in medians]
    # weights = ['bold', 'semibold']

    # plt.figtext(0.92, 0.93, 'Median:', size='small', color='black', weight='roman')
    # plt.figtext(0.93, 0.915, upperLabels[0], size='small', backgroundcolor=boxColors[0], weight=weights[0], color='black') # 23 - train
    # plt.figtext(0.93, 0.865, upperLabels[1], size='small', backgroundcolor=boxColors[1], weight=weights[0], color='black') # 20
    # plt.figtext(0.93, 0.815, upperLabels[2], size='small', backgroundcolor=boxColors[2], weight=weights[0], color='black') # 21
    # plt.figtext(0.93, 0.765, upperLabels[3], size='small', backgroundcolor=boxColors[3], weight=weights[0], color='black') # 22
    # plt.figtext(0.93, 0.715, upperLabels[4], size='small', backgroundcolor=boxColors[4], weight=weights[0], color='black') # 23
    # plt.figtext(0.93, 0.665, upperLabels[5], size='small', backgroundcolor=boxColors[5], weight=weights[0], color='black') # 24
    # plt.figtext(0.93, 0.615, upperLabels[6], size='small', backgroundcolor=boxColors[6], weight=weights[0], color='black') # 25
    # plt.figtext(0.93, 0.565, upperLabels[7], size='small', backgroundcolor=boxColors[7], weight=weights[0], color='black') # 26
    # plt.figtext(0.93, 0.515, upperLabels[8], size='small', backgroundcolor=boxColors[8], weight=weights[0], color='black') # 20
    # plt.figtext(0.93, 0.465, upperLabels[9], size='small', backgroundcolor=boxColors[9], weight=weights[0], color='black') # 21
    # plt.figtext(0.93, 0.415, upperLabels[10], size='small', backgroundcolor=boxColors[10], weight=weights[0], color='black') # 22
    # plt.figtext(0.93, 0.365, upperLabels[11], size='small', backgroundcolor=boxColors[11], weight=weights[0], color='black') # 23
    # plt.figtext(0.93, 0.315, upperLabels[12], size='small', backgroundcolor=boxColors[12], weight=weights[0], color='black') # 24
    # plt.figtext(0.93, 0.265, upperLabels[13], size='small', backgroundcolor=boxColors[13], weight=weights[0], color='black') # 25
    # plt.figtext(0.93, 0.22, upperLabels[14], size='small', backgroundcolor=boxColors[14], weight=weights[0], color='black') # 26
    # plt.figtext(0.92, 0.18, 'o', color='black', weight='roman', size='medium')
    # plt.figtext(0.94, 0.18, 'Mean', color='black', weight='roman', size='small')

    if type == 'rotation':
        # ax.set_ylim(-0.1, 30)
        plt.gca().yaxis.set_minor_formatter(NullFormatter())
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_func_y_degree2))

    ax.set_ylabel(label_y)
    ax.set_xlabel(' ')
    ax.set_title(figtitle)
    # plt.savefig(os.path.join(figure_path, figname))
    plt.savefig(os.path.join(figure_path, figname), format='png')
    plt.close(fig)

# -----------------------------------------------------------------------------

def visualize_errors_boxplot_weekly_old(error_train, error_weekly_list ,
                                      figname, figtitle, label_y, figure_path):

    data_to_plot = [error_train] + error_weekly_list
    labels = ['23w', '20w', '21w', '22w', '23w', '24w', '25w', '26w']
    fig, ax = plt.subplots()
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_axisbelow(True)

    bp = plt.boxplot(data_to_plot, notch=False, vert=1, whis=1.5, showfliers=True)
    plt.setp(bp['boxes'], color='black', linewidth=1)
    plt.setp(bp['whiskers'], color='black', linestyle='--', linewidth=1)
    plt.setp(bp['means'], color='black', linewidth=1)
    plt.setp(bp['medians'], color='red', linewidth=1)
    plt.setp(bp['caps'], color='black', linewidth=1)
    plt.setp(bp['fliers'], color='black', marker='+')

    boxColors = [color_train, color_test, color_test, color_test, color_test, color_test, color_test, color_test]
    numBoxes = len(labels)
    medians = list(range(numBoxes))
    for i in range(numBoxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        # Alternate between Dark Khaki and Royal Blue
        k = i % numBoxes
        boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
        ax.add_patch(boxPolygon)
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            plt.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        plt.plot([np.average(med.get_xdata())], [np.average(data_to_plot[i])], color='w', marker='o', markeredgecolor='k')

    # ax.set_xlim(0.5, numBoxes + 0.5)
    ax.set_xlim(0.5, numBoxes + 1.0)
    # ax.set_ylim(-0.5, None)
    ax.set_xticklabels([labels[0], labels[1], labels[2], labels[3], labels[4], labels[5], labels[6], labels[7]])

    pos = np.arange(numBoxes) + 1
    upperLabels = [str(np.round(s, 2)) for s in medians]
    weights = ['bold', 'semibold']

    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_func_y_degree2))

    # plt.figtext(0.87,
    #             0.45,
    #             'Median:',
    #             color='black',
    #             weight='roman',
    #             size='medium')
    # plt.figtext(0.88,
    #             0.40,
    #             upperLabels[0],
    #             size='medium',
    #             backgroundcolor=boxColors[0],
    #             weight=weights[0],
    #             color='black')
    # plt.figtext(0.88,
    #             0.33,
    #             upperLabels[1],
    #             size='medium',
    #             backgroundcolor=boxColors[1],
    #             weight=weights[0],
    #             color='black')
    # plt.figtext(0.88,
    #             0.26,
    #             upperLabels[2],
    #             size='medium',
    #             backgroundcolor=boxColors[1],
    #             weight=weights[0],
    #             color='black')
    # plt.figtext(0.87, 0.17, 'o', color='black', weight='roman', size='medium')
    # plt.figtext(0.89,
    #             0.175,
    #             'Average',
    #             color='black',
    #             weight='roman',
    #             size='medium')
    # plt.figtext(0.89,
    #             0.14,
    #             'Value',
    #             color='black',
    #             weight='roman',
    #             size='medium')

    ax.set_ylabel(label_y)
    ax.set_xlabel(' ')
    ax.set_title(figtitle)
    # plt.savefig(os.path.join(figure_path, figname))
    plt.savefig(os.path.join(figure_path, figname), format='png')
    plt.close(fig)

# -----------------------------------------------------------------------------

def visualize_errors_boxplot_weekly_2(error_train, error_weekly_list_rp, error_weekly_list_sp, figname, figtitle, label_y, figure_path, type):

    data_to_plot = [error_train] + error_weekly_list_rp + error_weekly_list_sp
    labels = ['23w', '23w', '39w', '39w_rp', '23w', '30w' '30w_sp']
    fig, ax = plt.subplots()
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_axisbelow(True)
    # custom_fliers = dict(markerfacecolor='#A2C4C9', marker='o')
    # custom_fliers = dict(markerfacecolor='red', color='k', marker='o')
    # custom_fliers = dict(markerfacecolor='black', color='red', marker='o')

    bp = plt.boxplot(
        data_to_plot,
        notch=False,
        # flierprops=custom_fliers,
        vert=1,
        whis=1.5,
        showfliers=True)
    plt.setp(bp['boxes'], color='black', linewidth=1)
    plt.setp(bp['whiskers'], color='black', linestyle='--', linewidth=1)
    plt.setp(bp['means'], color='black', linewidth=1)
    plt.setp(bp['medians'], color='red', linewidth=1)
    plt.setp(bp['caps'], color='black', linewidth=1)
    plt.setp(bp['fliers'], color='black', marker='+')

    boxColors = [color_train, color_test, color_test, color_test, color_test_sp, color_test_sp, color_test_sp]
    numBoxes = len(labels)
    medians = list(range(numBoxes))
    for i in range(numBoxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        # Alternate between Dark Khaki and Royal Blue
        k = i % numBoxes
        boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
        ax.add_patch(boxPolygon)
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            plt.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        plt.plot([np.average(med.get_xdata())], [np.average(data_to_plot[i])], color='w', marker='o', markeredgecolor='k')

    # Set the axes ranges and axes labels
    # ax.set_xlim(0.5, numBoxes + 0.5)
    ax.set_xlim(0.5, numBoxes + 1.0)
    # ax.set_ylim(-0.5, None)
    ax.set_xticklabels([labels[0], labels[1], labels[2], labels[3], labels[4], labels[5], labels[6]], rotation=90)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(numBoxes) + 1
    upperLabels = [str(np.round(s, 2)) for s in medians]
    weights = ['bold', 'semibold']
    # for tick, label in zip(range(numBoxes), ax.get_xticklabels()):
    #     k = tick % 2
    #     ax.text(pos[tick], -20.00, upperLabels[tick], horizontalalignment='center', size='x-small', backgroundcolor=boxColors[k], weight=weights[k], color='black')

    if type == 'rotation':
        plt.gca().yaxis.set_minor_formatter(NullFormatter())
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_func_y_degree2))

    plt.figtext(0.93, 0.57, 'Median:', size='small', color='black', weight='roman')
    plt.figtext(0.93, 0.52, upperLabels[0], size='small', backgroundcolor=boxColors[0], weight=weights[0], color='black') # 23 - train
    plt.figtext(0.93, 0.47, upperLabels[1], size='small', backgroundcolor=boxColors[1], weight=weights[0], color='black') # 23 - rp
    plt.figtext(0.93, 0.42, upperLabels[4], size='small', backgroundcolor=boxColors[4], weight=weights[0], color='black') # 23 - sp
    plt.figtext(0.93, 0.37, upperLabels[5], size='small', backgroundcolor=boxColors[5], weight=weights[0], color='black') # 39 - rp
    plt.figtext(0.93, 0.32, upperLabels[6], size='small', backgroundcolor=boxColors[6], weight=weights[0], color='black') # 39 - sp
    plt.figtext(0.93, 0.27, upperLabels[7], size='small', backgroundcolor=boxColors[7], weight=weights[0], color='black') # 39 - rp - 6 dof
    plt.figtext(0.93, 0.22, upperLabels[8], size='small', backgroundcolor=boxColors[8], weight=weights[0], color='black') # 30 - sp - 6 dof
    plt.figtext(0.92, 0.18, 'o', color='black', weight='roman', size='medium')
    plt.figtext(0.94, 0.18, 'Mean', color='black', weight='roman', size='small')

    ax.set_ylabel(label_y)
    ax.set_xlabel(' ')
    ax.set_title(figtitle)
    # plt.savefig(os.path.join(figure_path, figname))
    plt.savefig(os.path.join(figure_path, figname), format='png')
    plt.close(fig)
