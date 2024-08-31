import os
import warnings

# Suppressing unnecessary warnings to make the output cleaner
warnings.filterwarnings("ignore")

# Setting the GPU device to be used
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Setting the working directory to the specific path for debbuging purposes
# Comment out if not needed
os.chdir('/home/chiara/workspace/proximity-to-sp-us-videos')

# Importing necessary libraries
import time
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
from moviepy.editor import VideoFileClip
from scipy.spatial.transform import Rotation as R

# Importing functions and classes from other files in the project
from main import inference_frames
from utils_videos import (ClassSegmentationModel, 
                          SegmentationModel, 
                          SegmentationDatasetVideoFrames,
                          SegmentationRegressionDataset, 
                          ResNet, 
                          FetalDatasetFrames,
                          compute_geodesic_distance_from_two_matrices)
from utils import (device, 
                   data_transforms, 
                   prepare_labels, 
                   save_results, 
                   save_errors_frames, 
                   save_files_frames, 
                   compute_metrics, 
                   compute_err_mm, 
                   prepare_labels_test)


# Setup directories for input videos and outputs.
video_folder = r'data/videos/'
output_folder = r'output/frames_posereg/masks/'
os.makedirs(output_folder, exist_ok=True)

def process_videos(video_file, video_folder, model_name_class, model_name_seg, model_name_reg, exp):

    video_path = os.path.join(video_folder, video_file)
    
    video_clip = VideoFileClip(video_path)

    for idx, frame in enumerate(video_clip.iter_frames(fps=10, dtype="uint8")):

        dataset = SegmentationDatasetVideoFrames(frames=[frame])

        model_class = ClassSegmentationModel().to(device)
        model_class.load_state_dict(torch.load(model_name_class))
        model_class.eval()

        with torch.no_grad():

            for idx in range(len(dataset)):

                image = dataset[idx]
                _, _, _, class_probs, _ = model_class(image.to(device).unsqueeze(0), None, None, None, None) # (C, H, W) -> (1, C, H, W)
                binary_predictions = (class_probs > 0.5).int().item()

                print('Starting classification...')

                if binary_predictions == 1:
                   
                    model_seg = SegmentationModel().to(device)
                    model_seg.load_state_dict(torch.load(model_name_seg))
                    model_seg.eval()

                    model_reg = ResNet().to(device)
                    model_reg.load_state_dict(torch.load(model_name_reg))
                    model_reg.eval()

                    dataset = SegmentationRegressionDataset(frames=[image])

                    margin_size = 30

                    data_loader = DataLoader(image[idx], batch_size=1, shuffle=False, num_workers=4)

                    key_values = case_dictionary[exp]
                    fold = key_values[0]
                    folder_name = key_values[1]
                    sheetname = key_values[2]

                    _, _, scaler = prepare_labels(
                    rf'data/csv/unity/23w_train.csv',
                    rf'data/csv/23w_train.csv',
                    rf'data/csv/unity/23w.csv',
                    rf'data/csv/23w.csv')

                    # Construct the output path
                    out_path = os.path.join(output_folder, folder_name, fold)
                    # Ensure the output directory exists
                    os.makedirs(out_path, exist_ok=True)

                    # Read pose data
                    pose_file = 'data/csv/sp_coords_maela.xlsx'
                    xls = pd.ExcelFile(pose_file)
                    df_poses = pd.read_excel(xls, sheet_name=sheetname)
                    
                    pose_row = df_poses.loc[df_poses['model'] == folder_name]

                    # Check if the pose_row is empty, which means the model_name was not found
                    if pose_row.empty:
                        print(f"No pose data found for model {folder_name}.")
                        return  # Exit the function as we cannot proceed without the pose data

                    mask_type = 'masks'

                    df = pd.DataFrame()

                    pos_xs, pos_ys, pos_zs, rot_xs, rot_ys, rot_zs = [], [], [], [], [], []

                    print('Starting segmentation and masked frame generation...')

                    with torch.no_grad():
                       
                        for i, image in enumerate(data_loader):
                                
                            image = image.to(device)
                            logits_masks, _, _ = model_seg(image)

                            pred_mask = (torch.sigmoid(logits_masks[i]) > 0.5).float()
                            original_image = (image[i].cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
                            mask = cv2.resize(pred_mask.cpu().numpy().squeeze(), (original_image.shape[1], original_image.shape[0]))
                            kernel = np.ones((margin_size, margin_size), np.uint8)
                            mask_dilated = cv2.dilate(mask, kernel, iterations=1)
                            masked_img = np.zeros_like(original_image)
                            masked_img[mask_dilated > 0] = original_image[mask_dilated > 0]

                            print('Masked image generated...')

                            pos_xs.append(pose_row['pos_x_tv'].values[0])
                            pos_ys.append(pose_row['pos_y_tv'].values[0])
                            pos_zs.append(pose_row['pos_z_tv'].values[0])
                            rot_xs.append(pose_row['rot_x_tv'].values[0])
                            rot_ys.append(pose_row['rot_y_tv'].values[0])
                            rot_zs.append(pose_row['rot_z_tv'].values[0])

                            # After collecting all values, assign them to the DataFrame
                            df['pos_x'] = pos_xs
                            df['pos_y'] = pos_ys
                            df['pos_z'] = pos_zs
                            df['rot_x'] = rot_xs
                            df['rot_y'] = rot_ys
                            df['rot_z'] = rot_zs

                        # Save the DataFrame to CSV
                        csv_output_path = rf'data/csv/{folder_name}/{fold}_{mask_type}.csv'
                        os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
                        df.to_csv(csv_output_path, index=False)

                        labels = prepare_labels_test(rf'data/csv/{folder_name}/{fold}_{mask_type}.csv', rf'data/csv/{folder_name}/{fold}_{mask_type}.csv', scaler)

                        dataset = FetalDatasetFrames(labels, masked_img, data_transforms['real_test'])
                        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=4)
                        
                        image_names_list = []
                        evaluation_res = open(os.path.join(out_path, 'res.txt'), 'w')
    
                        for batch in data_loader:
                            image_names, labels = batch
                            image_names_list.extend(image_names)
                                
                        labels_transl_list_eval, out_transl_list_eval, labels_rot_list_eval, out_angles_list_eval, transl_errors_list_eval, geodesic_errors_list_eval, rot_norm_errors_list_eval = compute_errors_frames(data_loader, model_reg, scaler, 'real')

                    save_results(transl_errors_list_eval, geodesic_errors_list_eval, evaluation_res)
                    save_files_frames(masked_img, labels_transl_list_eval, out_transl_list_eval, scaler, labels_rot_list_eval, out_angles_list_eval, 'gt.csv', 'pred.csv', out_path)
                    save_errors_frames(masked_img, transl_errors_list_eval, rot_norm_errors_list_eval, 'transl_err.csv', 'rot_err.csv', out_path)

def compute_errors_frames(data_loader, model, scaler, data_type):
    
    labels_transl_list, labels_rot_list = [], []
    out_transl_list, out_angles_list = [], []
    geodesic_errors_list = np.array([])
    transl_errors_list = np.array([])
    rot_norm_errors_list = np.array([])
    
    model.eval()
    
    with torch.no_grad():

        for image_batch, label_batch in data_loader:

            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            label_batch_transl = label_batch[:, :3]
            label_batch_rot = label_batch[:, -3:]
            
            label_batch_rotmat = R.from_euler('xyz', label_batch_rot.detach().cpu())
            label_batch_rotmat = label_batch_rotmat.as_matrix()
            
            output_batch_transl, output_batch_rotmat = model(image_batch)

            out_angles = R.from_matrix(output_batch_rotmat.detach().cpu())
            out_angles = out_angles.as_euler('xyz')

            labels_transl_list.append(label_batch_transl.detach().cpu().numpy())
            out_transl_list.append(output_batch_transl.detach().cpu().numpy())
            labels_rot_list.append(label_batch_rot.detach().cpu().numpy())
            out_angles_list.append(out_angles)

            _, _, err_mm = compute_metrics(labels_list=label_batch_transl.detach().cpu().numpy(), predictions_list=output_batch_transl.detach().cpu().numpy(), scaler=scaler, data_type = data_type)

            geodesic_errors = np.array(compute_geodesic_distance_from_two_matrices(output_batch_rotmat.float(), torch.Tensor(label_batch_rotmat).to(device)).data.tolist())
            geodesic_errors = geodesic_errors * 180 / np.pi 

            # - First Column: Represents the new x-axis in terms of the original coordinate system. 
            # That is, it shows where the unit vector along the x-axis of the rotated system will 
            # end up in the original coordinate system.
            # - Second Column: Represents the new y-axis in the same way, showing where the unit 
            # vector along the y-axis of the rotated system points in the original system.
            # Third Column: Represents the new z-axis. 
            # The z-axis in many conventions (especially in right-handed coordinate systems) is perpendicular 
            # to the x-y plane of the coordinate system. Thus, the third column of a rotation matrix can be seen 
            # as the normal vector to the plane formed by the new x and y axes after rotation.
            normal_vector_output = output_batch_rotmat[:, :, 2].detach().cpu().numpy()
            normal_vector_label = label_batch_rotmat[:, :, 2]
            # Compute the cosine of the angle between normal vectors
            dot_product = np.sum(normal_vector_output * normal_vector_label, axis=1)
            norms_output = np.linalg.norm(normal_vector_output, axis=1)
            norms_label = np.linalg.norm(normal_vector_label, axis=1)
            cos_angle = dot_product / (norms_output * norms_label)
            rot_norm_errors = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
            
            transl_errors_list = np.append(transl_errors_list, err_mm)
            geodesic_errors_list = np.append(geodesic_errors_list, geodesic_errors)
            rot_norm_errors_list = np.append(rot_norm_errors_list, rot_norm_errors)

    return labels_transl_list, out_transl_list, labels_rot_list, out_angles_list, transl_errors_list, geodesic_errors_list, rot_norm_errors_list


#---------------------------------------------------------------------

# Case dictionary
case_dictionary = {
    '1': ['Operator_1_Novice', '23w_train_out', 'Operator1'],
}

#---------------------------------------------------------------------

# Loop to process only specific videos defined in case_dictionary, rpocess them and runs inference
for exp, op in case_dictionary.items():
    video_filename = f'{op[0]}.mp4' 
    print(f'Processing video: {video_filename}')
    video_file_path = os.path.join(video_folder, video_filename)
    classification_model = f'models/miccai_loocv_ss_class_{op[1]}.pt'
    segmentation_model = f'models/miccai_loocv_ss_{op[1]}.pt'
    posereg_model = f'models/miccai_loocv_ss_posereg_{op[1]}_new.pt'

    if os.path.exists(video_file_path):
        process_videos(video_filename, video_folder, classification_model, segmentation_model, posereg_model, exp)
    else:
        print(f"Video file {video_filename} not found in directory.")
        
# Notify when processing is complete for all specified setups.
print("Completed")