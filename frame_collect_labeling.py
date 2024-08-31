import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


gt_folder = r'output/cleaned_gt/'
frame_col_folder = r'output/frame_collect/'
case_dictionary = {
    # '0': ['Operator_17_Expert', '23w_train_out', 'Operator17', 420, 0, 1313, 1080],
    # '1': ['Operator_16_Expert', '23w_train_out', 'Operator16', 320, 50, 1036, 777],
    # '2': ['Operator_15_Novice', '23w_train_out', 'Operator15', 320, 50, 1036, 777],
    # '3': ['Operator_14_Expert', '23w_train_out', 'Operator14', 320, 50, 1036, 777],
    # '4': ['Operator_13_Novice', '23w_train_out', 'Operator13', 320, 50, 1036, 777],
    # '5': ['Operator_12_Novice', '23w_train_out', 'Operator12', 320, 50, 1036, 777],
    # '6': ['Operator_11_Novice', '23w_train_out', 'Operator11', 320, 50, 1036, 777],
    '7': ['Operator_10_Novice', '23w_train_out', 'Operator10', 320, 50, 1036, 777],
    # '8': ['Operator_9_Novice', '23w_train_out', 'Operator9', 320, 50, 1036, 777],
    # '9': ['Operator_8_Novice', '23w_train_out', 'Operator8', 320, 50, 1036, 777],
    # '10': ['Operator_7_Novice', '23w_train_out', 'Operator7', 320, 50, 1036, 777],
    # '11': ['Operator_6_Novice', '23w_train_out', 'Operator6', 320, 50, 1036, 777],
    # '12': ['Operator_5_Novice', '23w_train_out', 'Operator5', 205, 0, 545, 374],
    # '13': ['Operator_4_Novice', '23w_train_out', 'Operator4', 320, 50, 1036, 777],
    # '14': ['Operator_3_Novice', '23w_train_out', 'Operator3', 320, 50, 1036, 777],
    # '15': ['Operator_2_Novice', '23w_train_out', 'Operator2', 180, 0, 576, 408],
    # '16': ['Operator_1_Novice', '23w_train_out', 'Operator1', 320, 50, 1036, 777]
}

for exp, op in case_dictionary.items():
    gt_filename = f'ground_truth_{op[0]}.csv'
    frame_col_filename = f'frame_collect_{op[0]}.csv'
    print(f'Processing: {op[0]}')
    gt_file_path = os.path.join(gt_folder, gt_filename)
    frame_col_path = os.path.join(frame_col_folder, frame_col_filename)

    gt_df = pd.read_csv(gt_file_path)
    fc_df = pd.read_csv(frame_col_path)
    merged_df = pd.merge(gt_df, fc_df, on='Index')
    print(merged_df)

    results = []

    for index, row in merged_df.iterrows():
        if row['Collected'] == 1:
            same_index = index
            most_recent_label = np.nan
            
            for i in range(same_index):
                if pd.notna(merged_df.loc[i, 'True Label']):
                    most_recent_label = merged_df.loc[i, 'True Label']
            
            results.append({'Index': same_index, 'Recent_True_Label': most_recent_label})

    result_df = pd.DataFrame(results)
    print(result_df)


    new_df = pd.DataFrame(columns=['Index', 'True Label', 'Same', 'New Label'])

    def get_latest_collected_before_label_change(df):
        result = pd.DataFrame(columns=df.columns)
        
        latest_collected_row = None
        last_label = None
        
        for index, row in df.iterrows():
            current_label = row['True Label']
            
            if pd.notna(current_label) and current_label != last_label:
                if latest_collected_row is not None:
                    result = pd.concat([result, pd.DataFrame([latest_collected_row])], ignore_index=True)
                
                last_label = current_label

            if row['Collected'] == 1:
                latest_collected_row = row
        
        if latest_collected_row is not None:
            result = pd.concat([result, pd.DataFrame([latest_collected_row])], ignore_index=True)
        
        return result

    change_rows_df = get_latest_collected_before_label_change(merged_df)

    def create_new_label_column(df, change_rows_df):
        new_label = []
        current_label = None
        change_indices = change_rows_df['Index'].tolist()
        change_index_set = set(change_indices)
        print(change_indices)
        for idx, row in df.iterrows():
            if idx in change_index_set:
                current_label = None
            elif pd.notna(row['True Label']):
                current_label = row['True Label']
            
            new_label.append(current_label)
        
        return new_label

    merged_df['New Label'] = create_new_label_column(merged_df, change_rows_df)

    print(merged_df)

    new_df = merged_df[['Index','Collected','True Label','New Label']]
    new_df.to_csv(f'/cs/student/projects1/aisd/2023/ckenney/proximity-to-sp-us-videos/output/cleaned_frame_collect_labels/clean_frame_collect_{op[0]}.csv', index=False)