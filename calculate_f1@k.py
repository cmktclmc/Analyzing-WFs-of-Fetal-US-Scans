import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from scipy.stats import mode

# gt_folder = r'output/cleaned_frame_collect_labels/'
# pred_folder = r'output/sononet/'
gt_folder = r'output/cleaned_frame_collect_labels'
pred_folder = r'output/sononet_block_final'

labels_df = pd.read_csv('/Users/Caitlin/Documents/GitHub/proximity-to-sp-us-videos/label_categories_15.csv')
#labels_df = pd.read_csv('/Users/Caitlin/Documents/GitHub/proximity-to-sp-us-videos/sono_label_categories.csv')
label_col_name = 'New Label'
gt_file_start = 'clean_frame_collect_'


case_dictionary = {'2': ['Operator_15_Novice', '23w_train_out', 'Operator15', 320, 50, 1036, 777, 'Breech'], 
                        '4': ['Operator_13_Novice', '23w_train_out', 'Operator13', 320, 50, 1036, 777, 'Breech'], 
                        '16': ['Operator_1_Novice', '23w_train_out', 'Operator1', 320, 50, 1036, 777, 'Cephalic'], 
                        '1': ['Operator_16_Expert', '23w_train_out', 'Operator16', 320, 50, 1036, 777, 'Cephalic']
                    }


import pandas as pd
import numpy as np

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
 
def compute_iou(segment_a, segment_b):
    """Compute the Intersection over Union (IoU) between two segments."""
    start_a, end_a = segment_a
    start_b, end_b = segment_b
    
    intersection = max(0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    
    if union == 0:
        return 0
    return intersection / union

def segment_labels(df, label_column):
    """Extract start and end indices for continuous segments of the same label."""
    segments = []
    current_label = None
    start_idx = None
    
    for idx, label in enumerate(df[label_column]):
        if label != current_label:
            if current_label is not None:
                segments.append((current_label, start_idx, idx-1))
            current_label = label
            start_idx = idx
    
    if current_label is not None:
        segments.append((current_label, start_idx, len(df) - 1))
    
    return segments

def compute_f1_at_k(df, iou_threshold):
    """Compute the F1@k metric for the given DataFrame and IoU threshold."""
    
    # Extract segments for ground truth and predictions
    ground_truth_segments = segment_labels(df, label_col_name)
    predicted_segments = segment_labels(df, 'Predicted Label')
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    used_predictions = set()
    
    # Evaluate each ground truth segment
    for gt_label, gt_start, gt_end in ground_truth_segments:
        gt_segment = (gt_start, gt_end)
        matched = False
        
        for pred_idx, (pred_label, pred_start, pred_end) in enumerate(predicted_segments):
            if pred_label == gt_label and pred_idx not in used_predictions:
                pred_segment = (pred_start, pred_end)
                iou = compute_iou(gt_segment, pred_segment) 
                
                if iou >= iou_threshold:
                    true_positives += 1
                    used_predictions.add(pred_idx)
                    matched = True
                    #print("pos ", pred_label)
                    break
        
        if not matched:
            false_negatives += 1
    
    # Any remaining predictions are false positives
    false_positives = len(predicted_segments) - len(used_predictions)
    
    # Compute precision, recall, and F1 score
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0
    
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0.0
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return f1, precision, recall


def main():
    for idx, (exp, op) in enumerate(case_dictionary.items()):
        gt_filename = f'{gt_file_start}{op[0]}.csv'
        pred_filename = f'sononet_prediction_{op[0]}.csv'
        print(f'Processing: {op[0]}')
        gt_file_path = os.path.join(gt_folder, gt_filename)
        pred_file_path = os.path.join(pred_folder, pred_filename)

        gt_df = pd.read_csv(gt_file_path)
        pred_df = pd.read_csv(pred_file_path)
        replace_dict = labels_df.set_index('Item')['Category'].to_dict()

        gt_df[label_col_name] = gt_df[label_col_name].replace(replace_dict)
        gt_df[label_col_name] = gt_df[label_col_name].fillna('Background')
        gt_df.loc[gt_df[label_col_name].str.contains('Other', case=False, na=False), label_col_name] = 'Background'


        pred_df = pd.read_csv(pred_file_path)

        window_size = 20  # You can adjust this size

        # Apply the mode filter
        def mode_temporal_filter(df, window_size, label_name):
            labels_filtered = []
            for i in range(len(df)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(df), i + window_size // 2 + 1)
                window_labels = df[label_name][start_idx:end_idx]
                mode_label = window_labels.mode()[0]
                labels_filtered.append(mode_label)
            return labels_filtered

        pred_df['Predicted Label'] = mode_temporal_filter(pred_df, window_size, 'Predicted Label')
        gt_df[label_col_name] = mode_temporal_filter(gt_df, window_size, label_col_name)

        combined_df = pd.merge(gt_df, pred_df, on='Index')

        # Calculate F1@k for k = 0.5 (or any other threshold)
        k_values = [0.01, 0.05, 0.1, 0.25, 0.5]
        f1_vals, per_vals, rec_vals = [], [], []
        for k in k_values:
            # Compute F1@k
            f1_score, precision, recall = compute_f1_at_k(combined_df, k)
            f1_score = round(f1_score, 2)
            precision = round(precision, 2)
            recall = round(recall, 2)

            f1_vals.append(f1_score)
            per_vals.append(precision)
            rec_vals.append(recall)
        
        print(f"k : {k_values}")
        print(f"F1: {f1_vals}")
        print(f"P : {per_vals}")
        print(f"R : {rec_vals}")
        print("")

if __name__ == '__main__':
    main()
