import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from scipy.stats import mode

#gt_folder = r'output/cleaned_gt/'
gt_folder = r'output/cleaned_frame_collect_labels/'
pred_folder = r'output/sononet_block_final/'
labels_df = pd.read_csv('/Users/Caitlin/Documents/GitHub/proximity-to-sp-us-videos/label_categories_15.csv')

case_dictionary = {'2': ['Operator_15_Novice', '23w_train_out', 'Operator15', 320, 50, 1036, 777, 'Breech'], 
                   '4': ['Operator_13_Novice', '23w_train_out', 'Operator13', 320, 50, 1036, 777, 'Breech'], 
                   '16': ['Operator_1_Novice', '23w_train_out', 'Operator1', 320, 50, 1036, 777, 'Cephalic'], 
                   '1': ['Operator_16_Expert', '23w_train_out', 'Operator16', 320, 50, 1036, 777, 'Cephalic']
                  }

def plot_workflow(df, ax, title, label_colors, show_legend=False):
    # Create a colored block for each row in the dataframe
    for idx, row in df.iterrows():
        true_label = row['New Label']
        pred_label = row['Predicted Label']
        
        true_color = label_colors[true_label]
        pred_color = label_colors[pred_label]

        ax.add_patch(mpatches.Rectangle((idx, 0), 1, 0.5, color=true_color))
        ax.add_patch(mpatches.Rectangle((idx, 0.5), 1, 0.5, color=pred_color))

    # Set the limits and labels
    ax.set_xlim(0, len(df))
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(['Ground Truth', 'Sononet Prediction'])

    # Remove the default title
    ax.set_title('')  # Clear any existing title

    # Place the title manually with rotation
    ax.text(
        -0.2,  # X position of the title (adjust as needed)
        0.5,   # Y position (middle of the plot)
        title,  # Title text
        fontsize=10,
        rotation=90,  # Rotate title 90 degrees
        verticalalignment='center',  # Centered vertically
        horizontalalignment='center',  # Centered horizontally
        transform=ax.transAxes  # Use axis coordinates
    )

    # Optionally create a legend
    if show_legend:
        legend_patches = [
            mpatches.Patch(color=color, label=label) 
            for label, color in label_colors.items() 
            if label != 'Background'
        ]
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 7})

def main():
    num_cases = len(case_dictionary)
    
    # Create a single column of subplots with reduced height
    fig, axs = plt.subplots(nrows=num_cases, ncols=1, figsize=(15, 2.2 * num_cases))
    fig.suptitle('Finetuned Sononet Model Predictions Compared to New Class Expanded Ground Truth Annotations', fontsize=16)

    # If only one subplot, axs is not an array, so convert it to an array for consistency
    if num_cases == 1:
        axs = [axs]

    # Extract unique labels and assign a color to each
    labels = pd.concat([labels_df['Category'], pd.Series(['Background'])]).unique()
    np.random.seed(0)

    # Define light grey color for "Background"
    background_color = 'black'

    # Create a colormap for other labels
    cmap = plt.get_cmap('tab20', len(labels) + 1)
    label_colors = {}
    other_index = 0
    for label in labels:
        if label == 'Background':
            label_colors[label] = background_color
        else:
            label_colors[label] = cmap(other_index)
            other_index += 1
    
    max_xlim = 0
    for idx, (exp, op) in enumerate(case_dictionary.items()):
        gt_filename = f'clean_frame_collect_{op[0]}.csv'
        pred_filename = f'sononet_prediction_{op[0]}.csv'
        print(f'Processing: {op[0]}')
        gt_file_path = os.path.join(gt_folder, gt_filename)
        pred_file_path = os.path.join(pred_folder, pred_filename)

        # Load the CSV data
        gt_df = pd.read_csv(gt_file_path)
        pred_df = pd.read_csv(pred_file_path)
        replace_dict = labels_df.set_index('Item')['Category'].to_dict()

        # Replace values in gt_df['True Label'] column using the mapping dictionary
        gt_df['New Label'] = gt_df['New Label'].replace(replace_dict)
        gt_df['New Label'] = gt_df['New Label'].fillna('Background')

        pred_df = pd.read_csv(pred_file_path)

        # Define the window size for the temporal filter
        window_size = 20  # You can adjust this size

        # Apply the mode filter
        def mode_temporal_filter(df, window_size, label_name):
            labels_filtered = []
            for i in range(len(df)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(df), i + window_size // 2 + 1)
                window_labels = df[label_name][start_idx:end_idx]
                # Use pandas mode function, which works with non-numeric types
                mode_label = window_labels.mode()[0]  # [0] gets the first mode in case of ties
                labels_filtered.append(mode_label)
            return labels_filtered

        pred_df['Predicted Label'] = mode_temporal_filter(pred_df, window_size, 'Predicted Label')
        gt_df['New Label'] = mode_temporal_filter(gt_df, window_size, 'New Label')

        # Merge dataframes on their index
        combined_df = pd.merge(gt_df, pred_df, on='Index')

        # Track the maximum x-axis limit
        max_xlim = max(max_xlim, len(combined_df))

        # Plot result
        show_legend = (idx == 0)  # Show legend only for the first subplot
        plot_workflow(combined_df, axs[idx], f'{op[0]}', label_colors, show_legend=show_legend)

    # Set the same x-axis limits for all subplots
    for ax in axs:
        ax.set_xlim(0, max_xlim)
        # Remove x-ticks from all but the last subplot
        if ax != axs[-1]:
            ax.set_xticks([])
    
    axs[-1].set_xlabel('Frame')
    # Adjust layout to remove extra white space
    plt.subplots_adjust(hspace=-0.5)  # Adjusted vertical space between subplots
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
