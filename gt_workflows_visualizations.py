import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np
import csv

gt_folder = r'output/extracted_gt/'
pred_folder = r'output/sononet/'
comb_folder = r'output/combined_gt_pred/'
gt_clean_folder = r'output/cleaned_gt/'
# Case dictionary
case_dictionary = {
    '0': ['Operator_17_Expert', '23w_train_out', 'Operator17'],
    '1': ['Operator_16_Expert', '23w_train_out', 'Operator16'],
    '2': ['Operator_15_Novice', '23w_train_out', 'Operator15'],
    '3': ['Operator_14_Expert', '23w_train_out', 'Operator14'],
    '4': ['Operator_13_Novice', '23w_train_out', 'Operator13'],
    '5': ['Operator_12_Novice', '23w_train_out', 'Operator12'],
    '6': ['Operator_11_Novice', '23w_train_out', 'Operator11'],
    '7': ['Operator_10_Novice', '23w_train_out', 'Operator10'],
    '8': ['Operator_9_Novice', '23w_train_out', 'Operator9'],
    '9': ['Operator_8_Novice', '23w_train_out', 'Operator8'],
    '10': ['Operator_7_Novice', '23w_train_out', 'Operator7'],
    '11': ['Operator_6_Novice', '23w_train_out', 'Operator6'],
    '12': ['Operator_5_Novice', '23w_train_out', 'Operator5'],
    '13': ['Operator_4_Novice', '23w_train_out', 'Operator4'],
    '14': ['Operator_3_Novice', '23w_train_out', 'Operator3'],
    '15': ['Operator_2_Novice', '23w_train_out', 'Operator2'],
    '16': ['Operator_1_Novice', '23w_train_out', 'Operator1']
}


def plot_workflow(df, labels_df):
    
    # Extract unique labels and assign a color to each
    labels = labels_df['Category'].unique()
    np.random.seed(0)

    # Define light grey color for "Background"
    background_color = 'black'

    # Create a colormap for other labels
    cmap = plt.get_cmap('tab20', len(labels) + 1)
    label_colors = {}
    for i, label in enumerate(labels):
        label_colors[label] = cmap(i)
        
    print(label_colors)
    label_colors['Background'] = background_color

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(15, 6))

    # Get unique sources and assign y positions with smaller gaps
    sources = combined_df['Source'].unique()
    max_length = 0
    source_positions = {}
    for i, source in enumerate(sources):
        source_df = combined_df[combined_df['Source'] == source].reset_index(drop=True)
        source_positions[source] = i * 0.5
        max_length = max(max_length, len(source_df))

        for idx, row in source_df.iterrows():
            true_label = row['True Label']
 
            category = category = labels_df.loc[labels_df['Item'] == true_label, 'Category']
            if not category.empty:
                if 'Other' in category.iloc[0]:
                    category.iloc[0] = 'Background'
                    true_color = label_colors['Background']
                else:
                    true_color = label_colors[category.iloc[0]]                
            else:
                true_color = label_colors['Background']

            ax.add_patch(mpatches.Rectangle((idx, source_positions[source] - 0.25), 1, 0.5, color=true_color))

    # Set the limits and labels
    ax.set_xlim(0, max_length)
    ax.set_ylim(-0.25, max(source_positions.values()) + 0.25)
    ax.set_yticks(list(source_positions.values()))
    ax.set_yticklabels(sources)
    ax.set_xlabel('Frame')
    ax.set_title('Workflows Over Time with Annotated Labels and New Classes')



    # Create a legend with adjusted layout
    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in label_colors.items()]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Ensure legend fits within the figure
    plt.tight_layout()

    # Show the plot
    plt.show()


combined_df = pd.DataFrame()

# Loop to process only specific videos defined in case_dictionary
for exp, op in case_dictionary.items():
    gt_filename = f'ground_truth_{op[0]}.csv'
    print(f'Processing: {op[0]}')
    gt_file_path = os.path.join(gt_clean_folder, gt_filename)

    gt_df = pd.read_csv(gt_file_path)

    gt_df['Source'] = op[0]

    combined_df = pd.concat([combined_df, gt_df], ignore_index=True)

print(combined_df)

#labels_df = pd.read_csv('sono_label_categories.csv')
labels_df = pd.read_csv('label_categories_15.csv')
print('All new labels: ')
for label in combined_df['True Label'].unique():
    if label not in list(labels_df['Item']):
        print("|",label,'|\n')

# Plot the combined data
plot_workflow(combined_df, labels_df)





















