import os
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# Define directories
data_directory = '/Users/Caitlin/Documents/GitHub/proximity-to-sp-us-videos/output/cleaned_frame_collect_labels'
category_file = '/Users/Caitlin/Documents/GitHub/proximity-to-sp-us-videos/label_categories.csv'

label_counts = Counter()
category_file_appearance = defaultdict(set)

category_df = pd.read_csv(category_file)
item_to_category = dict(zip(category_df['Item'], category_df['Category']))

file_lengths = {}
files_missing_categories = defaultdict(list)

required_categories = {"Brain with Skull Head and Neck", "Nose and Lips", "Abdomen", "Spine", "Femur"}

# Iterate over all files in the data directory
for filename in os.listdir(data_directory):
    if filename.endswith('.csv'):
        # Read the CSV file
        filepath = os.path.join(data_directory, filename)
        df = pd.read_csv(filepath)
        
        file_lengths[filename] = len(df)

        label_counts.update(df['New Label'])

        seen_categories = set()
        for label in df['New Label']:
            category = item_to_category.get(label, 'Unknown')
            seen_categories.add(category)

        missing_categories = required_categories - seen_categories
        if missing_categories:
            files_missing_categories[filename] = missing_categories
        
        for category in seen_categories:
            category_file_appearance[category].add(filename)

print("\nFiles Missing Required Categories:")
for filename, missing in files_missing_categories.items():
    missing_str = ', '.join(missing)
    print(f'{filename}: Missing categories -> {missing_str}')

print("\nLength of Each File:")
for filename, length in file_lengths.items():
    print(f'{filename}: {length} rows')

category_counts = defaultdict(int)
category_file_count = {}

for label, count in label_counts.items():
    category = item_to_category.get(label, 'Background')
    category_counts[category] += count

category_file_count = {category: len(files) for category, files in category_file_appearance.items()}

sorted_category_counts = sorted(category_counts.items(), key=lambda item: item[1], reverse=True)

categories, counts = zip(*sorted_category_counts)

total = sum(counts)

percentages = [(count / total) * 100 for count in counts]

print("\nTop Categories:")
for category, count, percentage in zip(categories, counts, percentages):
    files_count = category_file_count.get(category, 0)
    print(f'{category}: Count = {count}, Percentage = {percentage:.2f}%, Appears in {files_count} file(s)')

print(f"\nTotal Count: {total}")

plt.figure(figsize=(12, 8))
bars = plt.bar(categories, percentages, color='skyblue')
plt.ylim(0, max(percentages) * 1.2)

for bar, percentage in zip(bars, percentages):
    yval = bar.get_height() +0.5
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{percentage:.2f}%', ha='center', va='bottom', rotation=90)

plt.xlabel('Class')
plt.ylabel('Percentage (%)')
plt.title('Percentage of Occurrences per Class')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Show the plot
plt.show()
