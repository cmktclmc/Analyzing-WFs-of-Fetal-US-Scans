import random

case_dictionary = {
    '0': ['Operator_17_Expert', '23w_train_out', 'Operator17', 420, 0, 1313, 1080],
    '1': ['Operator_16_Expert', '23w_train_out', 'Operator16', 320, 50, 1036, 777],
    '2': ['Operator_15_Novice', '23w_train_out', 'Operator15', 320, 50, 1036, 777],
    '3': ['Operator_14_Expert', '23w_train_out', 'Operator14', 320, 50, 1036, 777],
    '4': ['Operator_13_Novice', '23w_train_out', 'Operator13', 320, 50, 1036, 777],
    '5': ['Operator_12_Novice', '23w_train_out', 'Operator12', 320, 50, 1036, 777],
    '6': ['Operator_11_Novice', '23w_train_out', 'Operator11', 320, 50, 1036, 777],
    '7': ['Operator_10_Novice', '23w_train_out', 'Operator10', 320, 50, 1036, 777],
    '8': ['Operator_9_Novice', '23w_train_out', 'Operator9', 320, 50, 1036, 777],
    '9': ['Operator_8_Novice', '23w_train_out', 'Operator8', 320, 50, 1036, 777],
    '10': ['Operator_7_Novice', '23w_train_out', 'Operator7', 320, 50, 1036, 777],
    '11': ['Operator_6_Novice', '23w_train_out', 'Operator6', 320, 50, 1036, 777],
    '12': ['Operator_5_Novice', '23w_train_out', 'Operator5', 205, 0, 545, 374],
    '13': ['Operator_4_Novice', '23w_train_out', 'Operator4', 320, 50, 1036, 777],
    '14': ['Operator_3_Novice', '23w_train_out', 'Operator3', 320, 50, 1036, 777],
    '15': ['Operator_2_Novice', '23w_train_out', 'Operator2', 180, 0, 576, 408],
    '16':['Operator_1_Novice', '23w_train_out', 'Operator1', 320, 50, 1036, 777]
    
}

# Separate novices and experts
novices = {k: v for k, v in case_dictionary.items() if 'Novice' in v[0]}
experts = {k: v for k, v in case_dictionary.items() if 'Expert' in v[0]}

# Set random seed for reproducibility
random.seed(42)

# Split novices into train and test (80%-20% split)
novice_keys = list(novices.keys())
random.shuffle(novice_keys)

split_index_novice = int(0.8 * len(novice_keys))
train_novices = {k: novices[k] for k in novice_keys[:split_index_novice]}
test_novices = {k: novices[k] for k in novice_keys[split_index_novice:]}

# Split experts into train and test (80%-20% split)
expert_keys = list(experts.keys())
random.shuffle(expert_keys)

split_index_expert = int(0.8 * len(expert_keys))
train_experts = {k: experts[k] for k in expert_keys[:split_index_expert]}
test_experts = {k: experts[k] for k in expert_keys[split_index_expert:]}

# Combine train and test sets
train_case_dictionary = {**train_novices, **train_experts}
test_case_dictionary = {**test_novices, **test_experts}

print("Train Set:", train_case_dictionary)
print("Test Set:", test_case_dictionary)