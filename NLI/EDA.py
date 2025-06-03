import pandas as pd
from src.config import load_config
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

def load_dataset_splits(config):
    train = pd.read_csv(config['dataset']['train_path'])
    val = pd.read_csv(config['dataset']['val_path'])
    test = pd.read_csv(config['dataset']['test_path'])

    merged = pd.concat([train, val, test], axis=0, ignore_index=True)
    return merged

def plot_labels(datasets, save_path):
    label_data = []

    for name, df in datasets.items():
        if name == 'SICK-NL':
            #  contradiction = 0, neutral = 1, entailment = 2
            label_names = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
        else:
            # XNLI and LingNLI: 0 = entailment, 1 = neutral, 2 = contradiction
            label_names = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        label_counts = df['label'].value_counts(normalize=True).to_dict()  # normalize=True gives proportions
        for label, proportion in label_counts.items():
            new_elem = {
                'Dataset': name,
                'Label': label_names.get(label, label),
                'Proportion': proportion
            }
            label_data.append(new_elem)

    # Convert to DataFrame
    plot_df = pd.DataFrame(label_data)

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x='Dataset', y='Proportion', hue='Label', palette='Paired')
    plt.title('Label Proportions Across Datasets')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.legend(title='Label')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # dpi for quality, bbox for layout
    plt.show()

    print(f"Plot saved to: {save_path}")
    for elem in label_data:
        print("\n")
        print(elem)

config_it = load_config("configs/base.yaml", "configs/Italian_translated.yaml")
config_nl = load_config("configs/base.yaml", "configs/Dutch_translated.yaml")
config_ru = load_config("configs/base.yaml", "configs/Russian_translated.yaml")
config_bg = load_config("configs/base.yaml", "configs/Bulgarian_translated.yaml")
config_ch = load_config("configs/base.yaml", "configs/Chinese_translated.yaml")

df_it = load_dataset_splits(config_it)
df_ru = load_dataset_splits(config_ru)
df_bg = load_dataset_splits(config_bg)
df_nl = load_dataset_splits(config_nl)
df_ch = load_dataset_splits(config_ch)

#print(df_nl['label'].value_counts())


datasets = {
    'XNLI-bg': df_bg,
    'SICK-NL': df_nl,
    'XNLI-zh': df_ch,
    'LingNLi': df_it,
    'XNLI-ru': df_ru
}

# Define the directory and filename
save_dir = "outputs/plots"
os.makedirs(save_dir, exist_ok=True)  # create the directory if it doesn't exist
filename_1 = "NLI_label_proportions_across_datasets.png"
save_path_1 = os.path.join(save_dir, filename_1)

# Dutch dataset: contradiction = 0, neutral = 1, entailment = 2
# XNLI and LingNLI: 0 = entailment, 1 = neutral, 2 = contradiction


plot_labels(datasets, save_path_1)