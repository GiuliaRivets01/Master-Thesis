import pandas as pd
from src.config import load_config
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

def load_dataset_splits(config):
    train = pd.read_csv(config['dataset']['output_path_train'])
    val = pd.read_csv(config['dataset']['output_path_val'])
    test = pd.read_csv(config['dataset']['output_path_test'])

    merged = pd.concat([train, val, test], axis=0, ignore_index=True)
    return merged

def plot_labels(datasets, save_path):

    label_names = {0: 'non-hateful', 1: 'hateful'}
    label_data = []

    for name, df in datasets.items():
        label_counts = df['labels'].value_counts(normalize=True).to_dict()  # normalize=True gives proportions
        for label, proportion in label_counts.items():
            label_data.append({
                'Dataset': name,
                'Label': label_names.get(label, label),
                'Proportion': proportion
            })

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

def text_length_analysis(text_col, datasets, save_path):
    # Collect text lengths per dataset
    length_data = []
    i = 0
    for name, df in datasets.items():
        df['text_length'] = df[text_col[i]].astype(str).apply(lambda x: len(x.split()))  # or len(x) for characters
        for length in df['text_length']:
            length_data.append({'Dataset': name, 'Text Length': length})
        i+=1

    # Create a DataFrame for plotting
    length_df = pd.DataFrame(length_data)

    # Plot using seaborn boxplot (good for comparing distributions)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Dataset', y='Text Length', data=length_df)
    plt.yscale('log')
    plt.title('Text Length Distribution Across Datasets (Log Scale)')
    plt.ylabel('Number of Words (Log Scale)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Save and show
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # dpi for quality, bbox for layout
    plt.show()

    summary = length_df.groupby("Dataset")["Text Length"].describe()[["count", "mean", "50%", "std", "min", "max"]]
    summary.columns = ["Count", "Mean", "Median", "Std", "Min", "Max"]
    print(summary)



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


datasets = {
    'HS-bg': df_bg,
    'Dutch HateCheck': df_nl,
    'COLD': df_ch,
    'HS-it': df_it,
    'South Park': df_ru
}

# Define the directory and filename
save_dir = "outputs/plots"
os.makedirs(save_dir, exist_ok=True)  # create the directory if it doesn't exist
filename_1 = "HSD_label_proportions_across_datasets.png"
save_path_1 = os.path.join(save_dir, filename_1)
filename_2 = "HSD_length_distribution.png"
save_path_2 = os.path.join(save_dir, filename_2)

plot_labels(datasets, save_path_1)

text_col = ["translated_text", "translated_test_case", "translated_TEXT", "translated_full_text", "translated_text"]
text_length_analysis(text_col, datasets, save_path_2)
