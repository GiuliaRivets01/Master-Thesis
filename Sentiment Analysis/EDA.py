import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.config import load_config
import os

config_it = load_config("configs/base.yaml", "configs/Italian_translated.yaml")
config_nl = load_config("configs/base.yaml", "configs/Dutch_translated.yaml")
config_ru = load_config("configs/base.yaml", "configs/Russian_translated.yaml")
config_bg = load_config("configs/base.yaml", "configs/Bulgarian_translated.yaml")
config_ch = load_config("configs/base.yaml", "configs/Chinese_translated.yaml")

def load_dataset_splits(config):
    train = pd.read_csv(config['dataset']['train_path'])
    val = pd.read_csv(config['dataset']['val_path'])
    test = pd.read_csv(config['dataset']['test_path'])

    merged = pd.concat([train, val, test], axis=0, ignore_index=True)
    return merged

def add_data(df, name, label_names, label_data, label_col):
    label_counts = df[label_col].value_counts(normalize=True).to_dict()  # normalize=True gives proportions
    for label, proportion in label_counts.items():
        new_datum = {
            'Dataset': name,
            'Label': label_names.get(label, label),
            'Proportion': proportion
        }
        label_data.append(new_datum)
    return label_data

def plot_labels(df_bg, df_ch, df_nl, df_it, df_ru, save_path):

    label_names_bg = {0: "negative", 1: "neutral", 2: "positive"}
    label_names_nl = {0: "negative", 1: "positive"}
    label_names_ch = {0: "negative", 1: "positive"}
    label_names_ru = {0: "negative", 1: "neutral", 2: "positive"}
    label_names_it = {0: "positive", 1: "negative", 2: "neutral", 3: "mixed"}

    label_data = []
    label_data = add_data(df_bg, 'Cinexio', label_names_bg, label_data, config_bg['dataset']['label_col'])
    label_data = add_data(df_ch, 'Weibo Senti 100k', label_names_ch, label_data, config_ch['dataset']['label_col'])
    label_data = add_data(df_nl, 'DBRD', label_names_nl, label_data, config_nl['dataset']['label_col'])
    label_data = add_data(df_it, 'Italian Tweets Dataset', label_names_it, label_data, config_it['dataset']['label_col'])
    label_data = add_data(df_ru, 'RuReviews', label_names_ru, label_data, config_ru['dataset']['label_col'])
    
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

df_it = load_dataset_splits(config_it)
df_ru = load_dataset_splits(config_ru)
df_bg = load_dataset_splits(config_bg)
df_nl = load_dataset_splits(config_nl)
df_ch = load_dataset_splits(config_ch)

sentiment_mapping_it = {"POSITIVE": 0, "NEGATIVE": 1, "NEUTRAL": 2, "MIXED": 3}
df_it["sentiment"] = df_it["sentiment"].map(sentiment_mapping_it)

sentiment_mapping_ru = {"negative": 0, "neautral": 1, "positive": 2}
df_ru["sentiment"] = df_ru["sentiment"].map(sentiment_mapping_ru)

save_dir = "outputs/plots"
os.makedirs(save_dir, exist_ok=True)  # create the directory if it doesn't exist
filename = "SA_label_proportions_across_datasets.png"
save_path = os.path.join(save_dir, filename)

plot_labels(df_bg, df_ch, df_nl, df_it, df_ru, save_path)