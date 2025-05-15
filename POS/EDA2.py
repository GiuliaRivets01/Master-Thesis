import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ast
from collections import Counter
from src.utils import load_config
import os

def grouped_bar_plot_pos_distribution(dataset_dict, save_path=None):
    """
    dataset_dict: dict with keys as dataset names and values as lists of strings (JSON-like POS-tagged sentences)
    save_path: where to save the plot
    """
    all_tag_set = set()
    proportions_list = []

    # Process each dataset
    for dataset_name, data in dataset_dict.items():
        parsed_data = [ast.literal_eval(sentence) for sentence in data]
        all_tags = [tag for sentence in parsed_data for _, tag in sentence]
        tag_counts = Counter(all_tags)
        total = sum(tag_counts.values())

        # Update global tag set
        all_tag_set.update(tag_counts.keys())

        # Store proportions
        for tag in tag_counts:
            proportions_list.append({
                "Dataset": dataset_name,
                "POS Tag": tag,
                "Proportion": tag_counts[tag] / total
            })
    print(proportions_list)

    # Make sure every tag appears in every dataset (with 0 if missing)
    full_data = []
    for tag in sorted(all_tag_set):
        for dataset_name in dataset_dict.keys():
            found = next((item for item in proportions_list if item["Dataset"] == dataset_name and item["POS Tag"] == tag), None)
            proportion = found["Proportion"] if found else 0.0
            full_data.append({
                "Dataset": dataset_name,
                "POS Tag": tag,
                "Proportion": proportion
            })

    df = pd.DataFrame(full_data)

    # Plot
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x="Dataset", y="Proportion", hue="POS Tag", palette="Paired")

    plt.title("POS Tag Distribution Across Datasets", fontsize=14)
    plt.ylabel("Proportion", fontsize=12)
    plt.xlabel("Dataset", fontsize=12)
    plt.legend(title="POS Tag", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def get_sentences(config):
    pos_train = pd.read_csv(config['dataset']['train_path_annotated'])
    pos_val = pd.read_csv(config['dataset']['val_path_annotated'])
    pos_test = pd.read_csv(config['dataset']['test_path_annotated'])


    sentences_list_train = pos_train["annotated_Sentence"].tolist()
    sentences_list_val = pos_val["annotated_Sentence"].tolist()
    sentences_list_test = pos_test["annotated_Sentence"].tolist()
    sentences = sentences_list_train + sentences_list_val + sentences_list_test
    return sentences


config_bg = load_config("configs/base.yaml", "configs/Bulgarian_translated.yaml")
config_ch = load_config("configs/base.yaml", "configs/Chinese_translated.yaml")
config_nl = load_config("configs/base.yaml", "configs/Dutch_translated.yaml")
config_it = load_config("configs/base.yaml", "configs/Italian_translated.yaml")
config_ru= load_config("configs/base.yaml", "configs/Russian_translated.yaml")

sentences_bg = get_sentences(config_bg)
sentences_ch = get_sentences(config_ch)
sentences_nl = get_sentences(config_nl)
sentences_it = get_sentences(config_it)
sentences_ru = get_sentences(config_ru)

dataset_dict = {
    "UD_Bulgarian-BTB": sentences_bg,
    "UD_Chinese-GSD": sentences_ch,
    "Alpino": sentences_nl,
    "PoSTWITA-UD": sentences_it,
    "Taiga": sentences_ru
}

save_dir = "outputs/plots"
os.makedirs(save_dir, exist_ok=True) 
filename = "POS_grouped_label_distribution.png"
save_path = os.path.join(save_dir, filename)

grouped_bar_plot_pos_distribution(dataset_dict, save_path)