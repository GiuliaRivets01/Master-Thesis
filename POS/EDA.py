import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.config import load_config
from collections import Counter
import ast
import os
from matplotlib import cm


def bar_plot_sentiment_distribution(data, save_path):
    parsed_data = [ast.literal_eval(sentence) for sentence in data]

    # Step 2: Flatten and extract POS tags
    all_tags = [tag for sentence in parsed_data for _, tag in sentence]
    tag_counts = Counter(all_tags)

    # Step 3: Fixed order of POS categories (customize as needed)
    fixed_categories = list(set(all_tags))

    # Step 4: Compute proportions, use 0 for missing tags
    label_counts = {tag: tag_counts.get(tag, 0) / sum(tag_counts.values()) for tag in fixed_categories}

    # Step 5: Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=fixed_categories,
        y=[label_counts[tag] for tag in fixed_categories],
        palette="viridis"
    )

    # Labels and title
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Proportion", fontsize=12)
    plt.title("Frequency of Each POS Tag", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def pie_chart_pos_distribution(data, label_color_map, save_path=None, threshold=3.0):
    import ast
    from collections import Counter

    # Step 1: Parse input data
    parsed_data = [ast.literal_eval(sentence) for sentence in data]

    # Step 2: Flatten and extract POS tags
    all_tags = [tag for sentence in parsed_data for _, tag in sentence]
    tag_counts = Counter(all_tags)
    total_count = sum(tag_counts.values())

    # Step 3: Separate main and small tags
    labels = []
    sizes = []
    other_tags = []
    other_size = 0

    for tag, count in tag_counts.items():
        percentage = (count / total_count) * 100
        if percentage < threshold:
            other_tags.append(tag)
            other_size += count
        else:
            labels.append(tag)
            sizes.append(count)

    # Add the "Other" category if applicable
    if other_tags:
        other_label = f"({', '.join(other_tags)})"
        labels.append(other_label)
        sizes.append(other_size)

    # Step 4: Assign colors consistently
    colors = [
        label_color_map.get(label, 'grey') if not label.startswith("Other") else 'lightgrey'
        for label in labels
    ]

    # Step 5: Plot
    plt.figure(figsize=(8, 8))
    plt.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors
    )

    plt.axis('equal')
    plt.title("Distribution of POS Tags", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



config = load_config("configs/base.yaml", "configs/Bulgarian_translated.yaml")
pos_train = pd.read_csv(config['dataset']['train_path_annotated'])
pos_val = pd.read_csv(config['dataset']['val_path_annotated'])
pos_test = pd.read_csv(config['dataset']['test_path_annotated'])

# Define the directory and filename
save_dir = "outputs/plots"
os.makedirs(save_dir, exist_ok=True)  # create the directory if it doesn't exist
filename = "POS_label_pie_chart_bg.png"
save_path = os.path.join(save_dir, filename)

sentences_list_train = pos_train["annotated_Sentence"].tolist()
sentences_list_val = pos_val["annotated_Sentence"].tolist()
sentences_list_test = pos_test["annotated_Sentence"].tolist()
sentences = sentences_list_train + sentences_list_val + sentences_list_test

#bar_plot_sentiment_distribution(sentences, save_path)

all_possible_labels = [
    'NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'NUM', 'CONJ',
    'PUNCT', 'X', 'PROPN', 'INTJ', 'SYM', 'PART', 'AUX', 'SCONJ', 'CCONJ'
]

# Generate a color map
base_cmap = cm.get_cmap('Set3', len(all_possible_labels))
label_color_map = {label: base_cmap(i) for i, label in enumerate(all_possible_labels)}

pie_chart_pos_distribution(sentences, label_color_map, save_path)
