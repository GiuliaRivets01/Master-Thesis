import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils import load_config, setup_logger

config_bg = load_config("configs/base.yaml", "configs/Bulgarian_translated.yaml")
config_ch = load_config("configs/base.yaml", "configs/Chinese_translated.yaml")
config_nl = load_config("configs/base.yaml", "configs/Dutch_translated.yaml")
config_it = load_config("configs/base.yaml", "configs/Italian_translated.yaml")
config_ru = load_config("configs/base.yaml", "configs/Russian_translated.yaml")

def merge_dataset_splits(train_path, val_path, test_path):
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    # Merge
    merged_df = pd.concat([train, val, test])
    return merged_df

df_italian = merge_dataset_splits(config_it['dataset']['cleaned_train_path'],
                                  config_it['dataset']['cleaned_val_path'],
                                  config_it['dataset']['cleaned_test_path'])

df_russian = merge_dataset_splits(config_ru['dataset']['cleaned_train_path'],
                                  config_ru['dataset']['cleaned_val_path'],
                                  config_ru['dataset']['cleaned_test_path'])

df_dutch = merge_dataset_splits(config_nl['dataset']['cleaned_train_path'],
                                  config_nl['dataset']['cleaned_val_path'],
                                  config_nl['dataset']['cleaned_test_path'])

df_chinese = merge_dataset_splits(config_ch['dataset']['cleaned_train_path'],
                                  config_ch['dataset']['cleaned_val_path'],
                                  config_ch['dataset']['cleaned_test_path'])

df_bulgarian = merge_dataset_splits(config_bg['dataset']['cleaned_train_path'],
                                  config_bg['dataset']['cleaned_val_path'],
                                  config_bg['dataset']['cleaned_test_path'])


# --- Add language label to each dataframe ---
df_italian['language'] = 'Italian'
df_dutch['language'] = 'Dutch'
df_russian['language'] = 'Russian'
df_bulgarian['language'] = 'Bulgarian'
df_chinese['language'] = 'Chinese'

# --- Combine into one dataframe ---
df_all = pd.concat([df_italian, df_dutch, df_russian, df_bulgarian, df_chinese], ignore_index=True)

# --- Compute answer lengths ---
df_all['answer_length'] = df_all['translated_answer'].apply(lambda x: len(str(x).split()))

# --- Plot KDEs on the same axes ---
plt.figure(figsize=(8, 5))
sns.set(style="whitegrid")

sns.kdeplot(data=df_all, x="answer_length", hue="language", fill=False, common_norm=False, bw_adjust=0.8)

plt.title("Answer Length Distribution by Language")
plt.xlabel("Answer Length (tokens)")
plt.ylabel("Density")
plt.xlim(0, 50)  # Adjust as needed for your data
plt.legend(title="Language")
plt.tight_layout()
#plt.savefig("plots/combined_answer_lengths.pdf")
plt.show()


logger = setup_logger("outputs/logs/")

def get_lengths(df, name):
    df["context_length"] = df["translated_context"].apply(lambda x: len(x.split()))
    df["answer_length"] = df["translated_answer"].apply(lambda x: len(x.split()))
    avg_context_length = df["context_length"].mean()
    avg_answer_length = df["answer_length"].mean()
    logger.info(f"Average context length in {name}: {avg_context_length}")
    logger.info(f"Average answer length in {name}: {avg_answer_length}\n")

get_lengths(df_bulgarian, 'HS-bg')
get_lengths(df_chinese, 'COLD')
get_lengths(df_dutch, 'Dutch HateCheck')
get_lengths(df_italian, 'HS-it')
get_lengths(df_russian, 'South Park')
