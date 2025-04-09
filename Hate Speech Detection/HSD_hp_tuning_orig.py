import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Preprocessing
from sklearn.utils.class_weight import compute_class_weight
import nltk
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import re
from transformers import AutoTokenizer
from torch.utils.data import Dataset

# Fine-tuning
import optuna
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Training
from transformers import BertForSequenceClassification
import os

# Results
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import seaborn as sns

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("language", help="Choose a language between Chinese, Dutch, Italian, Russian and Bulgarian")
args = parser.parse_args()
lang = args.language

# Download necessary NLTK data for preprocessing
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

def load_dataset_splits(train_path, val_path, test_path):
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    return train, val, test

def split_dataset(df):
    """
        Split a given dataset into train, test and validation sets, with
        80% train, 10% test and 10% validation
    """
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

    # Split the 20% into 10% validation and 10% test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    return train_df, val_df, test_df

def remove_failed_translations(df_train, df_val, df_test, col_name):
    # Train
    rows_train = df_train.shape[0]
    df_train = df_train.dropna(subset=[col_name])
    print("Nan values found in train set: ", rows_train - df_train.shape[0])

    # Validation
    rows_val = df_val.shape[0]
    df_val = df_val.dropna(subset=[col_name])
    print("Nan values found in validation set: ", rows_val - df_val.shape[0])

    # Test
    rows_test = df_test.shape[0]
    df_test = df_test.dropna(subset=[col_name])
    print("Nan values found in test set: ", rows_test - df_test.shape[0])

    return df_train, df_val, df_test

def drop_columns(train, val, test, cols, label_col):
    if cols != None:
        train = train.drop(columns=cols)
        val = val.drop(columns=cols)
        test = test.drop(columns=cols)

    train["labels"] = train[label_col]
    val["labels"] = val[label_col]
    test["labels"] = test[label_col]

    train = train.drop(columns=[label_col])
    val = val.drop(columns=[label_col])
    test = test.drop(columns=[label_col])

    return train, val, test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Compute class weights
def get_class_weights(train_labels, classes, device):
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_labels
    )
    return torch.tensor(class_weights, dtype=torch.float).to(device)

def compute_class_weights(train_df, classes, device):
    # Get weights for loss function
    train_labels = train_df['labels'].values
    class_weights = get_class_weights(train_labels, classes, device)

    # Define weighted loss function
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    return loss_fn

def clean_text(text):
    """Preprocess a single text instance."""
    if pd.isna(text):
        return ""  # Handle missing values

    if lang != "ch":
        text = text.lower()  # Lowercase all text

    # For all languages
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs

    if lang == "it" or lang == "nl":
        text = re.sub(r"[^\x00-\x7F]+", "", text)  # Remove non-ASCII characters
        text = re.sub(r"([a-z])\1{2,}", r"\1", text)  # Reduce repeated letters
        text = re.sub(r"[^\w\s]", " ", text)  # Remove excessive punctuation

    if lang == "bg":
        text = re.sub(r"([а-яА-Я])\1{2,}", r"\1", text)
        text = re.sub(r"[^\w\s]", " ", text)  # Remove excessive punctuation
    
    if lang == "ru":
        text = re.sub(r"([a-zа-яА-Я])\1{2,}", r"\1", text)
        text = re.sub(r"[^\w\sа-яА-ЯёЁ]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace

    tokens = word_tokenize(text)  # Tokenization

    return " ".join(tokens)  # Reconstruct cleaned text

def preprocess_dataframe(df, text_col):
    """Apply preprocessing to an entire dataset."""
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    return df

def preprocess_dataset(train, val, test, textCol):
    train = preprocess_dataframe(train, textCol)
    val = preprocess_dataframe(val, textCol)
    test = preprocess_dataframe(test, textCol)
    return train, val, test

bg_model_path = "AIaLT-IICT/bert_bg_lit_web_base_uncased"
ch_model_path = "google-bert/bert-base-chinese"
nl_model_path = "GroNLP/bert-base-dutch-cased"
it_model_path = "dbmdz/bert-base-italian-uncased"
ru_model_path = "deepvk/bert-base-uncased"

if lang == "bg":
    print("Loading Bulgarian Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(bg_model_path)

elif lang == "ch":
    print("Loading Chinese Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(ch_model_path)

elif lang == "nl":
    print("Loading Dutch Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(nl_model_path)

elif lang == "it":
    print("Loading Italian Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(it_model_path)

elif lang == "ru":
    print("Loading Russian Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(ru_model_path)

# Now use the Hugging Face tokenizer for the final tokenization
def tokenize_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")


def tokenization(train, val, test, labelCol, textCol):
    # Ensure 'translated_text' column is converted to a list of strings
    train_texts = train[textCol].astype(str).tolist()  # Convert to list of strings
    val_texts = val[textCol].astype(str).tolist()      # Convert to list of strings
    test_texts = test[textCol].astype(str).tolist()    # Convert to list of strings

    # Tokenize the preprocessed data
    train_encodings = tokenize_texts(train_texts)
    val_encodings = tokenize_texts(val_texts)
    test_encodings = tokenize_texts(test_texts)

    # Convert labels to the format expected by the model
    train_encodings[labelCol] = torch.tensor(pd.Series(train[labelCol]).values)
    val_encodings[labelCol] = torch.tensor(pd.Series(val[labelCol]).values)
    test_encodings[labelCol] = torch.tensor(pd.Series(test[labelCol]).values)

    return train_encodings, val_encodings, test_encodings

class HateSpeechDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
    
def load_model(df):
    num_labels = len(set(df['labels']))

    if lang == "bg":
        print("Loading Bulgarian model")
        model = BertForSequenceClassification.from_pretrained(bg_model_path, num_labels=num_labels)
    
    elif lang == "ch":
        print("Loading Chinese model")
        model = BertForSequenceClassification.from_pretrained(ch_model_path, num_labels=num_labels)
      
    elif lang == "nl":
        print("Loading Dutch model")
        model = BertForSequenceClassification.from_pretrained(nl_model_path, num_labels=num_labels)

    elif lang == "it":
        print("Loading Italian model")
        model = BertForSequenceClassification.from_pretrained(it_model_path, num_labels=num_labels)

    elif lang == "ru":
        print("Loading Russian model")
        model = BertForSequenceClassification.from_pretrained(ru_model_path, num_labels=num_labels)

    
    return model.to(device)

class CustomTrainer(Trainer):
    def __init__(self, *args, loss_fn=None, **kwargs):
      super().__init__(*args, **kwargs)
      self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")  # Get the true labels
        outputs = model(**inputs)  # Get model outputs
        logits = outputs.get("logits")  # Get the logits (raw predictions)

        # Compute the weighted loss
        loss = self.loss_fn(logits, labels)

        # Return loss, and optionally, the outputs (for debugging/metrics)
        return (loss, outputs) if return_outputs else loss
    
def compute_metrics(p):
    preds = p.predictions.argmax(-1)  # Get predicted labels
    labels = p.label_ids  # True labels

    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)

    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def objective(trial, train_dataset, val_dataset, model_ch, tokenizer, loss_fn):
    # Define hyperparameters to be tuned
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 6)
    weight_decay = trial.suggest_uniform("weight_decay", 0.01, 0.1)
    
    # Set training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        #eval_steps=500,
        save_strategy="epoch",
        #save_steps=500,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        report_to="none",
        learning_rate=learning_rate,
    )
    
    if loss_fn != None:
        trainer = CustomTrainer(
            model=model_ch,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics ,
            loss_fn=loss_fn
        )
    else:
        trainer = Trainer(
            model=model_ch,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics  # Assuming this function is available
        )

    # Train and evaluate the model
    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results["eval_f1"]  # Return F1 score to optimize


class DatasetPipeline():
    def __init__(self, lang, train_path, val_path, test_path, text_col, label_col, split_again,
                 cols_to_drop, classes):
        self.lang = lang
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.text_col = text_col
        self.label_col = label_col
        self.split_again = split_again
        self.cols_to_drop = cols_to_drop
        self.classes = classes
      
    def prepare_dataset(self, train, val, test):

        # Change the labels of the Bulgarian dataset so that 0 = non-hateful and 1 = hateful
        if self.lang == "bg":
            train.loc[train[self.label_col] >= 1, self.label_col] = 1
            val.loc[val[self.label_col] >= 1, self.label_col] = 1
            test.loc[test[self.label_col] >= 1, self.label_col] = 1
      
        # Split the dataset into train (80%), validation (10%) and test (10%)
        if self.split_again == True:
            # Merge the current train, test and validation splits into a single dataframe
            merged = pd.concat([train, val, test], axis=0, ignore_index=True)
            # Split
            train, df_temp = train_test_split(merged, test_size=0.2, random_state=42)
            val, test = train_test_split(df_temp, test_size=0.5, random_state=42)
        
        # Removed failed translations
        train, val, test = remove_failed_translations(train, val, test, self.text_col) 

        # Remove unnecessary columns
        train, val, test = drop_columns(train, val, test, self.cols_to_drop, self.label_col)

        # Replace string labels with numbers
        if self.lang == "nl":
            sentiment_map = {'non-hateful': 0, 'hateful': 1}
            train['labels'] = train['labels'].map(sentiment_map)
            val['labels'] = val['labels'].map(sentiment_map)
            test['labels'] = test['labels'].map(sentiment_map)
    
        if self.lang == "ru":
            train['labels'] = train['labels'].astype(int)
            val['labels'] = val['labels'].astype(int)
            test['labels'] = test['labels'].astype(int)
        
        return train, val, test

    def preprocess_dataset(self, train, val, test):

        # Compute the weighted loss function
        loss_fn = compute_class_weights(train, self.classes, device)

        # Preprocess
        print("Preprocessing...")
        train_cleaned, val_cleaned, test_cleaned = preprocess_dataset(train, val, test, self.text_col)

        # Tokenization
        print("Tokenizing...")
        train_encodings, val_encodings, test_encodings = tokenization(train, val, test, 'labels', self.text_col)

        return train_encodings, val_encodings, test_encodings, loss_fn

    def main(self):
        # Load the dataset
        train, val, test = load_dataset_splits(self.train_path, self.val_path, self.test_path)

        # Prepare the dataset
        train, val, test = self.prepare_dataset(train, val, test)

        # Preprocess the dataset
        train_encodings, val_encodings, test_encodings, loss_fn = self.preprocess_dataset(train, val, test)

        # Create the HSD Dataset
        train_dataset = HateSpeechDataset(train_encodings, train_encodings['labels'])
        val_dataset = HateSpeechDataset(val_encodings, val_encodings['labels'])
        test_dataset = HateSpeechDataset(test_encodings, test_encodings['labels'])

        # Load the model
        model= load_model(train)

        # Hyperparameter tuning
        print("Hyperparameter tuning started...")
        # Create a study object for optimization
        study = optuna.create_study(direction="maximize")  # Optimize for maximum F1 score
        # Pass the datasets and model into the optimization process
        study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, model, tokenizer, loss_fn), n_trials=10)
        # Print the best hyperparameters found
        print("Best hyperparameters: ", study.best_params)

if lang == "en":
    en_dataset_pipeline = DatasetPipeline(lang="en", train_path="HSD_bg_train.csv", 
                                        val_path="HSD_bg_val.csv", test_path="HSD_bg_test.csv", 
                                        text_col="text", label_col="Type", split_again=True, 
                                        cols_to_drop=["Positivity"], classes=np.array([0, 1]))
    en_dataset_pipeline.main()


elif lang == "bg":
    bg_dataset_pipeline = DatasetPipeline(lang="bg", train_path="HSD_bg_train.csv", 
                                        val_path="HSD_bg_val.csv", test_path="HSD_bg_test.csv", 
                                        text_col="text", label_col="Type", split_again=True, 
                                        cols_to_drop=["Positivity"], classes=np.array([0, 1]))
    bg_dataset_pipeline.main()

elif lang == "ch":
    ch_dataset_pipeline = DatasetPipeline(lang="ch", train_path="train.csv", 
                                        val_path="dev.csv", test_path="test.csv", 
                                        text_col="TEXT", label_col="label", split_again=True, 
                                        cols_to_drop=["split", "topic"], classes=np.array([0, 1]))
    ch_dataset_pipeline.main()

elif lang == "nl":
    nl_dataset_pipeline = DatasetPipeline(lang="nl", train_path="HSD_nl_train.csv", 
                                        val_path="HSD_nl_val.csv", test_path="HSD_nl_test.csv", 
                                        text_col="test_case", label_col="label_gold", split_again=False, 
                                        cols_to_drop=[], classes=np.array([0, 1]))
    nl_dataset_pipeline.main()


elif lang == "it":
    it_dataset_pipeline = DatasetPipeline(lang="it", train_path="HSD_it_train_translated.csv", 
                                        val_path="HSD_it_val_translated.csv", test_path="HSD_it_test_translated.csv", 
                                        text_col="full_text", label_col="hs", split_again=False, 
                                        cols_to_drop=["id", "stereotype", "translated_full_text"], classes=np.array([0, 1]))
    it_dataset_pipeline.main()

elif lang == "ru":
    ru_dataset_pipeline = DatasetPipeline(lang="ru", train_path="HSD_ru_train_translated.csv", 
                                        val_path="HSD_ru_val_translated.csv", test_path="HSD_ru_test_translated.csv", 
                                        text_col="text", label_col="hate_speech", split_again=False, 
                                        cols_to_drop=["Unnamed: 0", "index", "sentiment", "translated_text"], classes=np.array([0, 1]))
    ru_dataset_pipeline.main()

