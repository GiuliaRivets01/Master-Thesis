from sklearn.utils.class_weight import compute_class_weight
import torch
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from torch.utils.data import Dataset
import spacy
import jieba
import stanza

# Download necessary NLTK data for preprocessing
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

class SA_Preprocessor():
    def __init__(self, train, val, test, config, device, tokenizer, args, logger):
        self.train = train
        self.val = val
        self.test = test
        self.config = config
        self.classes = config['dataset']['classes']
        self.device = device
        self.tokenizer = tokenizer
        self.text_col = config['dataset']['text_col']
        self.lang = config['dataset']['language']
        self.args = args
        self.logger = logger

    # Compute class weights
    def get_class_weights(self, train_labels, classes, device):
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=train_labels
        )
        return torch.tensor(class_weights, dtype=torch.float).to(device)

    def compute_class_weights(self, train_df, classes, device):
        # Get weights for loss function
        train_labels = train_df['labels'].values
        class_weights = self.get_class_weights(train_labels, classes, device)

        # Define weighted loss function
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        return loss_fn
    
    def clean_text(self, text):
        """Preprocess a single text instance."""
        if pd.isna(text):
            return ""  # Handle missing values

        text = text.lower()  # Lowercase all text

        text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags

        text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs

        if self.lang == 'it':
            text = re.sub(r"@[A-Za-z0-9_]+", "", text) # Remove usernames

        text = re.sub(r"[^\x00-\x7F]+", "", text)  # Remove non-ASCII characters

        text = re.sub(r"([a-z])\1{2,}", r"\1", text)  # Reduce repeated letters (e.g., "gooood" → "good")

        text = re.sub(r"[^\w\s]", " ", text)  # Remove excessive punctuation

        text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace

        tokens = word_tokenize(text)  # Tokenization

        return " ".join(tokens)  # Reconstruct cleaned text

    def clean_text_bg_original(self, text):
        """Preprocess a single text instance."""
        nlp = stanza.Pipeline(lang="bg", processors="tokenize,lemma")

        if pd.isna(text):
            return ""  # Handle missing values

        text = text.lower()  # Lowercase all text
        text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
        text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
        text = re.sub(r"[^\x00-\x7F]+", "", text)  # Remove non-ASCII characters
        text = re.sub(r"([a-z])\1{2,}", r"\1", text)  # Reduce repeated letters
        text = re.sub(r"[^\w\s]", " ", text)  # Remove excessive punctuation
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace

        # Tokenization and Lemmatization using spaCy
        doc = nlp(text)
        tokens = [token.text for token in doc]

        return " ".join(tokens)  # Reconstruct cleaned text

    def clean_text_original(self, text):
        if self.args.language == "it":
            nlp = spacy.load("it_core_news_sm")
        elif self.args.language == "nl":
            nlp = spacy.load("nl_core_news_sm")
        elif self.args.language == "ch":
            nlp = spacy.load("zh_core_web_sm")
        elif self.args.language == "ru":
            nlp = spacy.load("ru_core_news_sm")

        """Preprocess a single text instance."""
        if pd.isna(text):
            return ""  # Handle missing values

        if self.args.language != "ch":
            text = text.lower()  # Lowercase all text
        text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
        text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs

        if self.args.language != "ch" and self.args.language != "ru":
            text = re.sub(r"[^\x00-\x7F]+", "", text)  # Remove non-ASCII characters
            text = re.sub(r"([a-z])\1{2,}", r"\1", text)  # Reduce repeated letters
            text = re.sub(r"[^\w\s]", " ", text)  # Remove excessive punctuation

        if self.args.language == "ru":
            text = re.sub(r"([a-zа-яА-Я])\1{2,}", r"\1", text)
            text = re.sub(r"[^\w\sа-яА-ЯёЁ]", " ", text)

        text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace

        # Tokenization and Lemmatization using spaCy
        if self.args.language == "ch":
            tokens = jieba.cut(text)  # Tokenize Chinese
        else:
            doc = nlp(text)
            tokens = [token.text for token in doc]

        return " ".join(tokens)  # Reconstruct cleaned text


    def tokenize_texts(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    def tokenization(self, train, val, test):
        # Ensure 'translated_text' column is converted to a list of strings
        text_col = self.config['dataset']['text_col']
        label_col = "labels"

        train_texts = train[text_col].astype(str).tolist()  # Convert to list of strings
        val_texts = val[text_col].astype(str).tolist()      
        test_texts = test[text_col].astype(str).tolist()    

        # Tokenize the preprocessed data
        train_encodings = self.tokenize_texts(train_texts)
        val_encodings = self.tokenize_texts(val_texts)
        test_encodings = self.tokenize_texts(test_texts)

        # Convert labels to the format expected by the model
        train_encodings[label_col] = torch.tensor(pd.Series(train[label_col]).values)
        val_encodings[label_col] = torch.tensor(pd.Series(val[label_col]).values)
        test_encodings[label_col] = torch.tensor(pd.Series(test[label_col]).values)

        return train_encodings, val_encodings, test_encodings

    
    def main(self):
        # Compute the weighted loss function
        if self.classes != False:
            print(np.array(self.classes))
            print(self.train['labels'].unique())
            input("C")
            loss = self.compute_class_weights(self.train, np.array(self.classes), self.device)
        else:
            loss = None

        # Clean the dataset
        if self.args.language == 'bg' and self.args.original == 'or':
            self.logger.info(f"Cleaning Bulgarian original dataset...")
            stanza.download("bg")
            self.train[self.text_col] = self.train[self.text_col].astype(str).apply(self.clean_text_bg_original)
            self.val[self.text_col] = self.val[self.text_col].astype(str).apply(self.clean_text_bg_original)
            self.test[self.text_col] = self.test[self.text_col].astype(str).apply(self.clean_text_bg_original)

        elif self.args.language != 'bg' and self.args.original == 'or':
            self.logger.info(f"Cleaning original dataset...")
            self.train[self.text_col] = self.train[self.text_col].astype(str).apply(self.clean_text_original)
            self.val[self.text_col] = self.val[self.text_col].astype(str).apply(self.clean_text_original)
            self.test[self.text_col] = self.test[self.text_col].astype(str).apply(self.clean_text_original)

        else:
            self.logger.info(f"Cleaning translated dataset...")
            self.train[self.text_col] = self.train[self.text_col].astype(str).apply(self.clean_text)
            self.val[self.text_col] = self.val[self.text_col].astype(str).apply(self.clean_text)
            self.val[self.text_col] = self.val[self.text_col].astype(str).apply(self.clean_text)

        # Tokenization
        train_encodings, val_encodings, test_encodings = self.tokenization(self.train, self.val, self.test)

        # Creating the dataset
        train_dataset = SentimentDataset(train_encodings, train_encodings['labels'])
        val_dataset = SentimentDataset(val_encodings, val_encodings['labels'])
        test_dataset = SentimentDataset(test_encodings, test_encodings['labels'])

        return train_dataset, val_dataset, test_dataset, loss