import nltk
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize

# Download necessary NLTK data for preprocessing
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

def clean_text_translated(text, lang):
    """Preprocess a single text instance."""
    if pd.isna(text):
        return ""  # Handle missing values

    text = text.lower()  # Lowercase all text

    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags

    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs

    text = re.sub(r"[^\x00-\x7F]+", "", text)  # Remove non-ASCII characters

    text = re.sub(r"([a-z])\1{2,}", r"\1", text)  # Reduce repeated letters (e.g., "gooood" → "good")

    text = re.sub(r"[^\w\s]", " ", text)  # Remove excessive punctuation

    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace

    tokens = word_tokenize(text)  # Tokenization


    return " ".join(tokens)  # Reconstruct cleaned text

def clean_text_original(text, lang):
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

def preprocess_dataframe(df, text_col, original, lang):
    """Apply preprocessing to an entire dataset."""

    if original:
        print("Cleaning original dataset")
        df[text_col] = df[text_col].astype(str).apply(lambda x: clean_text_original(x, lang))
    else:
        print("Cleaning translated dataset")
        df[text_col] = df[text_col].astype(str).apply(lambda x: clean_text_translated(x, lang))
    return df

def preprocess_dataset(train, val, test, textCol, original, lang):
    train = preprocess_dataframe(train, textCol, original, lang)
    val = preprocess_dataframe(val, textCol, original, lang)
    test = preprocess_dataframe(test, textCol, original, lang)
    return train, val, test