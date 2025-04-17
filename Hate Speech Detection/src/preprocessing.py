from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd

class Preprocessor():
    def __init__(self, train, val, test, classes, text_col, model_path, device):
        self.train = train
        self.val = val
        self.test = test
        self.classes = classes
        self.text_col = text_col
        self.model_path = model_path
        self.device = device

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

    def get_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return tokenizer

    # Now use the Hugging Face tokenizer for the final tokenization
    def tokenize_texts(self, texts, tokenizer):
        return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    def tokenization(self, train, val, test, labelCol, textCol, tokenizer):
        # Ensure 'translated_text' column is converted to a list of strings
        train_texts = train[textCol].astype(str).tolist()  # Convert to list of strings
        val_texts = val[textCol].astype(str).tolist()      # Convert to list of strings
        test_texts = test[textCol].astype(str).tolist()    # Convert to list of strings

        # Tokenize the preprocessed data
        train_encodings = self.tokenize_texts(train_texts, tokenizer)
        val_encodings = self.tokenize_texts(val_texts, tokenizer)
        test_encodings = self.tokenize_texts(test_texts, tokenizer)

        # Convert labels to the format expected by the model
        train_encodings[labelCol] = torch.tensor(pd.Series(train[labelCol]).values)
        val_encodings[labelCol] = torch.tensor(pd.Series(val[labelCol]).values)
        test_encodings[labelCol] = torch.tensor(pd.Series(test[labelCol]).values)

        return train_encodings, val_encodings, test_encodings
        
    def main(self):
        # Compute weighted loss
        loss_fn = self.compute_class_weights(self.train, self.classes, self.device)

        tokenizer = self.get_tokenizer(self.model_path)

        # Tokenization
        print("Tokenizing...")
        train_encodings, val_encodings, test_encodings = self.tokenization(self.train, self.val, self.test, 'labels', self.text_col, tokenizer)
        return train_encodings, val_encodings, test_encodings, loss_fn


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