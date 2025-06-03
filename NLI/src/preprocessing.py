from datasets import Dataset
import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class NLI_Preprocessor():
    def __init__(self, tokenizer, config, train, val, test, device):
        self.tokenizer = tokenizer
        self.premise = config['dataset']['premise_col']
        self.hypothesis = config['dataset']['hypothesis_col']
        self.cols_to_drop = config['dataset']['cols_to_drop']
        self.lang = config['dataset']['language']
        self.train = train
        self.val = val
        self.test = test
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

    def clean_dataset(self, df):
        df = df.drop(columns=self.cols_to_drop)
        df = df.dropna()
        return df

    def tokenize_function(self, example):
        return self.tokenizer(
            example[self.premise],
            example[self.hypothesis],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
    
    def main(self):
        if self.lang == 'nl':
            classes = np.array([0, 1, 2]) 
            loss = self.compute_class_weights(self.train, classes, self.device)
        else: 
            loss = None

        train = self.clean_dataset(self.train)
        val = self.clean_dataset(self.val)
        test = self.clean_dataset(self.test)
        print(train.columns)
        print(val.columns)
        print(test.columns)
        input("C")

        train = Dataset.from_pandas(train)
        val = Dataset.from_pandas(val)
        test = Dataset.from_pandas(test)

        tokenized_train = train.map(self.tokenize_function, batched=True)
        tokenized_val = val.map(self.tokenize_function, batched=True)
        tokenized_test = test.map(self.tokenize_function, batched=True)

        return tokenized_train, tokenized_val, tokenized_test, loss
