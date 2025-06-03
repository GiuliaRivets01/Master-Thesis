import pandas as pd
from sklearn.model_selection import train_test_split

class SA_DataLoader():
    def __init__(self, config, logger):
        self.train_path = config['dataset']['train_path']
        self.val_path = config['dataset']['val_path']
        self.test_path = config['dataset']['test_path']
        self.text_col = config['dataset']['text_col']
        self.label_col = config['dataset']['label_col']
        self.cols_to_drop = config['dataset']['cols_to_drop']
        self.logger = logger
    
    def load_dataset_splits(self):
        train = pd.read_csv(self.train_path)
        val = pd.read_csv(self.val_path)
        test = pd.read_csv(self.test_path)

        merged = pd.concat([train, val, test])

        train, temp = train_test_split(merged, test_size=0.2, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)

        return train, val, test
    
    def remove_failed_translations(self, df_train, df_val, df_test):
        # Train
        rows_train = df_train.shape[0]
        df_train = df_train.dropna(subset=[self.text_col])
        self.logger.info(f"Nan values found in train set: {rows_train - df_train.shape[0]}")

        # Validation
        rows_val = df_val.shape[0]
        df_val = df_val.dropna(subset=[self.text_col])
        self.logger.info(f"Nan values found in validation set: {rows_val - df_val.shape[0]}")

        # Test
        rows_test = df_test.shape[0]
        df_test = df_test.dropna(subset=[self.text_col])
        self.logger.info(f"Nan values found in test set: {rows_test - df_test.shape[0]}")

        return df_train, df_val, df_test
    
    def drop_columns(self, train, val, test):
        if self.cols_to_drop != []:
            train = train.drop(columns=self.cols_to_drop)
            val = val.drop(columns=self.cols_to_drop)
            test = test.drop(columns=self.cols_to_drop)

        train["labels"] = train[self.label_col]
        val["labels"] = val[self.label_col]
        test["labels"] = test[self.label_col]

        train = train.drop(columns=[self.label_col])
        val = val.drop(columns=[self.label_col])
        test = test.drop(columns=[self.label_col])

        return train, val, test
    
    def main(self):
        train, val, test = self.load_dataset_splits()
        train, val, test = self.remove_failed_translations(train, val, test)
        train, val, test = self.drop_columns(train, val, test)
        return train, val, test
