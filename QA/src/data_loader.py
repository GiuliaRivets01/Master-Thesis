
from sklearn.model_selection import train_test_split
import pandas as pd

class DataLoader():
    def __init__(self, train_path, val_path, test_path):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

    def load_dataset_splits(self):
        train = pd.read_csv(self.train_path)
        val = pd.read_csv(self.val_path)
        test = pd.read_csv(self.test_path)

        return train, val, test

    def resplit_dataset(self, train, val, test):
        # Merge the current train, test and validation splits into a single dataframe
        merged = pd.concat([train, val, test], axis=0, ignore_index=True)

        train, df_temp = train_test_split(merged, test_size=0.2, random_state=42)
        val, test = train_test_split(df_temp, test_size=0.5, random_state=42)

        return train, val, test

    def reformat_to_squad(self, train, val, test):
        # Reformat the answers column such that it is a ditctionary with two lists: one for the text of
        # the answer and the other one for the start position of the answer within the context
        train["answers"] = train.apply(lambda row: {"text": [row["predicted_answer"]], "answer_start": [row["predicted_start"]]}, axis=1)
        val["answers"] = val.apply(lambda row: {"text": [row["predicted_answer"]], "answer_start": [row["predicted_start"]]}, axis=1)
        test["answers"] = test.apply(lambda row: {"text": [row["predicted_answer"]], "answer_start": [row["predicted_start"]]}, axis=1)
        return train, val, test

    def main(self):
        train, val, test = self.load_dataset_splits()
        train, val, test = self.resplit_dataset(train, val, test)
        train, val, test = self.reformat_to_squad(train, val, test)

        train.to_csv("train_check.csv", index=False)
        val.to_csv("val_check.csv", index=False)
        test.to_csv("test_check.csv", index=False)

        return train, val, test
