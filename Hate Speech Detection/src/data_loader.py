import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset_splits(train_path, val_path, test_path):
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    return train, val, test

def remove_nan_values(df_train, df_val, df_test, col_name):
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

class DatasetPreparer:
    def __init__(self, train_path, val_path, test_path, lang, text_col, label_col, cols_to_drop=None):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.lang = lang
        self.text_col = text_col
        self.label_col = label_col
        self.cols_to_drop = cols_to_drop

    def prepare_dataset(self):

        # Load the dataset
        train, val, test = load_dataset_splits(self.train_path, self.val_path, self.test_path)

        # Change the labels of the Bulgarian dataset so that 0 = non-hateful and 1 = hateful
        if self.lang == "bg":
            train.loc[train[self.label_col] >= 1, self.label_col] = 1
            val.loc[val[self.label_col] >= 1, self.label_col] = 1
            test.loc[test[self.label_col] >= 1, self.label_col] = 1

        # Split the dataset into train (80%), validation (10%) and test (10%)
        # Merge the current train, test and validation splits into a single dataframe
        merged = pd.concat([train, val, test], axis=0, ignore_index=True)
        # Split
        train, df_temp = train_test_split(merged, test_size=0.2, random_state=42)
        val, test = train_test_split(df_temp, test_size=0.5, random_state=42)

        # Removed failed translations
        train, val, test = remove_nan_values(train, val, test, self.text_col)

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