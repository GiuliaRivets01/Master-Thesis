from datasets import load_dataset

def get_datastes(train_path, val_path, test_path):
    train = load_dataset("json", data_files=train_path)['train']
    val = load_dataset("json", data_files=val_path)['train']
    test = load_dataset("json", data_files=test_path)['train']

    return train, val, test