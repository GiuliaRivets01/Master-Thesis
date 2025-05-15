from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
import torch

class POSDataset(TorchDataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = {key: torch.tensor(self.dataset[idx][key]) for key in ["input_ids", "attention_mask", "labels"]}
        return item

class POS_Preprocessor():
    def __init__(self, config, sentences_list_train, sentences_list_val, sentences_list_test, tokenizer, logger):
        self.config = config
        self.sentences_list_train = sentences_list_train
        self.sentences_list_val = sentences_list_val
        self.sentences_list_test = sentences_list_test
        self.tokenizer = tokenizer
        self.logger = logger

    def build_dictionary_tagged(self, sentences_list_train, sentences_list_val, sentences_list_test):
        tags = set([item for sublist in sentences_list_train+sentences_list_val+sentences_list_test for _, item in sublist])
        self.logger.info('TOTAL TAGS: ', len(tags))

        tag2int = {}
        int2tag = {}
        for i, tag in enumerate(sorted(tags)):
            tag2int[tag] = i+1
            int2tag[i+1] = tag

        # Special character for the tags
        tag2int['-PAD-'] = 0
        int2tag[0] = '-PAD-'

        n_tags = len(tag2int)
        print('Total tags:', n_tags)

        # Create the disctionaries
        dataset_dict_train = Dataset.from_dict({
        "sentences": [[(word, tag) for word, tag in sentence] for sentence in sentences_list_train],
        })

        dataset_dict_val = Dataset.from_dict({
            "sentences": [[(word, tag) for word, tag in sentence] for sentence in sentences_list_val],
        })

        dataset_dict_test = Dataset.from_dict({
            "sentences": [[(word, tag) for word, tag in sentence] for sentence in sentences_list_test],
        })

        return dataset_dict_train, dataset_dict_val, dataset_dict_test, n_tags, tag2int, int2tag

    def tokenize_and_align_labels(self, examples, tag2int):
        tokenized_inputs = self.tokenizer(
            [list(map(lambda x: x[0], sentence)) for sentence in examples["sentences"]],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=128
        )

        labels = []
        for i, sentence in enumerate(examples["sentences"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to words
            label_ids = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # Ignore special tokens
                elif word_idx != previous_word_idx:
                    label_ids.append(tag2int[sentence[word_idx][1]])  # Assign the correct label
                else:
                    label_ids.append(tag2int[sentence[word_idx][1]])  # Use same label for subwords

                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def tokenize_dataset(self, dataset_dict_train, dataset_dict_val, dataset_dict_test, tag2int):
        encoded_dataset_train = dataset_dict_train.map(
            self.tokenize_and_align_labels,
            batched=True,
            fn_kwargs={"tag2int": tag2int}  # Pass tag2int to the function
        )
        encoded_dataset_val = dataset_dict_val.map(
            self.tokenize_and_align_labels,
            batched=True,
            fn_kwargs={"tag2int": tag2int}  # Pass tag2int to the function
        )
        encoded_dataset_test = dataset_dict_test.map(
            self.tokenize_and_align_labels,
            batched=True,
            fn_kwargs={"tag2int": tag2int}  # Pass tag2int to the function
        )

        return encoded_dataset_train, encoded_dataset_val, encoded_dataset_test

    def main(self, return_tag_dicts=False):
        dict_train, dict_val, dict_test, n_tags, tag2int, int2tag = self.build_dictionary_tagged(
            self.sentences_list_train, self.sentences_list_val, self.sentences_list_test)

        encoded_train, encoded_val, encoded_test = self.tokenize_dataset(dict_train, dict_val, dict_test, tag2int)
        
        train_dataset = POSDataset(encoded_train, self.tokenizer)
        val_dataset = POSDataset(encoded_val, self.tokenizer)
        test_dataset = POSDataset(encoded_test, self.tokenizer)

        """
        for split_name, dataset in zip(['train', 'val', 'test'], [train_dataset, val_dataset, test_dataset]):
            all_labels = torch.cat([example['labels'] for example in dataset])
            print(f"{split_name} labels: min={all_labels.min().item()}, max={all_labels.max().item()}, unique={torch.unique(all_labels)}")
        print("N tags: ", n_tags)
        print("Tag to int: ", tag2int)
        input("C") """


        if return_tag_dicts:
            return train_dataset, val_dataset, test_dataset, n_tags, tag2int
        return train_dataset, val_dataset, test_dataset, n_tags