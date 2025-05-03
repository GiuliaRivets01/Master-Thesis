
class Preprocessor():
    def __init__(self, train, val, test, logger, label_names, tokenizer, original):
        self.train = train
        self.val = val
        self.test = test
        self.logger = logger
        self.label_names = label_names
        self.tokenizer = tokenizer
        self.original = original

    def clean_translated_tokens(self, example):
        example["translated_tokens"] = [
            word[:-1] if word.endswith(".") else word
            for word in example["translated_tokens"]
        ]
        return example
    
    # Find rows where the lengths of 'translated_tokens' and 'ner_tags' don't match
    def check_length_mismatch(self, example):
        if len(example["translated_tokens"]) != len(example["ner_tags"]):
            return True
    
    def check_correct_translation(self):
        # Apply the check to each dataset
        mismatched_train = self.train.filter(self.check_length_mismatch)
        mismatched_val = self.val.filter(self.check_length_mismatch)
        mismatched_test = self.test.filter(self.check_length_mismatch)

        # Print summary
        self.logger.info(f"Train set mismatches: {len(mismatched_train)}")
        self.logger.info(f"Validation set mismatches: {len(mismatched_val)}")
        self.logger.info(f"Test set mismatches: {len(mismatched_test)}")

        # If mismatches exist, print some examples
        if len(mismatched_train) > 0:
            self.logger.info("Example mismatch from train set:", mismatched_train[0])


    def tokenize_adjust_labels(self,examples):
        tokenized_inputs = self.tokenizer(
            examples["translated_tokens"],
            is_split_into_words=True,
            padding="max_length",
            max_length=128,
            truncation=True,
        )

        all_labels = []
        for i, word_ids in enumerate(tokenized_inputs.word_ids(batch_index=i) for i in range(len(examples["translated_tokens"]))):
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(examples["ner_tags"][i][word_idx])
                else:
                    # For subword tokens, we replicate the label — you could change this to -100 if needed
                    label_ids.append(examples["ner_tags"][i][word_idx])
                previous_word_idx = word_idx
            all_labels.append(label_ids)

        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    def main(self):
        if self.original == False:
            # Clean the translated data
            train = self.train.map(self.clean_translated_tokens)
            val = self.val.map(self.clean_translated_tokens)
            test = self.test.map(self.clean_translated_tokens)

            print(train[0])
            self.check_correct_translation()
        

        # Tokenization
        tokenized_train = train.map(
            lambda x: self.tokenize_adjust_labels(x),
            batched=True
        )
        tokenized_val = val.map(
            lambda x: self.tokenize_adjust_labels(x),
            batched=True
        )
        tokenized_test = test.map(
            lambda x: self.tokenize_adjust_labels(x),
            batched=True
        )

        if self.original == False:
            tokenized_train = tokenized_train.remove_columns(['tokens', 'langs', 'spans', 'translated_tokens', 'ner_tags'])
            tokenized_val = tokenized_val.remove_columns(['tokens', 'langs', 'spans', 'translated_tokens', 'ner_tags'])
            tokenized_test = tokenized_test.remove_columns(['tokens', 'langs', 'spans', 'translated_tokens', 'ner_tags'])
        elif self.original == True:
            tokenized_train = tokenized_train.remove_columns(['tokens', 'langs', 'spans', 'ner_tags'])
            tokenized_val = tokenized_val.remove_columns(['tokens', 'langs', 'spans', 'ner_tags'])
            tokenized_test = tokenized_test.remove_columns(['tokens', 'langs', 'spans', 'ner_tags'])
        
        return tokenized_train, tokenized_val, tokenized_test
