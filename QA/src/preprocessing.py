from datasets import Dataset

class QA_Preprocessor():
    def __init__(self, train, val, test, args, tokenizer, max_length, stride):
        self.train = train
        self.val = val
        self.test = test
        self.args = args
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
    
    def adjust_dataset(self):
        self.train['id'] = self.train.index
        self.val['id'] = self.val.index
        self.test['id'] = self.test.index

        self.train['id'] = self.train['id'].astype(str)
        self.val['id'] = self.val['id'].astype(str)
        self.test['id'] = self.test['id'].astype(str)
        if self.args.language == "it":
            self.train = self.train.dropna()

        if self.args.language == "ch" or self.args.language == "it":
            cols_to_drop = ['context', 'question', 'answer', 'predicted_start', 'translated_answer', 'predicted_answer']
            cols_to_rename = {'translated_question': 'question', 'translated_context': 'context'}
        elif self.args.language == "bg" or self.args.language == "nl":
            cols_to_drop = ['predicted_start', 'translated_answer', 'predicted_answer']
            cols_to_rename = {'translated_question': 'question', 'translated_context': 'context'}
        elif self.args.language == "ru":
            cols_to_drop = ["context", "question", "answer", "translated_answer", "predicted_answer", "predicted_start"]
            cols_to_rename = {'translated_question': 'question', 'translated_context': 'context'}

        train = self.train.drop(columns=cols_to_drop)
        val = self.val.drop(columns=cols_to_drop)
        test = self.test.drop(columns=cols_to_drop)

        train = train.rename(columns=cols_to_rename)
        val = val.rename(columns=cols_to_rename)
        test = test.rename(columns=cols_to_rename)

        dataset_train = Dataset.from_pandas(train)
        dataset_val = Dataset.from_pandas(val)
        dataset_test = Dataset.from_pandas(test)

        return dataset_train, dataset_val, dataset_test
    
    def adjust_original_dataset(self):
        self.train['id'] = self.train.index
        self.val['id'] = self.val.index
        self.test['id'] = self.test.index
        self.train['id'] = self.train['id'].astype(str)
        self.val['id'] = self.val['id'].astype(str)
        self.test['id'] = self.test['id'].astype(str)

        if self.args.language == "nl":
            cols_to_drop = ['title', 'is_impossible', 'answer', 'answer_start']
        else:
            cols_to_drop = ['answer', 'answer_start']

        train = self.train.drop(columns=cols_to_drop)
        val = self.val.drop(columns=cols_to_drop)
        test = self.test.drop(columns=cols_to_drop)

        #train = train.rename(columns=cols_to_rename)
        #val = val.rename(columns=cols_to_rename)
        #test = test.rename(columns=cols_to_rename)

        dataset_train = Dataset.from_pandas(train)
        dataset_val = Dataset.from_pandas(val)
        dataset_test = Dataset.from_pandas(test)

        return dataset_train, dataset_val, dataset_test
    
    def preprocess_training_examples(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_validation_examples(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs
    
    def main(self):
        if self.args.original == 'tr':
            dataset_train, dataset_val, dataset_test = self.adjust_dataset()
        elif self.args.original == 'or':
            dataset_train, dataset_val, dataset_test = self.adjust_original_dataset()

        train_dataset = dataset_train.map(self.preprocess_training_examples,
                                        batched=True,
                                        remove_columns=dataset_train.column_names,
                                        )

        validation_dataset = dataset_val.map(self.preprocess_validation_examples,
                                            batched=True,
                                            remove_columns=dataset_val.column_names,
        )

        test_dataset = dataset_test.map(self.preprocess_validation_examples,
                                            batched=True,
                                            remove_columns=dataset_test.column_names,
        )
        return train_dataset, validation_dataset, test_dataset, dataset_val, dataset_test