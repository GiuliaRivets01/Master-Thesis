from transformers import TrainingArguments, Trainer

class QA_Trainer():
    def __init__(self, train_dataset, validation_dataset, config, model, tokenizer):
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.config = config
        self.model = model
        self.tokenizer = tokenizer

    def main(self):
        args = TrainingArguments(
            "bert-finetuned-squad",
            eval_strategy="no",
            save_strategy="epoch",
            learning_rate=self.config['parameters']['learning_rate'],
            per_device_train_batch_size = self.config['parameters']['per_device_train_batch_size'],
            per_device_eval_batch_size = self.config['parameters']['per_device_eval_batch_size'],
            num_train_epochs=self.config['parameters']['num_train_epochs'],
            weight_decay=self.config['parameters']['weight_decay'],
            fp16=True,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            tokenizer=self.tokenizer,
        )
        return trainer