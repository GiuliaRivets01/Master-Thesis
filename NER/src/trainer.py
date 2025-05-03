import evaluate 
import numpy as np
from transformers import TrainingArguments, Trainer
import os
import yaml
from src.utils import compute_metrics

class NER_Trainer():
    def __init__(self, label_names, model, tokenizer, train_dataset, validation_dataset, data_collator, config, logger):
        self.label_names = label_names
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.data_collator = data_collator
        self.config = config
        self.logger = logger
    
    def main(self):
        training_args_1 = self.config['training']
        training_args_2 = self.config['parameters']
        training_args_1.update(training_args_2)
        self.logger.info(f"Training args: {training_args_1}")

        training_args  = TrainingArguments(**training_args_1)

        # Look for the latest checkpoint
        checkpoint_dir = training_args.output_dir
        last_checkpoint = None

        if os.path.isdir(checkpoint_dir):
            checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                last_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1])
        if last_checkpoint:
            print(f"Resuming training from checkpoint: {last_checkpoint}")
        else:
            print("No checkpoint found, starting from scratch.")

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )

        return trainer, last_checkpoint
