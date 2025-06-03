from transformers import TrainingArguments
import os
import argparse
import logging
from src.utils import get_trainer


class SA_Trainer():
    def __init__(self, device, config, train_dataset, val_dataset, train_df, loss_fn, logger, model, tokenizer):
        self.device = device
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_df = train_df
        self.loss_fn = loss_fn
        self.logger = logger
        self.model = model
        self.tokenizer = tokenizer

    def main(self):
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            learning_rate=1e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.05,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            report_to="none"  # Disables wandb logging
        )

        # Look for the latest checkpoint
        checkpoint_dir = training_args.output_dir
        last_checkpoint = None

        if os.path.isdir(checkpoint_dir):
            checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                last_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1])
        if last_checkpoint:
            self.logger.info(f"Resuming training from checkpoint: {last_checkpoint}")
        else:
            self.logger.info("No checkpoint found, starting from scratch.")

        trainer = get_trainer(self.loss_fn, self.model, training_args, self.tokenizer, self.train_dataset, self.val_dataset)
        return trainer, last_checkpoint