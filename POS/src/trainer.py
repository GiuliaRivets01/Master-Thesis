import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from transformers import TrainingArguments, Trainer
import os
from src.utils import CustomTrainer, compute_metrics

class POS_Trainer:
    def __init__(self, model, data_collator, train_dataset, val_dataset, config, loss_fn):
        self.model = model
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.loss_fn = loss_fn

    def main(self):
        training_args = TrainingArguments(
            output_dir=self.config['general']['output_dir'],
            eval_strategy=self.config['general']['evaluation_strategy'],
            eval_steps=self.config['general']['eval_steps'],
            save_strategy=self.config['general']['save_strategy'],
            save_steps=self.config['general']['save_steps'],
            save_total_limit=self.config['general']['save_total_limit'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['batch_size'],
            num_train_epochs=self.config['training']['epochs'],
            weight_decay=self.config['training']['weight_decay'],
            learning_rate=self.config['training']['learning_rate'],
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
            print(f"Resuming training from checkpoint: {last_checkpoint}")
        else:
            print("No checkpoint found, starting from scratch.")


        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
            loss_fn=self.loss_fn
        )
        return trainer