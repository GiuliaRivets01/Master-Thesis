from transformers import Trainer, TrainingArguments
import os
from src.config import load_config
import argparse
import logging
from src.utils import compute_metrics, CustomTrainer, get_trainer

parser = argparse.ArgumentParser()
parser.add_argument("--cleaning", type=str, help="Skip data cleaning or not?", default="yes")
parser.add_argument("--language", type=str, help="Dataset language", default="bg")
parser.add_argument("--original", type=str, help="Original (or) or translated (tr) dataset?", default="tr")
parser.add_argument("--tuning", type=str, help="Hyperparameter tuning (hp) or fine-tuning (ft)?", default="ft")
args = parser.parse_args()

if args.original == "or":
        if args.language == "bg":
                config = load_config("configs/base.yaml", "configs/Bulgarian_original.yaml")
        elif args.language == "ch":
                config = load_config("configs/base.yaml", "configs/Chinese_original.yaml")
        elif args.language == "nl":
                config = load_config("configs/base.yaml", "configs/Dutch_original.yaml")
        elif args.language == "it":
                config = load_config("configs/base.yaml", "configs/Italian_original.yaml")
        elif args.language == "ru":
                config = load_config("configs/base.yaml", "configs/Russian_original.yaml")

elif args.original == "tr":
        if args.language == "bg":
                config = load_config("configs/base.yaml", "configs/Bulgarian_translated.yaml")
        elif args.language == "ch":
                config = load_config("configs/base.yaml", "configs/Chinese_translated.yaml")
        elif args.language == "nl":
                config = load_config("configs/base.yaml", "configs/Dutch_translated.yaml")
        elif args.language == "it":
                config = load_config("configs/base.yaml", "configs/Italian_translated.yaml")
        elif args.language == "ru":
                config = load_config("configs/base.yaml", "configs/Russian_translated.yaml")


class TrainingInitializer():
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
            learning_rate=self.config['training']['learning_rate'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['batch_size'],
            num_train_epochs=self.config['training']['epochs'],
            weight_decay=self.config['training']['weight_decay'],
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
