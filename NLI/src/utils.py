from sklearn.metrics import f1_score
import os
import logging
import argparse
from src.config import load_config
from transformers import Trainer
from sklearn.metrics import f1_score, precision_score, recall_score

def setup_logger(output_dir, log_name='training.log'):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_name)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Avoid duplicate logs
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        # Optional: also log to console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def create_commandline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, help="Dataset language ('bg', 'ch', 'nl', 'it', or 'ru')", default="bg")
    parser.add_argument("--original", type=str, help="Original ('or') or translated ('tr') dataset?", default="tr")
    parser.add_argument("--tuning", type=str, help="Hyperparameter tuning ('hp') or fine-tuning ('ft')?", default="ft")
    args = parser.parse_args()
    return args

def create_config(args):
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
    return config


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    return {
        "accuracy": (predictions == labels).mean(),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
        "micro_f1": f1_score(labels, predictions, average="micro"),
        "macro_precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, predictions, average="macro", zero_division=0),
        "weighted_precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "weighted_recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "micro_precision": precision_score(labels, predictions, average="micro", zero_division=0),
        "micro_recall": recall_score(labels, predictions, average="micro", zero_division=0)
    }

class CustomTrainer(Trainer):
    def __init__(self, *args, loss_fn=None, **kwargs):
      super().__init__(*args, **kwargs)
      self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")  # Get the true labels
        outputs = model(**inputs)  # Get model outputs
        logits = outputs.get("logits")  # Get the logits (raw predictions)

        # Compute the weighted loss
        loss = self.loss_fn(logits, labels)

        # Return loss, and optionally, the outputs (for debugging/metrics)
        return (loss, outputs) if return_outputs else loss