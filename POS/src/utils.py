import logging
import os
import argparse
from src.config import load_config
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import torch
from transformers import Trainer

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
    predictions = np.argmax(logits, axis=-1)

    # Flatten the labels and predictions to calculate overall metrics
    flattened_labels = labels.flatten()
    flattened_predictions = predictions.flatten()

    # Remove -100 (which represents padding) from labels and predictions
    non_padding_idx = flattened_labels != -100
    flattened_labels = flattened_labels[non_padding_idx]
    flattened_predictions = flattened_predictions[non_padding_idx]

    precision, recall, f1, support = precision_recall_fscore_support(
        flattened_labels, flattened_predictions, average='macro'
    )

    accuracy = (flattened_labels == flattened_predictions).mean()

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

def compute_pos_class_weights(sentences_list_train, tag2int, device):
    non_pad_tag2int = {k: v for k, v in tag2int.items() if k != '-PAD-'}
    all_tags = [
        non_pad_tag2int[tag]
        for sentence in sentences_list_train
        for _, tag in sentence
        if tag != '-PAD-'
    ]
    classes = np.array(list(non_pad_tag2int.values()))
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=all_tags)
    # Insert a dummy weight for the PAD class at index 0
    weights = np.insert(weights, 0, 0.0)
    return torch.tensor(weights, dtype=torch.float).to(device)

def get_loss(sentences_list_train, tag2int, device):
    class_weights = compute_pos_class_weights(sentences_list_train, tag2int, device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    return loss_fn

class CustomTrainer(Trainer):
    def __init__(self, *args, loss_fn=None, **kwargs):
      super().__init__(*args, **kwargs)
      self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")           # shape: [batch_size, seq_len]
        outputs = model(**inputs)               # output includes logits
        logits = outputs.get("logits")          # shape: [batch_size, seq_len, num_labels]

        # Reshape for CrossEntropyLoss
        logits = logits.view(-1, logits.shape[-1])  # [batch_size * seq_len, num_labels]
        labels = labels.view(-1)                    # [batch_size * seq_len]

        # Optional: mask out -100 labels (used for padding usually)
        if (labels == -100).any():  # Hugging Face convention for ignoring tokens
            loss = self.loss_fn(logits, labels)
        else:
            loss = self.loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss