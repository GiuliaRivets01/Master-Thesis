from datasets import load_dataset, Dataset
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import re
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import evaluate
import os
import torch
import argparse
from src.config import load_config
import logging
from src.data_loader import get_datastes
import random
from src.preprocessing import Preprocessor
from src.trainer import NER_Trainer
import optuna
from src.hp_tuning import objective
from src.utils import get_class_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, help="Dataset language", default="bg")
parser.add_argument("--original", type=str, help="Original (or) or translated (tr) dataset?", default="tr")
parser.add_argument("--tuning", type=str, help="Hyperparameter tuning (hp) or fine-tuning (ft)?", default="ft")
args = parser.parse_args()

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

seed = config['general']['seed']
random.seed(seed)
log_dir = config["training_dir"]["logs_dir"]
logger = setup_logger(log_dir)

dataset = load_dataset("wikiann", "en")
label_names = dataset["train"].features["ner_tags"].feature.names

tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'], use_fast=True)
data_collator = DataCollatorForTokenClassification(tokenizer, padding=True, return_tensors="pt")
model = AutoModelForTokenClassification.from_pretrained(config['model']['base_model'], num_labels=len(label_names))

# Load dataset
train, val, test = get_datastes(config['dataset']['train_path'],
                                         config['dataset']['val_path'],
                                         config['dataset']['test_path'])


# Preprocessing
tokenized_train, tokenized_val, tokenized_test = Preprocessor(train, val, test, logger, label_names, 
                                                              tokenizer, 
                                                              config['dataset']['original'],
                                                              config['dataset']['token_col']).main()

label_ids = [0, 1, 2, 3, 4, 5, 6]
class_weights = get_class_weights(train_dataset=tokenized_train, label_list=label_ids, device=device)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

# HP Tuning
if args.tuning == "hp":
    print("Hyperparameter tuning started...")
    # Create a study object for optimization
    study = optuna.create_study(direction="maximize")  # Optimize for maximum F1 score
    # Pass the datasets and model into the optimization process
    study.optimize(lambda trial: objective(trial, tokenized_train, tokenized_val, model, tokenizer, data_collator, logger, loss_fn), n_trials=10)
    # Print the best hyperparameters found
    logger.info(f"Best hyperparameters: {study.best_params}")

# Training
if args.tuning == "ft":
    trainer, last_checkpoint = NER_Trainer(label_names, model, tokenizer, 
                                        tokenized_train, tokenized_val, data_collator, config, logger, loss_fn).main()

    trainer.train(resume_from_checkpoint=last_checkpoint)

    metrics = trainer.evaluate()
    logger.info(f"Validation metrics: {metrics}")

    test_metrics = trainer.evaluate(tokenized_test)
    logger.info(f"Test metrics: {test_metrics}")