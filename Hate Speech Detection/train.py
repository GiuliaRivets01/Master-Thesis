from src.config import load_config
from src.data_loader import DatasetPreparer
from src.cleaner import preprocess_dataset
import pandas as pd
import argparse
from src.preprocessing import Preprocessor, HateSpeechDataset
import numpy as np
import torch
from src.trainer import TrainingInitializer
import os
import logging
from transformers import AutoTokenizer
import optuna
from src.hp_tuning import objective
import random
from src.utils import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--cleaning", type=str, help="Skip data cleaning or not?", default="yes")
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

text_col = config['dataset']['text_column']
label_col = config['dataset']['label_column']
original = config['dataset']['original']
language = config['dataset']['language']
log_dir = config["training"]["logs_dir"]
seed = config['general']['seed']
random.seed(seed)
logger = setup_logger(log_dir)

if not args.cleaning:
        # Load and prepare dataset
        train, val, test = DatasetPreparer(
                config['dataset']['train_path'],
                config['dataset']['val_path'],
                config['dataset']['test_path'],
                config['dataset']['language'], 
                text_col,
        label_col,
                config['dataset']['cols_to_drop']
        ).prepare_dataset()

        # Clean dataset
        train, val, test = preprocess_dataset(train, val, test, text_col, original, language)
        # Save cleaned dataset
        train.to_csv(config['dataset']['output_path_train'], index=False)
        val.to_csv(config['dataset']['output_path_val'], index=False)
        test.to_csv(config['dataset']['output_path_test'], index=False)


# Preprocessing and tokenization
train = pd.read_csv(config['dataset']['output_path_train'])
val = pd.read_csv(config['dataset']['output_path_val'])
test = pd.read_csv(config['dataset']['output_path_test'])
classes = np.array([0, 1])
model_path = config['model']['base_model']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
logger.info(f"Using device: {device}")

train_encodings, val_encodings, test_encodings, loss_fn = Preprocessor(train, val, test, classes, 
                                                              text_col, model_path, device
                                                              ).main()

# Create the HSD Dataset
train_dataset = HateSpeechDataset(train_encodings, train_encodings['labels'])
val_dataset = HateSpeechDataset(val_encodings, val_encodings['labels'])
test_dataset = HateSpeechDataset(test_encodings, test_encodings['labels'])

# Fine tuning
# Load the model
model = load_model(train, model_path, device)
tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])

# Hyperparameter tuning
if args.tuning == "hp":
        print("Hyperparameter tuning started...")
        # Create a study object for optimization
        study = optuna.create_study(direction="maximize")  # Optimize for maximum F1 score
        # Pass the datasets and model into the optimization process
        study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, model, tokenizer, loss_fn, logger), n_trials=10)
        # Print the best hyperparameters found
        logger.info(f"Best hyperparameters: {study.best_params}")

# Training
elif args.tuning == "ft":
        trainer, last_checkpoint = TrainingInitializer(device, config, train_dataset, 
                                                val_dataset, train, loss_fn, logger, model, tokenizer
                                                ).main()

        trainer.train(resume_from_checkpoint=last_checkpoint)

        metrics = trainer.evaluate()
        logger.info(f"Validation metrics: {metrics}")

        test_metrics = trainer.evaluate(test_dataset)
        logger.info(f"Test metrics: {test_metrics}")
