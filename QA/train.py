from src.data_loader import DataLoader
from src.utils import create_commandline_args, create_config, setup_logger, compute_metrics
import torch
import random
from transformers import AutoTokenizer
from src.preprocessing import QA_Preprocessor
from transformers import AutoModelForQuestionAnswering
from src.trainer import QA_Trainer
import optuna
from src.hp_tuning import objective

args = create_commandline_args()
config = create_config(args)

seed = config['general']['seed']
random.seed(seed)
logger = setup_logger(config['training_dir']['logs_dir'])
logger.info(f"Dataset {config['dataset']['name']} | Language: {config['dataset']['language']} | Original: {config['dataset']['original']}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. Load the dataset
train_path = config['dataset']['cleaned_train_path']
val_path = config['dataset']['cleaned_val_path']
test_path = config['dataset']['cleaned_test_path']

train, val, test = DataLoader(train_path, val_path, test_path).main()

model_checkpoint = config['model']['base_model']
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint).to(device)

train_dataset, validation_dataset, test_dataset, dataset_val, dataset_test = QA_Preprocessor(train, val, test, args, tokenizer, 
                                                                config['model']['max_length'], 
                                                                config['model']['stride']).main()


if args.tuning == "hp":
    print("Hyperparameter tuning started...")
    # Create a study object for optimization
    study = optuna.create_study(direction="maximize")  # Optimize for maximum F1 score
    # Pass the datasets and model into the optimization process
    study.optimize(lambda trial: objective(trial, train_dataset, validation_dataset, dataset_val, model, tokenizer, logger, config), n_trials=10)
    # Print the best hyperparameters found
    logger.info(f"Best hyperparameters: {study.best_params}")

elif args.tuning == "ft":
    trainer = QA_Trainer(train_dataset, validation_dataset, config, model, tokenizer).main()
    trainer.train()

    predictions, _, _ = trainer.predict(validation_dataset)
    start_logits, end_logits = predictions
    res = compute_metrics(start_logits, end_logits, validation_dataset, dataset_val)
    logger.info(f"Validation results: {res}")

    predictions_test, _, _ = trainer.predict(test_dataset)
    start_logits_test, end_logits_test = predictions_test
    res_test = compute_metrics(start_logits_test, end_logits_test, test_dataset, dataset_test)
    logger.info(f"Test results: {res_test}")
