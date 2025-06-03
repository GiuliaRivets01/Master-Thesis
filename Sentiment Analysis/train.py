from src.utils import create_commandline_args, create_config, setup_logger
import random
import torch
from src.data_loader import SA_DataLoader
from src.preprocessing import SA_Preprocessor
from transformers import AutoTokenizer, BertForSequenceClassification
import optuna
from src.hp_tuning import objective
from src.trainer import SA_Trainer

def load_model(df, config, device):
    num_labels = len(set(df['labels']))
    model = BertForSequenceClassification.from_pretrained(config['model']['base_model'], num_labels=num_labels)
    return model.to(device)

args = create_commandline_args()
config = create_config(args)

seed = config['general']['seed']
random.seed(seed)
logger = setup_logger(config['training']['logs_dir'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device {device}")

# Print info
logger.info(f"Dataset {config['dataset']['name']} | Language: {config['dataset']['language']} | Original: {config['dataset']['original']}")

# Load Data
train, val, test = SA_DataLoader(config, logger).main()
print(f"Train: {len(train)}, val: {len(val)}, test: {len(test)}")
input("C")

if config['dataset']['language'] == 'it':
    sentiment_map = {'POSITIVE': 0, 'NEGATIVE': 1, 'NEUTRAL': 2, 'MIXED': 3}
    train['labels'] = train['labels'].map(sentiment_map)
    val['labels'] = val['labels'].map(sentiment_map)
    test['labels'] = test['labels'].map(sentiment_map)

    train = train.dropna(subset=['labels'])
    train['labels'] = train['labels'].astype(int)
    val = val.dropna(subset=['labels'])
    val['labels'] = val['labels'].astype(int)
    test = test.dropna(subset=['labels'])
    test['labels'] = test['labels'].astype(int)

if config['dataset']['language'] == 'ru':
    sentiment_map = {'positive': 0, 'negative': 1, 'neautral': 2}
    # Replace sentiment values with numbers
    train['labels'] = train['labels'].map(sentiment_map)
    val['labels'] = val['labels'].map(sentiment_map)
    test['labels'] = test['labels'].map(sentiment_map)

# Preprocess data
tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])
train_dataset, val_dataset, test_dataset, loss = SA_Preprocessor(train, val, test, config, device, tokenizer, args, logger).main()

# Load model
model = load_model(train, config, device)

# Hyperparameter tuning
if args.tuning == 'hp':
    print("Hyperparameter tuning started...")
    # Create a study object for optimization
    study = optuna.create_study(direction="maximize")  # Optimize for maximum F1 score
    # Pass the datasets and model into the optimization process
    study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, model, tokenizer, loss, logger), n_trials=10)
    # Print the best hyperparameters found
    logger.info(f"Best hyperparameters: {study.best_params}")

# Training
elif args.tuning == 'ft':
        logger.info(f"Fine-tuning BERT...")
        trainer, last_checkpoint = SA_Trainer(device, config, train_dataset, 
                                                val_dataset, train, loss, logger, model, tokenizer
                                                ).main()

        trainer.train(resume_from_checkpoint=last_checkpoint)

        metrics = trainer.evaluate()
        logger.info(f"Validation metrics: {metrics}")

        test_metrics = trainer.evaluate(test_dataset)
        logger.info(f"Test metrics: {test_metrics}")
