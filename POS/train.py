from src.utils import setup_logger, create_commandline_args, create_config, get_loss
from src.config import load_config
import random
import pandas as pd
from src.prepropcessing import POS_Preprocessor
from transformers import BertForTokenClassification, DataCollatorForTokenClassification, BertTokenizerFast
from src.trainer import POS_Trainer
import optuna
from src.hp_tuning import objective
import numpy as np
import torch
import ast

def adjust_original_sentences(pos_df, logger):
    sentences_list = []
    count = 0
    for idx, row in pos_df.iterrows():
        words = row['Sentence'].split()
        pos_tags = row['UPOSS'].split()
        sentence = []
        if len(words) != len(pos_tags):
            count += 1
        else:
            for i in range(len(words)):
                sentence.append((words[i], pos_tags[i]))
        sentences_list.append(sentence)
    logger.info(f"Dropping examples where original text does not match original UPOSS. Dropped {count} examples.")
    return sentences_list

def get_stats(train_sentences, val_sentences, test_sentences, logger):
    logger.info("Tagged sentences in train set: ", len(train_sentences))
    logger.info("Tagged words in train set:", len([item for sublist in train_sentences for item in sublist]))
    logger.info(40*'=')
    logger.info("Tagged sentences in dev set: ", len(val_sentences))
    logger.info("Tagged words in dev set:", len([item for sublist in val_sentences for item in sublist]))
    logger.info(40*'=')
    logger.info("Tagged sentences in test set: ", len(test_sentences))
    logger.info("Tagged words in test set:", len([item for sublist in test_sentences for item in sublist]))
    logger.info(40*'*')
    logger.info("Total sentences in dataset:", len(train_sentences)+len(val_sentences)+len(test_sentences))

def get_model(n_tags):
    model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=n_tags)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    return model.to(device), data_collator

def get_model_original(n_tags, config):
    model = BertForTokenClassification.from_pretrained(config['model']['base_model'], num_labels=n_tags)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    return model.to(device), data_collator

args = create_commandline_args()
config = create_config(args)

seed = config['general']['seed']
random.seed(seed)
logger = setup_logger(config['training']['logs_dir'])
logger.info(f"Dataset {config['dataset']['name']} | Language: {config['dataset']['language']} | Original: {config['dataset']['original']}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load translated and annotated dataset
pos_train = pd.read_csv(config['dataset']['train_path_annotated'])
pos_val = pd.read_csv(config['dataset']['val_path_annotated'])
pos_test = pd.read_csv(config['dataset']['test_path_annotated'])

if args.original == 'tr':
    parsed_data_train = [ast.literal_eval(sentence) for sentence in pos_train['annotated_Sentence']]
    parsed_data_val = [ast.literal_eval(sentence) for sentence in pos_val['annotated_Sentence']]
    parsed_data_test = [ast.literal_eval(sentence) for sentence in pos_test['annotated_Sentence']]

    # Extract the 'annotated_Sentence' column into a list
    sentences_list_train = parsed_data_train
    sentences_list_val = parsed_data_val
    sentences_list_test = parsed_data_test
    count = 1

elif args.original == 'or':
    sentences_list_train = adjust_original_sentences(pos_train, logger)
    sentences_list_val = adjust_original_sentences(pos_val, logger)
    sentences_list_test = adjust_original_sentences(pos_test, logger)

get_stats(sentences_list_train, sentences_list_val, sentences_list_test, logger)

# PREPROCESSING
tokenizer = BertTokenizerFast.from_pretrained(config['model']['base_model'])
train_dataset, val_dataset, test_dataset, n_tags, tag2int  = POS_Preprocessor(config, 
                                                                     sentences_list_train, 
                                                                     sentences_list_val, 
                                                                     sentences_list_test,
                                                                     tokenizer,
                                                                     logger
                                                                     ).main(return_tag_dicts=True)

if args.original == 'tr':
    model, data_collator = get_model(n_tags)
elif args.original == 'or':
    model, data_collator = get_model_original(n_tags, config)

loss_fn = get_loss(sentences_list_train, tag2int, device)

if args.tuning == "hp":
    print("Hyperparameter tuning started...")
    # Create a study object for optimization
    study = optuna.create_study(direction="maximize")  # Optimize for maximum F1 score
    # Pass the datasets and model into the optimization process
    study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, model, tokenizer, data_collator, logger, config, loss_fn), n_trials=10)
    # Print the best hyperparameters found
    logger.info(f"Best hyperparameters: {study.best_params}")


elif args.tuning == "ft":
    trainer = POS_Trainer(model, data_collator, train_dataset, val_dataset, config, loss_fn).main()

    trainer.train()

    metrics = trainer.evaluate()
    logger.info(f"Validation metrics: {metrics}")

    test_metrics = trainer.evaluate(test_dataset)
    logger.info(f"Test metrics: {test_metrics}")
