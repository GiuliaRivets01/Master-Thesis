import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from src.preprocessing import NLI_Preprocessor
from src.utils import compute_metrics, create_commandline_args, create_config, CustomTrainer, setup_logger
from src.config import load_config
import torch
import random
import os

def get_last_checkpoints(training_args, logger):
    checkpoint_dir = training_args.output_dir
    last_checkpoint = None

    if os.path.isdir(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1])
    if last_checkpoint:
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}")
    else:
        logger.info("No checkpoint found, starting from scratch.")

    if loss != None:
        logger.info(f"Loss computed.")
    else:
        logger.info("Loss not computed.")
    return last_checkpoint

args = create_commandline_args()
config = create_config(args)
random.seed(42)
logger = setup_logger(config['training']['logs_dir'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = config['model']['base_model']
tokenizer = AutoTokenizer.from_pretrained(model_name)

train = pd.read_csv(config['dataset']['train_path'])
val = pd.read_csv(config['dataset']['val_path'])
test = pd.read_csv(config['dataset']['test_path'])
print(train.columns)
print(val.columns)
print(test.columns)
input("C")
label_col = config['dataset']['label_col']

train["labels"] = train[label_col]
val["labels"] = val[label_col]
test["labels"] = test[label_col]

if args.original == 'or':
    dataset_type = 'Original'
else:
    dataset_type = 'Translated'
logger.info(f"Dataset: {config['dataset']['name']} | language: {config['dataset']['language']} | {dataset_type}")
logger.info(f"Model: {model_name}")
tokenized_train, tokenized_val, tokenized_test, loss = NLI_Preprocessor(tokenizer, 
                                                                  config, 
                                                                  train, 
                                                                  val, 
                                                                  test,
                                                                  device).main()
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.05,
    load_best_model_at_end=True,
    report_to="none",
)

# Look for the latest checkpoint
last_checkpoint = get_last_checkpoints(training_args, logger)

if loss != None:
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        loss_fn=loss
    )
else:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

trainer.train(resume_from_checkpoint=last_checkpoint)

result = trainer.evaluate()
logger.info(f"Validation results: {result}")

result_test = trainer.evaluate(tokenized_test)
logger.info(f"Test results: {result_test}")