from transformers import Trainer, TrainingArguments
from src.utils import compute_metrics

def objective(trial, train_dataset, validation_dataset, dataset_val, model, tokenizer, logger, config):
    # Define hyperparameters to be tuned
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 6)
    weight_decay = trial.suggest_uniform("weight_decay", 0.01, 0.1)
    
    # Set training arguments
    training_args = TrainingArguments(
        "bert-finetuned-squad",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        eval_strategy="no",
        save_strategy="epoch",
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
    )


    # Train and evaluate the model
    trainer.train()

    predictions, _, _ = trainer.predict(validation_dataset)
    start_logits, end_logits = predictions
    res = compute_metrics(start_logits, end_logits, validation_dataset, dataset_val)

    # Log the trial details
    logger.info(f"Trial {trial.number} | Params: {trial.params} | Metrics: {res}")

    return res["f1"]  # Return F1 score to optimize