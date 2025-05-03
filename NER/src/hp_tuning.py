from transformers import Trainer, TrainingArguments
from src.utils import compute_metrics

def objective(trial, train_dataset, val_dataset, model, tokenizer, data_collator, logger):
    # Define hyperparameters to be tuned
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 6)
    weight_decay = trial.suggest_uniform("weight_decay", 0.01, 0.1)
    
    # Set training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        report_to="none",
        learning_rate=learning_rate,
    )
    
    trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )


    # Train and evaluate the model
    trainer.train()
    eval_results = trainer.evaluate()

    # Log the trial details
    logger.info(f"Trial {trial.number} | Params: {trial.params} | Metrics: {eval_results}")

    return eval_results["eval_overall_f1"]  # Return F1 score to optimize
