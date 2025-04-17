from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from transformers import BertForSequenceClassification

def load_model(df, model_path, device):
    num_labels = len(set(df['labels']))
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    # "bert-base-uncased"
    return model.to(device)

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

def compute_metrics(p):
    preds = p.predictions.argmax(-1)  # Get predicted labels
    labels = p.label_ids  # True labels

    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)

    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def get_trainer(loss_fn, model, training_args, tokenizer, train_dataset, val_dataset):
    if loss_fn != None:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            loss_fn=loss_fn
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
    return trainer