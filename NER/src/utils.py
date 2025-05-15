import evaluate
import numpy as np
from transformers import Trainer
from datasets import load_dataset
import torch
from sklearn.utils.class_weight import compute_class_weight

metric = evaluate.load("seqeval")

dataset = load_dataset("wikiann", "en")
label_names = dataset["train"].features["ner_tags"].feature.names

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    flattened_results = {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "overall_accuracy": results["overall_accuracy"],
    }
    for k in results.keys():
        if(k not in flattened_results.keys()):
            flattened_results[k+"_f1"]=results[k]["f1"]

    return flattened_results


def get_class_weights(train_dataset, label_list, ignore_index=-100, device="cpu"):
    """
    Compute class weights for NER task.

    Args:
        train_dataset: A Hugging Face Dataset object where each example has 'labels' field (list of ints).
        label_list: List of all unique label indices (e.g. list(range(len(label_names)))).
        ignore_index: Index used to mask tokens for loss computation (default -100).
        device: Device to move the tensor to.

    Returns:
        torch.tensor of class weights
    """
    # Flatten all label sequences, removing ignored indices
    all_labels = [
        label for example in train_dataset["labels"] for label in example if label != ignore_index
    ]

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(label_list),
        y=np.array(all_labels)
    )

    return torch.tensor(class_weights, dtype=torch.float).to(device)


class CustomTrainer(Trainer):
    def __init__(self, *args, loss_fn=None, **kwargs):
      super().__init__(*args, **kwargs)
      self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")  # Get the true labels
        outputs = model(**inputs)  # Get model outputs
        logits = outputs.get("logits")  # Get the logits (raw predictions)
        logits = logits.view(-1, logits.size(-1))  # Shape: [16*128, 7]
        labels = labels.view(-1)  # Shape: [16*128]

        # Compute the weighted loss
        loss = self.loss_fn(logits, labels)

        # Return loss, and optionally, the outputs (for debugging/metrics)
        return (loss, outputs) if return_outputs else loss
