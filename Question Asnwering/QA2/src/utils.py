import os
import logging
import argparse
from src.config import load_config
import evaluate
from tqdm import tqdm
import collections
import numpy as np

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

def create_commandline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, help="Dataset language ('bg', 'ch', 'nl', 'it', or 'ru')", default="bg")
    parser.add_argument("--original", type=str, help="Original ('or') or translated ('tr') dataset?", default="tr")
    parser.add_argument("--tuning", type=str, help="Hyperparameter tuning ('hp') or fine-tuning ('ft')?", default="ft")
    args = parser.parse_args()
    return args

def create_config(args):
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
    return config


n_best = 20
max_answer_length = 30

metric = evaluate.load("squad")

def safe_clean_answer(answer):
    if not answer or not isinstance(answer, dict):
        return {"text": [""], "answer_start": [0]}
    if not answer.get("text") or answer["text"][0] is None:
        return {"text": [""], "answer_start": [0]}
    return answer

def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [
    {"id": ex["id"], "answers": safe_clean_answer(ex["answers"])}
    for ex in examples
]
    predicted_answers = [
    {"id": pred["id"], "prediction_text": pred.get("prediction_text", "") or ""}
    for pred in predicted_answers
]
    print("Theoretical answers: ", theoretical_answers)
    print("Predicted answers: ", predicted_answers)
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)
