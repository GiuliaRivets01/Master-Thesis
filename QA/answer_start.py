from transformers import pipeline
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

# Use a model that supports Italian
qa_pipeline = pipeline("question-answering", model="deepset/xlm-roberta-large-squad2")

def get_answer_start_original(question, context):
    try:
        result = qa_pipeline(question=question, context=context)
        answer = result['answer']
        start = result['start']
        return answer, start
    except:
        return None, None


train_it_or_2 = pd.read_csv("data/Italian/original/raw/QA_it_train_original.csv")
val_it_or_2 = pd.read_csv("data/Italian/original/raw/QA_it_val_original.csv")
test_it_or_2 = pd.read_csv("data/Italian/original/raw/QA_it_test_original.csv")

train_it_or_2[['predicted_answer', 'predicted_start']] = train_it_or_2.progress_apply(
    lambda row: pd.Series(get_answer_start_original(row['question'], row['context'])), axis=1
)

val_it_or_2[['predicted_answer', 'predicted_start']] = val_it_or_2.progress_apply(
    lambda row: pd.Series(get_answer_start_original(row['question'], row['context'])), axis=1
)

test_it_or_2[['predicted_answer', 'predicted_start']] = test_it_or_2.progress_apply(
    lambda row: pd.Series(get_answer_start_original(row['question'], row['context'])), axis=1
)

train_it_or_2.to_csv("train_res_it.csv", index=False)
val_it_or_2.to_csv("val_res_it.csv", index=False)
test_it_or_2.to_csv("test_res_it.csv", index=False)