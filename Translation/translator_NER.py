import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
from datasets import load_dataset
from datasets import Dataset


class TranslatorNER():
    def __init__(self, dataset_name, source_lang, target_lang, target_columns):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.dataset_name = dataset_name
        self.target_columns = target_columns

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self. device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(self.device)
    
    def initialize_translator(self):
        # Load the translation model and tokenizer
        MODEL_NAME = f"Helsinki-NLP/opus-mt-{self.source_lang}-{self.target_lang}"  
        tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        model = MarianMTModel.from_pretrained(MODEL_NAME).to(self.device)

        return model, tokenizer

    def translate_tokens(self, tokens, tokenizer, model):
        """
        Translates a list of tokens (one row of tokens) individually and returns a list of translated tokens.
        """
        if isinstance(tokens, list):  # Ensure the tokens are in list format
            translated_tokens = []
            for token in tokens:
                if pd.isna(token) or token.strip() == "":
                    translated_tokens.append(token)  # Skip empty or NaN values
                else:
                    # Translate each token
                    inputs = tokenizer(token, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    
                    with torch.no_grad():
                        translated = model.generate(**inputs, max_length=512)
                    
                    translated_token = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
                    translated_tokens.append(translated_token)
            
            return translated_tokens
        return tokens  # If it's not a list, return as is

    def translation(self):
        # Load dataset from Hugging Face
        dataset_split = "test"
        dataset_full = load_dataset(self.dataset_name, self.source_lang, split=dataset_split)
        start_idx = 4000
        end_idx = 7000
        dataset = Dataset.from_dict(dataset_full[start_idx:end_idx])


        # Load tokenizer and model
        model, tokenizer = self.initialize_translator()

        print(f"Translating {self.dataset_name} {dataset_split} from {start_idx} to {end_idx}...")

        # For each column that needs translation
        for col in self.target_columns:
            print(f"Translating column: {col}")
            # Apply the translation function to each row (not batched)
            dataset = dataset.map(
                lambda example: {
                    f"translated_{col}": self.translate_tokens(example[col], tokenizer, model)
                },
                num_proc=1  # Use only 1 process to make sure it's row by row (not batched)
            )

        # Save the translated dataset (in this case, we'll save it as CSV or other formats if needed)
        output_file = f"{self.dataset_name}_{self.source_lang}_{dataset_split}_translated_{start_idx}-{end_idx}.json"
        dataset.to_json(output_file)

        print(f"Translation completed! Translated dataset saved as {output_file}")


def main():
    dataset_name = "wikiann"
    target_columns = ["tokens"]  # We are specifically interested in the tokens for translation

    TranslatorNER(dataset_name, "ru", "en", target_columns).translation()

if __name__ == '__main__':
    main()
