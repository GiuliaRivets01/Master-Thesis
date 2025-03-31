import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import torch
from tqdm import tqdm

class Translator():
    def __init__(self, file_path, source_lang, target_lang, target_columns):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.file_path = file_path
        self.target_columns = target_columns
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def initialize_translator(self):
        # Load the translation model and tokenizer
        MODEL_NAME = f"Helsinki-NLP/opus-mt-{self.source_lang}-{self.target_lang}"  
        tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        model = MarianMTModel.from_pretrained(MODEL_NAME).to(self.device)
        return model, tokenizer

    def translate_token(self, token, tokenizer, model):
        """
        Translates a single token.
        """
        if pd.isna(token) or token.strip() == "":
            return token  # Return unchanged if token is empty or NaN

        inputs = tokenizer(token, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            translated = model.generate(**inputs, max_length=10)  # Limit token length

        translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        return translated_text

    def translate_sentence(self, sentence, tokenizer, model):
        """
        Translates a sentence word-by-word to preserve token order.
        """
        tokens = sentence.split()  # Tokenize by whitespace
        translated_tokens = [self.translate_token(token, tokenizer, model) for token in tokens]
  
        return " ".join(translated_tokens)  # Reconstruct the sentence

    def translation(self):
        # Load input file
        INPUT_CSV = self.file_path
        OUTPUT_CSV = f"{self.file_path[:-4]}_translated.csv"
        df = pd.read_csv(INPUT_CSV)

        # Load tokenizer and model
        model, tokenizer = self.initialize_translator()

        print(f"Translating {INPUT_CSV}...")
        for col in self.target_columns:
            print(f"Translating column: {col}")
            tqdm.pandas(desc=f"Translating {col}")

            # Translate each sentence word-by-word
            df[f"translated_{col}"] = df[f"{col}"].progress_apply(lambda x: self.translate_sentence(str(x), tokenizer, model))

            # Save the translated dataset
            df.to_csv(OUTPUT_CSV, index=False)

        print(f"Translation completed! Translated dataset saved as {OUTPUT_CSV}")

def main():
    file_path = "POS_it_test_adjusted.csv"
    target_columns = ["Sentence"]  # Column containing sentences

    Translator(file_path, "it", "en", target_columns).translation()

if __name__ == '__main__':
    main()
