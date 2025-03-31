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

    def translate_long_text(self, text, tokenizer, model, max_chunk_size=450, overlap=50):
        """
        Translates long text by splitting it into overlapping chunks.
        """
        if pd.isna(text) or text.strip() == "":
            return text  # Skip empty or NaN values

        tokens = tokenizer.encode(text, return_tensors="pt", truncation=False)[0]
        
        translated_chunks = []
        for i in range(0, len(tokens), max_chunk_size - overlap):
            chunk = tokens[i:i + max_chunk_size]
            input_text = tokenizer.decode(chunk, skip_special_tokens=True)
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)

            with torch.no_grad():
                translated = model.generate(**inputs, max_length=512)
            
            translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            translated_chunks.append(translated_text)

        return " ".join(translated_chunks)  # Reassemble translated chunks


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
            df[f"translated_{col}"] = df[f"{col}"].progress_apply(lambda x: self.translate_long_text(x, tokenizer, model))

            # Save the translated dataset
            df.to_csv(OUTPUT_CSV, index=False)

        print(f"Translation completed! Translated dataset saved as {OUTPUT_CSV}")

def main():
    file_path = "HSD_nl_val.csv"
    target_columns = ["test_case"]

    Translator(file_path, "nl", "en", target_columns).translation()

if __name__ == '__main__':
    main()
