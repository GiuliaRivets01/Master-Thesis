import pandas as pd
import spacy
import re

nlp = spacy.load("en_core_web_sm")

class PrepareForTraining:
    def __init__(self, config, lang):
        self.config = config
        self.lang = lang

    def separate_punctuation(self, text):
        if pd.isna(text) or text.strip() == "":
            return text  # Return unchanged if empty

        # Do not modify URLs (words starting with 'http')
        words = text.split()  # Split text into words
        processed_words = []

        for word in words:
            if word.startswith("http"):
                processed_words.append(word)  # Keep URLs unchanged
            else:
                # Add spaces around punctuation, but keep apostrophes as part of words
                word = re.sub(r"([.,!?;:()\"])", r" \1 ", word)  # Add spaces around punctuation
                word = re.sub(r"\s+", " ", word).strip()  # Remove extra spaces
                processed_words.append(word)

        return " ".join(processed_words)  # Reconstruct sentence

    # Function to annotate POS tags using Universal Dependencies
    def annotate_pos_tags(self, text):
        if pd.isna(text) or text.strip() == "":
            return text  # Return unchanged if empty

        # Process the text with spaCy
        doc = nlp(text)

        # Create a list of tuples (word, POS tag)
        annotated_words = [(token.text, token.pos_) for token in doc]

        return annotated_words

    def main(self):
        pos_train = pd.read_csv(self.config['dataset']['train_path_raw'])
        pos_val= pd.read_csv(self.config['dataset']['val_path_raw'])
        pos_test = pd.read_csv(self.config['dataset']['test_path_raw'])

        # Punctuation marks are not on their own, but they are part of the previous word
        # So we separate them
        pos_train["translated_Sentence"] = pos_train["translated_Sentence"].apply(self.separate_punctuation)
        pos_val["translated_Sentence"] = pos_val["translated_Sentence"].apply(self.separate_punctuation)
        pos_test["translated_Sentence"] = pos_test["translated_Sentence"].apply(self.separate_punctuation)


        # Add column "annotated_Sentence", which will consists in tuples: (word, POS tag)
        pos_train["annotated_Sentence"] = pos_train["translated_Sentence"].apply(self.annotate_pos_tags)
        pos_val["annotated_Sentence"] = pos_val["translated_Sentence"].apply(self.annotate_pos_tags)
        pos_test["annotated_Sentence"] = pos_test["translated_Sentence"].apply(self.annotate_pos_tags)

        pos_train.to_csv(self.config['dataset']['train_path_annotated'], index=False)
        pos_val.to_csv(self.config['dataset']['val_path_annotated'], index=False)
        pos_test.to_csv(self.config['dataset']['test_path_annotated'], index=False)
        print(f"Dataset splits have been annotated and saved to:\n{self.config['dataset']['train_path_annotated']}\n{self.config['dataset']['val_path_annotated']}\n{self.config['dataset']['test_path_annotated']}")
