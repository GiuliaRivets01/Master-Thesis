import pyconll
import pandas as pd

class PrepareForTranslation:
    def __init__(self, train_path, val_path, test_path, lang):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.lang = lang

    def adjust_dataset(self, file_path):
        train = pyconll.load_from_file(file_path)

        all_sentences = []
        all_sentences_tokens = []
        all_sentences_upos = []
        for sentence in train:
            single_sentence = []
            sentence_tokens = []
            sentence_upos = []
            for token in sentence:
                if token.upos == None:
                    token_upos = "None"
                else:
                    token_upos = token.upos

                single_sentence.append((token.form, token_upos))
                sentence_tokens.append(token.form)
                sentence_upos.append(token_upos)
            all_sentences.append(single_sentence)
            all_sentences_tokens.append(sentence_tokens)
            all_sentences_upos.append(sentence_upos)
        return all_sentences, all_sentences_tokens, all_sentences_upos

    def create_custom_dataframe(self, all_sentences_tokens, all_sentences_upos):

        # Get a list of all the sentences
        sent = []
        for i in all_sentences_tokens:
            phrase = ' '.join(i)
            sent.append(phrase)

        # Get a list of all upos
        uposss = []
        for i in all_sentences_upos:
            phrase = ' '.join(i)
            uposss.append(phrase)
        return sent, uposss

    def main(self):
        # Get the list of lists
        # Each element of the list represents a sentence
        # Each sentence is a list of tuples (word, pos)
        all_sentences_train, all_sentences_tokens_train, all_sentences_upos_train = self.adjust_dataset(self.train_path)
        all_sentences_val, all_sentences_tokens_val, all_sentences_upos_val = self.adjust_dataset(self.val_path)
        all_sentences_test, all_sentences_tokens_test, all_sentences_upos_test = self.adjust_dataset(self.test_path)

        # Create a dataframe with these lists of sentences
        train = pd.DataFrame({"Sentence": all_sentences_train})
        val = pd.DataFrame({"Sentence": all_sentences_val})
        test = pd.DataFrame({"Sentence": all_sentences_test})

        # Create sentences of strings, where tokens are separated by whitespaces
        sent_train, uposss_train = self.create_custom_dataframe(all_sentences_tokens_train, all_sentences_upos_train)
        sent_val, uposss_val = self.create_custom_dataframe(all_sentences_tokens_val, all_sentences_upos_val)
        sent_test, uposss_test = self.create_custom_dataframe(all_sentences_tokens_test, all_sentences_upos_test)

        # Add this to the dataframe
        df_train = pd.DataFrame({"Sentence": sent_train, "UPOSS": uposss_train})
        df_val = pd.DataFrame({"Sentence": sent_val, "UPOSS": uposss_val})
        df_test = pd.DataFrame({"Sentence": sent_test, "UPOSS": uposss_test})

        # Save the dataframes, which are not ready for translation
        df_train.to_csv(f"POS_{self.lang}_train_adjusted.csv", index=False)
        df_val.to_csv(f"POS_{self.lang}_val_adjusted.csv", index=False)
        df_test.to_csv(f"POS_{self.lang}_test_adjusted.csv", index=False)

        print(f"Train saved at: POS_{self.lang}_train_adjusted.csv")
        print(f"Val saved at: POS_{self.lang}_val_adjusted.csv")
        print(f"Test saved at: POS_{self.lang}_test_adjusted.csv")