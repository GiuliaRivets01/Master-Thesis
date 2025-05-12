# Master Thesis Computer Science - AI: Enhancing NLP for Low-Resource Language with Translation
BERT models are widely used for processing unstructured text across various languages and fields.
However, creating high-quality BERT models for non-English languages often demands extensive
computational resources and large datasets, which are not always available for minority languages.
A promising alternative to developing language-specific BERT models is to translate non-English text
into English and then fine-tune English BERT models. Initial studies have shown that this approach
can produce comparable or even superior results to native-language BERT models. For example, in
clinical contexts, fine-tuning English BERT models on translated Dutch clinical texts has demonstrated
comparable performance with Dutch-specific models trained on native data [4]. Similar outcomes
were observed in another study, where German clinical text translated to English and processed with
English BERT models outperformed the use of German-specific BERT models [1].
This translation-based approach offers potential benefits, as English BERT models are typically
trained on larger and more diverse datasets, which may enable better generalization and robustness.
This could be particularly valuable for minority languages, which often lack extensive datasets for
model training. Despite these promising initial findings, research on using translation to extend
English BERT models to non-English tasks remains limited, with most NLP efforts focused on building
separate models for each language. This master’s thesis seeks to examine whether a translation- based approach can consistently yield enhanced performance across various NLP tasks in different
minority languages.