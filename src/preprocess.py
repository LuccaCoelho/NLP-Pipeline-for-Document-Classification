import spacy
import re

def tokenizer(text):
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    # Remove non-ASCII
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    doc = nlp(text)

    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and token.is_alpha:
            tokens.append(token)

    return tokens

def process_dict(doc_dict):
    processed = {}

    for name, text in doc_dict.items():
        tokenized_text = tokenizer(text)
        processed[name] = tokenized_text

    return processed
