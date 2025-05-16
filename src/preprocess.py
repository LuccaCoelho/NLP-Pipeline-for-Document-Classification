import spacy
import re
from sklearn.datasets import fetch_20newsgroups

def clean_text(text):
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove newline chars
    text = re.sub(r"\n", " ", text)

    # Remove page numbers
    text = re.sub(r"Page \d+", "", text)

    doc = nlp(text.lower())
    # Lemmatize and remove stop words and non-alpha tokens
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

    # Load and clean the 20 Newsgroups dataset


def load_and_clean_data():
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    cleaned = [clean_text(text) for text in newsgroups.data]

    return cleaned, newsgroups.target, newsgroups.target_names
