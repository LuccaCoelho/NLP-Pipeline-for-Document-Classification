from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Option 1: TF-IDF

def extract_tfidf_features(data, max_features = 500):
    vectorizer = TfidfVectorizer(max_features=max_features)

    return vectorizer.fit_transform(data), vectorizer

# Option 2: Sentence embeddings using pre-trained transformer

def extract_sentence_embeddings(data, model_name = "all-MiniLM-L6-v2"):
    embedder = SentenceTransformer(model_name)

    embeddings = embedder.encode(data, show_progress_bar=True, convert_to_tensor=False)
    return np.array(embeddings)