from sentence_transformers import SentenceTransformer

def embed_documents(doct_dict, embedding_model= "sentence-transformers/all-MiniLM-L6-v2"):
    embedding_model = SentenceTransformer(embedding_model)

    embedded_dict = {}

    for name, token in doct_dict.items():
        joined_text = " ".join(token)

        embeddings = embedding_model.encode(joined_text)

        embedded_dict[name] = embeddings

    return embedded_dict