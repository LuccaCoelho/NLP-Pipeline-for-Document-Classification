import spacy

nlp = spacy.load("en_core_web_sm")

def tokenizer(data):
    doc = nlp(data)

    for token in doc:
        print(token, "| ", token.pos_, "| ", token.lemma_)


tokenizer("I'm very cute, but my girlfriend didn't want to be with me")
