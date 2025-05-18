import spacy
import json

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
input_path = "/home/lucca-coelho/nlp-pipeline/data/pdf_dataset.json"
output_path = "/home/lucca-coelho/nlp-pipeline/data/processed_data.json"

def tokenizer(text):
    doc = nlp(text.lower().strip())

    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

    return " ".join(tokens)
def clean_text(infile_path, outfile_path):
    infile_path = input_path
    outfile_path = output_path
    index = 0
    with open(infile_path, "r", encoding="utf-8") as infile, open(outfile_path, "w", encoding="utf-8") as outfile:
        data = json.load(infile)
        while index < len(data):
            cleaned_text = tokenizer(data[index]["text"])

            data[index]["text"] = cleaned_text

            json.dump(data, outfile)
            index += 1

clean_text(input_path, output_path)