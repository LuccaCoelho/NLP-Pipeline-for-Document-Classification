import spacy
import json

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
input_path = "/home/lucca-coelho/nlp-pipeline/data/pdf_dataset.json"
output_path = "/home/lucca-coelho/nlp-pipeline/data/processed_data.json"

def tokenizer(text):
    doc = nlp(text.lower().strip())

    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and len(token) > 2]

    return " ".join(tokens)
def clean_text(infile_path, outfile_path):
    with open(infile_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    for entry in data:
        cleaned_text = tokenizer(entry["text"])
        entry["text"] = cleaned_text

    with open(outfile_path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=2)

    print("✅ The processed_data.json file was successfully written.")

clean_text(input_path, output_path)