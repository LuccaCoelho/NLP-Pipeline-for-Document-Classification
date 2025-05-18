import os
import pdfplumber
import json

output_path = "/home/lucca-coelho/nlp-pipeline/data/pdf_dataset.json"

def get_label_file(filename):
    current = ""
    count = 0

    while filename[count] != "_":
        current += filename[count]
        count += 1

    return current

def extract_text(pdf_path):
    text = " "
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    return text

def extract_text_directory(directory_path):
    pdf_texts = {}
    for file in os.listdir(directory_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(directory_path, file)
            text = extract_text(file_path)
            pdf_texts[file] = text

    return pdf_texts

def save_data_json(files_directory):
    data = extract_text_directory(files_directory)

    json_data = [{
        "filename": filename,
        "label": get_label_file(filename),
        "text": text
    }
        for filename, text in data.items()
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False)

    print("✅ The pdf_dataset.json was updated.")

save_data_json("/home/lucca-coelho/nlp-pipeline/data/raw_pdfs")