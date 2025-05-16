import os
import re
import pdfplumber

def extract_text(pdf_path):
    text = " "
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    return text

def extract_text_directory(directory_path):
    pdf_texts = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            text = extract_text(file_path)
            pdf_texts["filename"] = text

    return pdf_texts