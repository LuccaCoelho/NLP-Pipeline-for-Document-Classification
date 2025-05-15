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

def clean_text(text):
    # Remove lines that look like page numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    # Remove lines that repeat too often (headers/footers)
    lines = text.split('\n')
    line_counts = {}
    for line in lines:
        line = line.strip()
        if line:
            line_counts[line] = line_counts.get(line, 0) + 1

    cleaned_lines = [line for line in lines if line_counts.get(line.strip(), 0) <= 2]

    cleaned_text = "\n".join(cleaned_lines)

    return cleaned_text.strip()