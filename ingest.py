import os
from typing import List

def ingest_documents(data_folder: str) -> List[str]:
    texts = []
    for fname in os.listdir(data_folder):
        full_path = os.path.join(data_folder, fname)
        if fname.endswith('.txt'):
            with open(full_path, 'r', encoding='utf-8') as f:
                texts.append(f.read().strip())
        elif fname.endswith('.pdf'):
            texts.append(extract_text_pdf(full_path))
        elif fname.endswith('.docx'):
            texts.append(extract_text_docx(full_path))
        elif fname.endswith('.eml'):
            texts.append(extract_text_email(full_path))
    return texts

def extract_text_pdf(file_path: str) -> str:
    import pdfplumber
    text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text())
    return "\n".join(filter(None, text))

def extract_text_docx(file_path: str) -> str:
    from docx import Document
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)

def extract_text_email(file_path: str) -> str:
    import email
    with open(file_path, 'r', encoding='utf-8') as f:
        msg = email.message_from_file(f)
    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                parts.append(part.get_payload(decode=True).decode('utf-8', errors='ignore'))
    else:
        parts.append(msg.get_payload(decode=True).decode('utf-8', errors='ignore'))
    return "\n".join(parts)

from pdf_loader import load_pdfs_from_folder

def ingest_documents(data_folder: str):
    return load_pdfs_from_folder(data_folder)
