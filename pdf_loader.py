import fitz  # PyMuPDF
import os

def load_pdfs_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(file_path)
            if text.strip():
                documents.append({
                    "filename": filename,
                    "text": text
                })
            else:
                print(f"[Warning] No text extracted from {filename}")
    return documents

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        return full_text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""
