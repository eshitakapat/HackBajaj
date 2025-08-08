import requests
import fitz
from io import BytesIO

def download_pdf_from_url(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    pdf_bytes = BytesIO(response.content)
    doc = fitz.open(stream=pdf_bytes.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def chunk_text(text, chunk_size=1200, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
