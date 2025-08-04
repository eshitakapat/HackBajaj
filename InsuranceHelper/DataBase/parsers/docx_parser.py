from docx import Document

def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    full_text = []
    for para in doc.paragraphs:
        if para.text:
            full_text.append(para.text)
    return "\n".join(full_text)
