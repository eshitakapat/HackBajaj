from .pdf_parser import extract_text_from_pdf
from .docx_parser import extract_text_from_docx
from .email_parser import extract_text_from_eml
from .text_processing import split_into_clauses

__all__ = [
    "extract_text_from_pdf",
    "extract_text_from_docx",
    "extract_text_from_eml",
    "split_into_clauses"
]
