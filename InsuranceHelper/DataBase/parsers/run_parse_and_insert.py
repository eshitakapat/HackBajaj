import os
import uuid
from .pdf_parser import extract_text_from_pdf
from .docx_parser import extract_text_from_docx
from .email_parser import extract_text_from_eml
from .text_processing import split_into_clauses
from ..database import connect_db  # Assuming you have a connect_db func there

def insert_clauses(document_id: str, clauses: list, conn):
    cursor = conn.cursor()
    try:
        for idx, clause in enumerate(clauses):
            # For now, dummy embedding vector of length 1536 filled with 0.001
            embedding = [0.001] * 1536
            cursor.execute("""
                INSERT INTO clauses (document_id, clause_text, clause_index, embedding)
                VALUES (%s, %s, %s, %s)
            """, (document_id, clause, idx, embedding))
        conn.commit()
        print(f"Inserted {len(clauses)} clauses.")
    except Exception as e:
        print("Error inserting clauses:", e)
        conn.rollback()
    finally:
        cursor.close()

def main():
    # Example: Change this to your test file path & type
    test_file_path = "/mnt/c/Users/KIIT0001/Desktop/Coding/HackBajaj/InsuranceHelper/sample.pdf"
    file_type = "pdf"  # Change to 'docx' or 'email' as needed

    if file_type == "pdf":
        text = extract_text_from_pdf(test_file_path)
    elif file_type == "docx":
        text = extract_text_from_docx(test_file_path)
    elif file_type == "email":
        text = extract_text_from_eml(test_file_path)
    else:
        raise ValueError("Unsupported file type")

    clauses = split_into_clauses(text)
    print(f"Extracted {len(clauses)} clauses.")

    conn = connect_db()
    cursor = conn.cursor()
    document_id = str(uuid.uuid4())

    try:
        # Insert document metadata
        cursor.execute("""
            INSERT INTO documents (id, title, source_type, domain, upload_time)
            VALUES (%s, %s, %s, %s, NOW())
        """, (document_id, os.path.basename(test_file_path), file_type, "insurance"))
        conn.commit()
        print("Inserted document metadata.")
    except Exception as e:
        print("Error inserting document metadata:", e)
        conn.rollback()
        conn.close()
        return

    insert_clauses(document_id, clauses, conn)
    conn.close()

if __name__ == "__main__":
    main()
