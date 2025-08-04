import os
import psycopg2
import uuid

def connect_db():
    password = os.getenv("DB_PASSWORD")
    if not password:
        raise ValueError("DB_PASSWORD environment variable not set.")
    return psycopg2.connect(
        dbname="intelligent_query",
        user="postgres",
        password=password,
        host="localhost",
        port=5432
    )

def insert_sample_clause():
    conn = None
    cursor = None
    try:
        conn = connect_db()
        cursor = conn.cursor()

        document_id = uuid.uuid4()
        document_id_str = str(document_id)
        clause_text = "This policy covers accidental damage and fire incidents."
        clause_index = 0
        embedding = [0.001] * 1536  # dummy embedding vector

        # Insert document
        cursor.execute("""
            INSERT INTO documents (id, title, source_type, domain, upload_time)
            VALUES (%s, %s, %s, %s, NOW())
        """, (document_id_str, "Sample Insurance Policy", "pdf", "insurance"))

        # Insert clause
        cursor.execute("""
            INSERT INTO clauses (document_id, clause_text, clause_index, embedding)
            VALUES (%s, %s, %s, %s)
        """, (document_id_str, clause_text, clause_index, embedding))

        conn.commit()
        print("✅ Sample clause inserted successfully.")

    except Exception as e:
        print("❌ Error:", e)

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Run the function if this file is executed directly
if __name__ == "__main__":
    insert_sample_clause()
