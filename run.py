import json
from llm_parser import parse_query_to_structured
from embedding_search import EmbeddingSearch
from clause_matching import ClauseMatcher
from logic_evaluation import evaluate_decision
from pdf_utils import download_pdf_from_url, chunk_text

# Generate mock answer based on top-matched clause
def generate_answer(question, reranked_clauses):
    if not reranked_clauses:
        return "No relevant clause found."

    top_clause = reranked_clauses[0]
    if top_clause["score"] < 0.2:
        return "Clause does not meet compliance criteria."

    return top_clause["clause_snippet"]

def main():
    # ✅ HARDCODED PDF URL
    pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

    # ✅ Still let user enter questions
    print("\n❓ Enter your questions (separated by commas):")
    question_input = input("> ").strip()
    questions = [q.strip() for q in question_input.split(",") if q.strip()]

    print("\n[1] Downloading and extracting PDF text...")
    full_text = download_pdf_from_url(pdf_url)
    print(f"[✓] Extracted {len(full_text)} characters of text")

    print("[2] Chunking document...")
    chunks = chunk_text(full_text)
    print(f"[✓] Created {len(chunks)} text chunks")

    # Prepare documents for embedding search
    docs = [{"text": c} for c in chunks]

    embed_search = EmbeddingSearch()
    embed_search.add_documents(docs)

    clause_matcher = ClauseMatcher()
    answers = []

    for question in questions:
        print(f"[→] Searching answer for: {question}")
        retrieved = embed_search.search(question, top_k=3)
        reranked = clause_matcher.rerank(question, retrieved)
        answer = generate_answer(question, reranked)
        answers.append(answer)

    output = {
        "answers": answers
    }

    print("\n📦 Final JSON output:")
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
