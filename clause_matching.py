from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class ClauseMatcher:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def rerank(self, query, candidates):
        if not candidates:
            return []

        q_emb = self.model.encode([query])
        c_emb = self.model.encode(candidates)
        sims = cosine_similarity(q_emb, c_emb)[0]

        def truncate_text(text, max_chars=150):
            # Truncate text to roughly 1-2 lines (150 chars or first newline)
            snippet = text.split('\n')[0]
            if len(snippet) > max_chars:
                snippet = snippet[:max_chars].rstrip() + "..."
            return snippet

        scored = list(zip(candidates, sims))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "clause": c,
                "clause_snippet": truncate_text(c["text"]),
                "score": float(s)
            }
            for c, s in scored
        ]
