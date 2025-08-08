from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class EmbeddingSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        self.index = faiss.IndexFlatIP(self.dimension)
        self.text_chunks = []

    def add_documents(self, docs):
        # Filter out empty or whitespace-only texts
        self.text_chunks = [doc['text'].strip() for doc in docs if doc['text'].strip()]
        if not self.text_chunks:
            print("[Warning] No valid text chunks to index.")
            return

        embeddings = self.model.encode(self.text_chunks, convert_to_numpy=True, normalize_embeddings=True)
        self.index.add(embeddings.astype('float32'))


    def search(self, query, top_k=5):
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        distances, indices = self.index.search(q_emb, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                "text": self.text_chunks[idx],
                "similarity_score": float(dist)
            })
        return results
