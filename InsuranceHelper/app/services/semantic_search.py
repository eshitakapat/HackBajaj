"""
Semantic search service using Pinecone for vector similarity search.
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from app.services.document_processor import document_processor
from app.services.vector_store import vector_store
from app.core.config import settings

logger = logging.getLogger(__name__)

class SemanticSearch:
    """Handles semantic search operations using Pinecone."""
    
    def __init__(self):
        self.vector_store = vector_store
        self.min_score = 0.5  # Minimum similarity score (0-1)
        self.top_k = 3  # Number of chunks to retrieve
    
    async def get_relevant_chunks(
        self, 
        question: str, 
        document_id: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a question using semantic search.
        
        Args:
            question: The question to search for
            document_id: Document ID to search within
            top_k: Number of chunks to return (default: 3)
            min_score: Minimum similarity score (0-1, default: 0.5)
            
        Returns:
            List of relevant chunks with metadata and similarity scores
        """
        if not question or not document_id:
            return []
            
        top_k = top_k or self.top_k
        min_score = min_score or self.min_score
        
        try:
            # 1. Get the question embedding
            query_embedding = await self._get_question_embedding(question)
            if query_embedding is None:
                logger.error("Failed to get question embedding")
                return []
            
            # 2. Search for similar vectors in Pinecone
            search_results = await self.vector_store.search_similar(
                query_vector=query_embedding,
                top_k=top_k,
                namespace=document_id,
                include_metadata=True
            )
            
            if not search_results:
                logger.info("No search results found")
                return []
            
            # 3. Process and filter results
            relevant_chunks = []
            for result in search_results:
                if result.get('score', 0) >= min_score:
                    metadata = result.get('metadata', {})
                    relevant_chunks.append({
                        'id': result.get('id'),
                        'score': float(result['score']),
                        'text': metadata.get('text', ''),
                        'metadata': metadata
                    })
            
            # Sort by score in descending order
            relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Found {len(relevant_chunks)} relevant chunks for question")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}", exc_info=True)
            return []
    
    async def get_context_block(
        self, 
        question: str, 
        document_id: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        separator: str = "\n\n---\n\n"
    ) -> str:
        """
        Get a formatted context block from relevant document chunks.
        
        Args:
            question: The question to search for
            document_id: Document ID to search within
            top_k: Number of chunks to include (default: 3)
            min_score: Minimum similarity score (0-1, default: 0.5)
            separator: String to separate chunks in the output
            
        Returns:
            Formatted context string with relevant chunks
        """
        chunks = await self.get_relevant_chunks(
            question=question,
            document_id=document_id,
            top_k=top_k,
            min_score=min_score
        )
        
        if not chunks:
            return ""
        
        # Format chunks with their scores and text
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Context {i}, Score: {chunk['score']:.3f}]\n"
                f"{chunk['text'].strip()}"
            )
        
        return separator.join(context_parts)
    
    async def _get_question_embedding(self, question: str) -> Optional[List[float]]:
        """Get embedding vector for a question."""
        try:
            # Initialize the embedding model if not already done
            document_processor._initialize_embedding_model()
            
            # Get embedding using the same model as document chunks
            embedding = document_processor.embedding_model.encode(
                question,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating question embedding: {str(e)}")
            return None

# Create a singleton instance
semantic_search = SemanticSearch()
