from typing import List, Union
import logging
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

class PolicyEmbeddingGenerator:
    """Handles generation and management of document embeddings."""
    
    def __init__(self, model: str = None):
        """Initialize the embedding generator.
        
        Args:
            model: The embedding model to use (defaults to the one in settings)
        """
        self.model = model
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for policy text.
        
        Args:
            text: Policy text or question
            
        Returns:
            List[float]: Vector embedding (dimensions depend on the model)
        """
        try:
            # Get embeddings for the text
            embeddings = await llm_service.get_embeddings([text], model=self.model)
            return embeddings[0] if embeddings else None
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return a zero vector if there's an error
            return [0.0] * 1536  # Default dimension for text-embedding-3-small
    
    async def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a batch.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of vector embeddings
        """
        if not texts:
            return []
            
        try:
            return await llm_service.get_embeddings(texts, model=self.model)
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {str(e)}")
            # Return zero vectors for all inputs if there's an error
            return [[0.0] * 1536 for _ in texts]

# Create a default instance for backward compatibility
embedding_generator = PolicyEmbeddingGenerator()

# For backward compatibility
async def generate_embedding(text: str) -> List[float]:
    """Generate a single embedding (legacy function)."""
    return await embedding_generator.generate_embedding(text)
