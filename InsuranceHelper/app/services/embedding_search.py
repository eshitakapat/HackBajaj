from typing import List, Optional
import logging
import torch
from sentence_transformers import SentenceTransformer
from app.core.config import settings

logger = logging.getLogger(__name__)

class PolicyEmbeddingGenerator:
    """Handles generation and management of document embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedding generator.
        
        Args:
            model_name: The sentence-transformers model to use (default: 'all-MiniLM-L6-v2')
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.model_name = model_name
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Initialized sentence transformer model: {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize model {self.model_name}: {str(e)}")
            raise
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for policy text.
        
        Args:
            text: Policy text or question
            
        Returns:
            List[float]: Vector embedding (dimensions depend on the model)
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding generation")
                return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2
                
            # Generate embedding
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return a zero vector if there's an error
            return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2
    
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
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                return []
                
            # Generate embeddings in a batch
            embeddings = self.model.encode(valid_texts, convert_to_tensor=False)
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {str(e)}")
            # Return zero vectors for all inputs if there's an error
            return [[0.0] * 384 for _ in texts]  # Default dimension for all-MiniLM-L6-v2

# Create a default instance
embedding_generator = PolicyEmbeddingGenerator()

def generate_embedding(text: str) -> List[float]:
    """Generate a single embedding."""
    return embedding_generator.generate_embedding(text)
