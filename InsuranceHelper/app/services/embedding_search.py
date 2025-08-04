from typing import List
from openai import OpenAI
from app.config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

class PolicyEmbeddingGenerator:
    @staticmethod
    def generate_embedding(text: str) -> List[float]:
        """
        Generates embeddings for policy text using text-embedding-3-small
        
        Args:
            text: Policy text or question
            
        Returns:
            List[float]: 1536-dimensional vector
        """
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
