"""LLM service using OpenRouter API."""
import logging
from typing import List, Dict, Any, Optional
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)

class OpenRouterService:
    """Service for interacting with OpenRouter API."""
    
    def __init__(self):
        """Initialize the OpenRouter service with settings."""
        self.api_key = settings.OPENROUTER_API_KEY.get_secret_value()
        self.api_base = settings.OPENROUTER_API_BASE
        self.default_model = settings.OPENROUTER_MODEL
        self.embedding_model = settings.EMBEDDING_MODEL
        
        if not self.api_key:
            logger.warning("OpenRouter API key not set. LLM features will be disabled.")
    
    async def _make_request(
        self, 
        endpoint: str, 
        payload: Dict[str, Any],
        headers_override: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make a request to the OpenRouter API."""
        if not self.api_key:
            raise ValueError("OpenRouter API key is not configured")
            
        url = f"{self.api_base.rstrip('/')}/{endpoint.lstrip('/')}"
        
        # Default headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/InsuranceHelper",
            "X-Title": "Insurance Helper"
        }
        
        if headers_override:
            headers.update(headers_override)
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPStatusError as e:
            error_msg = f"OpenRouter API request failed with status {e.response.status_code}: {e.response.text}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
            
        except Exception as e:
            logger.error(f"Error making request to OpenRouter: {str(e)}")
            raise
    
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text using the specified model."""
        if not self.api_key:
            logger.warning("OpenRouter API key not set")
            return "LLM service is not configured"
            
        model = model or self.default_model
        
        try:
            # Use chat format for modern models
            messages = [{"role": "user", "content": prompt}]
            response = await self._make_request(
                "chat/completions",
                {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    **kwargs
                }
            )
            return response["choices"][0]["message"]["content"]
                
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            return f"Error generating response: {str(e)}"

# Create a singleton instance
llm_service = OpenRouterService()