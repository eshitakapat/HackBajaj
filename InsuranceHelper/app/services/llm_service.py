"""LLM service using OpenRouter API with support for text and vision models."""
import os
import json
import logging
import base64
from typing import List, Dict, Any, Optional, Union, Literal
import httpx
from pydantic import HttpUrl
from app.config import settings

logger = logging.getLogger(__name__)

class OpenRouterService:
    """Service for interacting with OpenRouter API with support for text and vision models."""
    
    def __init__(self):
        """Initialize the OpenRouter service with settings."""
        self.api_key = settings.OPENROUTER_API_KEY
        self.api_base = str(settings.OPENROUTER_API_BASE)
        # Set default model to Mistral instruct if not already set
        self.default_model = getattr(settings, 'OPENROUTER_MODEL', None) or 'mistralai/mistral-small-3.1-24b-instruct:free'
        self.embedding_model = settings.EMBEDDING_MODEL
        
        # Vision models configuration
        self.vision_models = {
            "mistralai/mistral-small-3.1-24b-instruct:free": {
                "supports_vision": True,
                "max_tokens": 4096,
                "supports_json": True
            },
            # Add more vision models as needed
        }
        
        if not self.api_key:
            logger.warning("OpenRouter API key not set. LLM features will be disabled.")
    
    async def _make_request(
        self, 
        endpoint: str, 
        payload: Dict[str, Any],
        headers_override: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Make a request to the OpenRouter API.
        
        Args:
            endpoint: API endpoint (e.g., 'chat/completions')
            payload: Request payload
            headers_override: Optional headers to override defaults
            
        Returns:
            JSON response from the API
        """
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
        
        # Apply any header overrides
        if headers_override:
            headers.update(headers_override)
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:  # Increased timeout for vision
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPStatusError as e:
            error_msg = (
                f"OpenRouter API request failed with status {e.response.status_code}: "
                f"{e.response.text}"
            )
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
        """Generate text using the specified model.
        
        Args:
            prompt: The input prompt
            model: Model to use (defaults to the one in settings)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-2)
            **kwargs: Additional model parameters
            
        Returns:
            Generated text
        """
        if not self.api_key:
            logger.warning("OpenRouter API key not set")
            return ""
            
        model = model or self.default_model
        
        try:
            # Check if model supports chat format
            if any(m in model.lower() for m in ['gpt', 'claude', 'llama', 'mistral']):
                # Chat format
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
            else:
                # Completion format
                response = await self._make_request(
                    "completions",
                    {
                        "model": model,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        **kwargs
                    }
                )
                return response["choices"][0]["text"]
                
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}", exc_info=True)
            return ""           
    
    async def analyze_image(
        self,
        image_url: str,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Analyze an image using a vision model through OpenRouter.
        
        Args:
            image_url: URL of the image to analyze
            prompt: The question or instruction about the image
            model: The vision model to use (defaults to Mistral vision model)
            temperature: Controls randomness (0.0 to 2.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The model's analysis of the image
        """
        if not self.api_key:
            return "LLM service is not configured (missing API key)"
            
        # Default to Mistral vision model if not specified
        model = model or "mistralai/mistral-small-3.1-24b-instruct:free"
        
        # Check if the model supports vision
        model_info = self.vision_models.get(model, {})
        if not model_info.get("supports_vision", False):
            logger.warning(f"Model {model} may not support vision capabilities")
        
        # Prepare the message with image
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }]
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": max(0.0, min(2.0, temperature)),
            "max_tokens": max(1, min(8000, max_tokens)),
            **kwargs
        }
        
        try:
            response = await self._make_request("chat/completions", payload)
            return response["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            raise
            
    async def analyze_local_image(
        self,
        image_path: str,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Analyze a local image file using a vision model through OpenRouter.
        
        Args:
            image_path: Path to the local image file
            prompt: The question or instruction about the image
            model: The vision model to use
            **kwargs: Additional parameters for the API call
            
        Returns:
            The model's analysis of the image
        """
        try:
            # Convert local image to base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
            # Default MIME type, can be made configurable if needed
            mime_type = "image/jpeg"
            if image_path.lower().endswith('.png'):
                mime_type = "image/png"
            elif image_path.lower().endswith('.gif'):
                mime_type = "image/gif"
                
            data_url = f"data:{mime_type};base64,{base64_image}"
            
            # Use the URL-based analysis with the data URL
            return await self.analyze_image(
                image_url=data_url,
                prompt=prompt,
                model=model,
                **kwargs
            )
            
        except FileNotFoundError:
            error_msg = f"Image file not found: {image_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            logger.error(f"Error processing local image: {str(e)}")
            raise
    
    async def embed_text(
        self, 
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """Get embedding for a single text.
        
        Args:
            text: Text to embed
            model: Model to use for embeddings
            
        Returns:
            Embedding vector as a list of floats
        """
        if not self.api_key or not text:
            return []
            
        model = model or self.embedding_model
        
        try:
            response = await self._make_request(
                "embeddings",
                {
                    "model": model,
                    "input": text
                }
            )
            
            # Return the first (and only) embedding
            return response["data"][0]["embedding"]
            
        except Exception as e:
            logger.error(f"Failed to get embedding: {str(e)}", exc_info=True)
            # Return a zero vector as fallback
            return [0.0] * 768  # Default dimension for all-MiniLM-L6-v2
            
    async def get_embeddings(
        self, 
        texts: List[str], 
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            model: Model to use for embeddings
            
        Returns:
            List of embedding vectors
        """
        if not self.api_key or not texts:
            return []
            
        model = model or self.embedding_model
        
        try:
            response = await self._make_request(
                "embeddings",
                {
                    "model": model,
                    "input": texts
                }
            )
            
            # Sort embeddings by index to maintain order
            embeddings = [None] * len(texts)
            for data in response["data"]:
                embeddings[data["index"]] = data["embedding"]
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to get embeddings: {str(e)}", exc_info=True)
            return []

# Create a singleton instance
llm_service = OpenRouterService()

# For backward compatibility and convenience
async def generate_text(*args, **kwargs) -> str:
    """Generate text using the LLM service (legacy function)."""
    return await llm_service.generate_text(*args, **kwargs)

async def get_embeddings(*args, **kwargs) -> List[List[float]]:
    """Get embeddings using the LLM service (legacy function)."""
    return await llm_service.get_embeddings(*args, **kwargs)

async def analyze_image(*args, **kwargs) -> str:
    """Analyze an image using a vision model (convenience function)."""
    return await llm_service.analyze_image(*args, **kwargs)

async def analyze_local_image(*args, **kwargs) -> str:
    """Analyze a local image file using a vision model (convenience function)."""
    return await llm_service.analyze_local_image(*args, **kwargs)
