"""Module for handling and processing policy queries."""
import json
import logging
from typing import Dict, List, Any, Optional
import httpx

from app.core.config import settings
from app.Models.schemas import ParsedQuery

logger = logging.getLogger(__name__)

class PolicyQueryParser:
    """Handles parsing and processing of policy-related queries."""
    
    @staticmethod
    async def parse_question(question: str) -> Dict[str, Any]:
        """
        Analyzes insurance policy questions and extracts structured information.
        
        Args:
            question: The policy question to analyze
            
        Returns:
            Dict containing parsed information about the question
        """
        try:
            headers = {
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY.get_secret_value()}",
                "HTTP-Referer": "https://your-site.com",  # Required by OpenRouter
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": settings.DEFAULT_LLM_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": """Analyze the insurance policy question and return a JSON with:
                        - policy_section: The relevant section (e.g., 'coverage', 'claims')
                        - query_type: Type of question (e.g., 'eligibility', 'coverage')
                        - key_terms: List of important terms
                        """
                    },
                    {"role": "user", "content": question}
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                
            content = result["choices"][0]["message"]["content"]
            parsed_data = json.loads(content)
            
            # Ensure required fields exist
            return {
                "policy_section": parsed_data.get("policy_section", "general"),
                "query_type": parsed_data.get("query_type", "general_inquiry"),
                "key_terms": parsed_data.get("key_terms", []),
                "raw_response": parsed_data
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            return {
                "policy_section": "general",
                "query_type": "general_inquiry",
                "key_terms": [],
                "error": "Failed to parse response"
            }
            
        except Exception as e:
            logger.error(f"Error in parse_question: {str(e)}")
            return {
                "policy_section": "general",
                "query_type": "general_inquiry",
                "key_terms": [],
                "error": str(e)
            }
