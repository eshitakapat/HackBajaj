"""Module for handling and processing policy queries."""
import json
import logging
from typing import Dict, List, Any, Optional

from openai import OpenAI, APIError

from app.config import settings
from app.Models.schemas import ParsedQuery

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=settings.OPENAI_API_KEY)

class PolicyQueryParser:
    """Handles parsing and processing of policy-related queries."""
    
    @staticmethod
    def parse_question(question: str) -> Dict[str, Any]:
        """
        Analyzes insurance policy questions and extracts structured information.
        
        Args:
            question: The policy question to analyze
            
        Returns:
            Dict containing parsed information about the question
        """
        try:
            response = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
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
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            parsed_data = json.loads(result)
            
            # Validate and return the parsed data
            return {
                "policy_section": parsed_data.get("policy_section", "general"),
                "query_type": parsed_data.get("query_type", "general_inquiry"),
                "key_terms": parsed_data.get("key_terms", [])
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response for question: {question}")
            return {
                "policy_section": "general",
                "query_type": "general_inquiry",
                "key_terms": []
            }
        except APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in parse_question: {str(e)}")
            return {
                "policy_section": "general",
                "query_type": "error",
                "key_terms": []
            }
