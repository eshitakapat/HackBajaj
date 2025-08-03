from fastapi import APIRouter, Depends, HTTPException, status, Header
from typing import Optional, List
import os
import logging
import httpx
import json

from app.services.query_handling import PolicyQueryParser
from app.services.embedding_search import PolicyEmbeddingGenerator
from app.services.document_processing import process_pdf_from_url
from app.Models.schemas import PolicyQuery, PolicyResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# Get API key from environment
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable not set")

async def verify_token(authorization: str = Header(...)) -> bool:
    """Verify the Authorization header"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split(" ")[1]
    if token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

@router.post(
    "/hackrx/run",
    response_model=PolicyResponse,
    status_code=status.HTTP_200_OK,
    summary="Process policy document and answer questions",
    response_description="List of answers to the provided questions"
)
async def process_policy_query(
    query: PolicyQuery,
    token_verified: bool = Depends(verify_token)
) -> PolicyResponse:
    """
    Process insurance policy document and answer questions about it.
    
    - **documents**: URL to the policy document (PDF)
    - **questions**: List of questions about the policy
    """
    try:
        # Process each question
        answers = []
        for question in query.questions:
            try:
                # Parse the question
                parsed = PolicyQueryParser.parse_question(question)
                
                # Generate embedding for semantic search
                embedding = PolicyEmbeddingGenerator.generate_embedding(question)
                
                # Download and process the document
                document_text = await process_pdf_from_url(query.documents)
                
                # Generate answer (simplified - implement your logic here)
                answer = f"Answer for: {question}"
                answers.append(answer)
                
            except Exception as e:
                logger.error(f"Error processing question '{question}': {str(e)}")
                answers.append("Error processing question")
        
        return PolicyResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error in process_policy_query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing your request"
        )
