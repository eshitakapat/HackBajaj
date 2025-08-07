"""HackRx specific API endpoint for hackathon submission."""
import logging
import time
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends, Header
from fastapi.responses import JSONResponse

from app.schemas.hackrx import HackRxRequest, HackRxResponse
from app.services.document_processor import document_processor
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Authentication token for hackathon
HACKRX_TOKEN = "0ed1b3e379e363e65b52c090f35648e913017fa88d757a36889962c787daad05"

def verify_hackrx_token(authorization: str = Header(None)):
    """Verify the hackathon authentication token."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required"
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization format. Use 'Bearer <token>'"
        )
    
    token = authorization.replace("Bearer ", "")
    if token != HACKRX_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    return token

@router.post("/run", response_model=HackRxResponse)
async def hackrx_run(
    request: HackRxRequest,
    token: str = Depends(verify_hackrx_token)
):
    """
    HackRx main endpoint: Process documents and answer questions.
    
    This endpoint:
    1. Downloads and processes the document from the provided URL
    2. Answers all the questions based on the document content
    3. Returns answers in the required format
    """
    start_time = time.time()
    
    try:
        logger.info("HackRx submission started")
        logger.info(f"Document URL: {request.documents}")
        logger.info(f"Number of questions: {len(request.questions)}")
        
        # Step 1: Process the document
        logger.info("Processing document...")
        try:
            doc_result = await document_processor.process_document(
                url=request.documents,
                file_name="hackrx_policy.pdf",
                content_type="application/pdf"
            )
            document_id = doc_result["document_id"]
            logger.info(f"Document processed successfully: {document_id}")
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process document: {str(e)}"
            )
        
        # Step 2: Answer all questions
        logger.info("Processing questions...")
        answers = []
        
        for i, question in enumerate(request.questions):
            try:
                logger.info(f"Processing question {i+1}/{len(request.questions)}: {question[:100]}...")
                
                # Get answer for this question
                qa_result = await document_processor.answer_question(
                    question=question,
                    document_id=document_id,
                    top_k=5,  # Get more context for better answers
                    model=None  # Use default model
                )
                
                answer = qa_result.get("answer", "").strip()
                if not answer:
                    answer = "I could not find information to answer this question in the provided document."
                
                answers.append(answer)
                logger.info(f"Question {i+1} answered successfully")
                
            except Exception as e:
                logger.error(f"Failed to answer question {i+1}: {str(e)}")
                # Provide a fallback answer instead of failing completely
                answers.append("I encountered an error while processing this question. Please try again.")
        
        processing_time = time.time() - start_time
        logger.info(f"HackRx submission completed in {processing_time:.2f} seconds")
        logger.info(f"Generated {len(answers)} answers")
        
        return HackRxResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"HackRx submission failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Submission processing failed: {str(e)}"
        )

@router.get("/health")
async def hackrx_health():
    """Health check for HackRx endpoint."""
    try:
        services_status = {}
        
        # Check document processor
        try:
            services_status["document_processor"] = "available"
        except Exception as e:
            services_status["document_processor"] = f"unavailable: {str(e)}"
        
        # Check LLM service
        try:
            services_status["llm_service"] = "available" if llm_service.available else "limited"
        except Exception as e:
            services_status["llm_service"] = f"unavailable: {str(e)}"
        
        # Check vector store
        try:
            from app.services.vector_store import vector_store
            services_status["vector_store"] = "available" if vector_store.available else "unavailable"
        except Exception as e:
            services_status["vector_store"] = f"unavailable: {str(e)}"
        
        return {
            "status": "ready",
            "endpoint": "/hackrx/run",
            "authentication": "Bearer token required",
            "services": services_status,
            "version": "1.0.0"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
