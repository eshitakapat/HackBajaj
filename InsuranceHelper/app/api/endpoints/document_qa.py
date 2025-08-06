"""Document QA API endpoints for processing insurance documents."""
from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks, UploadFile, File, Form, Security
from typing import List, Dict, Any, Optional, Union
import logging
from pydantic import BaseModel, Field, HttpUrl, validator
from datetime import datetime
import os
import httpx
import uuid
from sqlalchemy.orm import Session
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ...models.user import User
from ...services.document_processor import DocumentProcessor
from ...services.document_store import DocumentStore
from ...services.vector_store import vector_store
from ...services.semantic_search import semantic_search
from ...dependencies import get_current_active_user, get_document_processor, get_document_store
from ...models.document_models import Document as DocumentModel, Question as QuestionModel
from ...core.config import settings

logger = logging.getLogger(__name__)

# Create routers
router = APIRouter(prefix="/documents", tags=["Documents"])
qa_router = APIRouter(prefix="/qa", tags=["Question Answering"])
hackrx_router = APIRouter(prefix="/hackrx", tags=["HackRx API"])

# Add security scheme for Bearer token
auth_scheme = HTTPBearer()

# Constants
DEFAULT_TOP_K = 3
MAX_CHUNK_LENGTH = 1000
SUPPORTED_CONTENT_TYPES = [
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
]

class DocumentProcessRequest(BaseModel):
    """Request model for document processing."""
    url: Optional[HttpUrl] = Field(
        None, 
        description="URL of the document to process. Required if file is not provided."
    )
    file_name: Optional[str] = Field(None, description="Original file name")
    content_type: Optional[str] = Field(None, description="MIME type of the document")
    
    @validator('content_type')
    def validate_content_type(cls, v, values):
        if v and v not in SUPPORTED_CONTENT_TYPES:
            raise ValueError(f"Unsupported content type. Must be one of: {', '.join(SUPPORTED_CONTENT_TYPES)}")
        return v
    
    @validator('url')
    def validate_url_or_file(cls, v, values):
        if v is None and 'file' not in values:
            raise ValueError("Either url or file must be provided")
        return v

class DocumentResponse(BaseModel):
    """Document response model."""
    id: str = Field(..., description="Internal database ID")
    document_id: str = Field(..., description="Unique document ID")
    url: str = Field(..., description="URL of the document")
    file_name: Optional[str] = Field(None, description="Original file name")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    content_type: Optional[str] = Field(None, description="MIME type of the document")
    status: str = Field(..., description="Processing status")
    processed_at: Optional[datetime] = Field(None, description="When the document was processed")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")

    class Config:
        orm_mode = True

class QuestionResponse(BaseModel):
    """Question response model."""
    id: str = Field(..., description="Internal database ID")
    document_id: str = Field(..., description="ID of the document this question is about")
    question: str = Field(..., description="The question that was asked")
    answer: str = Field(..., description="The generated answer")
    model_used: str = Field(..., description="The model used to generate the answer")
    created_at: datetime = Field(..., description="When the question was asked")
    processing_time_ms: int = Field(..., description="Time taken to answer in milliseconds")
    error: Optional[str] = Field(None, description="Error message if question failed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        orm_mode = True

class DocumentProcessResponse(BaseModel):
    """Response model for document processing."""
    document_id: str = Field(..., description="Unique ID of the processed document")
    status: str = Field(..., description="Processing status")
    url: str = Field(..., description="URL of the processed document")
    file_name: Optional[str] = Field(None, description="Original file name")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    content_type: Optional[str] = Field(None, description="MIME type of the document")
    processed_at: Optional[datetime] = Field(None, description="When the document was processed")

class HackRxRequest(BaseModel):
    """Request model for HackRx API."""
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    """Response model for HackRx API."""
    answers: List[str]

class DocumentSearchRequest(BaseModel):
    """Request model for document search and question answering."""
    document_id: str = Field(..., description="ID of the document to search")
    search_query: str = Field(..., description="Search query or question")
    top_k: int = Field(3, ge=1, le=10, description="Number of relevant chunks to retrieve")
    min_score: float = Field(0.5, ge=0, le=1, description="Minimum relevance score for chunks")
    model: str = Field("mistralai/mistral-7b-instruct", description="LLM model to use for answering")
    
    class Config:
        schema_extra = {
            "example": {
                "document_id": "doc_1234567890abcdef",
                "search_query": "What is the sum insured?",
                "top_k": 3,
                "min_score": 0.5,
                "model": "mistralai/mistral-7b-instruct"
            }
        }

async def process_document(url: str) -> Dict[str, Any]:
    """Process a document by downloading and extracting text."""
    logger.info(f"Processing document from: {url}")
    try:
        # Download and process the document
        content = await document_processor.download_document(url)
        content_type = 'application/pdf'  # Could be determined from URL or content
        
        # Process the document (extract text and chunk it)
        logger.info("Processing document content")
        doc_id, chunks = await document_processor.process_document(content, content_type)
        
        # Store in vector database
        logger.info(f"Upserting {len(chunks)} chunks to vector store")
        await vector_store.upsert_document(doc_id, chunks)
        
        return {
            'document_id': doc_id,
            'num_chunks': len(chunks),
            'chunks': chunks
        }
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}", exc_info=True)
        raise

async def answer_question(
    question: str, 
    doc_id: str, 
    top_k: int = DEFAULT_TOP_K,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """Answer a question about a processed document."""
    try:
        # 1. Perform semantic search to find relevant chunks
        logger.info(f"Searching for relevant chunks for question: {question[:100]}...")
        relevant_chunks = await vector_store.semantic_search(question, doc_id, top_k=top_k)
        
        if not relevant_chunks:
            return {
                'question': question,
                'answer': "No relevant information found in the document.",
                'sources': []
            }
        
        # 2. Build context from chunks
        context = "\n\n".join(
            f"[Document excerpt {i+1}]\n{chunk.get('text', '')[:MAX_CHUNK_LENGTH]}"
            for i, chunk in enumerate(relevant_chunks)
        )
        
        # 3. Generate answer using LLM
        logger.info("Generating answer using LLM")
        prompt = f"""You are an AI assistant that answers questions based on the provided document excerpts.
        
        Document excerpts:
        {context}
        
        Instructions:
        - Answer the question based ONLY on the document excerpts above.
        - If the answer cannot be found in the excerpts, say "I couldn't find a clear answer in the document."
        - Be concise and to the point.
        - Format your response in clear, readable paragraphs.
        
        Question: {question}
        
        Answer:"""
        
        answer = await llm_service.generate_text(
            prompt=prompt,
            model=model,
            max_tokens=1000,
            temperature=0.3,
        )
        
        # 4. Format sources
        sources = [
            {
                'text': chunk.get('text', '')[:500] + '...' if len(chunk.get('text', '')) > 500 else chunk.get('text', ''),
                'score': round(chunk.get('score', 0), 4),
                'position': chunk.get('chunk_index', i)
            }
            for i, chunk in enumerate(relevant_chunks)
        ]
        
        return {
            'question': question,
            'answer': answer.strip(),
            'sources': sources
        }
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}", exc_info=True)
        return {
            'question': question,
            'answer': "An error occurred while processing your question.",
            'sources': []
        }

@router.post(
    "/process",
    response_model=DocumentQAResponse,
    summary="Process a document and answer questions about it",
    response_description="Document processing and QA results"
)
async def process_document_qa(
    request: DocumentQARequest
) -> Dict[str, Any]:
    """Process a document and answer questions about it.
    
    This endpoint:
    1. Downloads the document from the provided URL
    2. Extracts and chunks the text
    3. Generates embeddings and stores them in the vector store
    4. Answers each question using the document content
    
    Example request:
    ```json
    {
        "url": "https://example.com/document.pdf",
        "questions": ["What is the sum insured?", "What are the exclusions?"],
        "top_k": 3
    }
    ```
    """
    try:
        logger.info(f"Starting document QA process for URL: {request.url}")
        
        # 1. Process the document
        doc_info = await process_document(str(request.url))
        doc_id = doc_info['document_id']
        
        # 2. Answer each question
        answers = []
        for question in request.questions:
            answer = await answer_question(
                question=question,
                doc_id=doc_id,
                top_k=request.top_k,
                model=request.model
            )
            answers.append(answer)
        
        # 3. Prepare response
        response = {
            'document_id': doc_id,
            'timestamp': datetime.utcnow().isoformat(),
            'answers': answers
        }
        
        logger.info(f"Completed document QA process for document: {doc_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document QA process failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )

async def process_document_background(
    document_processor: DocumentProcessor,
    document_store: DocumentStore,
    document_id: str,
    url: Optional[str] = None,
    file_path: Optional[str] = None,
    file_name: Optional[str] = None,
    content_type: Optional[str] = None,
    user_id: Optional[str] = None
):
    """Background task to process a document asynchronously."""
    try:
        # Update status to processing
        await document_store.update_document_status(
            document_id=document_id,
            status="processing"
        )
        
        # Process the document
        result = await document_processor.process_document(
            url=url,
            file_path=file_path,
            file_name=file_name,
            content_type=content_type
        )
        
        # Update document with processing results
        await document_store.update_document_after_processing(
            document_id=document_id,
            status="processed",
            chunk_count=result.get("chunk_count", 0),
            error_message=None
        )
        
        # Clean up temporary file if it exists
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Error removing temporary file {file_path}: {str(e)}")
                
    except Exception as e:
        error_msg = f"Background document processing failed: {str(e)}"
        logger.error(error_msg)
        
        # Update document with error status
        try:
            await document_store.update_document_after_processing(
                document_id=document_id,
                status="failed",
                error_message=error_msg
            )
        except Exception as update_error:
            logger.error(f"Failed to update document status after error: {str(update_error)}")
            
        # Clean up temporary file if it exists
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_error:
                logger.error(f"Error removing temporary file after error {file_path}: {str(cleanup_error)}")

@router.post("/process", response_model=DocumentProcessResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_document(
    request: DocumentProcessRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    document_store: DocumentStore = Depends(get_document_store)
):
    """Process a document from a URL or file upload.
    
    This endpoint starts an asynchronous background task to process the document.
    It returns immediately with the document details while processing continues in the background.
    """
    try:
        # Generate a document ID
        document_id = f"doc_{uuid.uuid4().hex}"
        
        # Create document record
        document = await document_store.create_document(
            url=str(request.url) if request.url else None,
            document_id=document_id,
            file_name=request.file_name,
            content_type=request.content_type,
            status="queued"
        )
        
        # Add background task to process the document
        background_tasks.add_task(
            process_document_background,
            document_processor=document_processor,
            document_store=document_store,
            document_id=document_id,
            url=str(request.url) if request.url else None,
            file_name=request.file_name,
            content_type=request.content_type,
            user_id=str(current_user.id) if current_user else None
        )
        
        return {
            "document_id": document.document_id,
            "status": document.status,
            "url": document.url,
            "file_name": document.file_name,
            "file_size": document.file_size,
            "content_type": document.content_type,
            "processed_at": document.processed_at
        }
        
    except Exception as e:
        logger.error(f"Error queuing document for processing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue document for processing: {str(e)}"
        )


@router.post("/upload", response_model=DocumentProcessResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_active_user),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    document_store: DocumentStore = Depends(get_document_store)
):
    """Upload and process a document file.
    
    This endpoint accepts a file upload, saves it temporarily, and processes it asynchronously.
    It returns immediately with the document details while processing continues in the background.
    """
    # Check file size
    file.file.seek(0, 2)  # Move to end of file
    file_size = file.file.tell()
    file.file.seek(0)  # Reset file pointer
    
    if file_size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {settings.MAX_FILE_SIZE / (1024 * 1024)}MB"
        )
    
    # Check file extension
    file_extension = file.filename.split('.')[-1].lower() if file.filename else ''
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    # Determine content type
    content_type = file.content_type or f"application/{file_extension}"
    
    # Create uploads directory if it doesn't exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    # Generate a unique filename
    file_id = str(uuid.uuid4())
    file_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}.{file_extension}")
    
    try:
        # Save the file temporarily
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Generate a document ID
        document_id = f"doc_{uuid.uuid4().hex}"
        
        # Create document record
        document = await document_store.create_document(
            url=None,
            document_id=document_id,
            file_name=file.filename,
            file_size=file_size,
            content_type=content_type,
            status="queued"
        )
        
        # Add background task to process the document
        background_tasks.add_task(
            process_document_background,
            document_processor=document_processor,
            document_store=document_store,
            document_id=document_id,
            file_path=file_path,
            file_name=file.filename,
            content_type=content_type,
            user_id=str(current_user.id) if current_user else None
        )
        
        return {
            "document_id": document.document_id,
            "status": document.status,
            "url": None,
            "file_name": document.file_name,
            "file_size": document.file_size,
            "content_type": document.content_type,
            "processed_at": document.processed_at
        }
        
    except Exception as e:
        # Clean up the temporary file if it was created
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_error:
                logger.error(f"Error removing temporary file {file_path}: {str(cleanup_error)}")
        
        logger.error(f"Error processing file upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process file upload: {str(e)}"
        )

@router.post("/semantic-search", response_model=List[Dict[str, Any]])
async def semantic_search_endpoint(
    request: DocumentSearchRequest,
    current_user: User = Depends(get_current_active_user),
):
    """
    Perform semantic search on a document.
    
    Args:
        request: Contains document_id and search_query
        
    Returns:
        List of relevant chunks with scores and metadata
    """
    try:
        # Get relevant chunks using semantic search
        relevant_chunks = await semantic_search.get_relevant_chunks(
            question=request.search_query,
            document_id=request.document_id,
            top_k=request.top_k or 3,
            min_score=request.min_score or 0.5
        )
        
        return relevant_chunks
        
    except Exception as e:
        logger.error(f"Semantic search failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform semantic search"
        )

@qa_router.post("/ask", response_model=Dict[str, Any])
async def ask_question(
    request: DocumentSearchRequest,
    current_user: User = Depends(get_current_active_user),
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
    """Ask a question about a processed document."""
    try:
        return await document_processor.answer_question(
            question=request.search_query,
            document_id=request.document_id,
            top_k=request.top_k,
            model=request.model
        )
    except Exception as e:
        logger.error(f"Error in ask endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process your question: {str(e)}"
        )

@hackrx_router.post(
    "/hackrx/run",
    response_model=HackRxResponse,
    summary="HackRx API endpoint for document QA",
    response_description="List of answers to the provided questions"
)
async def hackrx_qa_endpoint(
    request: HackRxRequest,
    credentials: HTTPAuthorizationCredentials = Security(auth_scheme)
):
    """
    Process a document and answer questions in HackRx format.
    """
    # Check Bearer token
    required_token = "0ed1b3e379e363e65b52c090f35648e913017fa88d757a36889962c787daad05"
    if credentials.credentials != required_token:
        raise HTTPException(status_code=401, detail="Invalid or missing token.")
    try:
        logger.info(f"Processing document: {request.documents}")
        # Process the document
        doc_info = await process_document(request.documents)
        doc_id = doc_info['document_id']
        # Answer each question
        answers = []
        for question in request.questions:
            try:
                result = await answer_question(
                    question=question,
                    doc_id=doc_id,
                    top_k=3
                )
                # Ensure the answer is a clean string
                answer = result.get('answer', '').strip()
                answers.append(answer)
            except Exception as e:
                logger.error(f"Error answering question: {str(e)}")
                answers.append("")
        logger.info("Successfully processed questions")
        return {"answers": answers}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        # Return empty answers on error to maintain response format
        return {"answers": [""] * len(request.questions) if 'request' in locals() else []}

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    document_store: DocumentStore = Depends(get_document_store)
):
    """Get document details and processing status."""
    document = await document_store.get_document_by_id(document_id)
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )
    
    return document

@router.get("/{document_id}/questions", response_model=List[QuestionResponse])
async def get_document_questions(
    document_id: str,
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(get_current_active_user),
    document_store: DocumentStore = Depends(get_document_store)
):
    """Get questions and answers for a document.
    
    Args:
        document_id: ID of the document
        limit: Maximum number of questions to return (max 1000)
        offset: Number of questions to skip
        
    Returns:
        List of questions and answers for the document
    """
    # Verify document exists and user has access
    document = await document_store.get_document_by_id(document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )
    
    # Get questions with pagination
    questions = await document_store.get_document_questions(
        document_id=document_id,
        limit=min(limit, 1000),  # Enforce a reasonable limit
        offset=offset
    )
    
    return questions

# Include the QA and HackRx routers
router.include_router(qa_router)
router.include_router(hackrx_router)
