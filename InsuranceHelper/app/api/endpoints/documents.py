"""Document processing endpoints."""
from typing import List
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.schemas.document import (
    DocumentProcessRequest, DocumentProcessResponse,
    QuestionRequest, QuestionResponse,
    SearchRequest, SearchResponse
)
from app.services.document_processor import document_processor

router = APIRouter()

@router.post("/process", response_model=DocumentProcessResponse)
async def process_document(request: DocumentProcessRequest):
    """Process a document from URL."""
    try:
        result = await document_processor.process_document(
            url=request.url,
            file_name=request.file_name,
            content_type=request.content_type
        )
        return DocumentProcessResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    file_name: str = Form(None)
):
    """Upload and process a document file."""
    try:
        # Save uploaded file temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Process the uploaded file
            result = await document_processor.process_document(
                url=tmp_path,
                file_name=file_name or file.filename,
                content_type=file.content_type
            )
            return DocumentProcessResponse(**result)
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload and processing failed: {str(e)}"
        )

@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about a document."""
    try:
        result = await document_processor.answer_question(
            question=request.question,
            document_id=request.document_id,
            top_k=request.top_k or 3,
            model=request.model
        )
        return QuestionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question answering failed: {str(e)}"
        )

@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search for relevant chunks in a document."""
    try:
        from app.services.vector_store import vector_store
        
        results = await vector_store.semantic_search(
            query=request.query,
            doc_id=request.document_id,
            top_k=request.top_k or 5,
            min_score=request.min_score or 0.5
        )
        
        return SearchResponse(
            results=results,
            query=request.query,
            document_id=request.document_id,
            total_found=len(results)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@router.get("/health")
async def documents_health():
    """Health check for document service."""
    try:
        from app.services.vector_store import vector_store
        return {
            "status": "healthy",
            "vector_store_available": vector_store.available,
            "using_pinecone": vector_store._use_pinecone
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }