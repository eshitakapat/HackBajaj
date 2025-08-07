"""Document processing schemas with better validation."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, HttpUrl, Field

class DocumentProcessRequest(BaseModel):
    url: str = Field(..., description="URL or path to the document")
    file_name: Optional[str] = Field(None, description="Optional file name")
    content_type: Optional[str] = Field(None, description="MIME type of the document")

class DocumentProcessResponse(BaseModel):
    document_id: str = Field(..., description="Unique document identifier")
    status: str = Field(..., description="Processing status")
    chunk_count: int = Field(..., description="Number of text chunks created")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    message: Optional[str] = Field(None, description="Additional message")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    content_type: Optional[str] = Field(None, description="Document MIME type")

class QuestionRequest(BaseModel):
    document_id: str = Field(..., description="Document ID to query")
    question: str = Field(..., min_length=1, description="Question about the document")
    model: Optional[str] = Field(None, description="LLM model to use")
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Number of context chunks")

class QuestionResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    document_id: str = Field(..., description="Document ID that was queried")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    context_chunks: List[Dict[str, Any]] = Field(..., description="Context chunks used")
    model_used: Optional[str] = Field(None, description="Model used for generation")

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    document_id: str = Field(..., description="Document ID to search within")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of results")
    min_score: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity score")

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    query: str = Field(..., description="Original search query")
    document_id: str = Field(..., description="Document ID searched")
    total_found: int = Field(..., description="Total number of results found")