"""Document processing schemas."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, HttpUrl

class DocumentProcessRequest(BaseModel):
    url: str
    file_name: Optional[str] = None
    content_type: Optional[str] = None

class DocumentProcessResponse(BaseModel):
    document_id: str
    status: str
    chunk_count: int
    processing_time_ms: int
    message: Optional[str] = None

class QuestionRequest(BaseModel):
    document_id: str
    question: str
    model: Optional[str] = None
    top_k: Optional[int] = 3

class QuestionResponse(BaseModel):
    answer: str
    document_id: str
    processing_time_ms: int
    context_chunks: List[Dict[str, Any]]
    model_used: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    document_id: str
    top_k: Optional[int] = 5
    min_score: Optional[float] = 0.5

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str
    document_id: str
    total_found: int