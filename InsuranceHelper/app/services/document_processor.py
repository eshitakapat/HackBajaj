"""Document processing service for handling document uploads and text extraction."""
import logging
import os
import re
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any
import httpx
from sentence_transformers import SentenceTransformer
import torch
from sqlalchemy.orm import Session

from app.services.document_store import DocumentStore
from app.services.vector_store import vector_store
from app.services.llm_prompt import llm_prompt_service
from app.models.document_models import Document, Question

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing and question answering."""
    
    def __init__(self, db: Session):
        self.vector_store = vector_store
        self.document_store = DocumentStore(db)
        self.embedding_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model for embeddings."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(
                'all-MiniLM-L6-v2',  # Lightweight but effective model
                device=self.device
            )
    
    async def download_document(self, url: str) -> bytes:
        """Download a document from a URL or read from local file.
        
        Args:
            url: URL or local file path of the document (can be file://, http://, https://, or local path)
            
        Returns:
            Document content as bytes
            
        Raises:
            FileNotFoundError: If local file doesn't exist
            httpx.HTTPStatusError: For HTTP request errors
            ValueError: For unsupported URL schemes
        """
        try:
            # Handle file:// URLs
            if url.startswith('file://'):
                file_path = url[7:]  # Remove 'file://' prefix
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                with open(file_path, 'rb') as f:
                    return f.read()
                    
            # Handle local file paths
            elif os.path.exists(url):
                with open(url, 'rb') as f:
                    return f.read()
                    
            # Handle relative paths from project root
            project_root = Path(__file__).parent.parent.parent
            abs_path = (project_root / url).resolve()
            if abs_path.exists():
                with open(abs_path, 'rb') as f:
                    return f.read()
                    
            # Handle HTTP/HTTPS URLs
            elif url.startswith(('http://', 'https://')):
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, follow_redirects=True, timeout=30.0)
                    response.raise_for_status()
                    return response.content
                    
            raise ValueError(f"Unsupported URL or file path format: {url}")
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {url}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} for URL: {url}")
            raise
        except Exception as e:
            logger.error(f"Error processing document from {url}: {str(e)}", exc_info=True)
            raise
    
    def extract_text(self, content: bytes, content_type: str) -> str:
        """Extract text from document content.
        
        Args:
            content: Document content as bytes
            content_type: MIME type of the document
            
        Returns:
            Extracted text as a string
        """
        try:
            if content_type == 'application/pdf':
                return self._extract_text_from_pdf(content)
            # Add support for other document types as needed
            else:
                return content.decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to extract text: {str(e)}")
            raise
    
    def _extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF content using PyMuPDF for better text extraction.
        
        Args:
            content: PDF document as bytes
            
        Returns:
            Extracted text as a string with preserved layout
        """
        try:
            # First try PyMuPDF for better text extraction
            with fitz.open(stream=content, filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text("text") + "\n\n"
                
                # If PyMuPDF returns very little text, try pdfplumber as fallback
                if len(text.strip()) < 100:  # Arbitrary threshold
                    with pdfplumber.open(BytesIO(content)) as pdf:
                        text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
                
                return text.strip()
                
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {str(e)}")
            # Fallback to PyPDF2 if both methods fail
            try:
                with BytesIO(content) as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    return "\n\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as e2:
                logger.error(f"Fallback PDF extraction also failed: {str(e2)}")
                raise
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks with overlapping windows.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks with metadata and embeddings
        """
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_chunk and current_length + sentence_length > 1000:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'sentences': current_chunk.copy(),
                    'char_start': text.find(chunk_text),
                    'char_end': text.find(chunk_text) + len(chunk_text)
                })
                
                # Keep last N sentences for overlap
                overlap_size = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    if overlap_size + len(sent) > 200 and overlap_sentences:
                        break
                    overlap_sentences.insert(0, sent)
                    overlap_size += len(sent) + 1  # +1 for space
                
                current_chunk = overlap_sentences
                current_length = sum(len(s) + 1 for s in current_chunk) - 1
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space
        
        # Add the last chunk if not empty
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'sentences': current_chunk.copy(),
                'char_start': text.find(chunk_text),
                'char_end': text.find(chunk_text) + len(chunk_text)
            })
        
        # Generate embeddings for each chunk
        self._initialize_embedding_model()
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Process in batches to avoid OOM errors with large documents
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            all_embeddings.extend(batch_embeddings)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = all_embeddings[i].tolist()
        
        return chunks
    
    async def answer_question(
        self,
        question: str,
        document_id: str,
        top_k: int = 3,
        model: str = "mistralai/mistral-7b-instruct"
    ) -> Dict[str, Any]:
        """Answer a question about a document."""
        import time
        
        start_time = time.time()
        
        try:
            # Get relevant chunks
            relevant_chunks = await self.vector_store.search_similar(
                query=question,
                namespace=document_id,
                top_k=top_k
            )
            
            if not relevant_chunks:
                return {
                    "answer": "I couldn't find any relevant information in the document to answer your question.",
                    "context_chunks": []
                }
            
            # Get LLM answer
            result = await llm_prompt_service.get_answer(
                question=question,
                context_chunks=relevant_chunks,
                model=model
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Log the question and answer
            await self.document_store.log_question(
                document_id=document_id,
                question=question,
                answer=result['answer'],
                model_used=model,
                processing_time_ms=processing_time,
                metadata={
                    'top_k': top_k,
                    'chunk_scores': [c['score'] for c in relevant_chunks]
                }
            )
            
            return {
                **result,
                "document_id": document_id,
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error answering question: {error_msg}", exc_info=True)
            
            # Log the error
            if 'document_store' in locals():
                await self.document_store.log_question(
                    document_id=document_id,
                    question=question,
                    answer="",
                    model_used=model,
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    error=error_msg
                )
            
            raise
    
    def generate_document_id(self, text: str) -> str:
        """Generate a unique ID for a document based on its content.
        
        Args:
            text: Document text
            
        Returns:
            Unique document ID
        """
        return hashlib.md5(text.encode()).hexdigest()
    
    async def process_document(
        self,
        url: str,
        file_name: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a document and store it in the database.
        
        Args:
            url: URL or path to the document
            file_name: Original file name (optional)
            content_type: MIME type of the document (optional)
            
        Returns:
            Dictionary containing processing results
        """
        import time
        
        start_time = time.time()
        document_id = f"doc_{uuid.uuid4().hex}"
        
        try:
            # Create document record
            await self.document_store.create_document(
                url=url,
                document_id=document_id,
                status="processing"
            )
            
            # Download the document
            content = await self.download_document(url)
            
            # Get file size if not provided
            file_size = len(content)
            
            # Try to determine content type if not provided
            if not content_type:
                if url.lower().endswith('.pdf'):
                    content_type = 'application/pdf'
                elif url.lower().endswith(('.txt', '.md')):
                    content_type = 'text/plain'
                else:
                    content_type = 'application/octet-stream'
            
            # Extract text from the document
            text = self.extract_text(content, content_type)
            
            if not text.strip():
                raise ValueError("No text could be extracted from the document")
            
            # Chunk the text
            chunks = self.chunk_text(text)
            
            # Store in vector database
            await self.vector_store.upsert_document(document_id, chunks)
            
            # Update document status
            await self.document_store.update_document_status(
                document_id=document_id,
                status="completed",
                file_name=file_name or os.path.basename(url),
                file_size=file_size,
                content_type=content_type
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                "document_id": document_id,
                "status": "completed",
                "chunk_count": len(chunks),
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing document: {error_msg}", exc_info=True)
            
            # Update document with error status
            if 'document_store' in locals():
                await self.document_store.update_document_status(
                    document_id=document_id,
                    status="failed",
                    error_message=error_msg
                )
            
            raise

# Singleton instance will be created in app/deps.py
document_processor = None
