"""Document processing service for handling document uploads and text extraction."""
import hashlib
import logging
import os
import re
import uuid
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import httpx
import torch

from app.services.embedding_search import PolicyEmbeddingGenerator

# PDF processing imports
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

from app.core.config import settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing and question answering."""
    
    def __init__(self):
        self.embedding_model = PolicyEmbeddingGenerator()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vector_store = None
        self.llm_service = None
        logger.info(f"DocumentProcessor initialized with device: {self.device}")
        
    def _initialize_services(self):
        """Initialize required services."""
        if self.vector_store is None:
            from app.services.vector_store import vector_store
            self.vector_store = vector_store
            
        if self.llm_service is None:
            from app.services.llm_service import llm_service
            self.llm_service = llm_service
        
    def _initialize_embedding_model(self):
        """Initialize the embedding model."""
        # No need to initialize here as it's done in the PolicyEmbeddingGenerator
        pass
    
    async def download_document(self, url: str) -> bytes:
        """Download a document from a URL or read from local file."""
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
            
        except Exception as e:
            logger.error(f"Error downloading document from {url}: {str(e)}")
            raise
    
    def extract_text(self, content: bytes, content_type: str) -> str:
        """Extract text from document content."""
        try:
            if content_type == 'application/pdf':
                return self._extract_text_from_pdf(content)
            else:
                # Try to decode as text
                try:
                    return content.decode('utf-8')
                except UnicodeDecodeError:
                    return content.decode('latin-1')
        except Exception as e:
            logger.error(f"Failed to extract text: {str(e)}")
            raise
    
    def _extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF content."""
        text = ""
        
        # Try PyMuPDF first (best quality)
        if HAS_PYMUPDF:
            try:
                with fitz.open(stream=content, filetype="pdf") as doc:
                    for page in doc:
                        text += page.get_text("text") + "\n\n"
                if text.strip():
                    return text.strip()
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed: {e}")
        
        # Fallback to pdfplumber
        if HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(BytesIO(content)) as pdf:
                    text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
                if text.strip():
                    return text.strip()
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Last resort: PyPDF2
        if HAS_PYPDF2:
            try:
                with BytesIO(content) as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
                if text.strip():
                    return text.strip()
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # If all methods fail
        raise ValueError("Could not extract text from PDF. No PDF libraries available or all extraction methods failed.")
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks with overlapping windows."""
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
                    'char_start': text.find(chunk_text) if chunk_text in text else 0,
                    'char_end': text.find(chunk_text) + len(chunk_text) if chunk_text in text else len(chunk_text)
                })
                
                # Keep last sentences for overlap
                overlap_size = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    if overlap_size + len(sent) > 200 and overlap_sentences:
                        break
                    overlap_sentences.insert(0, sent)
                    overlap_size += len(sent) + 1
                
                current_chunk = overlap_sentences
                current_length = sum(len(s) + 1 for s in current_chunk) - 1 if current_chunk else 0
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1
        
        # Add the last chunk if not empty
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'sentences': current_chunk.copy(),
                'char_start': text.find(chunk_text) if chunk_text in text else 0,
                'char_end': text.find(chunk_text) + len(chunk_text) if chunk_text in text else len(chunk_text)
            })
        
        # Generate embeddings for each chunk
        self._initialize_embedding_model()
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Process in batches to avoid OOM errors
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
    
    def generate_document_id(self, text: str) -> str:
        """Generate a unique ID for a document based on its content."""
        return hashlib.md5(text.encode()).hexdigest()[:16]  # Shorter ID
    
    async def process_document(
        self,
        url: str,
        file_name: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a document and store it in Pinecone."""
        start_time = time.time()
        document_id = f"doc_{uuid.uuid4().hex[:8]}"  # Shorter ID
        
        try:
            self._initialize_services()
            
            # Download the document
            logger.info(f"Downloading document from: {url}")
            content = await self.download_document(url)
            
            # Get file size
            file_size = len(content)
            
            # Determine content type if not provided
            if not content_type:
                if url.lower().endswith('.pdf'):
                    content_type = 'application/pdf'
                elif url.lower().endswith(('.txt', '.md')):
                    content_type = 'text/plain'
                else:
                    content_type = 'application/octet-stream'
            
            # Extract text from the document
            logger.info("Extracting text from document")
            text = self.extract_text(content, content_type)
            
            if not text.strip():
                raise ValueError("No text could be extracted from the document")
            
            # Chunk the text
            logger.info("Chunking text and generating embeddings")
            chunks = self.chunk_text(text)
            
            # Store in vector database
            logger.info(f"Storing {len(chunks)} chunks in Pinecone")
            success = await self.vector_store.upsert_document(document_id, chunks)
            
            if not success:
                raise Exception("Failed to store document in vector database")
            
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.info(f"Document {document_id} processed successfully in {processing_time}ms")
            
            return {
                "document_id": document_id,
                "status": "completed",
                "chunk_count": len(chunks),
                "processing_time_ms": processing_time,
                "file_size": file_size,
                "content_type": content_type
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing document: {error_msg}", exc_info=True)
            raise
    
    async def answer_question(
        self,
        question: str,
        document_id: str,
        top_k: int = 3,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Answer a question about a document."""
        start_time = time.time()
        
        try:
            self._initialize_services()
            
            # Get relevant chunks from vector store
            logger.info(f"Searching for relevant chunks for question: {question[:50]}...")
            relevant_chunks = await self.vector_store.semantic_search(
                query=question,
                doc_id=document_id,
                top_k=top_k,
                min_score=0.3  # Lower threshold for better recall
            )
            
            if not relevant_chunks:
                return {
                    "answer": "I couldn't find any relevant information in the document to answer your question.",
                    "context_chunks": [],
                    "document_id": document_id,
                    "processing_time_ms": int((time.time() - start_time) * 1000)
                }
            
            # Prepare context for LLM
            context_text = "\n\n".join([
                f"[Context {i+1}] {chunk['text']}" 
                for i, chunk in enumerate(relevant_chunks)
            ])
            
            # Create prompt for LLM
            prompt = f"""Based on the following document context, please answer the question. If the information is not available in the context, please say so.

Context:
{context_text}

Question: {question}

Answer:"""
            
            # Get answer from LLM
            model = model or settings.OPENROUTER_MODEL
            logger.info(f"Getting answer from LLM model: {model}")
            
            answer = await self.llm_service.generate_text(
                prompt=prompt,
                model=model,
                max_tokens=1000,
                temperature=0.3
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                "answer": answer.strip(),
                "context_chunks": relevant_chunks,
                "document_id": document_id,
                "processing_time_ms": processing_time,
                "model_used": model
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}", exc_info=True)
            raise

# Global instance
document_processor = DocumentProcessor()