""
Tests for the document processing pipeline.
"""
import os
import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from app.services.document_processor import (
    process_document,
    answer_question,
    document_processor
)
from app.services.vector_store import VectorStore

# Sample test data
SAMPLE_PDF_URL = "https://example.com/sample.pdf"
SAMPLE_PDF_CONTENT = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << >> /Contents 4 0 R>>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 24 Tf\n100 700 Td\n(This is a test PDF) Tj\nET\nendstream\nendobj\n\nxref\n0 5\n0000000000 65535 f \n0000000015 00000 n \n0000000069 00000 n \n0000000122 00000 n \n0000000202 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n295\n%%EOF"

SAMPLE_TEXT = """
INSURANCE POLICY DOCUMENT

POLICY TERMS AND CONDITIONS

1. Sum Insured: The maximum amount covered under this policy is $1,000,000.

2. Exclusions: The following are excluded from coverage:
   - Pre-existing conditions
   - Intentional damage
   - Acts of war

3. Claims Process:
   - Notify the insurer within 24 hours of the incident
   - Submit all required documents
   - An adjuster will be assigned to your case
"""

@pytest.fixture
def mock_httpx_client():
    with patch('httpx.AsyncClient') as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = SAMPLE_PDF_CONTENT
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
        yield mock_client

@pytest.fixture
def mock_pymupdf():
    with patch('fitz.open') as mock_fitz:
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = SAMPLE_TEXT
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__iter__.return_value = [mock_page]
        mock_fitz.return_value = mock_doc
        yield mock_fitz

@pytest.fixture
def mock_vector_store():
    with patch('app.services.vector_store.VectorStore') as mock_vs:
        mock_instance = AsyncMock()
        mock_vs.return_value = mock_instance
        mock_instance.search_similar.return_value = [
            {
                'id': 'chunk1',
                'score': 0.95,
                'metadata': {
                    'text': 'The maximum amount covered under this policy is $1,000,000.',
                    'document_id': 'test_doc'
                }
            }
        ]
        yield mock_instance

@pytest.fixture
def mock_llm_service():
    with patch('app.services.llm_service.llm_service') as mock_llm:
        mock_llm.generate_text.return_value = "The sum insured is $1,000,000."
        yield mock_llm

@pytest.mark.asyncio
async def test_process_document(mock_httpx_client, mock_pymupdf, mock_vector_store):
    """Test document processing with a mock PDF."""
    # Call the function
    result = await process_document(SAMPLE_PDF_URL)
    
    # Assertions
    assert 'document_id' in result
    assert 'chunks' in result
    assert len(result['chunks']) > 0
    assert 'text' in result['chunks'][0]
    assert 'embedding' in result['chunks'][0]
    
    # Verify the PDF was processed
    mock_pymupdf.assert_called_once()
    
    # Verify the vector store was updated
    mock_vector_store.return_value.upsert_document.assert_awaited_once()

@pytest.mark.asyncio
async def test_answer_question(mock_vector_store, mock_llm_service):
    """Test question answering with mock vector store and LLM."""
    # Call the function
    result = await answer_question(
        question="What is the sum insured?",
        doc_id="test_doc"
    )
    
    # Assertions
    assert 'answer' in result
    assert 'contexts' in result
    assert len(result['contexts']) > 0
    assert 'score' in result['contexts'][0]
    assert 'text' in result['contexts'][0]
    
    # Verify the vector store was queried
    mock_vector_store.return_value.search_similar.assert_awaited_once()
    
    # Verify the LLM was called
    mock_llm_service.generate_text.assert_awaited_once()

def test_chunk_text():
    """Test text chunking functionality."""
    # Setup
    from app.services.document_processor import DocumentProcessor
    processor = DocumentProcessor()
    
    # Test with sample text
    chunks = processor.chunk_text(SAMPLE_TEXT)
    
    # Assertions
    assert len(chunks) > 0
    for chunk in chunks:
        assert 'text' in chunk
        assert 'embedding' in chunk
        assert len(chunk['embedding']) == 384  # all-MiniLM-L6-v2 embedding size
