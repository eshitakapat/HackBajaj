"""Integration tests for the Insurance Helper application."""
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# Try to import sentence-transformers for real embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False

from app.services.document_processor import DocumentProcessor
from app.services.vector_store import vector_store

# Test document path
TEST_DOC_PATH = Path(__file__).parent.parent / "dataSamples" / "Arogya Sanjeevani Policy - CIN - U10200WB1906GOI001713 1.pdf"

# Skip if test document doesn't exist
if not TEST_DOC_PATH.exists():
    pytest.skip(f"Test document not found at {TEST_DOC_PATH}", allow_module_level=True)

# Initialize services
document_processor = DocumentProcessor()

# Initialize embedding model if available
if EMBEDDING_AVAILABLE:
    # Using a small, fast model for testing
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def generate_embedding(text):
        """Generate real embeddings using sentence-transformers."""
        return embedding_model.encode(text, convert_to_numpy=True).tolist()
else:
    # Fallback to mock embeddings
    def generate_embedding(text, dim=1024, seed=42):
        """Generate a deterministic mock embedding for testing."""
        np.random.seed(hash(text) % 10000 + seed)
        return (np.random.rand(dim) * 2 - 1).tolist()
    
    print("\nWarning: Using mock embeddings. Install sentence-transformers for real embeddings.")
    print("Run: pip install sentence-transformers\n")

@pytest.mark.asyncio
async def test_document_processing_and_qa():
    """Test the full document processing and QA pipeline."""
    # 1. Process the document
    print(f"\nProcessing document: {TEST_DOC_PATH.name}")
    
    try:
        # Read document content
        with open(TEST_DOC_PATH, 'rb') as f:
            document_content = f.read()
        
        # Process document into chunks
        doc_id, chunks = await document_processor.process_document(
            document_content, 
            content_type='application/pdf'
        )
        assert len(chunks) > 0, "No chunks were generated from the document"
        print(f"✓ Document processed into {len(chunks)} chunks")
        
        # 2. Generate embeddings for each chunk
        print("\nGenerating embeddings...")
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings (real or mock)
        embeddings = [generate_embedding(text) for text in chunk_texts]
        print(f"✓ Generated {len(embeddings)} embeddings with dimension {len(embeddings[0]) if embeddings else 0}")
        
        # 3. Store in vector database
        document_id = f"test_doc_{hash(str(chunks[0]['text']))}"
        
        # Prepare vectors for upsert
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{document_id}_{i}"
            metadata = {
                'text': chunk['text'][:1000],  # Truncate long text
                'document_id': document_id,
                'chunk_index': i,
                'char_start': chunk.get('char_start', 0),
                'char_end': chunk.get('char_end', 0)
            }
            vectors.append((vector_id, embedding, metadata))
        
        # Skip vector store operations if not available
        if not vector_store.available:
            print("Skipping vector store operations (not available)")
            return
            
        print(f"Attempting to upsert {len(vectors)} vectors to vector store...")
        success = await vector_store.upsert_vectors(vectors)
        assert success, "Failed to upsert vectors to vector store"
        print(f"✓ Successfully stored {len(vectors)} vectors in vector store")
        
        # 4. Test similarity search
        # Use the first chunk's text as the query to ensure we get a match
        query_text = chunks[0]['text'][:200]  # First 200 chars of first chunk
        print(f"\nQuery text: {query_text[:100]}...")
        
        # Generate embedding for the query using the same model
        print("Generating query embedding...")
        query_embedding = generate_embedding(query_text)
        
        print("Performing similarity search...")
        results = await vector_store.search_similar(
            query_embedding,
            top_k=min(3, len(vectors)),
            include_metadata=True
        )
        
        print(f"Search returned {len(results)} results")
        if results:
            print(f"\nTop result:")
            print(f"- Score: {results[0].get('score', 0):.4f}")
            print(f"- Text: {results[0].get('metadata', {}).get('text', '')[:200]}...")
        
        assert len(results) > 0, "No results found in vector search"
        print(f"✓ Found {len(results)} similar vectors")
        
        # 5. Clean up test data
        if vector_store.available and 'document_id' in locals():
            print("Cleaning up test vectors...")
            await vector_store.delete_vectors(
                ids=[v[0] for v in vectors],
                namespace=None
            )
            print("✓ Cleaned up test vectors")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--asyncio-mode=auto"])
