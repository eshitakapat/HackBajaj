"""
Test script for semantic search functionality.
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from dotenv import load_dotenv
from app.services.semantic_search import semantic_search
from app.services.document_processor import process_document

# Load environment variables
load_dotenv()

async def test_semantic_search(pdf_url: str, questions: List[str], top_k: int = 3):
    """Test the semantic search functionality."""
    print(f"\n{'='*50}")
    print("Testing Semantic Search")
    print(f"Document URL: {pdf_url}")
    print(f"{'='*50}\n")
    
    try:
        # Step 1: Process the document
        print("Processing document...")
        doc_info = await process_document(pdf_url)
        doc_id = doc_info['document_id']
        print(f"Document processed. ID: {doc_id}")
        
        # Step 2: Test semantic search for each question
        for question in questions:
            print(f"\nQuestion: {question}")
            print("-" * 80)
            
            # Get relevant chunks
            chunks = await semantic_search.get_relevant_chunks(
                question=question,
                document_id=doc_id,
                top_k=top_k
            )
            
            if not chunks:
                print("No relevant chunks found.")
                continue
                
            # Print results
            for i, chunk in enumerate(chunks, 1):
                print(f"\nChunk {i} (Score: {chunk['score']:.3f}):")
                print("-" * 50)
                print(chunk['text'])
                print("-" * 50)
            
            # Get context block
            context = await semantic_search.get_context_block(
                question=question,
                document_id=doc_id,
                top_k=top_k
            )
            
            print("\nContext Block:")
            print("=" * 80)
            print(context)
            print("=" * 80)
        
        return True
    
    except Exception as e:
        print(f"Error in semantic search test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test with a sample insurance policy PDF
    test_pdf_url = "https://www.irdai.gov.in/ADMINCMS/cms/Uploadedfiles/Regulation/Non-Life/IRDAI/Eng/IRDAI-NonLife-Eng-01042020.pdf"
    test_questions = [
        "What is the sum insured?",
        "What are the exclusions in this policy?",
        "What is the claim process?"
    ]
    
    # Run the test
    asyncio.run(test_semantic_search(test_pdf_url, test_questions))
