"""
Test script for the enhanced document processing pipeline.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from dotenv import load_dotenv
from app.services.document_processor import process_document, answer_question
from app.services.vector_store import VectorStore

# Load environment variables
load_dotenv()

async def test_document_processor(pdf_url: str, questions: list[str]):
    """Test the document processing pipeline."""
    print(f"\n{'='*50}")
    print("Testing Document Processing Pipeline")
    print(f"Document URL: {pdf_url}")
    print(f"Questions: {questions}")
    print(f"{'='*50}\n")
    
    try:
        # Step 1: Process the document
        print("Processing document...")
        doc_info = await process_document(pdf_url)
        doc_id = doc_info['document_id']
        print(f"Document processed successfully. ID: {doc_id}")
        print(f"Number of chunks: {len(doc_info['chunks'])}")
        
        # Step 2: Answer questions
        print("\nAnswering questions...")
        for question in questions:
            print(f"\nQuestion: {question}")
            answer = await answer_question(question, doc_id)
            print(f"Answer: {answer.get('answer', 'No answer found')}")
            
            # Show relevant chunks
            if 'contexts' in answer and answer['contexts']:
                print("\nRelevant context:")
                for i, ctx in enumerate(answer['contexts'], 1):
                    print(f"\nContext {i} (Score: {ctx['score']:.3f}):")
                    print(f"{ctx['text']}\n")
                    print(f"-" * 80)
        
        return True
    
    except Exception as e:
        print(f"Error in document processing: {str(e)}", file=sys.stderr)
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
    asyncio.run(test_document_processor(test_pdf_url, test_questions))
