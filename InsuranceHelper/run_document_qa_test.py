"""
Run document QA test with a local PDF file.

This script:
1. Starts the FastAPI server in a separate process
2. Uploads the test document
3. Asks questions and displays answers
4. Cleans up when done
"""
import os
import sys
import time
import json
import uvicorn
import httpx
import asyncio
from pathlib import Path
from multiprocessing import Process
from typing import List, Dict, Any

# Configuration
HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"
TEST_DOC_PATH = Path("dataSamples/Arogya Sanjeevani Policy - CIN - U10200WB1906GOI001713 1.pdf").absolute()

# Ensure the test document exists
if not TEST_DOC_PATH.exists():
    raise FileNotFoundError(f"Test document not found at: {TEST_DOC_PATH}")
TEST_QUESTIONS = [
    "What is the sum insured under this policy?",
    "What are the policy exclusions?",
    "What is the waiting period for pre-existing diseases?",
    "What is the policy term?",
    "What is the process for making a claim?"
]

class DocumentQATester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def upload_document(self, file_path: Path) -> str:
        """Upload a document for processing."""
        url = f"{self.base_url}/api/upload"
        
        try:
            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f, "application/pdf")}
                response = await self.client.post(url, files=files)
                response.raise_for_status()
                return response.json()["file_path"]
        except Exception as e:
            print(f"Error uploading document: {str(e)}")
            raise
    
    async def upload_document(self, file_path: Path) -> str:
        """Upload a document for processing."""
        url = f"{self.base_url}/api/upload"
        
        try:
            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f, "application/pdf")}
                response = await self.client.post(url, files=files)
                response.raise_for_status()
                return response.json()["file_path"]
        except Exception as e:
            print(f"Error uploading document: {str(e)}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            raise
    
    async def ask_questions(self, document_path: str, questions: List[str]) -> List[Dict[str, Any]]:
        """Ask questions about the uploaded document."""
        url = f"{self.base_url}/api/hackrx/run"
        
        try:
            # Use the absolute path directly
            document_url = os.path.abspath(document_path)
            
            payload = {
                "url": document_url,
                "questions": questions,
                "top_k": 3
            }
            
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error asking questions: {str(e)}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            raise
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

def print_qa_results(results: Dict[str, Any]):
    """Print the QA results in a readable format."""
    print(f"\n{'='*80}")
    print(f"Document ID: {results.get('document_id')}")
    print(f"Timestamp: {results.get('timestamp')}")
    print(f"Questions Processed: {len(results.get('answers', []))}")
    print(f"{'='*80}\n")
    
    for i, answer in enumerate(results.get('answers', []), 1):
        print(f"{i}. Question: {answer.get('question')}")
        print(f"   Answer: {answer.get('answer', 'No answer provided.')}")
        
        sources = answer.get('sources', [])
        if sources:
            print(f"\n   Sources (Top {len(sources)} relevant chunks):")
            for j, source in enumerate(sources, 1):
                score = source.get('score', 0)
                text = source.get('text', '').replace('\n', ' ').strip()
                print(f"   {j}. [Relevance: {score:.2f}] {text[:200]}{'...' if len(text) > 200 else ''}")
        
        print(f"\n{'─'*80}")

async def run_tests():
    """Run the document QA tests."""
    if not TEST_DOC_PATH.exists():
        print(f"Error: Test document not found at {TEST_DOC_PATH}")
        return
    
    tester = DocumentQATester(BASE_URL)
    
    try:
        print(f"Starting document QA test with: {TEST_DOC_PATH.name}")
        
        # 1. Upload the document
        print("\n1. Uploading document...")
        document_path = await tester.upload_document(TEST_DOC_PATH)
        print(f"   Document uploaded successfully: {document_path}")
        
        # Give the server a moment to process the upload
        await asyncio.sleep(2)
        
        # 2. Ask questions
        print(f"\n2. Asking {len(TEST_QUESTIONS)} questions about the document...")
        results = await tester.ask_questions(document_path, TEST_QUESTIONS)
        
        # 3. Print results
        print_qa_results(results)
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
    finally:
        await tester.close()

def start_server():
    """Start the FastAPI server."""
    uvicorn.run("app.main:app", host=HOST, port=PORT, log_level="info")

if __name__ == "__main__":
    import multiprocessing
    
    # Start the server in a separate process
    server_process = multiprocessing.Process(target=start_server)
    server_process.start()
    
    try:
        # Give the server a moment to start
        time.sleep(2)
        
        # Run the tests
        asyncio.run(run_tests())
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        # Clean up the server process
        server_process.terminate()
        server_process.join()
        print("\nTest server stopped")
