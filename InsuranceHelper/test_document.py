import os
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
import httpx

# Add project root to path
project_root = str(Path(__file__).parent.absolute())
sys.path.append(project_root)

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = "http://localhost:8000"  # Update if your API runs on a different port
API_KEY = os.getenv("API_KEY")
DOCUMENT_PATH = os.path.join("dataSamples", "Arogya Sanjeevani Policy - CIN - U10200WB1906GOI001713 1.pdf")

async def test_document_processing():
    """Test the document processing and question answering."""
    if not API_KEY:
        print("Error: API_KEY not found in environment variables")
        return
    
    if not os.path.exists(DOCUMENT_PATH):
        print(f"Error: Document not found at {DOCUMENT_PATH}")
        return
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }
    
    async with httpx.AsyncClient() as client:
        # Test 1: Upload and process document
        print("\n=== Uploading and Processing Document ===")
        files = {"file": open(DOCUMENT_PATH, "rb")}
        
        try:
            response = await client.post(
                f"{BASE_URL}/api/upload",
                headers=headers,
                files=files,
                timeout=60.0  # Increased timeout for file upload
            )
            response.raise_for_status()
            document_id = response.json().get("document_id")
            print(f"Document uploaded successfully. ID: {document_id}")
            
            # Test 2: Ask questions about the document
            print("\n=== Asking Questions ===")
            questions = [
                "What is the sum insured under this policy?",
                "What are the waiting periods mentioned?",
                "Does this policy cover pre-existing conditions?"
            ]
            
            for question in questions:
                print(f"\nQuestion: {question}")
                response = await client.post(
                    f"{BASE_URL}/api/ask",
                    headers={"Content-Type": "application/json", **headers},
                    json={
                        "document_id": document_id,
                        "question": question
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                answer = response.json().get("answer", "No answer found")
                print(f"Answer: {answer}")
                
        except httpx.HTTPStatusError as e:
            print(f"Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
        finally:
            files["file"].close()

if __name__ == "__main__":
    asyncio.run(test_document_processing())
