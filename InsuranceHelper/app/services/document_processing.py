"""Module for processing policy documents."""
import io
import logging
from typing import Optional

import httpx
import PyPDF2

logger = logging.getLogger(__name__)

async def process_pdf_from_url(url: str) -> str:
    """
    Download and extract text from a PDF file at the given URL.
    
    Args:
        url: URL of the PDF document
        
    Returns:
        Extracted text from the PDF
        
    Raises:
        HTTPException: If there's an error downloading or processing the PDF
    """
    try:
        async with httpx.AsyncClient() as client:
            # Download the PDF
            response = await client.get(url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            
            # Read PDF content
            with io.BytesIO(response.content) as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = []
                
                # Extract text from each page
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text.strip())
                
                return "\n\n".join(text)
                
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error downloading PDF from {url}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error downloading document: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error processing PDF from {url}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing the document"
        )