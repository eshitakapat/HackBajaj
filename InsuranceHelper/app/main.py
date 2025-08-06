from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Generator
import os

from sqlalchemy.orm import Session

# Import database and models
from .db.session import SessionLocal, engine
from .models.base import Base
from .models import document_models  # noqa: F401

# Import API router
from .api import router as api_router

# Import services and config
from .services.document_processor import DocumentProcessor
from .core.config import settings

# Initialize database tables
Base.metadata.create_all(bind=engine)

def get_db() -> Generator[Session, None, None]:
    """Dependency for getting DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_document_processor() -> DocumentProcessor:
    """Dependency for getting document processor."""
    return DocumentProcessor(SessionLocal())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Insurance Document QA API",
    description="API for processing and querying insurance documents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router with version prefix
app.include_router(api_router, prefix=settings.API_V1_STR)

# Health check endpoint
@app.get(f"{settings.API_V1_STR}/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )