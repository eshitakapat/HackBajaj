"""FastAPI application entry point."""
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.endpoints import documents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Insurance Helper API...")
    
    # Initialize services
    try:
        from app.services.vector_store import vector_store
        from app.services.document_processor import document_processor
        from app.services.llm_service import llm_service
        
        logger.info(f"Vector store available: {vector_store.available}")
        logger.info(f"Using Pinecone: {vector_store._use_pinecone}")
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Insurance Helper API...")

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for processing policy documents and answering insurance questions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(
    documents.router,
    prefix=f"{settings.API_V1_STR}/documents",
    tags=["documents"]
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        from app.services.vector_store import vector_store
        
        return {
            "status": "ok",
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "services": {
                "vector_store": "available" if vector_store.available else "unavailable",
                "pinecone": "enabled" if vector_store._use_pinecone else "disabled"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Insurance Helper API",
        "docs": "/docs",
        "health": "/health",
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