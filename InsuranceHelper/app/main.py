"""FastAPI application entry point - Updated for HackRx."""
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings

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
    logger.info("Starting Insurance Helper API for HackRx...")
    
    # Initialize services with error handling
    services_status = {}
    
    try:
        from app.services.vector_store import vector_store
        services_status["vector_store"] = "available" if vector_store.available else "unavailable"
        logger.info(f"Vector store: {services_status['vector_store']}")
    except Exception as e:
        logger.warning(f"Vector store initialization failed: {e}")
        services_status["vector_store"] = "unavailable"
    
    try:
        from app.services.document_processor import document_processor
        services_status["document_processor"] = "available"
        logger.info("Document processor initialized")
    except Exception as e:
        logger.warning(f"Document processor initialization failed: {e}")
        services_status["document_processor"] = "unavailable"
    
    try:
        from app.services.llm_service import llm_service
        services_status["llm_service"] = "available" if llm_service.available else "limited"
        logger.info("LLM service initialized")
    except Exception as e:
        logger.warning(f"LLM service initialization failed: {e}")
        services_status["llm_service"] = "unavailable"
    
    # Store services status in app state
    app.state.services_status = services_status
    logger.info("✅ HackRx API ready for submissions!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down HackRx API...")

# Create FastAPI app
app = FastAPI(
    title="Insurance Helper API - HackRx",
    description="API for processing insurance documents and answering questions - HackRx Submission",
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

# Include API routes with error handling
try:
    from app.api.endpoints import documents
    app.include_router(
        documents.router,
        prefix=f"{settings.API_V1_STR}/documents",
        tags=["documents"]
    )
    logger.info("Documents router loaded")
except ImportError as e:
    logger.error(f"Failed to import documents router: {e}")

# Include HackRx router
try:
    from app.api.endpoints import hackrx
    app.include_router(
        hackrx.router,
        prefix=f"{settings.API_V1_STR}/hackrx",
        tags=["hackrx"]
    )
    logger.info("HackRx router loaded")
except ImportError as e:
    logger.error(f"Failed to import hackrx router: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for HackRx deployment."""
    try:
        services_status = getattr(app.state, 'services_status', {})
        
        # Determine overall health
        critical_services = ["document_processor"]
        is_healthy = any(
            services_status.get(service) == "available" 
            for service in critical_services
        )
        
        status_code = 200 if is_healthy else 503
        
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "healthy" if is_healthy else "degraded",
                "version": "1.0.0",
                "hackrx_endpoint": f"{settings.API_V1_STR}/hackrx/run",
                "environment": os.getenv("ENVIRONMENT", "production"),
                "services": services_status,
                "message": "HackRx API is ready!" if is_healthy else "Some services unavailable"
            }
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "version": "1.0.0"
            }
        )

# HackRx root endpoint
@app.get("/")
async def root():
    """Root endpoint for HackRx submission."""
    return {
        "message": "Insurance Helper API - HackRx Submission",
        "status": "ready",
        "hackrx_endpoint": f"{settings.API_V1_STR}/hackrx/run",
        "authentication": "Bearer 0ed1b3e379e363e65b52c090f35648e913017fa88d757a36889962c787daad05",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "internal_error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )