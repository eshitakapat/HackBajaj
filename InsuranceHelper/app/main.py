from fastapi import FastAPI 
from app.api.routes import router as api_router 
import logging 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

app = FastAPI(
    title="Policy Q&A API",
    description="API for processing policy documents and answering insurance questions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.include_router(api_router)