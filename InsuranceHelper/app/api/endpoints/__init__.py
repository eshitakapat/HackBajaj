"""API endpoints package."""
from fastapi import APIRouter
from . import document_qa

# Create a router for all endpoints
router = APIRouter()
router.include_router(document_qa.router, prefix="/documents", tags=["documents"])

__all__ = ["document_qa", "router"]
