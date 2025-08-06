"""API package."""
from fastapi import APIRouter

router = APIRouter()

# Import all endpoints here to register them with the router
from .endpoints import document_qa

# Include the router from document_qa
router.include_router(document_qa.router)

__all__ = ["router"]
