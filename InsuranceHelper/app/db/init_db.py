"""Initialize the database with tables."""
import logging
from sqlalchemy.orm import Session

# Relative imports
from .session import engine, Base
from ..models.document_models import Document, Question
from ..models.user import User  # Import other models as needed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db() -> None:
    """Initialize the database by creating all tables."""
    try:
        # Import all models here to ensure they are registered with SQLAlchemy
        from app.models.user import User  # noqa: F401
        from app.models.document_models import Document, Question  # noqa: F401
        
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting database initialization...")
    init_db()
