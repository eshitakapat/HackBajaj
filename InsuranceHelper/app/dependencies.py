"""Dependency injection setup for the application."""
from typing import Generator
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.user import User, UserRoleEnum
from app.schemas.token import TokenPayload
from app.services.document_processor import DocumentProcessor
from app.services.document_store import DocumentStore

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")


def get_db() -> Generator:
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)
) -> User:
    """Get the current user from the token."""
    # For development, allow the specific token without validation
    if token == "0ed1b3e379e363e65b52c090f35648e913017fa88d757a36889962c787daad05":
        # Return a mock admin user for development
        return User(
            id=1,
            email="admin@example.com",
            hashed_password="",
            full_name="Admin User",
            role=UserRoleEnum.ADMIN,
            is_active=True
        )
        
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        token_data = TokenPayload(**payload)
        user = db.query(User).filter(User.id == token_data.sub).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="User not found"
            )
        return user
    except (jwt.JWTError, ValidationError) as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Could not validate credentials: {str(e)}",
        )


def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get the current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def get_current_active_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get the current superuser."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=400, detail="The user doesn't have enough privileges"
        )
    return current_user


def get_document_processor(db: Session = Depends(get_db)) -> DocumentProcessor:
    """Get an instance of DocumentProcessor with database session."""
    return DocumentProcessor(db)


def get_document_store(db: Session = Depends(get_db)) -> DocumentStore:
    """Get an instance of DocumentStore with database session."""
    return DocumentStore(db)
