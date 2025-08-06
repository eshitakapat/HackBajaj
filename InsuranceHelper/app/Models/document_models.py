"""Database models for document storage and question-answering."""
import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base

class Document(Base):
    """Stores information about processed documents."""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url = Column(String, nullable=True, unique=True)  # Nullable for file uploads
    document_id = Column(String, nullable=False, unique=True)  # Our internal document ID
    file_name = Column(String)
    file_size = Column(Integer)  # Size in bytes
    content_type = Column(String)  # e.g., 'application/pdf'
    chunk_count = Column(Integer, default=0)  # Number of chunks after processing
    status = Column(String, default="queued")  # queued, processing, processed, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)  # When processing completed
    
    # Relationships
    questions = relationship("Question", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_document_id', 'document_id', unique=True),
        Index('idx_document_url', 'url', unique=True, postgresql_where=(url.isnot(None))),
        Index('idx_document_status', 'status'),
        Index('idx_document_processed_at', 'processed_at'),
    )
    
    def __repr__(self):
        return f"<Document(id={self.id}, url={self.url[:50]}...)>"


class Question(Base):
    """Stores questions and answers for documents."""
    __tablename__ = "questions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    model_used = Column(String, nullable=False)  # e.g., "mistralai/mistral-7b-instruct"
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    processing_time_ms = Column(Integer)  # Time taken to process in milliseconds
    error = Column(Text, nullable=True)
    
    # Metadata
    metadata_ = Column('metadata', Text, nullable=True)  # JSON string of additional metadata
    
    # Relationships
    document = relationship("Document", back_populates="questions")
    
    # Indexes
    __table_args__ = (
        Index('idx_questions_document_id', 'document_id'),
        Index('idx_questions_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Question(id={self.id}, question={self.question[:50]}...)>"
    
    def to_dict(self):
        """Convert the question to a dictionary."""
        return {
            'id': str(self.id),
            'document_id': str(self.document_id),
            'question': self.question,
            'answer': self.answer,
            'model_used': self.model_used,
            'created_at': self.created_at.isoformat(),
            'processing_time_ms': self.processing_time_ms,
            'error': self.error,
            'metadata': self.metadata_
        }
