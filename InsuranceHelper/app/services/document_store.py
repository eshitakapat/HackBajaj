"""Service for storing and retrieving documents and questions."""
import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import or_

from app.models.document_models import Document, Question
from app.core.logging import logger

class DocumentStore:
    """Handles storage and retrieval of documents and questions."""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def create_document(
        self,
        url: str,
        document_id: str,
        file_name: Optional[str] = None,
        file_size: Optional[int] = None,
        content_type: Optional[str] = None,
        status: str = "processing"
    ) -> Document:
        """Create a new document record.
        
        Args:
            url: URL or path of the document
            document_id: Unique identifier for the document
            file_name: Original name of the file
            file_size: Size of the file in bytes
            content_type: MIME type of the document
            status: Initial processing status
            
        Returns:
            The created Document object
            
        Raises:
            SQLAlchemyError: If there's a database error
        """
        try:
            # Check if document with same URL or document_id already exists
            existing_doc = self.db.query(Document).filter(
                or_(
                    Document.url == url,
                    Document.document_id == document_id
                )
            ).first()
            
            if existing_doc:
                # Update existing document
                existing_doc.file_name = file_name or existing_doc.file_name
                existing_doc.file_size = file_size or existing_doc.file_size
                existing_doc.content_type = content_type or existing_doc.content_type
                existing_doc.status = status
                self.db.commit()
                self.db.refresh(existing_doc)
                return existing_doc
            
            # Create new document
            document = Document(
                url=url,
                document_id=document_id,
                file_name=file_name,
                file_size=file_size,
                content_type=content_type,
                status=status
            )
            self.db.add(document)
            self.db.commit()
            self.db.refresh(document)
            return document
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating/updating document: {str(e)}")
            raise
    
    async def update_document_status(
        self,
        document_id: str,
        status: str,
        error_message: Optional[str] = None,
        file_name: Optional[str] = None,
        file_size: Optional[int] = None,
        content_type: Optional[str] = None
    ) -> Optional[Document]:
        """Update document status and error message.
        
        Args:
            document_id: ID of the document to update
            status: New status for the document
            error_message: Optional error message if processing failed
            file_name: Optional file name to update
            file_size: Optional file size in bytes to update
            content_type: Optional content type to update
            
        Returns:
            The updated Document object or None if not found
        """
        try:
            document = self.db.query(Document).filter(
                Document.document_id == document_id
            ).first()
            if document:
                document.status = status
                if error_message:
                    document.error_message = error_message
                if file_name:
                    document.file_name = file_name
                if file_size is not None:
                    document.file_size = file_size
                if content_type:
                    document.content_type = content_type
                document.updated_at = datetime.utcnow()
                self.db.commit()
                self.db.refresh(document)
                return document
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating document status: {str(e)}")
            raise
            
    async def update_document_after_processing(
        self,
        document_id: str,
        status: str,
        chunk_count: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> Optional[Document]:
        """Update document after processing is complete.
        
        Args:
            document_id: ID of the document to update
            status: Final status of processing (e.g., 'processed', 'failed')
            chunk_count: Number of chunks created during processing
            error_message: Error message if processing failed
            
        Returns:
            The updated Document object or None if not found
        """
        try:
            document = self.db.query(Document).filter(
                Document.document_id == document_id
            ).first()
            
            if document:
                document.status = status
                document.processed_at = datetime.utcnow()
                document.updated_at = datetime.utcnow()
                
                if chunk_count is not None:
                    document.chunk_count = chunk_count
                    
                if error_message:
                    document.error_message = error_message
                    
                self.db.commit()
                self.db.refresh(document)
                return document
                
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating document after processing: {str(e)}")
            raise
    
    async def log_question(
        self,
        document_id: str,
        question: str,
        answer: str,
        model_used: str,
        processing_time_ms: int,
        error: Optional[str] = None,
        metadata: Optional[Union[Dict[str, Any], str]] = None
    ) -> Question:
        """Log a question and its answer."""
        try:
            # Get the document
            document = self.db.query(Document).filter(
                Document.document_id == document_id
            ).first()
            
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # Create the question
            metadata_str = json.dumps(metadata) if isinstance(metadata, dict) else metadata
            
            question_obj = Question(
                document_id=document_id,
                question=question,
                answer=answer,
                model_used=model_used,
                processing_time_ms=processing_time_ms,
                error=error,
                metadata=metadata_str
            )
            
            self.db.add(question_obj)
            self.db.commit()
            self.db.refresh(question_obj)
            return question_obj
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error logging question: {str(e)}")
            raise
    
    async def get_document_questions(
        self,
        document_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Question]:
        """Get all questions for a document.
        
        Args:
            document_id: ID of the document
            limit: Maximum number of questions to return
            offset: Number of questions to skip
            
        Returns:
            List of Question objects
        """
        try:
            questions = self.db.query(Question).join(Document).filter(
                Document.document_id == document_id
            ).order_by(
                Question.created_at.desc()
            ).offset(offset).limit(limit).all()
            
            return questions
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting document questions: {str(e)}")
            raise
    
    async def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """Get a document by its ID.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            Document object if found, None otherwise
        """
        try:
            document = self.db.query(Document).filter(
                Document.document_id == document_id
            ).first()
            
            return document
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting document by ID: {str(e)}")
            raise
            
    async def get_document_by_url(self, url: str) -> Optional[Document]:
        """Get a document by its URL.
        
        Args:
            url: URL of the document to retrieve
            
        Returns:
            Document object if found, None otherwise
        """
        try:
            return self.db.query(Document).filter(
                Document.url == url
            ).first()
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting document by URL: {str(e)}")
            raise
            
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its associated questions.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if document was deleted, False if not found
        """
        try:
            document = await self.get_document_by_id(document_id)
            if not document:
                return False
                
            # Delete the document (cascade will delete questions)
            self.db.delete(document)
            self.db.commit()
            return True
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error deleting document: {str(e)}")
            raise
            
    async def search_documents(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[Document]:
        """Search documents by URL or file name.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            List of matching Document objects
        """
        if not query.strip():
            return []
            
        try:
            return self.db.query(Document).filter(
                or_(
                    Document.url.ilike(f"%{query}%"),
                    Document.file_name.ilike(f"%{query}%")
                )
            ).order_by(
                Document.processed_at.desc()
            ).offset(offset).limit(limit).all()
            
        except SQLAlchemyError as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise
            
    async def get_recent_documents(
        self,
        limit: int = 10,
        offset: int = 0
    ) -> List[Document]:
        """Get most recently processed documents.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of Document objects ordered by processing time
        """
        try:
            return self.db.query(Document).order_by(
                Document.processed_at.desc()
            ).offset(offset).limit(limit).all()
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting recent documents: {str(e)}")
            raise
