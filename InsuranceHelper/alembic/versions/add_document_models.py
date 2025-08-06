"""Add document and question models

Revision ID: 1a2b3c4d5e6f
Revises: 
Create Date: 2025-08-06 15:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '1a2b3c4d5e6f'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create documents table
    op.create_table(
        'documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('url', sa.String, nullable=False),
        sa.Column('document_id', sa.String, unique=True, nullable=False, index=True),
        sa.Column('file_name', sa.String, nullable=True),
        sa.Column('file_size', sa.Integer, nullable=True),
        sa.Column('content_type', sa.String, nullable=True),
        sa.Column('processed_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('status', sa.String, default='processing'),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Index('idx_documents_document_id', 'document_id', unique=True)
    )
    
    # Create questions table
    op.create_table(
        'questions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('question', sa.Text, nullable=False),
        sa.Column('answer', sa.Text, nullable=False),
        sa.Column('model_used', sa.String, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('processing_time_ms', sa.Integer, nullable=True),
        sa.Column('error', sa.Text, nullable=True),
        sa.Column('metadata', sa.Text, nullable=True),
        sa.Index('idx_questions_document_id', 'document_id'),
        sa.Index('idx_questions_created_at', 'created_at')
    )

def downgrade():
    op.drop_table('questions')
    op.drop_table('documents')
