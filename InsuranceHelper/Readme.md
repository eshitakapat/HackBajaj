# Insurance Helper API

A FastAPI-based service for processing insurance policy documents and answering questions using advanced NLP and vector search.

## Features

- **Document Processing**: Upload and process insurance documents (PDF, DOCX, TXT)
- **Semantic Search**: Find relevant document chunks using vector embeddings
- **Q&A System**: Ask natural language questions about your insurance documents
- **Mistral Integration**: Powered by Mistral 7B via OpenRouter for high-quality responses
- **Document Management**: Track document processing status and history
- **Secure API**: JWT token authentication
- **Asynchronous Processing**: Background processing for document ingestion
- **Structured Responses**: Consistent JSON output format
- **Database Storage**: PostgreSQL for document and question storage

## Prerequisites

- Python 3.8+
- PostgreSQL database
- Python virtual environment (recommended)
- CUDA-enabled GPU (recommended for faster processing)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd InsuranceHelper
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your configuration:
   ```ini
   # Required
   OPENROUTER_API_KEY=your-openrouter-api-key
   DEFAULT_LLM_MODEL=mistralai/mistral-7b-instruct

   # Database
   POSTGRES_SERVER=localhost
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=postgres
   POSTGRES_DB=insurance_helper

   # Pinecone
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_ENVIRONMENT=your-pinecone-environment
   PINECONE_INDEX=insurance-helper
   ```

5. Initialize the database:
   ```bash
   python -m scripts.init_database
   ```

## API Endpoints

### Document Management
- `POST /api/documents/`: Upload a new document
- `GET /api/documents/`: List all documents
- `GET /api/documents/{doc_id}`: Get document details
- `DELETE /api/documents/{doc_id}`: Delete a document

### Question Answering
- `POST /api/ask`: Ask a question about your documents
  ```json
  {
    "question": "What is the coverage limit for natural disasters?",
    "document_ids": ["doc1", "doc2"]
  }
  ```

### Semantic Search
- `POST /api/search`: Search for relevant document chunks
  ```json
  {
    "query": "natural disaster coverage",
    "top_k": 5
  }
  ```

### Document Processing

- `POST /api/documents/process` - Process a document from a URL
- `POST /api/documents/upload` - Upload and process a document file
- `GET /api/documents/{document_id}` - Get document details
- `GET /api/documents/{document_id}/questions` - Get questions for a document

### HackRx API

- `POST /api/hackrx` - Process documents and answer questions in HackRx format

## Running the Application

1. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

2. Access the API documentation at http://localhost:8000/docs

## Document Processing Pipeline

The document processing pipeline consists of the following steps:

1. **Document Ingestion**:
   - Accepts documents via URL or file upload
   - Validates file type and size
   - Tracks document status in the database

2. **Text Extraction**:
   - Supports multiple document formats (PDF, DOCX, TXT, etc.)
   - Uses appropriate libraries for each format
   - Preserves document structure and formatting

3. **Text Chunking**:
   - Splits text into meaningful chunks with overlap
   - Preserves context across chunk boundaries
   - Handles various document structures

4. **Embedding Generation**:
   - Uses sentence-transformers for vector embeddings
   - Generates high-dimensional vectors
   - Normalized for cosine similarity

5. **Vector Storage**:
   - Stores vectors in Pinecone vector database
   - Supports fast similarity search
   - Maintains document metadata and relationships

## Project Structure

```
InsuranceHelper/
├── app/
│   ├── api/
│   │   ├── endpoints/        # API endpoint modules
│   │   └── __init__.py
│   ├── core/                 # Core application code
│   │   └── config.py         # Application configuration
│   ├── db/                   # Database configuration
│   │   ├── __init__.py
│   │   ├── session.py        # Database session management
│   │   └── init_db.py        # Database initialization
│   ├── models/               # Database models
│   │   ├── __init__.py
│   │   ├── document_models.py # Document and Question models
│   │   └── user.py           # User authentication models
│   ├── services/             # Business logic
│   │   ├── __init__.py
│   │   ├── document_processor.py # Document processing logic
│   │   ├── document_store.py # Database operations
│   │   ├── semantic_search.py # Vector search functionality
│   │   └── vector_store.py   # Vector database interface
│   ├── schemas/              # Pydantic schemas
│   │   └── __init__.py
│   └── main.py               # Application entry point
├── scripts/                  # Utility scripts
│   └── init_database.py      # Database initialization script
├── tests/                    # Test files
├── .env                      # Environment variables
├── .gitignore
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `DEBUG` | Enable debug mode | No | `False` |
| `OPENROUTER_API_KEY` | OpenRouter API key | Yes | - |
| `DEFAULT_LLM_MODEL` | Default LLM model | Yes | - |
| `POSTGRES_SERVER` | PostgreSQL server | Yes | - |
| `POSTGRES_USER` | PostgreSQL user | Yes | - |
| `POSTGRES_PASSWORD` | PostgreSQL password | Yes | - |
| `POSTGRES_DB` | PostgreSQL database | Yes | - |
| `PINECONE_API_KEY` | Pinecone API key | Yes | - |
| `PINECONE_ENVIRONMENT` | Pinecone environment | Yes | - |
| `PINECONE_INDEX` | Pinecone index | Yes | - |

## Development

1. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run tests:
   ```bash
   pytest
   ```

3. Format code:
   ```bash
   black .
   isort .
   ```

4. Check code quality:
   ```bash
   flake8 .
   mypy .
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.