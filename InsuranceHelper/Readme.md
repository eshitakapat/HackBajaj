# Insurance Helper API

A FastAPI-based service for processing insurance policy documents and answering questions about them using natural language processing.

## Features

- Process insurance policy documents from PDF URLs
- Answer natural language questions about policy details
- Secure API with Bearer token authentication
- Asynchronous processing for better performance
- Structured response format

## Prerequisites

- Python 3.8+
- OpenAI API key
- Python virtual environment (recommended)

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
   # Application Settings
   DEBUG=True
   API_KEY=your_secure_api_key_here
   
   # OpenAI Settings
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4
   EMBEDDING_MODEL=text-embedding-3-small
   ```

## Running the Application

1. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

2. The API will be available at `http://localhost:8000`

3. Access the interactive API documentation at `http://localhost:8000/docs`

## API Endpoints

### POST /hackrx/run

Process an insurance policy document and answer questions about it.

**Headers:**
```
Authorization: Bearer <your_api_key>
Content-Type: application/json
```

**Request Body:**
```json
{
    "documents": "https://example.com/policy.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "What is covered under this policy?"
    ]
}
```

**Response:**
```json
{
    "answers": [
        "The grace period is 30 days.",
        "The policy covers medical expenses up to the sum insured."
    ]
}
```

## Project Structure

```
InsuranceHelper/
├── app/
│   ├── api/
│   │   └── routes.py         # API routes and endpoints
│   ├── Models/
│   │   ├── __init__.py
│   │   ├── schemas.py        # Pydantic models
│   │   └── db_models.py      # Database models (if needed)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── query_handling.py # Question parsing and processing
│   │   ├── embedding_search.py # Text embedding generation
│   │   ├── document_processing.py # PDF text extraction
│   │   └── json_output.py    # Response formatting
│   ├── config.py             # Application configuration
│   └── main.py               # Application entry point
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
| `API_KEY` | API key for authentication | Yes | - |
| `OPENAI_API_KEY` | OpenAI API key | Yes | - |
| `OPENAI_MODEL` | OpenAI model to use | No | `gpt-4` |
| `EMBEDDING_MODEL` | OpenAI embedding model | No | `text-embedding-3-small` |

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