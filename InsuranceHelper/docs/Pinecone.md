# Pinecone Vector Store Integration

This guide explains how to set up and use Pinecone as a vector store for semantic search in the Insurance Helper application.

## Prerequisites

1. A Pinecone account (sign up at [https://www.pinecone.io/](https://www.pinecone.io/))
2. Python 3.8+
3. Required Python packages (install with `pip install -r requirements.txt`)

## Setup

1. **Get Your Pinecone API Key**
   - Log in to your Pinecone account
   - Navigate to the API Keys section
   - Copy your API key and environment name (e.g., `us-west1-gcp`)

2. **Configure Environment Variables**
   Update your `.env` file with your Pinecone credentials:
   ```
   # Pinecone Configuration
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_ENVIRONMENT=your-pinecone-environment  # e.g., 'us-west1-gcp'
   PINECONE_INDEX=insurance-helper
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The `VectorStore` class provides a simple interface to interact with Pinecone:

```python
from app.services.vector_store import vector_store

# Check if Pinecone is available
if vector_store.available:
    # Upsert vectors
    vectors = [
        {
            "id": "doc_1",
            "values": [0.1, 0.2, ...],  # Your embedding vector
            "metadata": {
                "text": "Your document text",
                "doc_id": 1,
                "type": "your_document_type"
            }
        }
    ]
    await vector_store.upsert_vectors(vectors)
    
    # Search for similar vectors
    query_vector = [0.1, 0.2, ...]  # Your query embedding
    results = await vector_store.search_similar(
        query_vector=query_vector,
        top_k=5,
        filter={"type": "your_document_type"}
    )
    
    # Delete vectors (optional)
    await vector_store.delete_vectors(ids=["doc_1"])
```

## Example

See `examples/pinecone_example.py` for a complete example of using the Pinecone vector store.

## Best Practices

1. **Batch Operations**: When upserting many vectors, do it in batches (e.g., 100 vectors per batch)
2. **Filtering**: Use metadata filters to narrow down searches
3. **Error Handling**: Always check `vector_store.available` before making API calls
4. **Clean Up**: Delete test vectors when they're no longer needed

## Troubleshooting

- **Connection Issues**: Verify your API key and environment
- **Dimension Mismatch**: Ensure your vectors match the dimension of the Pinecone index
- **Rate Limiting**: Implement retries for rate-limited requests

For more information, see the [Pinecone documentation](https://docs.pinecone.io/).
