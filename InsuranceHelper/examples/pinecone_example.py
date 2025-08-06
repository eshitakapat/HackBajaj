"""
Example script demonstrating how to use the Pinecone vector store.

This script shows how to:
1. Initialize the vector store
2. Upsert vectors
3. Search for similar vectors
4. Clean up
"""
import asyncio
import numpy as np
from app.services.vector_store import vector_store
from app.services.embedding_search import get_embeddings

async def main():
    """Run the Pinecone example."""
    print("=== Pinecone Vector Store Example ===")
    
    if not vector_store.available:
        print("Error: Pinecone is not properly configured.")
        print("Please check your .env file and ensure you have set:")
        print("- PINECONE_API_KEY")
        print("- PINECONE_ENVIRONMENT")
        print("- PINECONE_INDEX")
        return
    
    # Example documents
    documents = [
        "Car insurance covers damage to your vehicle in case of an accident.",
        "Home insurance protects your property against natural disasters.",
        "Health insurance helps cover medical expenses and treatments.",
        "Life insurance provides financial security for your family.",
        "Travel insurance covers trip cancellations and medical emergencies abroad."
    ]
    
    try:
        # 1. Generate embeddings for the documents
        print("\n1. Generating embeddings...")
        embeddings = await get_embeddings(documents)
        
        # 2. Prepare vectors for Pinecone
        print("2. Preparing vectors for Pinecone...")
        vectors = [
            {
                "id": f"doc_{i}",
                "values": embedding,
                "metadata": {
                    "text": text,
                    "doc_id": i,
                    "type": "insurance_info"
                }
            }
            for i, (text, embedding) in enumerate(zip(documents, embeddings))
        ]
        
        # 3. Upsert vectors to Pinecone
        print("3. Upserting vectors to Pinecone...")
        success = await vector_store.upsert_vectors(vectors)
        if not success:
            print("Failed to upsert vectors to Pinecone")
            return
        print(f"Successfully upserted {len(vectors)} vectors")
        
        # 4. Search for similar vectors
        print("\n4. Searching for similar vectors...")
        query = "What does car insurance cover?"
        print(f"Query: {query}")
        
        # Get embedding for the query
        query_embedding = (await get_embeddings([query]))[0]
        
        # Search for similar vectors
        results = await vector_store.search_similar(
            query_vector=query_embedding,
            top_k=2,
            filter={"type": "insurance_info"}
        )
        
        # Display results
        print("\nTop matches:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Similarity: {result['score']:.4f}")
            print(f"   Text: {result['metadata']['text']}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        # 5. Clean up (optional)
        print("\n5. Cleaning up...")
        if vector_store.available:
            # Delete all vectors (in a real app, you might want to be more selective)
            await vector_store.delete_vectors(filter={"type": "insurance_info"})
            print("Cleaned up test vectors")

if __name__ == "__main__":
    asyncio.run(main())
