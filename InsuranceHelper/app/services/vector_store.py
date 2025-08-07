"""Vector store service using Pinecone for efficient similarity search."""
import logging
import json
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
import numpy as np

# Try to import Pinecone
PINECONE_AVAILABLE = False
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    pass

from app.core.config import settings

logger = logging.getLogger(__name__)

class VectorStore:
    """A class to handle vector storage and similarity search using Pinecone."""
    
    def __init__(self):
        self.metric = "cosine"
        self.dimension = 384  # all-MiniLM-L6-v2 embedding size
        self._use_pinecone = False
        self._pc = None
        self._index = None
        self._in_memory_store = defaultdict(list)
        
        if PINECONE_AVAILABLE and settings.PINECONE_API_KEY:
            try:
                self._initialize_pinecone()
                self.available = True
                logger.info("Pinecone vector store initialized")
            except Exception as e:
                logger.error(f"Pinecone init failed: {e}")
                logger.warning("Falling back to in-memory vector store")
                self.available = True
        else:
            logger.warning("Using in-memory vector store (not recommended for production)")
            self.available = True
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and index."""
        try:
            self._use_pinecone = True
            self._pc = Pinecone(api_key=settings.PINECONE_API_KEY.get_secret_value())
            
            # Get or create index
            existing_indexes = [index.name for index in self._pc.list_indexes()]
            
            if settings.PINECONE_INDEX not in existing_indexes:
                # Create new index
                self._pc.create_index(
                    name=settings.PINECONE_INDEX,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"  # Default region
                    )
                )
                logger.info(f"Created new Pinecone index: {settings.PINECONE_INDEX}")
            
            # Connect to the index
            self._index = self._pc.Index(settings.PINECONE_INDEX)
            logger.info(f"Connected to Pinecone index: {settings.PINECONE_INDEX}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            self._use_pinecone = False
            raise
    
    async def upsert_document(self, doc_id: str, chunks: List[Dict[str, Any]]) -> bool:
        """Upsert document chunks with embeddings to vector store."""
        if not self.available or not chunks:
            return False
            
        try:
            vectors = []
            for i, chunk in enumerate(chunks):
                if 'embedding' not in chunk:
                    logger.warning(f"Chunk {i} missing embedding")
                    continue
                    
                metadata = {
                    'text': chunk.get('text', ''),
                    'document_id': doc_id,
                    'chunk_index': i,
                    'char_start': chunk.get('char_start', 0),
                    'char_end': chunk.get('char_end', 0)
                }
                
                if 'sentences' in chunk:
                    metadata['sentences'] = json.dumps(chunk['sentences'])
                
                vectors.append((
                    f"{doc_id}_{i}",
                    chunk['embedding'],
                    metadata
                ))
            
            return await self.upsert_vectors(vectors, namespace=doc_id)
            
        except Exception as e:
            logger.error(f"Error upserting document {doc_id}: {e}")
            return False
    
    async def upsert_vectors(self, vectors: List[Tuple[str, List[float], Dict]], namespace: str = None) -> bool:
        """Upsert vectors to the vector store."""
        if not self.available or not vectors:
            return False
            
        try:
            if self._use_pinecone and self._index is not None:
                # Convert to Pinecone format
                pinecone_vectors = []
                for vector_id, vector, metadata in vectors:
                    # Ensure vector has correct dimension
                    if len(vector) != self.dimension:
                        if len(vector) < self.dimension:
                            padded_vector = list(vector) + [0.0] * (self.dimension - len(vector))
                        else:
                            padded_vector = vector[:self.dimension]
                    else:
                        padded_vector = vector
                        
                    pinecone_vectors.append({
                        'id': str(vector_id),
                        'values': padded_vector,
                        'metadata': metadata or {}
                    })
                
                # Upsert vectors
                self._index.upsert(vectors=pinecone_vectors, namespace=namespace)
                return True
                
            else:
                # Fallback to in-memory storage
                store = self._in_memory_store[namespace or 'default']
                for vector_id, vector, metadata in vectors:
                    existing = next((v for v in store if v['id'] == vector_id), None)
                    if existing:
                        existing.update({
                            'values': vector,
                            'metadata': metadata or {}
                        })
                    else:
                        store.append({
                            'id': str(vector_id),
                            'values': vector,
                            'metadata': metadata or {}
                        })
                return True
                
        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            return False
    
    async def semantic_search(
        self, 
        query: str, 
        doc_id: str, 
        top_k: int = 5,
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Semantic search using query embedding and document context."""
        try:
            if not self.available:
                return []
                
            # Get query embedding
            from app.services.document_processor import document_processor
            document_processor._initialize_embedding_model()
            query_embedding = document_processor.embedding_model.encode(
                query,
                show_progress_bar=False,
                normalize_embeddings=True
            ).tolist()
            
            if not query_embedding:
                return []
            
            # Search for similar vectors
            results = await self.search_similar(
                query_vector=query_embedding,
                top_k=top_k,
                namespace=doc_id,
                include_metadata=True
            )
            
            # Process and filter results
            processed_results = []
            for result in results:
                if result['score'] >= min_score:
                    metadata = result.get('metadata', {})
                    
                    # Parse sentences if available
                    if 'sentences' in metadata and isinstance(metadata['sentences'], str):
                        try:
                            metadata['sentences'] = json.loads(metadata['sentences'])
                        except json.JSONDecodeError:
                            metadata['sentences'] = []
                    
                    processed_results.append({
                        'id': result['id'],
                        'score': result['score'],
                        'text': metadata.get('text', ''),
                        'metadata': metadata
                    })
            
            return sorted(processed_results, key=lambda x: x['score'], reverse=True)
            
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []
    
    async def search_similar(
        self, 
        query_vector: List[float], 
        top_k: int = 5, 
        namespace: str = None,
        include_metadata: bool = True
    ) -> List[Dict]:
        """Search for similar vectors in the vector store."""
        if not self.available or not query_vector:
            return []
            
        try:
            if self._use_pinecone and self._index is not None:
                # Ensure query vector has correct dimension
                if len(query_vector) != self.dimension:
                    if len(query_vector) < self.dimension:
                        query_vector = list(query_vector) + [0.0] * (self.dimension - len(query_vector))
                    else:
                        query_vector = query_vector[:self.dimension]
                
                # Perform search
                response = self._index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=include_metadata,
                    namespace=namespace
                )
                
                # Format results
                return [{
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata or {}
                } for match in response.matches]
                
            else:
                # In-memory similarity search (cosine similarity)
                query_norm = np.linalg.norm(query_vector)
                if query_norm == 0:
                    return []
                    
                similarities = []
                for item in self._in_memory_store.get(namespace or 'default', []):
                    vector = item['values']
                    if len(vector) != len(query_vector):
                        continue
                        
                    # Calculate cosine similarity
                    dot_product = np.dot(query_vector, vector)
                    vector_norm = np.linalg.norm(vector)
                    
                    if vector_norm == 0:
                        similarity = 0.0
                    else:
                        similarity = dot_product / (query_norm * vector_norm)
                    
                    similarities.append({
                        'id': item['id'],
                        'score': float(similarity),
                        'metadata': item.get('metadata', {})
                    })
                
                # Sort by score (descending) and return top_k
                return sorted(
                    similarities, 
                    key=lambda x: x['score'], 
                    reverse=True
                )[:top_k]
                
        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            return []

# Create a singleton instance
vector_store = VectorStore()