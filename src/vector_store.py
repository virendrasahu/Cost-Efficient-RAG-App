import os
from typing import List, Dict, Any, Optional, Set
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from src.logger import logger
from src.config import settings

class VectorStoreManager:
    """Manages embedded ChromaDB vector store operations."""
    
    def __init__(self, db_path: str = settings.VECTOR_STORE_PATH, collection_name: str = "rag_chunks"):
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initializing embedding model: {settings.EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        
    def get_existing_ids(self) -> Set[str]:
        """Fetch all existing chunk IDs currently stored in the collection."""
        results = self.collection.get(include=[])
        return set(results['ids']) if results and 'ids' in results else set()

    def add_chunks(self, records: List[Dict[str, Any]]) -> int:
        """Add new chunk records to the vector store."""
        if not records:
            return 0
            
        ids = [rec["id"] for rec in records]
        texts = [rec["text"] for rec in records]
        metadatas = [
            {
                "source": rec["source"],
                "chunk_index": rec["chunk_index"],
                "file_type": rec["file_type"]
            }
            for rec in records
        ]
        
        # Generate embeddings locally
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False).tolist()
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        logger.info(f"Successfully inserted {len(records)} new vectors into vector store.")
        return len(records)

    def search(
        self,
        query: str,
        top_k: int = 5,
        file_type_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for top_k similar chunks matching query with optional metadata filter."""
        query_vector = self.embedding_model.encode([query], show_progress_bar=False).tolist()[0]
        
        where_clause = None
        if file_type_filter:
            where_clause = {"file_type": file_type_filter}
            
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved = []
        if results and results.get('ids') and results['ids'][0]:
            ids = results['ids'][0]
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            distances = results['distances'][0]
            
            for i in range(len(ids)):
                # Cosine distance to similarity conversion: similarity = 1 - cosine_distance
                similarity = 1.0 - distances[i] if distances[i] is not None else 0.0
                retrieved.append({
                    "id": ids[i],
                    "text": docs[i],
                    "source": metas[i].get("source"),
                    "chunk_index": metas[i].get("chunk_index"),
                    "file_type": metas[i].get("file_type"),
                    "similarity": similarity,
                    "distance": distances[i]
                })
                
        return retrieved
