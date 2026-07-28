import os
from typing import List, Dict, Any, Optional, Set

import chromadb
from chromadb.utils import embedding_functions

from src.logger import logger
from src.config import settings


class VectorStoreManager:
    """ChromaDB Vector Store using lightweight ONNX embeddings."""

    def __init__(
        self,
        db_path: str = settings.VECTOR_STORE_PATH,
        collection_name=settings.COLLECTION_NAME,
    ):
        os.makedirs(db_path, exist_ok=True)

        self.client = chromadb.PersistentClient(path=db_path)

        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.EMBEDDING_MODEL_NAME
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info("Vector store initialized successfully.")

    def get_existing_ids(self) -> Set[str]:
        results = self.collection.get(include=[])

        if results and "ids" in results:
            return set(results["ids"])

        return set()

    def add_chunks(self, records: List[Dict[str, Any]]) -> int:

        if not records:
            return 0

        ids = [r["id"] for r in records]

        texts = [r["text"] for r in records]

        metadatas = [
            {
                "source": r["source"],
                "chunk_index": r["chunk_index"],
                "file_type": r["file_type"],
            }
            for r in records
        ]

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(f"Inserted {len(records)} vectors.")

        return len(records)

    def search(
        self,
        query: str,
        top_k: int = 5,
        file_type_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        where = None

        if file_type_filter:
            where = {"file_type": file_type_filter}

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        retrieved = []

        if results["ids"] and results["ids"][0]:

            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for i in range(len(ids)):

                similarity = 1 - distances[i]

                retrieved.append(
                    {
                        "id": ids[i],
                        "text": docs[i],
                        "source": metas[i]["source"],
                        "chunk_index": metas[i]["chunk_index"],
                        "file_type": metas[i]["file_type"],
                        "similarity": similarity,
                        "distance": distances[i],
                    }
                )

        return retrieved
