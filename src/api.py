import traceback
import os
import shutil
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from src.config import settings
from src.logger import logger
from src.ingestion import process_and_deduplicate
from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline

app = FastAPI(
    title="Cost-Efficient RAG API",
    description="Low-cost embedded vector store (ChromaDB) RAG system with idempotent ingestion and grounded QA.",
    version="1.0.0"
)

# -------------------------
# Lazy-loaded global objects
# -------------------------
vector_store = None
rag_pipeline = None


def get_vector_store():
    global vector_store

    if vector_store is None:
        logger.info("Initializing Vector Store...")
        vector_store = VectorStoreManager()

    return vector_store


def get_rag_pipeline():
    global rag_pipeline

    if rag_pipeline is None:
        rag_pipeline = RAGPipeline(get_vector_store())

    return rag_pipeline


class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the architecture of the RAG system?")
    top_k: int = Field(default=3, ge=1, le=20)
    file_type_filter: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    retrieved_chunks_count: int
    retrieval_time_sec: float
    generation_time_sec: float
    token_usage: Dict[str, Any]
    fallback_triggered: bool


class IngestResponse(BaseModel):
    status: str
    file_name: str
    new_chunks_added: int
    total_existing_chunks: int


@app.get("/")
async def root():
    return {
        "status": "success",
        "message": "Cost-Efficient RAG API is running."
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }



@app.get("/stats")
async def stats():
    try:
        store = get_vector_store()
        existing_ids = store.get_existing_ids()

        return {
            "vector_store_path": settings.VECTOR_STORE_PATH,
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "total_vectors": len(existing_ids)
        }

    except Exception:
        traceback.print_exc()
        logger.exception("Stats endpoint failed")
        raise

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    temp_dir = "./data/raw_documents"
    os.makedirs(temp_dir, exist_ok=True)

    temp_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        store = get_vector_store()

        existing_ids = store.get_existing_ids()

        new_records = process_and_deduplicate(
            temp_path,
            existing_ids
        )

        added_count = store.add_chunks(new_records)

        return IngestResponse(
            status="success",
            file_name=file.filename,
            new_chunks_added=added_count,
            total_existing_chunks=len(existing_ids) + added_count
        )

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        pipeline = get_rag_pipeline()

        result = pipeline.query(
            query_text=request.query,
            top_k=request.top_k,
            file_type_filter=request.file_type_filter
        )

        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            citations=result["citations"],
            retrieved_chunks_count=result["retrieved_chunks_count"],
            retrieval_time_sec=result["retrieval_time_sec"],
            generation_time_sec=result["generation_time_sec"],
            token_usage=result["token_usage"],
            fallback_triggered=result["fallback_triggered"]
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# import os
# import shutil
# from typing import Optional, Dict, Any, List
# from fastapi import FastAPI, UploadFile, File, HTTPException, Query
# from pydantic import BaseModel, Field

# from src.config import settings
# from src.logger import logger
# from src.ingestion import process_and_deduplicate
# from src.vector_store import VectorStoreManager
# from src.rag_pipeline import RAGPipeline

# app = FastAPI(
#     title="Cost-Efficient RAG API",
#     description="Low-cost embedded vector store (ChromaDB) RAG system with idempotent ingestion and grounded QA.",
#     version="1.0.0"
# )

# # Global instances
# vector_store = VectorStoreManager()
# rag_pipeline = RAGPipeline(vector_store)

# class QueryRequest(BaseModel):
#     query: str = Field(..., example="What is the architecture of the RAG system?")
#     top_k: int = Field(default=3, ge=1, le=20)
#     file_type_filter: Optional[str] = Field(default=None, example="pdf")

# class QueryResponse(BaseModel):
#     query: str
#     answer: str
#     citations: List[Dict[str, Any]]
#     retrieved_chunks_count: int
#     retrieval_time_sec: float
#     generation_time_sec: float
#     token_usage: Dict[str, Any]
#     fallback_triggered: bool

# class IngestResponse(BaseModel):
#     status: str
#     file_name: str
#     new_chunks_added: int
#     total_existing_chunks: int

# @app.get("/")
# def read_root():
#     return {"message": "Cost-Efficient RAG API is running."}

# @app.get("/stats")
# def get_stats():
#     existing_ids = vector_store.get_existing_ids()
#     return {
#         "vector_store_path": settings.VECTOR_STORE_PATH,
#         "embedding_model": settings.EMBEDDING_MODEL_NAME,
#         "total_vectors": len(existing_ids)
#     }

# @app.post("/ingest", response_model=IngestResponse)
# async def ingest_document(file: UploadFile = File(...)):
#     """Ingest an uploaded PDF, HTML, or Markdown file with idempotent deduplication."""
#     temp_dir = "./data/raw_documents"
#     os.makedirs(temp_dir, exist_ok=True)
#     temp_path = os.path.join(temp_dir, file.filename)
    
#     try:
#         with open(temp_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
            
#         existing_ids = vector_store.get_existing_ids()
#         new_records = process_and_deduplicate(temp_path, existing_ids)
#         added_count = vector_store.add_chunks(new_records)
        
#         return IngestResponse(
#             status="success",
#             file_name=file.filename,
#             new_chunks_added=added_count,
#             total_existing_chunks=len(existing_ids) + added_count
#         )
#     except Exception as e:
#         logger.error(f"Ingestion failed for {file.filename}: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/query", response_model=QueryResponse)
# async def query_rag(request: QueryRequest):
#     """Retrieve top-k chunks and generate a grounded answer."""
#     try:
#         result = rag_pipeline.query(
#             query_text=request.query,
#             top_k=request.top_k,
#             file_type_filter=request.file_type_filter
#         )
#         return QueryResponse(
#             query=result["query"],
#             answer=result["answer"],
#             citations=result["citations"],
#             retrieved_chunks_count=result["retrieved_chunks_count"],
#             retrieval_time_sec=result["retrieval_time_sec"],
#             generation_time_sec=result["generation_time_sec"],
#             token_usage=result["token_usage"],
#             fallback_triggered=result["fallback_triggered"]
#         )
#     except Exception as e:
#         logger.error(f"Query processing failed: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
