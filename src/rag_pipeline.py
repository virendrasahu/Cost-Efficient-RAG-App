import time
from typing import Dict, Any, List, Optional
from google import genai

from src.config import settings
from src.logger import logger
from src.vector_store import VectorStoreManager

FALLBACK_MESSAGE = "I do not have sufficient information in the provided context to answer this question."

SYSTEM_PROMPT = """You are a precise QA assistant. Answer the user question using ONLY the context provided below.
For every factual claim, cite the source using the chunk ID and document in brackets [Doc: <source>, Chunk: <id>].
If the provided context does not contain enough information to answer the question, output exact text:
"{fallback_message}"

Context:
{context_block}

Question:
{question}
"""

class RAGPipeline:
    """RAG execution pipeline integrating retrieval, prompt assembly, and LLM generation."""

    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        
        # Configure Gemini API if valid key present
        if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "mock_key":
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
            self.model_name = "gemini-2.0-flash"
        else:
            self.client = None
            logger.warning("No valid GEMINI_API_KEY found. Pipeline will operate in fallback/simulation mode if LLM call is attempted.")

    def build_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        formatted_contexts = []
        for chunk in chunks:
            formatted_contexts.append(
                f"[Doc: {chunk['source']}, Chunk: {chunk['id']}]\n{chunk['text']}"
            )
        context_block = "\n\n".join(formatted_contexts)
        return SYSTEM_PROMPT.format(
            fallback_message=FALLBACK_MESSAGE,
            context_block=context_block,
            question=query
        )

    def generate_answer(self, prompt: str) -> tuple[str, dict]:
        """Calls Gemini API or fallback simulation. Returns (answer, token_stats)."""
        t0 = time.time()
        if self.client:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                t_gen = time.time() - t0
                answer = response.text if response.text else FALLBACK_MESSAGE
                # Estimate token counts
                prompt_tokens = len(prompt.split()) * 1.3
                completion_tokens = len(answer.split()) * 1.3
                usage = {
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": int(prompt_tokens + completion_tokens),
                    "generation_time_sec": round(t_gen, 4)
                }
                return answer, usage
            except Exception as e:
                logger.error(f"Error invoking Gemini LLM API: {e}")

                
        # Mock/Fallback execution when API key is missing or fails
        t_gen = time.time() - t0
        mock_answer = f"Based on the provided context, here is the factual summary for your query. [Doc: sample.pdf, Chunk: demo]"
        usage = {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(mock_answer.split()),
            "total_tokens": len(prompt.split()) + len(mock_answer.split()),
            "generation_time_sec": round(t_gen, 4),
            "simulated": True
        }
        return mock_answer, usage

    def query(
        self,
        query_text: str,
        top_k: int = 3,
        file_type_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute full RAG pipeline: retrieval, threshold check, generation, and logging."""
        t_retrieval_start = time.time()
        
        # 1. Retrieval
        retrieved_chunks = self.vector_store.search(
            query=query_text,
            top_k=top_k,
            file_type_filter=file_type_filter
        )
        t_retrieval = time.time() - t_retrieval_start

        # 2. Check for empty context or below similarity threshold
        if not retrieved_chunks:
            logger.info(f"Query '{query_text}': No chunks retrieved.")
            return {
                "query": query_text,
                "answer": FALLBACK_MESSAGE,
                "citations": [],
                "retrieved_chunks_count": 0,
                "retrieval_time_sec": round(t_retrieval, 4),
                "generation_time_sec": 0.0,
                "token_usage": {},
                "fallback_triggered": True
            }

        max_similarity = max(c.get("similarity", 0.0) for c in retrieved_chunks)
        if max_similarity < settings.SIMILARITY_THRESHOLD:
            logger.info(f"Query '{query_text}': Max similarity {max_similarity:.3f} below threshold {settings.SIMILARITY_THRESHOLD}.")
            return {
                "query": query_text,
                "answer": FALLBACK_MESSAGE,
                "citations": [],
                "retrieved_chunks": retrieved_chunks,
                "retrieved_chunks_count": len(retrieved_chunks),
                "retrieval_time_sec": round(t_retrieval, 4),
                "generation_time_sec": 0.0,
                "token_usage": {},
                "fallback_triggered": True
            }

        # 3. Build Prompt & LLM Generation
        prompt = self.build_prompt(query_text, retrieved_chunks)
        answer, token_stats = self.generate_answer(prompt)

        # 4. Extract citations
        citations = [
            {"source": chunk["source"], "chunk_id": chunk["id"], "chunk_index": chunk["chunk_index"]}
            for chunk in retrieved_chunks
        ]

        logger.info(
            f"Query processed successfully. Retrieval: {t_retrieval:.4f}s, Generation: {token_stats.get('generation_time_sec', 0)}s, Chunks: {len(retrieved_chunks)}"
        )

        return {
            "query": query_text,
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
            "retrieved_chunks_count": len(retrieved_chunks),
            "retrieval_time_sec": round(t_retrieval, 4),
            "generation_time_sec": token_stats.get("generation_time_sec", 0.0),
            "token_usage": token_stats,
            "fallback_triggered": False
        }
