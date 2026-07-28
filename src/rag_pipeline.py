import time
from typing import Dict, Any, List, Optional

from google import genai

from src.config import settings
from src.logger import logger
from src.vector_store import VectorStoreManager


FALLBACK_MESSAGE = (
    "I do not have sufficient information in the provided context to answer this question."
)

SYSTEM_PROMPT = """You are a precise QA assistant.

Answer the user's question using ONLY the context provided below.

For every factual claim, cite the source using:
[Doc: <source>, Chunk: <id>]

If the context does not contain enough information, return EXACTLY:

"{fallback_message}"

Context:
{context_block}

Question:
{question}
"""


class RAGPipeline:
    """Retrieval-Augmented Generation Pipeline."""

    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

        if (
            settings.GEMINI_API_KEY
            and settings.GEMINI_API_KEY != "mock_key"
        ):
            self.client = genai.Client(
                api_key=settings.GEMINI_API_KEY
            )

            # Lightweight & faster model
            self.model_name = "gemini-2.5-flash-lite"

        else:
            self.client = None
            logger.warning(
                "No GEMINI_API_KEY found. Running in fallback mode."
            )

    def build_prompt(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> str:

        context = []

        for chunk in chunks:
            context.append(
                f"[Doc: {chunk['source']}, Chunk: {chunk['id']}]\n{chunk['text']}"
            )

        return SYSTEM_PROMPT.format(
            fallback_message=FALLBACK_MESSAGE,
            context_block="\n\n".join(context),
            question=query,
        )

    def generate_answer(self, prompt: str):

        start = time.time()

        if self.client:

            try:

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt.strip(),
                )

                answer = (
                    response.text.strip()
                    if response.text
                    else FALLBACK_MESSAGE
                )

                generation_time = round(
                    time.time() - start,
                    4,
                )

                prompt_tokens = int(len(prompt.split()) * 1.3)
                completion_tokens = int(len(answer.split()) * 1.3)

                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "generation_time_sec": generation_time,
                }

                return answer, usage

            except Exception as e:

                logger.error(f"Gemini Error: {e}")

        generation_time = round(
            time.time() - start,
            4,
        )

        answer = (
            "Based on the retrieved context, this is the available information. "
            "[Doc: sample.pdf, Chunk: demo]"
        )

        usage = {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(answer.split()),
            "total_tokens": len(prompt.split()) + len(answer.split()),
            "generation_time_sec": generation_time,
            "simulated": True,
        }

        return answer, usage

    def query(
        self,
        query_text: str,
        top_k: int = 3,
        file_type_filter: Optional[str] = None,
    ) -> Dict[str, Any]:

        retrieval_start = time.time()

        retrieved_chunks = self.vector_store.search(
            query=query_text,
            top_k=top_k,
            file_type_filter=file_type_filter,
        )

        retrieval_time = round(
            time.time() - retrieval_start,
            4,
        )

        if not retrieved_chunks:

            return {
                "query": query_text,
                "answer": FALLBACK_MESSAGE,
                "citations": [],
                "retrieved_chunks_count": 0,
                "retrieval_time_sec": retrieval_time,
                "generation_time_sec": 0.0,
                "token_usage": {},
                "fallback_triggered": True,
            }

        max_similarity = max(
            chunk.get("similarity", 0)
            for chunk in retrieved_chunks
        )

        if max_similarity < settings.SIMILARITY_THRESHOLD:

            return {
                "query": query_text,
                "answer": FALLBACK_MESSAGE,
                "citations": [],
                "retrieved_chunks": retrieved_chunks,
                "retrieved_chunks_count": len(retrieved_chunks),
                "retrieval_time_sec": retrieval_time,
                "generation_time_sec": 0.0,
                "token_usage": {},
                "fallback_triggered": True,
            }

        prompt = self.build_prompt(
            query_text,
            retrieved_chunks,
        )

        answer, token_usage = self.generate_answer(prompt)

        citations = [
            {
                "source": chunk["source"],
                "chunk_id": chunk["id"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in retrieved_chunks
        ]

        logger.info(
            f"Query completed | Retrieval={retrieval_time}s | "
            f"Generation={token_usage.get('generation_time_sec',0)}s | "
            f"Chunks={len(retrieved_chunks)}"
        )

        return {
            "query": query_text,
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
            "retrieved_chunks_count": len(retrieved_chunks),
            "retrieval_time_sec": retrieval_time,
            "generation_time_sec": token_usage.get(
                "generation_time_sec",
                0.0,
            ),
            "token_usage": token_usage,
            "fallback_triggered": False,
        }
