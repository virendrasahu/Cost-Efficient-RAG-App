from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # =========================
    # Gemini Configuration
    # =========================
    GEMINI_API_KEY: str = "mock_key"
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"

    # =========================
    # Vector Store Configuration
    # =========================
    VECTOR_STORE_PATH: str = "./data/chroma_db"
    COLLECTION_NAME: str = "rag_chunks"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # =========================
    # Chunking Configuration
    # =========================
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50

    # =========================
    # Retrieval Configuration
    # =========================
    DEFAULT_TOP_K: int = 3
    SIMILARITY_THRESHOLD: float = 0.30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
