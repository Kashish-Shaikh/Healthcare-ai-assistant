"""
config.py - Centralized configuration management using Pydantic BaseSettings.

All settings are read from environment variables (or .env file).
Every module imports the `settings` singleton — never hardcode values.

Integration: Imported by every backend module as:
    from backend.config import settings
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    # ── Application ───────────────────────────────────────────────────────────
    APP_NAME: str = "Healthcare AI Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Ollama / LLM ──────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "mistral"      # swap to phi3:mini or llama3.2:1b for faster CPU
    LLM_TEMPERATURE: float = 0.1       # Low temp → deterministic, reduces hallucination
    LLM_MAX_TOKENS: int = 512          # Reduced from 1024 — halves generation time on CPU
    LLM_REQUEST_TIMEOUT: int = 180     # seconds

    # ── Embeddings ────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"      # "cuda" if GPU available

    # ── FAISS Vector Store ────────────────────────────────────────────────────
    FAISS_INDEX_PATH: str = "data/faiss_index"
    FAISS_TOP_K: int = 3               # Reduced from 5 — less context = faster LLM response

    # ── RAG Pipeline ──────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MIN_CONFIDENCE_THRESHOLD: float = 0.25   # Below → refuse to answer
    SIMILARITY_SCORE_THRESHOLD: float = 0.50

    # ── Document Storage ──────────────────────────────────────────────────────
    DOCUMENTS_DIR: str = "data/documents"
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".txt", ".md", ".docx"]

    # ── API ───────────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]

    # ── Safety ────────────────────────────────────────────────────────────────
    REFUSE_DIAGNOSIS: bool = True
    MAX_QUERY_LENGTH: int = 2000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — same Settings object reused across the process."""
    return Settings()


settings = get_settings()