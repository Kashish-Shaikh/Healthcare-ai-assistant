"""
embeddings.py - Embedding model wrapper for the Healthcare AI Assistant.

Model: sentence-transformers/all-MiniLM-L6-v2
  • 384-dimensional dense vectors
  • 80MB model — fast inference on CPU
  • Strong semantic similarity for English healthcare text
  • Open-source, no API key required

The singleton pattern ensures the model is loaded ONCE at startup
and reused across all requests — critical for latency in production.

Integration:
  - vector_store.py passes get_embeddings() to FAISS at index creation
  - rag.py uses the same instance for query embedding
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import settings
from backend.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load and cache the MiniLM-L6-v2 embedding model.

    The @lru_cache ensures this function's return value is stored after the
    first call — subsequent calls return the cached instance instantly.
    """
    logger.info(
        "Loading embedding model",
        extra={"model": settings.EMBEDDING_MODEL, "device": settings.EMBEDDING_DEVICE},
    )

    model_kwargs = {"device": settings.EMBEDDING_DEVICE}
    encode_kwargs = {
        "normalize_embeddings": True,   # L2-normalise → cosine similarity via dot product
        "batch_size": 32,
    }

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    logger.info("Embedding model loaded successfully", extra={"model": settings.EMBEDDING_MODEL})
    return embeddings


def embed_query(text: str) -> list[float]:
    """
    Convenience function: embed a single query string.
    Used for ad-hoc similarity calculations outside the LangChain chain.
    """
    return get_embeddings().embed_query(text)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Convenience function: batch-embed a list of text strings.
    """
    return get_embeddings().embed_documents(texts)
