"""
vector_store.py - FAISS index manager for the Healthcare AI Assistant.

Responsibilities:
  1. Build a FAISS index from LangChain Documents.
  2. Persist the index to disk (survives container restarts).
  3. Load an existing index from disk on startup.
  4. Expose a LangChain VectorStoreRetriever for the RAG chain.
  5. Report index statistics (document count) for the health endpoint.

Why FAISS?
  • Facebook AI Similarity Search — battle-tested, production-grade.
  • Pure in-process library — no network round-trip for retrieval.
  • Supports millions of vectors with sub-millisecond query latency on CPU.
  • IndexFlatL2 (exact search) is appropriate for < 100k chunks; can swap
    to IndexIVFFlat for approximate-nearest-neighbour at larger scale.
  • Serialisable to disk — index survives pod restarts without re-embedding.

Integration:
  - rag.py          calls get_retriever() to build the RAG chain.
  - routers/ingest.py calls build_index() after loading documents.
  - routers/health.py calls get_document_count() for metrics.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever

from backend.config import settings
from backend.embeddings import get_embeddings
from backend.logger import get_logger

logger = get_logger(__name__)

# Module-level cache — holds the in-memory FAISS instance
_faiss_store: Optional[FAISS] = None


# ─────────────────────────────────────────────────────────────────────────────
# BUILD & PERSIST
# ─────────────────────────────────────────────────────────────────────────────

def build_index(documents: List[Document], force_reload: bool = False) -> int:
    """
    Chunk documents → embed → build FAISS index → save to disk.

    Args:
        documents:    Raw LangChain Documents from DocumentLoader.
        force_reload: If True, delete any existing index before rebuilding.

    Returns:
        Number of chunks stored in the index.
    """
    global _faiss_store

    index_path = Path(settings.FAISS_INDEX_PATH)

    if force_reload and index_path.exists():
        shutil.rmtree(index_path)
        logger.info("Existing FAISS index cleared", extra={"path": str(index_path)})

    # ── 1. Chunk documents ────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    logger.info(
        "Documents chunked",
        extra={
            "raw_docs": len(documents),
            "chunks": len(chunks),
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
        },
    )

    if not chunks:
        raise ValueError("No chunks produced — documents may be empty or unsupported.")

    # ── 2. Embed + build FAISS ────────────────────────────────────────────────
    embeddings = get_embeddings()
    logger.info("Building FAISS index — this may take a moment …")

    _faiss_store = FAISS.from_documents(chunks, embeddings)

    # ── 3. Persist to disk ────────────────────────────────────────────────────
    index_path.mkdir(parents=True, exist_ok=True)
    _faiss_store.save_local(str(index_path))

    logger.info(
        "FAISS index saved",
        extra={"path": str(index_path), "chunks": len(chunks)},
    )
    return len(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD FROM DISK
# ─────────────────────────────────────────────────────────────────────────────

def load_index() -> bool:
    """
    Load a previously saved FAISS index from disk into memory.
    Called once at application startup.

    Returns:
        True if loaded successfully, False if no index exists yet.
    """
    global _faiss_store

    index_path = Path(settings.FAISS_INDEX_PATH)

    # Check for the actual index file, not just the directory.
    # The directory is pre-created by Docker volume mount and is always
    # present even on a fresh container — only the file confirms a real index.
    index_file = index_path / "index.faiss"
    if not index_file.exists():
        logger.warning(
            "No FAISS index file found on disk — call POST /ingest to build it.",
            extra={"expected_path": str(index_file)},
        )
        return False

    try:
        embeddings = get_embeddings()
        _faiss_store = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,  # Safe: we wrote this file ourselves
        )
        count = get_document_count()
        logger.info(
            "FAISS index loaded from disk",
            extra={"chunks": count, "path": str(index_path)},
        )
        return True
    except Exception as exc:
        logger.error(
            "Failed to load FAISS index — starting with empty index",
            extra={"error": str(exc), "path": str(index_path)},
        )
        _faiss_store = None
        return False


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def get_retriever() -> VectorStoreRetriever:
    """
    Return a LangChain VectorStoreRetriever backed by the in-memory FAISS index.

    Uses MMR (Maximum Marginal Relevance) to balance relevance and diversity
    in the top-K results, reducing redundant context in the prompt.

    Raises:
        RuntimeError: If the index has not been loaded or built yet.
    """
    if _faiss_store is None:
        raise RuntimeError(
            "FAISS index is not loaded. "
            "Call POST /ingest to build it, or ensure load_index() ran at startup."
        )

    return _faiss_store.as_retriever(
        search_type="mmr",                   # Maximum Marginal Relevance
        search_kwargs={
            "k": settings.FAISS_TOP_K,       # Return top-K chunks
            "fetch_k": settings.FAISS_TOP_K * 3,  # MMR candidate pool
            "lambda_mult": 0.7,              # 0 = max diversity, 1 = max relevance
        },
    )


def similarity_search_with_scores(
    query: str, k: int | None = None
) -> list[tuple[Document, float]]:
    """
    Run a similarity search and return (Document, score) tuples.
    Score is cosine similarity (0–1, higher = more similar).
    Used by rag.py to compute the per-answer confidence score.
    """
    if _faiss_store is None:
        raise RuntimeError("FAISS index is not loaded.")

    top_k = k or settings.FAISS_TOP_K
    results = _faiss_store.similarity_search_with_relevance_scores(query, k=top_k)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def get_document_count() -> int:
    """Return the number of chunks stored in the in-memory FAISS index."""
    if _faiss_store is None:
        return 0
    try:
        return _faiss_store.index.ntotal
    except Exception:
        return 0


def is_index_loaded() -> bool:
    """True if the FAISS index is in memory and ready to serve queries."""
    return _faiss_store is not None