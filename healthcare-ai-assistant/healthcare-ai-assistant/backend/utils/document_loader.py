"""
document_loader.py - Multi-format healthcare document loader.

Supports: .txt, .md, .pdf, .docx
Attaches metadata to every loaded Document so the RAG pipeline
can surface accurate source citations.

Integration:
  - vector_store.py calls DocumentLoader.load_directory()
  - Returns List[Document] ready for chunking by rag.py
"""

import os
import time
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

from backend.config import settings
from backend.logger import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """Loads all supported documents from a directory, attaching metadata."""

    LOADER_MAP = {
        ".txt":  TextLoader,
        ".md":   UnstructuredMarkdownLoader,
        ".pdf":  PyPDFLoader,
        ".docx": Docx2txtLoader,
    }

    def __init__(self, documents_dir: str | None = None):
        self.documents_dir = Path(documents_dir or settings.DOCUMENTS_DIR)
        self.allowed_extensions = set(settings.ALLOWED_EXTENSIONS)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_directory(self) -> List[Document]:
        """
        Scan self.documents_dir and load every supported file.
        Returns a flat list of LangChain Documents with metadata.
        """
        if not self.documents_dir.exists():
            logger.error("Documents directory not found", extra={"path": str(self.documents_dir)})
            raise FileNotFoundError(f"Documents directory not found: {self.documents_dir}")

        file_paths = self._discover_files()
        if not file_paths:
            logger.warning("No supported documents found", extra={"dir": str(self.documents_dir)})
            return []

        all_docs: List[Document] = []
        for file_path in file_paths:
            docs = self._load_file(file_path)
            all_docs.extend(docs)

        logger.info(
            "Directory loaded",
            extra={
                "directory": str(self.documents_dir),
                "files_found": len(file_paths),
                "documents_loaded": len(all_docs),
            },
        )
        return all_docs

    def load_file(self, file_path: str) -> List[Document]:
        """Load a single file by path."""
        return self._load_file(Path(file_path))

    # ── Private helpers ───────────────────────────────────────────────────────

    def _discover_files(self) -> List[Path]:
        """Return all files in the directory with allowed extensions."""
        files: List[Path] = []
        for item in sorted(self.documents_dir.iterdir()):
            if item.is_file() and item.suffix.lower() in self.allowed_extensions:
                files.append(item)
        logger.debug("Files discovered", extra={"count": len(files), "files": [f.name for f in files]})
        return files

    def _load_file(self, file_path: Path) -> List[Document]:
        """Load one file, returning a list of Documents with attached metadata."""
        ext = file_path.suffix.lower()
        loader_cls = self.LOADER_MAP.get(ext)

        if loader_cls is None:
            logger.warning("Unsupported file type — skipping", extra={"file": str(file_path)})
            return []

        t0 = time.perf_counter()
        try:
            loader = loader_cls(str(file_path))
            docs = loader.load()
            elapsed = round(time.perf_counter() - t0, 3)

            # Enrich metadata on every chunk produced by this file
            file_stat = file_path.stat()
            for doc in docs:
                doc.metadata.update(
                    {
                        "source":      file_path.name,          # used for citations
                        "file_path":   str(file_path),
                        "file_ext":    ext,
                        "file_size_kb": round(file_stat.st_size / 1024, 2),
                        "load_time_s": elapsed,
                    }
                )

            logger.info(
                "File loaded",
                extra={
                    "file": file_path.name,
                    "chunks": len(docs),
                    "size_kb": round(file_stat.st_size / 1024, 2),
                    "elapsed_s": elapsed,
                },
            )
            return docs

        except Exception as exc:
            logger.error(
                "Failed to load file",
                extra={"file": str(file_path), "error": str(exc)},
                exc_info=True,
            )
            return []  # Skip broken files rather than failing the whole ingest
