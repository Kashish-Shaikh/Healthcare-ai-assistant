"""
ingest.py - POST /ingest endpoint.

Accepts a documents directory path, loads all supported files,
builds/rebuilds the FAISS vector index, and returns ingestion metrics.

This is the first endpoint to call when setting up the system.
Typical call flow:
  POST /ingest {"documents_dir": "data/documents", "force_reload": false}
  → DocumentLoader scans directory
  → RecursiveCharacterTextSplitter chunks documents
  → MiniLM-L6-v2 embeds all chunks
  → FAISS index built and saved to disk
  → Returns document/chunk counts and timing
"""

import time
from fastapi import APIRouter, HTTPException, status

from backend.config import settings
from backend.logger import get_logger
from backend.models.schemas import IngestRequest, IngestResponse
from backend.utils.document_loader import DocumentLoader
from backend.vector_store import build_index

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest Healthcare Documents",
    description=(
        "Load healthcare documents from the specified directory, chunk them, "
        "embed with MiniLM-L6-v2, and store in the FAISS vector index. "
        "Call this endpoint once before querying with POST /ask."
    ),
    tags=["Document Management"],
)
async def ingest_documents(request: IngestRequest) -> IngestResponse:
    t0 = time.perf_counter()

    documents_dir = request.documents_dir or settings.DOCUMENTS_DIR

    logger.info(
        "Ingest request received",
        extra={"documents_dir": documents_dir, "force_reload": request.force_reload},
    )

    # ── Load documents ────────────────────────────────────────────────────────
    try:
        loader = DocumentLoader(documents_dir)
        documents = loader.load_directory()
    except FileNotFoundError as exc:
        logger.error("Documents directory not found", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Documents directory not found: {documents_dir}",
        )
    except Exception as exc:
        logger.error("Document loading failed", extra={"error": str(exc)}, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document loading failed: {str(exc)}",
        )

    if not documents:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"No supported documents found in '{documents_dir}'. "
                f"Supported formats: {settings.ALLOWED_EXTENSIONS}"
            ),
        )

    # ── Build FAISS index ─────────────────────────────────────────────────────
    try:
        chunks_created = build_index(documents, force_reload=request.force_reload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Index build failed", extra={"error": str(exc)}, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"FAISS index build failed: {str(exc)}",
        )

    processing_time = round(time.perf_counter() - t0, 3)

    logger.info(
        "Ingest completed",
        extra={
            "documents_processed": len(documents),
            "chunks_created": chunks_created,
            "processing_time_s": processing_time,
        },
    )

    return IngestResponse(
        success=True,
        message=(
            f"Successfully ingested {len(documents)} document(s) "
            f"into {chunks_created} chunks."
        ),
        documents_processed=len(documents),
        chunks_created=chunks_created,
        index_path=settings.FAISS_INDEX_PATH,
        processing_time_seconds=processing_time,
    )
