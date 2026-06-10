"""
ask.py - POST /ask endpoint.

Accepts a healthcare query, routes it through the agent (RAG or appointment
scheduling), and returns a structured response with answer, confidence,
sources, and safety disclaimer.

Request flow:
  POST /ask {"query": "...", "conversation_id": "...", "include_sources": true}
  → Input validation (Pydantic)
  → agent.route_query()
      ├─ APPOINTMENT → mock scheduling tool → Mistral
      └─ KNOWLEDGE   → FAISS retrieval → RAG prompt → Mistral
  → AskResponse (answer, confidence, sources, disclaimer, timing)
"""

import uuid
from fastapi import APIRouter, HTTPException, status

from backend.agent import route_query
from backend.logger import get_logger, request_id_var
from backend.models.schemas import AskRequest, AskResponse
from backend.vector_store import is_index_loaded

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a Healthcare Question",
    description=(
        "Submit a healthcare question. The system routes the query through "
        "the appropriate pipeline (RAG knowledge base or appointment scheduling) "
        "and returns an answer with confidence score and source citations.\n\n"
        "**Requires:** Documents must be ingested first via POST /ingest."
    ),
    tags=["Question Answering"],
)
async def ask_question(request: AskRequest) -> AskResponse:
    # Attach a request ID for distributed tracing
    req_id = str(uuid.uuid4())[:8]
    request_id_var.set(req_id)

    logger.info(
        "Ask request received",
        extra={
            "request_id": req_id,
            "query_preview": request.query[:80],
            "conversation_id": request.conversation_id,
        },
    )

    # ── Guard: index must be loaded ───────────────────────────────────────────
    if not is_index_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The knowledge base is not initialised. "
                "Please call POST /ingest to load documents before querying."
            ),
        )

    # ── Route and execute ─────────────────────────────────────────────────────
    try:
        response = route_query(request)
    except RuntimeError as exc:
        logger.error("Pipeline runtime error", extra={"error": str(exc)}, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Unexpected pipeline error", extra={"error": str(exc)}, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your query. Please try again.",
        )

    # ── Strip sources if caller opted out ────────────────────────────────────
    if not request.include_sources:
        response.sources = []

    logger.info(
        "Ask request completed",
        extra={
            "request_id": req_id,
            "confidence": response.confidence.value,
            "confidence_score": response.confidence_score,
            "query_type": response.query_type.value,
            "processing_time_s": response.processing_time_seconds,
        },
    )

    return response
