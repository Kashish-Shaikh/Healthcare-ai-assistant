"""
health.py - GET /health endpoint.

Returns real-time status of:
  - FAISS vector store (index loaded + chunk count)
  - Ollama LLM server (reachable + model present)
  - Overall system health (healthy / degraded / unhealthy)

Used by:
  - Docker HEALTHCHECK
  - Streamlit frontend status indicator
  - Monitoring / alerting systems
"""

import time
from fastapi import APIRouter

from backend.config import settings
from backend.llm import check_ollama_health
from backend.logger import get_logger
from backend.models.schemas import ComponentStatus, HealthResponse, HealthStatus
from backend.vector_store import get_document_count, is_index_loaded

logger = get_logger(__name__)
router = APIRouter()

# Track when the API started (for uptime reporting)
_START_TIME = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health Check",
    description="Returns the health status of all system components.",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    logger.info("Health check requested")

    # ── Check FAISS ───────────────────────────────────────────────────────────
    index_loaded = is_index_loaded()
    doc_count = get_document_count()

    faiss_status = ComponentStatus(
        status=HealthStatus.HEALTHY if index_loaded else HealthStatus.DEGRADED,
        detail=(
            f"Index loaded with {doc_count} chunks."
            if index_loaded
            else "Index not loaded — call POST /ingest to initialise."
        ),
    )

    # ── Check Ollama ──────────────────────────────────────────────────────────
    ollama_result = check_ollama_health()
    ollama_status = ComponentStatus(
        status=HealthStatus.HEALTHY if ollama_result["healthy"] else HealthStatus.UNHEALTHY,
        detail=ollama_result["detail"],
        latency_ms=ollama_result.get("latency_ms"),
    )

    # ── Overall status ────────────────────────────────────────────────────────
    if ollama_status.status == HealthStatus.UNHEALTHY:
        overall = HealthStatus.UNHEALTHY
    elif faiss_status.status == HealthStatus.DEGRADED:
        overall = HealthStatus.DEGRADED
    else:
        overall = HealthStatus.HEALTHY

    uptime = round(time.time() - _START_TIME, 1)

    return HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        components={"faiss": faiss_status, "ollama": ollama_status},
        index_document_count=doc_count,
        uptime_seconds=uptime,
    )
