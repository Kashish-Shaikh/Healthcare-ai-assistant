"""
schemas.py - All Pydantic request/response models for the Healthcare AI Assistant.

These schemas are the API contract. FastAPI uses them to:
  - Validate and parse incoming requests automatically.
  - Serialise outgoing responses.
  - Auto-generate the OpenAPI / Swagger docs at /docs.

Integration:
  - backend/routers/ingest.py → IngestRequest, IngestResponse
  - backend/routers/ask.py   → AskRequest, AskResponse
  - backend/routers/health.py → HealthResponse
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# ENUMERATIONS
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceLevel(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"
    NONE   = "NONE"


class QueryType(str, Enum):
    KNOWLEDGE   = "KNOWLEDGE"
    APPOINTMENT = "APPOINTMENT"


class HealthStatus(str, Enum):
    HEALTHY   = "healthy"
    DEGRADED  = "degraded"
    UNHEALTHY = "unhealthy"


# ─────────────────────────────────────────────────────────────────────────────
# INGEST SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """POST /ingest — request body."""

    documents_dir: Optional[str] = Field(
        default=None,
        description=(
            "Path to a directory containing documents to ingest. "
            "Falls back to the configured DOCUMENTS_DIR when omitted."
        ),
        examples=["data/documents"],
    )
    force_reload: bool = Field(
        default=False,
        description="Clear the existing FAISS index and re-ingest everything.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"documents_dir": "data/documents", "force_reload": False}
        }
    }


class IngestResponse(BaseModel):
    """POST /ingest — response body."""

    success: bool
    message: str
    documents_processed: int = Field(ge=0)
    chunks_created: int = Field(ge=0)
    index_path: str
    processing_time_seconds: float = Field(ge=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# ASK SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class SourceCitation(BaseModel):
    """One source document chunk that contributed to an answer."""

    document_name: str = Field(description="Filename of the source document.")
    page_or_section: Optional[str] = Field(
        default=None,
        description="Page number or section heading if available.",
    )
    relevant_excerpt: str = Field(
        max_length=500,
        description="Short excerpt from the chunk that supported the answer.",
    )
    similarity_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Cosine similarity between the chunk and the query (0–1).",
    )


class AskRequest(BaseModel):
    """POST /ask — request body."""

    query: str = Field(
        min_length=3,
        max_length=2000,
        description="The healthcare question to answer.",
        examples=["What is the telehealth appointment policy?"],
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for multi-turn conversation tracking.",
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source citations in the response.",
    )

    @field_validator("query")
    @classmethod
    def sanitise_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty or whitespace.")
        # Block naive prompt-injection attempts
        injection_patterns = r"(ignore previous|disregard all|you are now|act as if|forget your instructions)"
        if re.search(injection_patterns, v, re.IGNORECASE):
            raise ValueError("Query contains disallowed content.")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What documents do I need for a telehealth appointment?",
                "conversation_id": "session-abc-123",
                "include_sources": True,
            }
        }
    }


class AskResponse(BaseModel):
    """POST /ask — response body."""

    query: str
    answer: str
    confidence: ConfidenceLevel
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Numeric confidence (0 = none, 1 = maximum).",
    )
    query_type: QueryType = Field(
        description="Pipeline that handled this query: KNOWLEDGE (RAG) or APPOINTMENT."
    )
    sources: List[SourceCitation] = Field(
        default_factory=list,
        description="Source documents that contributed to this answer.",
    )
    disclaimer: str = Field(
        default=(
            "⚠️ This AI assistant provides general healthcare information only. "
            "It does not constitute medical advice, diagnosis, or treatment. "
            "Always consult a qualified healthcare professional for medical decisions."
        )
    )
    processing_time_seconds: float = Field(ge=0.0)
    conversation_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ComponentStatus(BaseModel):
    """Health status of one system component."""

    status: HealthStatus
    detail: Optional[str] = None
    latency_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """GET /health — response body."""

    status: HealthStatus
    version: str
    components: dict[str, ComponentStatus]
    index_document_count: int = Field(
        ge=0,
        description="Number of chunks currently stored in the FAISS index.",
    )
    uptime_seconds: float = Field(ge=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# ERROR SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    """Standard error envelope returned on 4xx / 5xx responses."""

    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
