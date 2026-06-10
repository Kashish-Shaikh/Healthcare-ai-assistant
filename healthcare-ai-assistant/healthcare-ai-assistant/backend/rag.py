"""
rag.py - Full RAG pipeline for the Healthcare AI Assistant.

Pipeline (left to right):
  Query → Safety Checks → FAISS Retriever → Context Assembly
        → RAG Prompt → Mistral LLM → Response Parser
        → Confidence Scorer → Citation Builder → AskResponse

Key design decisions:
  1. Safety checks run BEFORE the LLM to block emergencies/diagnosis fast.
  2. Confidence is computed from raw FAISS similarity scores (not self-reported).
  3. Structured output (ANSWER / CONFIDENCE / SOURCES / DISCLAIMER) parsed
     with regex fallbacks — the system degrades gracefully if Mistral
     deviates from the format.
  4. Sources are deduplicated and capped at FAISS_TOP_K.

Integration:
  - agent.py calls run_rag_pipeline(query) for KNOWLEDGE queries.
  - routers/ask.py calls agent.route_query() which delegates here.
"""

import re
import time
from typing import Any

from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from backend.config import settings
from backend.embeddings import get_embeddings
from backend.llm import get_llm
from backend.logger import get_logger
from backend.models.schemas import AskResponse, SourceCitation, ConfidenceLevel, QueryType
from backend.prompts import (
    RAG_PROMPT,
    REFUSAL_RESPONSE,
    EMERGENCY_RESPONSE,
    DIAGNOSIS_REFUSAL_RESPONSE,
    EMERGENCY_KEYWORDS,
    DIAGNOSIS_KEYWORDS,
)
from backend.vector_store import similarity_search_with_scores, get_retriever

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SAFETY CHECKS  (run before any LLM call)
# ─────────────────────────────────────────────────────────────────────────────

def _is_emergency(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in EMERGENCY_KEYWORDS)


def _is_diagnosis_request(query: str) -> bool:
    if not settings.REFUSE_DIAGNOSIS:
        return False
    q = query.lower()
    return any(kw in q for kw in DIAGNOSIS_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE PARSER
# ─────────────────────────────────────────────────────────────────────────────

def _parse_llm_response(raw: str) -> dict[str, Any]:
    """
    Parse the structured LLM output into a dict with keys:
      answer, confidence, sources, disclaimer.
    Falls back gracefully when the model doesn't follow the format exactly.
    """
    result: dict[str, Any] = {
        "answer":     "",
        "confidence": "LOW",
        "sources":    [],
        "disclaimer": (
            "⚠️ This AI assistant provides general healthcare information only. "
            "Always consult a qualified healthcare professional for medical decisions."
        ),
    }

    # ── ANSWER ────────────────────────────────────────────────────────────────
    answer_match = re.search(
        r"ANSWER:\s*(.*?)(?=CONFIDENCE:|SOURCES:|DISCLAIMER:|$)",
        raw, re.DOTALL | re.IGNORECASE,
    )
    if answer_match:
        result["answer"] = answer_match.group(1).strip()
    else:
        # Fallback: the whole response is the answer
        result["answer"] = raw.strip()

    # ── CONFIDENCE ────────────────────────────────────────────────────────────
    conf_match = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW|NONE)", raw, re.IGNORECASE)
    if conf_match:
        result["confidence"] = conf_match.group(1).upper()

    # ── SOURCES ───────────────────────────────────────────────────────────────
    sources_match = re.search(
        r"SOURCES:\s*(.*?)(?=DISCLAIMER:|$)", raw, re.DOTALL | re.IGNORECASE
    )
    if sources_match:
        raw_sources = sources_match.group(1).strip()
        # Split on commas or newlines, strip whitespace
        result["sources"] = [
            s.strip() for s in re.split(r"[,\n]+", raw_sources) if s.strip()
        ]

    # ── DISCLAIMER ────────────────────────────────────────────────────────────
    disc_match = re.search(r"DISCLAIMER:\s*(.*?)$", raw, re.DOTALL | re.IGNORECASE)
    if disc_match:
        disc_text = disc_match.group(1).strip()
        if disc_text:
            result["disclaimer"] = disc_text

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE SCORING
# ─────────────────────────────────────────────────────────────────────────────

_CONFIDENCE_MAP = {
    "HIGH":   0.9,
    "MEDIUM": 0.65,
    "LOW":    0.35,
    "NONE":   0.0,
}


def _compute_confidence_score(
    llm_confidence: str,
    similarity_scores: list[float],
) -> float:
    """
    Blend the LLM's self-reported confidence with the mean FAISS similarity score.

    This prevents the model from claiming HIGH confidence when retrieved
    chunks are only weakly related to the query.
    """
    llm_score = _CONFIDENCE_MAP.get(llm_confidence.upper(), 0.35)

    if similarity_scores:
        # Use top-3 scores to avoid outliers dragging the mean
        top_scores = sorted(similarity_scores, reverse=True)[:3]
        faiss_score = sum(top_scores) / len(top_scores)
    else:
        faiss_score = 0.0

    # Weighted blend: 60% FAISS evidence, 40% LLM self-report
    blended = 0.6 * faiss_score + 0.4 * llm_score
    return round(min(max(blended, 0.0), 1.0), 4)


def _score_to_level(score: float) -> ConfidenceLevel:
    if score >= 0.75:
        return ConfidenceLevel.HIGH
    if score >= 0.50:
        return ConfidenceLevel.MEDIUM
    if score >= settings.MIN_CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.NONE


# ─────────────────────────────────────────────────────────────────────────────
# CITATION BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_citations(
    docs_with_scores: list[tuple[Document, float]],
    cited_sources: list[str],
) -> list[SourceCitation]:
    """
    Build SourceCitation objects from retrieved FAISS chunks.
    Only include documents that the LLM cited, in similarity order.
    """
    citations: list[SourceCitation] = []
    seen: set[str] = set()

    for doc, score in sorted(docs_with_scores, key=lambda x: x[1], reverse=True):
        source_name = doc.metadata.get("source", "unknown")

        # Deduplicate by source file name
        if source_name in seen:
            continue
        seen.add(source_name)

        # Include if LLM cited it OR if no explicit citations were parsed
        should_include = (
            not cited_sources
            or any(source_name.lower() in s.lower() for s in cited_sources)
            or any(s.lower() in source_name.lower() for s in cited_sources)
        )

        if should_include:
            excerpt = doc.page_content[:400].strip()
            citations.append(
                SourceCitation(
                    document_name=source_name,
                    page_or_section=doc.metadata.get("page") or doc.metadata.get("section"),
                    relevant_excerpt=excerpt,
                    similarity_score=round(float(score), 4),
                )
            )

    return citations[:settings.FAISS_TOP_K]   # Cap at top-K


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

def _format_context(docs_with_scores: list[tuple[Document, float]]) -> str:
    """Format retrieved chunks into the context block injected into the prompt."""
    parts: list[str] = []
    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(
            f"[Document {i} | Source: {source} | Relevance: {score:.2f}]\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_rag_pipeline(query: str, conversation_id: str | None = None) -> AskResponse:
    """
    Execute the full RAG pipeline for a healthcare knowledge query.

    Steps:
      1. Safety checks (emergency / diagnosis keywords)
      2. FAISS similarity search with scores
      3. Low-confidence guard (return REFUSAL if max score < threshold)
      4. Build prompt context from top-K chunks
      5. Invoke Mistral via LangChain LCEL chain
      6. Parse structured response
      7. Compute blended confidence score
      8. Build source citations
      9. Return AskResponse

    Args:
        query:           The user's healthcare question.
        conversation_id: Optional session ID for logging/tracking.

    Returns:
        AskResponse with answer, confidence, sources, and disclaimer.
    """
    t0 = time.perf_counter()

    log_extra = {"query_preview": query[:80], "conversation_id": conversation_id}
    logger.info("RAG pipeline started", extra=log_extra)

    # ── Step 1: Safety checks ────────────────────────────────────────────────
    if _is_emergency(query):
        logger.warning("Emergency query detected — returning canned response", extra=log_extra)
        return AskResponse(
            query=query,
            query_type=QueryType.KNOWLEDGE,
            conversation_id=conversation_id,
            processing_time_seconds=round(time.perf_counter() - t0, 3),
            **EMERGENCY_RESPONSE,
        )

    if _is_diagnosis_request(query):
        logger.warning("Diagnosis request detected — returning refusal", extra=log_extra)
        return AskResponse(
            query=query,
            query_type=QueryType.KNOWLEDGE,
            conversation_id=conversation_id,
            processing_time_seconds=round(time.perf_counter() - t0, 3),
            **DIAGNOSIS_REFUSAL_RESPONSE,
        )

    # ── Step 2: Retrieve similar chunks with scores ──────────────────────────
    try:
        docs_with_scores = similarity_search_with_scores(query, k=settings.FAISS_TOP_K)
    except RuntimeError as exc:
        logger.error("FAISS retrieval failed", extra={**log_extra, "error": str(exc)})
        return AskResponse(
            query=query,
            query_type=QueryType.KNOWLEDGE,
            conversation_id=conversation_id,
            processing_time_seconds=round(time.perf_counter() - t0, 3),
            **REFUSAL_RESPONSE,
        )

    # ── Step 3: Low-confidence guard ─────────────────────────────────────────
    similarity_scores = [score for _, score in docs_with_scores]
    max_similarity = max(similarity_scores) if similarity_scores else 0.0

    logger.info(
        "FAISS retrieval complete",
        extra={
            **log_extra,
            "chunks_retrieved": len(docs_with_scores),
            "max_similarity": round(max_similarity, 4),
        },
    )

    if max_similarity < settings.MIN_CONFIDENCE_THRESHOLD:
        logger.warning(
            "Low similarity scores — refusing to answer",
            extra={**log_extra, "max_similarity": max_similarity},
        )
        return AskResponse(
            query=query,
            query_type=QueryType.KNOWLEDGE,
            conversation_id=conversation_id,
            processing_time_seconds=round(time.perf_counter() - t0, 3),
            **REFUSAL_RESPONSE,
        )

    # ── Step 4: Format context ────────────────────────────────────────────────
    context = _format_context(docs_with_scores)

    # ── Step 5: Build and invoke LangChain LCEL chain ────────────────────────
    llm = get_llm()

    # LCEL: prompt | llm | string output parser
    chain = RAG_PROMPT | llm | StrOutputParser()

    logger.info("Invoking LLM", extra=log_extra)
    raw_response = chain.invoke({"context": context, "question": query})

    logger.info(
        "LLM responded",
        extra={**log_extra, "response_length": len(raw_response)},
    )

    # ── Step 6: Parse structured response ────────────────────────────────────
    parsed = _parse_llm_response(raw_response)

    # ── Step 7: Compute blended confidence score ──────────────────────────────
    confidence_score = _compute_confidence_score(
        parsed["confidence"], similarity_scores
    )
    confidence_level = _score_to_level(confidence_score)

    # If the model self-reported NONE, override to REFUSAL
    if parsed["confidence"] == "NONE" or confidence_level == ConfidenceLevel.NONE:
        logger.info("LLM reported NONE confidence — returning refusal", extra=log_extra)
        return AskResponse(
            query=query,
            query_type=QueryType.KNOWLEDGE,
            conversation_id=conversation_id,
            processing_time_seconds=round(time.perf_counter() - t0, 3),
            **REFUSAL_RESPONSE,
        )

    # ── Step 8: Build citations ───────────────────────────────────────────────
    citations = _build_citations(docs_with_scores, parsed["sources"])

    processing_time = round(time.perf_counter() - t0, 3)

    logger.info(
        "RAG pipeline completed",
        extra={
            **log_extra,
            "confidence_level": confidence_level.value,
            "confidence_score": confidence_score,
            "citations": len(citations),
            "processing_time_s": processing_time,
        },
    )

    # ── Step 9: Return structured response ───────────────────────────────────
    return AskResponse(
        query=query,
        answer=parsed["answer"],
        confidence=confidence_level,
        confidence_score=confidence_score,
        query_type=QueryType.KNOWLEDGE,
        sources=citations,
        disclaimer=parsed["disclaimer"],
        processing_time_seconds=processing_time,
        conversation_id=conversation_id,
    )
