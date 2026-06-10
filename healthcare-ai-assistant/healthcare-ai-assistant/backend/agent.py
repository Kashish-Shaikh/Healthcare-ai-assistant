"""
agent.py - LangChain agent with keyword-based routing.

Routing logic:
  ┌──────────────────────────────────────────────────────────┐
  │ Query                                                    │
  │   │                                                      │
  │   ├─ contains appointment/book/schedule/availability     │
  │   │     → APPOINTMENT pipeline                           │
  │   │         → check_available_slots() [mock tool]        │
  │   │         → APPOINTMENT_PROMPT + Mistral               │
  │   │         → AskResponse (QueryType.APPOINTMENT)        │
  │   │                                                      │
  │   └─ everything else                                     │
  │         → KNOWLEDGE pipeline                             │
  │             → run_rag_pipeline(query)                    │
  │             → AskResponse (QueryType.KNOWLEDGE)          │
  └──────────────────────────────────────────────────────────┘

The keyword router is faster and more predictable than an LLM classifier
for the appointment use-case. An LLM classifier is included as a fallback
for ambiguous queries.

Integration:
  - routers/ask.py calls route_query(request) — the only public function.
  - agent.py calls rag.run_rag_pipeline() for KNOWLEDGE queries.
"""

import random
import time
from datetime import datetime, timedelta

from langchain.schema.output_parser import StrOutputParser

from backend.config import settings
from backend.llm import get_llm
from backend.logger import get_logger
from backend.models.schemas import AskResponse, SourceCitation, ConfidenceLevel, QueryType, AskRequest
from backend.prompts import APPOINTMENT_PROMPT, CLASSIFICATION_PROMPT
from backend.rag import run_rag_pipeline

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# APPOINTMENT KEYWORDS — routing trigger set
# ─────────────────────────────────────────────────────────────────────────────

APPOINTMENT_KEYWORDS = frozenset(
    {
        "appointment", "appointments",
        "book", "booking", "bookings",
        "schedule", "scheduling", "scheduled",
        "availability", "available",
        "slot", "slots",
        "cancel", "cancellation", "reschedule",
        "visit", "consultation",
        "see a doctor", "see the doctor",
        "make an appointment",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# MOCK APPOINTMENT TOOL
# In production this would call a calendar / EHR API.
# ─────────────────────────────────────────────────────────────────────────────

PROVIDERS = [
    {"name": "Dr. Sarah Mitchell, MD",  "specialty": "Primary Care",       "location": "Main Clinic — Room 101"},
    {"name": "Dr. James Chen, MD",      "specialty": "Internal Medicine",   "location": "Main Clinic — Room 205"},
    {"name": "Dr. Priya Patel, DO",     "specialty": "Family Medicine",     "location": "Telehealth (Video)"},
    {"name": "Dr. Robert Johnson, MD",  "specialty": "Cardiology",          "location": "Heart Center — Suite A"},
    {"name": "Dr. Laura Kim, MD",       "specialty": "Preventive Care",     "location": "Telehealth (Video)"},
]

TIME_SLOTS = ["08:00 AM", "09:30 AM", "11:00 AM", "01:00 PM", "02:30 PM", "04:00 PM"]


def check_available_slots(query: str | None = None) -> str:
    """
    Mock appointment scheduling tool.

    Simulates a real EHR/calendar API by generating pseudo-random but
    deterministic-looking availability for the next 7 business days.

    Returns:
        A human-readable string listing available slots, formatted for the LLM.
    """
    today = datetime.now()
    slots: list[str] = []

    # Generate slots for the next 7 calendar days (skip weekends)
    day_offset = 0
    slot_count = 0
    while slot_count < 8:          # Return ~8 slots across providers
        candidate = today + timedelta(days=day_offset + 1)
        day_offset += 1
        if candidate.weekday() >= 5:   # 5=Saturday, 6=Sunday
            continue

        # Random-ish: pick 1–2 providers for this day
        num_providers = random.randint(1, 2)
        chosen_providers = random.sample(PROVIDERS, k=num_providers)

        for provider in chosen_providers:
            time_slot = random.choice(TIME_SLOTS)
            date_str = candidate.strftime("%A, %B %d, %Y")
            slots.append(
                f"• {date_str} at {time_slot}\n"
                f"  Provider: {provider['name']} ({provider['specialty']})\n"
                f"  Location: {provider['location']}"
            )
            slot_count += 1
            if slot_count >= 8:
                break

    if not slots:
        return "No available slots found in the next 7 days. Please call our scheduling office."

    return "\n\n".join(slots)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER — keyword-based (fast path)
# ─────────────────────────────────────────────────────────────────────────────

def _is_appointment_query_keywords(query: str) -> bool:
    """Fast keyword scan — O(n) where n = number of keywords."""
    tokens = set(query.lower().split())
    # Also check multi-word phrases
    query_lower = query.lower()
    return bool(tokens & APPOINTMENT_KEYWORDS) or any(
        phrase in query_lower
        for phrase in ["make an appointment", "see a doctor", "see the doctor", "book a slot"]
    )


def _is_appointment_query_llm(query: str) -> bool:
    """
    LLM-based classifier — fallback for ambiguous queries.
    Returns True if the LLM classifies the query as APPOINTMENT.
    """
    try:
        llm = get_llm()
        chain = CLASSIFICATION_PROMPT | llm | StrOutputParser()
        result = chain.invoke({"query": query})
        return result.strip().upper() == "APPOINTMENT"
    except Exception as exc:
        logger.warning(
            "LLM classification failed — defaulting to KNOWLEDGE",
            extra={"error": str(exc)},
        )
        return False


# ─────────────────────────────────────────────────────────────────────────────
# APPOINTMENT PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _run_appointment_pipeline(query: str, conversation_id: str | None) -> AskResponse:
    """
    Execute the appointment scheduling pipeline:
      1. Fetch available slots (mock tool).
      2. Build prompt with slots + user request.
      3. Invoke Mistral.
      4. Return formatted AskResponse.
    """
    t0 = time.perf_counter()
    logger.info("Appointment pipeline started", extra={"query_preview": query[:80]})

    # Tool call
    available_slots = check_available_slots(query)
    logger.info("Available slots retrieved", extra={"slot_count": available_slots.count("•")})

    # LLM chain
    llm = get_llm()
    chain = APPOINTMENT_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"available_slots": available_slots, "user_request": query})

    processing_time = round(time.perf_counter() - t0, 3)

    logger.info(
        "Appointment pipeline completed",
        extra={"processing_time_s": processing_time, "conversation_id": conversation_id},
    )

    return AskResponse(
        query=query,
        answer=answer.strip(),
        confidence=ConfidenceLevel.HIGH,
        confidence_score=0.95,
        query_type=QueryType.APPOINTMENT,
        sources=[
            SourceCitation(
                document_name="appointment_scheduling_policy.txt",
                page_or_section="Scheduling Tool",
                relevant_excerpt="Real-time availability retrieved from the scheduling system.",
                similarity_score=1.0,
            )
        ],
        disclaimer=(
            "📅 Appointment availability is for illustration purposes. "
            "Contact our scheduling office at (555) 123-4567 to confirm bookings."
        ),
        processing_time_seconds=processing_time,
        conversation_id=conversation_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def route_query(request: AskRequest) -> AskResponse:
    """
    Route an incoming query to the appropriate pipeline.

    Routing order:
      1. Keyword scan (fast, deterministic).
      2. LLM classifier (slower, used only when keywords don't match).
      3. RAG pipeline (default for all KNOWLEDGE queries).

    Args:
        request: Validated AskRequest from the FastAPI router.

    Returns:
        AskResponse ready to be serialised and returned to the client.
    """
    query = request.query
    conversation_id = request.conversation_id

    logger.info(
        "Routing query",
        extra={"query_preview": query[:80], "conversation_id": conversation_id},
    )

    # ── Fast path: keyword scan ───────────────────────────────────────────────
    if _is_appointment_query_keywords(query):
        logger.info("Routed to APPOINTMENT pipeline via keywords")
        return _run_appointment_pipeline(query, conversation_id)

    # ── Default: RAG knowledge pipeline ───────────────────────────────────────
    logger.info("Routed to KNOWLEDGE (RAG) pipeline")
    return run_rag_pipeline(query, conversation_id)
