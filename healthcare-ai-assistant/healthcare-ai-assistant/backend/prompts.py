"""
prompts.py - All LLM prompt templates for the Healthcare AI Assistant.

Design principles:
  1. Ground every answer strictly in retrieved context — no external knowledge.
  2. Explicitly instruct the model to refuse when context is insufficient.
  3. Block diagnosis / unsafe medical advice at the prompt level.
  4. Request structured output (ANSWER / CONFIDENCE / SOURCES / DISCLAIMER)
     so the response parser can extract fields reliably.

Integration:
  - rag.py     imports RAG_PROMPT
  - agent.py   imports APPOINTMENT_PROMPT, CLASSIFICATION_PROMPT
"""

from langchain.prompts import PromptTemplate

# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY RAG PROMPT
# The main prompt used by the RAG chain for knowledge queries.
# ─────────────────────────────────────────────────────────────────────────────

_RAG_TEMPLATE = """You are a Healthcare AI Assistant. Answer using ONLY the context below.
Rules: (1) Only use provided context. (2) If answer not in context, say "I don't have enough information in the available documents to answer this. Please consult a healthcare professional." (3) Never diagnose or prescribe. (4) Cite document names.

Context:
{context}

Question: {question}

ANSWER:
[Answer here, cite sources inline e.g. (Source: filename.txt)]

CONFIDENCE: [HIGH|MEDIUM|LOW|NONE]
SOURCES: [filenames]
DISCLAIMER: [one sentence]
"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=_RAG_TEMPLATE,
)


# ─────────────────────────────────────────────────────────────────────────────
# APPOINTMENT BOOKING PROMPT
# Used when the agent routes to the appointment scheduling tool.
# ─────────────────────────────────────────────────────────────────────────────

_APPOINTMENT_TEMPLATE = """You are a friendly Healthcare Scheduling Assistant.
The system has retrieved the following available appointment slots:

{available_slots}

User's scheduling request: {user_request}

Instructions:
- Present the available slots clearly (date, time, provider, location/modality).
- If the user specified preferences (day, time, provider), highlight matching slots.
- If no slots match, apologise and suggest the closest alternatives.
- Keep the tone warm, professional, and concise.
- Always include these reminders at the end:
  • Cancellations require at least 24 hours' notice.
  • Bring a valid photo ID and your insurance card to in-person visits.
  • For urgent medical concerns, contact your provider directly or visit urgent care.

Response:"""

APPOINTMENT_PROMPT = PromptTemplate(
    input_variables=["available_slots", "user_request"],
    template=_APPOINTMENT_TEMPLATE,
)


# ─────────────────────────────────────────────────────────────────────────────
# QUERY CLASSIFICATION PROMPT
# Lightweight single-token classifier: APPOINTMENT or KNOWLEDGE.
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIFICATION_TEMPLATE = """Classify the following healthcare query into exactly one category.

Categories:
  APPOINTMENT — The user wants to book, cancel, reschedule, or check the
                availability of a medical appointment.
  KNOWLEDGE   — The user is asking for healthcare information, policies,
                procedures, insurance details, medications, or general
                health guidance.

Query: {query}

Respond with ONE WORD only — either APPOINTMENT or KNOWLEDGE.
Do not include punctuation, explanation, or any other text."""

CLASSIFICATION_PROMPT = PromptTemplate(
    input_variables=["query"],
    template=_CLASSIFICATION_TEMPLATE,
)


# ─────────────────────────────────────────────────────────────────────────────
# CANNED RESPONSES
# Returned without calling the LLM to protect safety boundaries.
# ─────────────────────────────────────────────────────────────────────────────

REFUSAL_RESPONSE = {
    "answer": (
        "I don't have enough information in the available documents to answer "
        "this question accurately. Please consult a qualified healthcare "
        "professional or contact our support team for assistance."
    ),
    "confidence": "NONE",
    "confidence_score": 0.0,
    "sources": [],
    "disclaimer": (
        "⚠️ This AI assistant provides general healthcare information only. "
        "It does not constitute medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare professional for medical decisions."
    ),
}

EMERGENCY_RESPONSE = {
    "answer": (
        "🚨 EMERGENCY DETECTED — If you or someone else is experiencing a "
        "medical emergency, call 911 (or your local emergency number) "
        "IMMEDIATELY. Do not rely on an AI assistant for emergency medical "
        "guidance. Stay on the line with emergency services."
    ),
    "confidence": "HIGH",
    "confidence_score": 1.0,
    "sources": [],
    "disclaimer": (
        "In a medical emergency always call emergency services (911) first."
    ),
}

DIAGNOSIS_REFUSAL_RESPONSE = {
    "answer": (
        "I'm not able to provide a medical diagnosis, prescribe medications, "
        "or recommend specific treatments for individual patients. "
        "Please schedule an appointment with a licensed healthcare provider "
        "who can evaluate your specific situation."
    ),
    "confidence": "HIGH",
    "confidence_score": 1.0,
    "sources": [],
    "disclaimer": (
        "⚠️ This system does not provide medical diagnosis or personalised "
        "treatment recommendations. Consult a licensed healthcare professional."
    ),
}

# Keywords that trigger emergency routing (checked before LLM call)
EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "stroke", "can't breathe", "cannot breathe",
    "difficulty breathing", "unconscious", "seizure", "severe bleeding",
    "allergic reaction", "anaphylaxis", "overdose", "suicidal", "suicide",
    "severe pain", "emergency",
]

# Keywords that trigger diagnosis refusal (checked before LLM call)
DIAGNOSIS_KEYWORDS = [
    "diagnose me", "do i have", "what disease do i have", "am i sick",
    "what's wrong with me", "prescribe", "what medication should i take",
    "treat my", "cure my", "is it cancer", "is it serious",
]