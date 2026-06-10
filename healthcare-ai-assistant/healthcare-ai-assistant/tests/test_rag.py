"""
test_rag.py - Unit tests for the RAG pipeline.

Tests:
  - Safety check functions (emergency / diagnosis detection)
  - LLM response parser
  - Confidence scoring and level mapping
  - Citation builder
  - Context formatter
  - Full pipeline integration (mocked LLM + FAISS)

Run:
  pytest tests/test_rag.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import Document


# ─────────────────────────────────────────────────────────────────────────────
# IMPORT RAG FUNCTIONS UNDER TEST
# Patch heavy imports before module load
# ─────────────────────────────────────────────────────────────────────────────

with patch("backend.embeddings.get_embeddings"), \
     patch("backend.llm.get_llm"):
    from backend.rag import (
        _is_emergency,
        _is_diagnosis_request,
        _parse_llm_response,
        _compute_confidence_score,
        _score_to_level,
        _build_citations,
        _format_context,
        run_rag_pipeline,
    )
    from backend.models.schemas import ConfidenceLevel, QueryType, SourceCitation


# ─────────────────────────────────────────────────────────────────────────────
# SAFETY CHECKS
# ─────────────────────────────────────────────────────────────────────────────

class TestSafetyChecks:

    def test_detects_chest_pain_emergency(self):
        assert _is_emergency("I have chest pain and shortness of breath") is True

    def test_detects_heart_attack_keyword(self):
        assert _is_emergency("My father is having a heart attack") is True

    def test_detects_stroke_keyword(self):
        assert _is_emergency("She is showing signs of a stroke") is True

    def test_detects_seizure_keyword(self):
        assert _is_emergency("patient had a seizure in the waiting room") is True

    def test_normal_question_not_emergency(self):
        assert _is_emergency("What is the HIPAA policy on patient records?") is False

    def test_appointment_query_not_emergency(self):
        assert _is_emergency("Can I book an appointment for next Tuesday?") is False

    def test_detects_diagnosis_request(self):
        assert _is_diagnosis_request("Can you diagnose me based on my symptoms?") is True

    def test_detects_do_i_have_pattern(self):
        assert _is_diagnosis_request("Do I have diabetes based on these symptoms?") is True

    def test_detects_prescribe_keyword(self):
        assert _is_diagnosis_request("Can you prescribe me something for my headache?") is True

    def test_normal_policy_query_not_diagnosis(self):
        assert _is_diagnosis_request("What is the telehealth visit policy?") is False

    def test_medication_refill_query_not_diagnosis(self):
        assert _is_diagnosis_request("How do I request a medication refill?") is False


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE PARSER
# ─────────────────────────────────────────────────────────────────────────────

class TestResponseParser:

    def test_parses_well_formed_response(self):
        raw = """
ANSWER:
Telehealth visits are available Monday through Friday from 7am to 8pm.

CONFIDENCE: HIGH
SOURCES: telehealth_policy.txt, appointment_scheduling_policy.txt
DISCLAIMER: Always consult a qualified healthcare professional.
"""
        result = _parse_llm_response(raw)
        assert "Monday through Friday" in result["answer"]
        assert result["confidence"] == "HIGH"
        assert "telehealth_policy.txt" in result["sources"]
        assert "Always consult" in result["disclaimer"]

    def test_parses_medium_confidence(self):
        raw = "ANSWER:\nPartial info.\n\nCONFIDENCE: MEDIUM\nSOURCES: doc.txt\nDISCLAIMER: Consult a doctor."
        result = _parse_llm_response(raw)
        assert result["confidence"] == "MEDIUM"

    def test_parses_none_confidence(self):
        raw = "ANSWER:\nI don't know.\n\nCONFIDENCE: NONE\nSOURCES:\nDISCLAIMER:"
        result = _parse_llm_response(raw)
        assert result["confidence"] == "NONE"

    def test_fallback_when_no_sections(self):
        raw = "The telehealth policy allows virtual visits."
        result = _parse_llm_response(raw)
        # Should not crash; answer should contain the full text
        assert result["answer"] != ""
        assert result["confidence"] in ("HIGH", "MEDIUM", "LOW", "NONE")

    def test_multiple_sources_parsed(self):
        raw = "ANSWER:\nSome info.\n\nCONFIDENCE: HIGH\nSOURCES: doc1.txt, doc2.txt, doc3.txt\nDISCLAIMER: N/A"
        result = _parse_llm_response(raw)
        assert len(result["sources"]) == 3

    def test_empty_sources_returns_list(self):
        raw = "ANSWER:\nInfo.\n\nCONFIDENCE: LOW\nSOURCES:\nDISCLAIMER: Consult a doctor."
        result = _parse_llm_response(raw)
        assert isinstance(result["sources"], list)

    def test_case_insensitive_confidence(self):
        raw = "ANSWER:\nInfo.\n\nCONFIDENCE: high\nSOURCES: doc.txt\nDISCLAIMER: ."
        result = _parse_llm_response(raw)
        assert result["confidence"] == "HIGH"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE SCORING
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceScoring:

    def test_high_llm_high_faiss_gives_high_score(self):
        score = _compute_confidence_score("HIGH", [0.95, 0.90, 0.88])
        assert score >= 0.80

    def test_none_llm_gives_low_score(self):
        score = _compute_confidence_score("NONE", [0.20, 0.15])
        assert score < 0.40

    def test_empty_faiss_scores_uses_llm_only(self):
        score = _compute_confidence_score("HIGH", [])
        assert score > 0.0  # Uses LLM weight only

    def test_score_clamped_to_0_1(self):
        score = _compute_confidence_score("HIGH", [1.0, 1.0, 1.0])
        assert 0.0 <= score <= 1.0

    def test_low_faiss_drags_down_high_llm(self):
        score_high_faiss = _compute_confidence_score("HIGH", [0.95, 0.90])
        score_low_faiss  = _compute_confidence_score("HIGH", [0.10, 0.05])
        assert score_high_faiss > score_low_faiss

    def test_score_to_level_high(self):
        assert _score_to_level(0.85) == ConfidenceLevel.HIGH

    def test_score_to_level_medium(self):
        assert _score_to_level(0.60) == ConfidenceLevel.MEDIUM

    def test_score_to_level_low(self):
        assert _score_to_level(0.35) == ConfidenceLevel.LOW

    def test_score_to_level_none(self):
        assert _score_to_level(0.10) == ConfidenceLevel.NONE


# ─────────────────────────────────────────────────────────────────────────────
# CITATION BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class TestCitationBuilder:

    def _make_doc(self, source: str, content: str = "Sample content") -> Document:
        return Document(page_content=content, metadata={"source": source})

    def test_builds_citation_from_docs(self):
        docs_with_scores = [
            (self._make_doc("telehealth_policy.txt", "Telehealth hours info"), 0.92),
        ]
        citations = _build_citations(docs_with_scores, ["telehealth_policy.txt"])
        assert len(citations) == 1
        assert citations[0].document_name == "telehealth_policy.txt"

    def test_deduplicates_same_source(self):
        doc = self._make_doc("doc.txt", "Content")
        docs = [(doc, 0.9), (doc, 0.8), (doc, 0.7)]
        citations = _build_citations(docs, [])
        assert len(citations) == 1

    def test_similarity_score_attached(self):
        docs = [(self._make_doc("doc.txt"), 0.87)]
        citations = _build_citations(docs, ["doc.txt"])
        assert citations[0].similarity_score == pytest.approx(0.87, abs=0.001)

    def test_returns_empty_for_no_docs(self):
        citations = _build_citations([], [])
        assert citations == []

    def test_excerpt_length_capped(self):
        long_content = "A" * 1000
        docs = [(self._make_doc("doc.txt", long_content), 0.85)]
        citations = _build_citations(docs, ["doc.txt"])
        assert len(citations[0].relevant_excerpt) <= 500


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

class TestContextFormatter:

    def test_formats_single_document(self):
        doc = Document(page_content="HIPAA policy content.", metadata={"source": "hipaa.txt"})
        context = _format_context([(doc, 0.92)])
        assert "hipaa.txt" in context
        assert "HIPAA policy content." in context
        assert "Document 1" in context

    def test_formats_multiple_documents(self):
        docs = [
            (Document(page_content=f"Content {i}", metadata={"source": f"doc{i}.txt"}), 0.9 - i * 0.1)
            for i in range(3)
        ]
        context = _format_context(docs)
        assert "Document 1" in context
        assert "Document 2" in context
        assert "Document 3" in context

    def test_includes_relevance_score(self):
        doc = Document(page_content="Content", metadata={"source": "doc.txt"})
        context = _format_context([(doc, 0.88)])
        assert "0.88" in context

    def test_empty_list_returns_empty_string(self):
        context = _format_context([])
        assert context == ""


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE INTEGRATION (mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGPipelineIntegration:

    def _make_doc(self, source: str) -> Document:
        return Document(
            page_content="Telehealth visits are available Monday to Friday 7am–8pm.",
            metadata={"source": source},
        )

    @patch("backend.rag.similarity_search_with_scores")
    @patch("backend.rag.get_llm")
    @patch("backend.rag.StrOutputParser")
    def test_pipeline_returns_ask_response(self, MockParser, mock_get_llm, mock_search):
        """Test that a valid query with good FAISS similarity produces an AskResponse."""
        mock_search.return_value = [(self._make_doc("telehealth_policy.txt"), 0.92)]

        mock_chain_output = (
            "ANSWER:\nTelehealth visits are Monday–Friday 7am–8pm.\n\n"
            "CONFIDENCE: HIGH\nSOURCES: telehealth_policy.txt\nDISCLAIMER: Consult a provider."
        )

        # Build a mock chain that returns our canned output when .invoke() is called
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_chain_output

        # LCEL: RAG_PROMPT | llm | parser  — patch the | operator on the prompt
        mock_prompt_with_llm = MagicMock()
        mock_prompt_with_llm.__or__ = MagicMock(return_value=mock_chain)

        with patch("backend.rag.RAG_PROMPT") as MockPrompt:
            MockPrompt.__or__ = MagicMock(return_value=mock_prompt_with_llm)
            result = run_rag_pipeline("What are the telehealth hours?")

        assert result.query == "What are the telehealth hours?"
        assert result.query_type == QueryType.KNOWLEDGE

    @patch("backend.rag.similarity_search_with_scores")
    def test_pipeline_returns_emergency_response(self, mock_search):
        result = run_rag_pipeline("I have chest pain and cannot breathe")
        assert "911" in result.answer or "emergency" in result.answer.lower()
        assert result.confidence_score == 1.0

    @patch("backend.rag.similarity_search_with_scores")
    def test_pipeline_refuses_diagnosis_request(self, mock_search):
        result = run_rag_pipeline("Can you diagnose me with diabetes?")
        assert result.confidence_score == 1.0  # HIGH confidence refusal
        assert "diagnosis" in result.answer.lower() or "provider" in result.answer.lower()

    @patch("backend.rag.similarity_search_with_scores")
    def test_pipeline_refuses_when_low_similarity(self, mock_search):
        mock_search.return_value = [
            (self._make_doc("doc.txt"), 0.10),  # Very low similarity
        ]
        result = run_rag_pipeline("What is quantum computing?")
        assert result.confidence == ConfidenceLevel.NONE

    @patch("backend.rag.similarity_search_with_scores")
    def test_pipeline_handles_faiss_runtime_error(self, mock_search):
        mock_search.side_effect = RuntimeError("FAISS index not loaded")
        result = run_rag_pipeline("What is HIPAA?")
        assert result.confidence == ConfidenceLevel.NONE
