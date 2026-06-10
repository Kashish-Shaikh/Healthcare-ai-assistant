"""
llm.py - Ollama/Mistral LLM wrapper for the Healthcare AI Assistant.

Why Mistral?
  • State-of-the-art 7B open-source model (Apache 2.0 licence)
  • Excellent instruction-following → reliable structured output parsing
  • Runs locally via Ollama → no data leaves the network (HIPAA-friendly)
  • Low hallucination rate vs comparable open models
  • Fits comfortably on a single GPU or fast CPU

Why Ollama?
  • Zero-configuration local LLM serving
  • Compatible with OpenAI-style API → easy LangChain integration
  • Supports model caching and quantisation out of the box

Integration:
  - rag.py   imports get_llm() for the RAG chain
  - agent.py imports get_llm() for the classification and appointment chains
"""

import time
from functools import lru_cache

import httpx
from langchain_ollama import OllamaLLM

from backend.config import settings
from backend.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> OllamaLLM:
    """
    Build and cache the Ollama/Mistral LLM instance.

    Parameters flow from config.py so they can be tuned via environment
    variables without touching code.
    """
    logger.info(
        "Initialising LLM",
        extra={
            "model": settings.OLLAMA_MODEL,
            "base_url": settings.OLLAMA_BASE_URL,
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
        },
    )

    llm = OllamaLLM(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
        num_predict=settings.LLM_MAX_TOKENS,
        timeout=settings.LLM_REQUEST_TIMEOUT,
        # Repeat-penalty reduces the model echoing the context back verbatim
        repeat_penalty=1.1,
    )

    logger.info("LLM initialised", extra={"model": settings.OLLAMA_MODEL})
    return llm


def check_ollama_health() -> dict:
    """
    Probe the Ollama server and verify the target model is available.
    Returns a dict with keys: 'healthy', 'latency_ms', 'detail'.
    Called by GET /health.
    """
    url = f"{settings.OLLAMA_BASE_URL}/api/tags"
    t0 = time.perf_counter()

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)

            data = resp.json()
            available_models = [m["name"] for m in data.get("models", [])]
            model_present = any(
                settings.OLLAMA_MODEL in name for name in available_models
            )

            if model_present:
                return {
                    "healthy": True,
                    "latency_ms": latency_ms,
                    "detail": f"Model '{settings.OLLAMA_MODEL}' is available.",
                }
            else:
                return {
                    "healthy": False,
                    "latency_ms": latency_ms,
                    "detail": (
                        f"Model '{settings.OLLAMA_MODEL}' not found. "
                        f"Available: {available_models}"
                    ),
                }

    except httpx.ConnectError:
        return {
            "healthy": False,
            "latency_ms": None,
            "detail": f"Cannot connect to Ollama at {settings.OLLAMA_BASE_URL}",
        }
    except Exception as exc:
        return {
            "healthy": False,
            "latency_ms": None,
            "detail": f"Ollama health check failed: {str(exc)}",
        }
