"""
logger.py - Structured JSON logging for every module.

Usage in any module:
    from backend.logger import get_logger
    logger = get_logger(__name__)
    logger.info("message", extra={"key": "value"})

Each log record is emitted as a single JSON line, making it trivial
to ingest into ELK, Datadog, or CloudWatch in a HIPAA-compliant setup.
"""

import logging
import json
import sys
from datetime import datetime, timezone
from contextvars import ContextVar
from typing import Any

from backend.config import settings

# Per-request trace ID injected by FastAPI middleware
request_id_var: ContextVar[str] = ContextVar("request_id", default="system")


class JSONFormatter(logging.Formatter):
    """Formats every LogRecord as a single-line JSON string."""

    LEVEL_NAMES = {
        logging.DEBUG:    "DEBUG",
        logging.INFO:     "INFO",
        logging.WARNING:  "WARNING",
        logging.ERROR:    "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    # Keys that belong to the standard LogRecord — we don't re-emit these raw
    _STDLIB_KEYS = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process", "message",
        "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()

        payload: dict[str, Any] = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "level":      self.LEVEL_NAMES.get(record.levelno, "UNKNOWN"),
            "logger":     record.name,
            "request_id": request_id_var.get("system"),
            "message":    record.message,
            "module":     record.module,
            "function":   record.funcName,
            "line":       record.lineno,
        }

        # Attach any caller-supplied `extra={...}` fields
        for key, value in record.__dict__.items():
            if key not in self._STDLIB_KEYS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """
    Return a JSON-structured logger for *name* (typically __name__).
    Idempotent — calling twice for the same name returns the same instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured — avoid duplicate handlers

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(JSONFormatter())

    logger.addHandler(handler)
    logger.propagate = False

    return logger
