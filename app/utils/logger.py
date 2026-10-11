"""
Structured JSON Observability Logger
-------------------------------------
Produces machine-readable JSON log lines consumable by log-aggregation
stacks (Loki, Datadog, ELK, Cloud Logging, etc.).

Every log entry includes:
  - timestamp  (ISO-8601 UTC)
  - level
  - logger name
  - message
  - any extra keyword context passed at call-site
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Custom JSON formatter
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """Renders each LogRecord as a single-line JSON object."""

    # Fields we want to pull directly from the LogRecord
    _RECORD_FIELDS = (
        "name",
        "levelname",
        "funcName",
        "lineno",
        "module",
        "pathname",
        "thread",
        "threadName",
        "process",
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach standard record metadata
        for field in self._RECORD_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        # Attach any extra context injected via the `extra={}` kwarg
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                payload[key] = value

        # Attach exception info when present
        if record.exc_info:
            payload["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "value": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Attach stack info
        if record.stack_info:
            payload["stack_info"] = record.stack_info

        return json.dumps(payload, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger that emits structured JSON to *stdout*.

    Args:
        name:  Logger name – use ``__name__`` at the call-site for automatic
               module-level scoping.
        level: Minimum log level (default: INFO).

    Returns:
        A configured :class:`logging.Logger` instance.

    Example::

        from app.utils.logger import get_logger
        log = get_logger(__name__)
        log.info("Request received", extra={"team_id": "eng-team-1", "tokens": 512})
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger was already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False  # prevent double-logging through the root logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    return logger


# ---------------------------------------------------------------------------
# Module-level convenience logger (used by other modules via import)
# ---------------------------------------------------------------------------

log = get_logger("llm_gateway")
