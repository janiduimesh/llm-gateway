from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
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

        for field in self._RECORD_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "value": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        if record.stack_info:
            payload["stack_info"] = record.stack_info

        return json.dumps(payload, default=str, ensure_ascii=False)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    return logger


log = get_logger("llm_gateway")
