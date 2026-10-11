from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from app.core.config import Settings
from app.utils.logger import get_logger

log = get_logger(__name__)


_PII_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        "SSN",
        re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
        "[REDACTED_SSN]",
    ),
    (
        "CREDIT_CARD",
        re.compile(
            r"\b(?:4[0-9]{12}(?:[0-9]{3})?|"
            r"5[1-5][0-9]{14}|"
            r"3[47][0-9]{13}|"
            r"3(?:0[0-5]|[68][0-9])[0-9]{11}|"
            r"6(?:011|5[0-9]{2})[0-9]{12}|"
            r"(?:2131|1800|35\d{3})\d{11})\b"
        ),
        "[REDACTED_CC]",
    ),
    (
        "PHONE",
        re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"),
        "[REDACTED_PHONE]",
    ),
    (
        "IPV4",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        "[REDACTED_IP]",
    ),
    (
        "SECRET",
        re.compile(r"(?i)\b(?:password|passwd|secret|token|api[_\-]?key)\s*[:=]\s*\S+"),
        "[REDACTED_SECRET]",
    ),
    (
        "DOB",
        re.compile(r"\b(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b"),
        "[REDACTED_DOB]",
    ),
]


_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.I),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(?:dan|jailbreak|evil|unrestricted)", re.I),
    re.compile(r"disregard\s+(your\s+)?(?:guidelines?|rules?|constraints?|training)", re.I),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+(?:an?\s+)?(?:ai|llm|assistant)?\s*without", re.I),
    re.compile(r"act\s+as\s+(?:if\s+you\s+(?:have\s+no|are\s+without)\s+(?:restrictions?|limits?))", re.I),
    re.compile(r"system\s*prompt\s*[:=]", re.I),
    re.compile(r"\bdo\s+anything\s+now\b", re.I),
    re.compile(r"override\s+(?:your\s+)?(?:safety|content)\s+(?:filters?|policies?)", re.I),
]


@dataclass
class GuardrailReport:
    blocked: bool = False
    block_reason: str = ""
    pii_detections: list[str] = field(default_factory=list)
    injection_detections: list[str] = field(default_factory=list)
    original_char_count: int = 0
    redacted_char_count: int = 0

    @property
    def had_pii(self) -> bool:
        return len(self.pii_detections) > 0

    @property
    def had_injection(self) -> bool:
        return len(self.injection_detections) > 0

    def to_dict(self) -> dict:
        return {
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "pii_types_detected": list(set(self.pii_detections)),
            "injection_signals": self.injection_detections,
            "original_char_count": self.original_char_count,
            "redacted_char_count": self.redacted_char_count,
        }


class Guardrails:
    def __init__(self, settings: Settings) -> None:
        self._enabled: bool = settings.guardrails_enabled
        self._max_length: int = settings.max_prompt_length
        self._pii_redaction: bool = settings.pii_redaction_enabled

    async def process(self, messages: List[dict]) -> Tuple[List[dict], GuardrailReport]:
        report = GuardrailReport()

        if not self._enabled:
            return messages, report

        full_text = " ".join(
            m.get("content", "") for m in messages if isinstance(m.get("content"), str)
        )
        report.original_char_count = len(full_text)

        if report.original_char_count > self._max_length:
            report.blocked = True
            report.block_reason = (
                f"Prompt length ({report.original_char_count:,} chars) exceeds "
                f"maximum ({self._max_length:,} chars)"
            )
            log.warning("Guardrail blocked: prompt too long", extra=report.to_dict())
            return messages, report

        for pattern in _INJECTION_PATTERNS:
            match = pattern.search(full_text)
            if match:
                report.injection_detections.append(match.group(0)[:80])

        if report.injection_detections:
            report.blocked = True
            report.block_reason = "Potential prompt-injection or jailbreak attempt detected"
            log.warning("Guardrail blocked: injection detected", extra=report.to_dict())
            return messages, report

        cleaned_messages: list[dict] = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, str):
                cleaned_messages.append(msg)
                continue

            cleaned_content = content
            for label, pattern, placeholder in _PII_PATTERNS:
                if pattern.search(cleaned_content):
                    report.pii_detections.append(label)
                    if self._pii_redaction:
                        cleaned_content = pattern.sub(placeholder, cleaned_content)

            cleaned_messages.append({**msg, "content": cleaned_content})

        report.redacted_char_count = sum(
            len(m.get("content", ""))
            for m in cleaned_messages
            if isinstance(m.get("content"), str)
        )

        if report.had_pii:
            log.info(
                "Guardrail: PII detected and redacted",
                extra={
                    "pii_types": list(set(report.pii_detections)),
                    "redaction_enabled": self._pii_redaction,
                    "chars_before": report.original_char_count,
                    "chars_after": report.redacted_char_count,
                },
            )

        return cleaned_messages, report

    @staticmethod
    def scan_output(text: str) -> GuardrailReport:
        report = GuardrailReport(original_char_count=len(text))
        for label, pattern, _ in _PII_PATTERNS:
            if pattern.search(text):
                report.pii_detections.append(label)
        return report
