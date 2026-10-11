from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict

from app.core.config import Settings
from app.utils.logger import get_logger

log = get_logger(__name__)


class CBState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


@dataclass
class ProviderState:
    name: str
    state: CBState = CBState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    opened_at: float = 0.0
    last_failure_reason: str = ""
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reset_counts(self) -> None:
        self.failure_count = 0
        self.success_count = 0

    def to_dict(self) -> dict:
        return {
            "provider": self.name,
            "state": self.state.name,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "last_failure_reason": self.last_failure_reason,
        }


class CircuitBreaker:
    def __init__(self, settings: Settings) -> None:
        self._failure_threshold: int = settings.cb_failure_threshold
        self._recovery_timeout: int = settings.cb_recovery_timeout
        self._success_threshold: int = settings.cb_success_threshold
        self._providers: Dict[str, ProviderState] = {}
        self._global_lock = threading.Lock()

    def _get_state(self, provider: str) -> ProviderState:
        with self._global_lock:
            if provider not in self._providers:
                self._providers[provider] = ProviderState(name=provider)
            return self._providers[provider]

    def _transition(self, ps: ProviderState, new_state: CBState) -> None:
        old_state = ps.state
        ps.state = new_state
        log.info(
            "Circuit breaker state transition",
            extra={
                "provider": ps.name,
                "from_state": old_state.name,
                "to_state": new_state.name,
                "failure_count": ps.failure_count,
                "success_count": ps.success_count,
            },
        )

    def can_attempt(self, provider: str) -> bool:
        ps = self._get_state(provider)
        with ps._lock:
            ps.total_requests += 1

            if ps.state == CBState.CLOSED:
                return True

            if ps.state == CBState.OPEN:
                elapsed = time.monotonic() - ps.opened_at
                if elapsed >= self._recovery_timeout:
                    log.info(
                        "Circuit entering HALF-OPEN – recovery probe allowed",
                        extra={"provider": provider, "elapsed_s": round(elapsed, 2)},
                    )
                    self._transition(ps, CBState.HALF_OPEN)
                    ps.reset_counts()
                    return True
                log.debug(
                    "Circuit is OPEN – request rejected",
                    extra={
                        "provider": provider,
                        "seconds_until_probe": round(self._recovery_timeout - elapsed, 2),
                    },
                )
                return False

            return True

    def record_success(self, provider: str) -> None:
        ps = self._get_state(provider)
        with ps._lock:
            ps.total_successes += 1

            if ps.state == CBState.HALF_OPEN:
                ps.success_count += 1
                if ps.success_count >= self._success_threshold:
                    ps.reset_counts()
                    self._transition(ps, CBState.CLOSED)
            elif ps.state == CBState.CLOSED:
                ps.failure_count = 0

    def record_failure(self, provider: str, reason: str = "") -> None:
        ps = self._get_state(provider)
        with ps._lock:
            ps.total_failures += 1
            ps.failure_count += 1
            ps.last_failure_reason = reason

            log.warning(
                "Circuit breaker: failure recorded",
                extra={
                    "provider": provider,
                    "failure_count": ps.failure_count,
                    "threshold": self._failure_threshold,
                    "reason": reason,
                },
            )

            if ps.state == CBState.HALF_OPEN:
                ps.opened_at = time.monotonic()
                ps.reset_counts()
                self._transition(ps, CBState.OPEN)

            elif ps.state == CBState.CLOSED:
                if ps.failure_count >= self._failure_threshold:
                    ps.opened_at = time.monotonic()
                    self._transition(ps, CBState.OPEN)

    def get_status(self) -> dict:
        return {name: ps.to_dict() for name, ps in self._providers.items()}

    def reset(self, provider: str) -> None:
        ps = self._get_state(provider)
        with ps._lock:
            ps.reset_counts()
            self._transition(ps, CBState.CLOSED)
        log.info("Circuit breaker manually reset", extra={"provider": provider})
