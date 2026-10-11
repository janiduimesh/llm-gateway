from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from app.core.config import Settings
from app.services.circuit_breaker import CircuitBreaker
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    default_model: str
    priority: int = 0


class LLMRouter:
    def __init__(self, settings: Settings, circuit_breaker: CircuitBreaker) -> None:
        self._settings = settings
        self._cb = circuit_breaker
        self._timeout = httpx.Timeout(
            connect=10.0,
            read=settings.llm_request_timeout,
            write=10.0,
            pool=5.0,
        )
        self._client: Optional[httpx.AsyncClient] = None
        self._providers: list[Provider] = sorted(
            self._build_providers(settings),
            key=lambda p: p.priority,
        )

    @staticmethod
    def _build_providers(settings: Settings) -> list[Provider]:
        providers: list[Provider] = []

        if settings.groq_api_key:
            providers.append(
                Provider(
                    name="groq",
                    base_url=settings.groq_base_url,
                    api_key=settings.groq_api_key,
                    default_model=settings.groq_default_model,
                    priority=0,
                )
            )
        if settings.openai_api_key:
            providers.append(
                Provider(
                    name="openai",
                    base_url=settings.openai_base_url,
                    api_key=settings.openai_api_key,
                    default_model=settings.openai_default_model,
                    priority=1,
                )
            )
        return providers

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            http2=True,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30,
            ),
        )
        log.info(
            "LLMRouter started",
            extra={"providers": [p.name for p in self._providers], "timeout_s": self._settings.llm_request_timeout},
        )

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
        log.info("LLMRouter shut down")

    def _resolve_model(self, provider: Provider, requested_model: Optional[str]) -> str:
        if requested_model:
            return requested_model
        return provider.default_model

    def _build_payload(
        self,
        provider: Provider,
        messages: List[dict],
        model: Optional[str],
        stream: bool,
        extra_params: Dict[str, Any],
    ) -> dict:
        return {
            "model": self._resolve_model(provider, model),
            "messages": messages,
            "stream": stream,
            **extra_params,
        }

    def _auth_headers(self, provider: Provider) -> dict:
        return {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }

    async def complete(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        **extra_params: Any,
    ) -> Dict[str, Any]:
        if not self._providers:
            raise RuntimeError("No LLM providers configured")

        last_error: Optional[Exception] = None

        for provider in self._providers:
            if not self._cb.can_attempt(provider.name):
                log.warning("LLMRouter: skipping provider (circuit open)", extra={"provider": provider.name})
                continue

            for attempt in range(1, self._settings.llm_max_retries + 2):
                try:
                    response = await self._call_provider(
                        provider, messages, model, stream=False, extra_params=extra_params
                    )
                    self._cb.record_success(provider.name)
                    log.info(
                        "LLMRouter: request completed",
                        extra={
                            "provider": provider.name,
                            "model": response.get("model"),
                            "attempt": attempt,
                            "usage": response.get("usage"),
                        },
                    )
                    return response

                except (httpx.TimeoutException, httpx.ConnectError) as exc:
                    last_error = exc
                    reason = f"{type(exc).__name__}: {exc}"
                    log.warning(
                        "LLMRouter: transient error",
                        extra={
                            "provider": provider.name,
                            "attempt": attempt,
                            "max_retries": self._settings.llm_max_retries + 1,
                            "error": reason,
                        },
                    )
                    if attempt <= self._settings.llm_max_retries:
                        await asyncio.sleep(0.5 * attempt)
                    else:
                        self._cb.record_failure(provider.name, reason=reason)
                        break

                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    reason = f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
                    log.error(
                        "LLMRouter: upstream HTTP error",
                        extra={
                            "provider": provider.name,
                            "status_code": exc.response.status_code,
                            "body": exc.response.text[:200],
                        },
                    )
                    self._cb.record_failure(provider.name, reason=reason)
                    break

                except Exception as exc:
                    last_error = exc
                    reason = f"{type(exc).__name__}: {exc}"
                    self._cb.record_failure(provider.name, reason=reason)
                    log.error("LLMRouter: unexpected error", extra={"provider": provider.name, "error": reason})
                    break

        raise RuntimeError(f"All LLM providers exhausted. Last error: {last_error}")

    async def stream(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        **extra_params: Any,
    ) -> AsyncIterator[str]:
        if not self._providers:
            raise RuntimeError("No LLM providers configured")

        for provider in self._providers:
            if not self._cb.can_attempt(provider.name):
                continue

            try:
                async for chunk in self._stream_provider(provider, messages, model, extra_params):
                    yield chunk
                self._cb.record_success(provider.name)
                return

            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                self._cb.record_failure(provider.name, reason=reason)
                log.error(
                    "LLMRouter: stream error, trying next provider",
                    extra={"provider": provider.name, "error": reason},
                )
                continue

        raise RuntimeError("All LLM providers exhausted during streaming")

    async def _call_provider(
        self,
        provider: Provider,
        messages: List[dict],
        model: Optional[str],
        stream: bool,
        extra_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        assert self._client is not None, "LLMRouter not started"

        payload = self._build_payload(provider, messages, model, stream=False, extra_params=extra_params)
        url = f"{provider.base_url.rstrip('/')}/chat/completions"

        t0 = time.perf_counter()
        resp = await self._client.post(url, json=payload, headers=self._auth_headers(provider))
        elapsed = time.perf_counter() - t0
        resp.raise_for_status()

        data = resp.json()
        data["_gateway_meta"] = {
            "provider": provider.name,
            "latency_ms": round(elapsed * 1000, 2),
        }
        return data

    async def _stream_provider(
        self,
        provider: Provider,
        messages: List[dict],
        model: Optional[str],
        extra_params: Dict[str, Any],
    ) -> AsyncIterator[str]:
        assert self._client is not None, "LLMRouter not started"

        payload = self._build_payload(provider, messages, model, stream=True, extra_params=extra_params)
        url = f"{provider.base_url.rstrip('/')}/chat/completions"

        async with self._client.stream("POST", url, json=payload, headers=self._auth_headers(provider)) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    yield line + "\n\n"

    def get_provider_status(self) -> list[dict]:
        return [
            {
                "name": p.name,
                "priority": p.priority,
                "default_model": p.default_model,
                "circuit": self._cb.get_status().get(p.name, {"state": "CLOSED"}),
            }
            for p in self._providers
        ]
