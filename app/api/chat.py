"""
/v1/chat/completions  –  Main Gateway Endpoint
------------------------------------------------
This module wires together every infrastructure service:

  Auth          → validate the gateway API key and resolve team config
  Rate Limiting → enforce per-team RPM caps atomically via Redis Lua
  Guardrails    → scan and sanitize input; block injections / redact PII
  Semantic Cache→ return a cached response when a similar prompt exists
  LLM Router    → route to Groq (primary) or OpenAI (fallback) with
                  circuit-breaker protection
  Logging       → structured observability on every request / response

Streaming support is included via FastAPI's ``StreamingResponse``.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.core.config import Settings, TeamConfig, get_settings
from app.services.circuit_breaker import CircuitBreaker
from app.services.guardrails import Guardrails
from app.services.rate_limiter import RateLimiter
from app.services.semantic_cache import SemanticCache
from app.services.llm_router import LLMRouter
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message author role: system | user | assistant")
    content: str = Field(..., description="Message text content")

    @field_validator("role")
    @classmethod
    def _validate_role(cls, v: str) -> str:
        allowed = {"system", "user", "assistant", "function", "tool"}
        if v not in allowed:
            raise ValueError(f"role must be one of {allowed}, got {v!r}")
        return v


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(
        default=None,
        description="Model identifier; uses provider default when omitted",
    )
    messages: List[ChatMessage] = Field(
        ..., min_length=1, description="Conversation history"
    )
    stream: bool = Field(default=False, description="Enable SSE streaming")
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None, ge=1, description="Maximum tokens in the completion"
    )
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    user: Optional[str] = Field(default=None, description="End-user identifier")

    def extra_params(self) -> Dict[str, Any]:
        """Return non-None optional fields for forwarding to the provider."""
        params: Dict[str, Any] = {}
        for field_name in ("temperature", "max_tokens", "top_p",
                           "frequency_penalty", "presence_penalty", "stop", "user"):
            value = getattr(self, field_name)
            if value is not None:
                params[field_name] = value
        return params


# ---------------------------------------------------------------------------
# Dependency: resolve & validate the gateway API key
# ---------------------------------------------------------------------------

async def get_current_team(
    request: Request,
    authorization: str = Header(..., description="Bearer <gateway-api-key>"),
    settings: Settings = Depends(get_settings),
) -> tuple[str, Optional[TeamConfig]]:
    """
    Extract the Bearer token from the Authorization header.

    Returns:
        A ``(api_key, TeamConfig | None)`` tuple.  When the key is not in the
        team registry the request is still allowed (using default limits) but
        ``TeamConfig`` is None.

    Raises:
        HTTPException 401: When the header is missing or malformed.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authorization header must use the 'Bearer <token>' scheme",
        )

    api_key = authorization.removeprefix("Bearer ").strip()
    if not api_key:
        raise HTTPException(status_code=401, detail="API key must not be empty")

    registry = settings.get_team_registry()
    team = registry.get(api_key)

    if not team and registry:
        # Registry is populated but key not found → reject
        log.warning(
            "Unauthorized API key attempt",
            extra={"key_prefix": api_key[:8] + "…"},
        )
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )

    return api_key, team


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/v1/chat/completions",
    summary="Chat Completions",
    description=(
        "OpenAI-compatible chat completions endpoint with rate limiting, "
        "PII guardrails, semantic caching, and multi-provider failover."
    ),
    response_description="OpenAI-format chat completion response",
    tags=["Chat"],
)
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    team_info: tuple = Depends(get_current_team),
) -> Any:
    """
    Main gateway endpoint for LLM chat completions.

    Pipeline:
      1. Authenticate & resolve team configuration
      2. Enforce rate limits
      3. Validate & sanitize messages through guardrails
      4. Check semantic cache for an existing similar response
      5. Route request through the provider failover chain
      6. Cache the response for future similar prompts
      7. Return the OpenAI-format response (or SSE stream)
    """
    request_id = str(uuid.uuid4())
    t_start = time.perf_counter()

    api_key, team = team_info
    team_id = team.name if team else api_key[:12]

    log.info(
        "Chat completion request received",
        extra={
            "request_id": request_id,
            "team_id": team_id,
            "model": body.model,
            "stream": body.stream,
            "message_count": len(body.messages),
        },
    )

    # ── 1. Rate limiting ───────────────────────────────────────────────
    rate_limiter: RateLimiter = request.app.state.rate_limiter
    limit_rpm = team.rate_limit_rpm if team else None

    rl_result = await rate_limiter.check(api_key, limit_rpm=limit_rpm)
    if not rl_result.allowed:
        log.warning(
            "Rate limit exceeded",
            extra={
                "request_id": request_id,
                "team_id": team_id,
                **rl_result.to_dict(),
            },
        )
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": "Too many requests. Please slow down.",
                "rate_limit": rl_result.to_dict(),
            },
            headers={
                "X-RateLimit-Limit": str(rl_result.limit),
                "X-RateLimit-Remaining": str(rl_result.remaining),
                "X-RateLimit-Reset": str(rl_result.reset_at),
                "Retry-After": str(rl_result.retry_after_seconds()),
            },
        )

    # ── 2. max_tokens enforcement ──────────────────────────────────────
    if team and body.max_tokens:
        if body.max_tokens > team.max_tokens_per_request:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "token_limit_exceeded",
                    "message": (
                        f"max_tokens ({body.max_tokens}) exceeds team limit "
                        f"({team.max_tokens_per_request})"
                    ),
                },
            )

    # ── 3. Guardrails ──────────────────────────────────────────────────
    guardrails: Guardrails = request.app.state.guardrails
    raw_messages = [m.model_dump() for m in body.messages]

    clean_messages, guard_report = await guardrails.process(raw_messages)

    if guard_report.blocked:
        log.warning(
            "Request blocked by guardrails",
            extra={
                "request_id": request_id,
                "team_id": team_id,
                "reason": guard_report.block_reason,
            },
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "guardrail_violation",
                "message": guard_report.block_reason,
                "details": guard_report.to_dict(),
            },
        )

    # ── 4. Semantic cache lookup ───────────────────────────────────────
    sem_cache: SemanticCache = request.app.state.semantic_cache
    cache_model = body.model or "default"

    if not body.stream:
        cached = await sem_cache.get(clean_messages, model=cache_model)
        if cached:
            log.info(
                "Serving response from semantic cache",
                extra={
                    "request_id": request_id,
                    "team_id": team_id,
                    "model": cache_model,
                    "latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
                },
            )
            cached["_gateway"] = {
                "request_id": request_id,
                "cache_hit": True,
                "guardrail_report": guard_report.to_dict(),
            }
            return cached

    # ── 5. LLM routing ────────────────────────────────────────────────
    llm_router: LLMRouter = request.app.state.llm_router
    extra_params = body.extra_params()

    if body.stream:
        return StreamingResponse(
            _stream_response(
                llm_router=llm_router,
                messages=clean_messages,
                model=body.model,
                extra_params=extra_params,
                request_id=request_id,
                team_id=team_id,
            ),
            media_type="text/event-stream",
            headers={
                "X-Request-ID": request_id,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    try:
        llm_response = await llm_router.complete(
            clean_messages,
            model=body.model,
            **extra_params,
        )
    except RuntimeError as exc:
        log.error(
            "All providers failed",
            extra={
                "request_id": request_id,
                "team_id": team_id,
                "error": str(exc),
            },
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "service_unavailable",
                "message": "All upstream LLM providers are currently unavailable",
            },
        )

    # ── 6. Cache the fresh response ────────────────────────────────────
    await sem_cache.set(clean_messages, model=cache_model, response=llm_response)

    # ── 7. Attach gateway metadata & return ───────────────────────────
    total_latency = round((time.perf_counter() - t_start) * 1000, 2)
    llm_response["_gateway"] = {
        "request_id": request_id,
        "cache_hit": False,
        "latency_ms": total_latency,
        "team_id": team_id,
        "guardrail_report": guard_report.to_dict(),
    }

    log.info(
        "Chat completion served",
        extra={
            "request_id": request_id,
            "team_id": team_id,
            "provider": llm_response.get("_gateway_meta", {}).get("provider"),
            "model": llm_response.get("model"),
            "total_latency_ms": total_latency,
            "usage": llm_response.get("usage"),
        },
    )

    return llm_response


# ---------------------------------------------------------------------------
# Streaming generator
# ---------------------------------------------------------------------------

async def _stream_response(
    llm_router: LLMRouter,
    messages: List[dict],
    model: Optional[str],
    extra_params: Dict[str, Any],
    request_id: str,
    team_id: str,
) -> AsyncIterator[str]:
    """Yield SSE chunks from the LLM provider."""
    try:
        async for chunk in llm_router.stream(messages, model=model, **extra_params):
            yield chunk
    except RuntimeError as exc:
        log.error(
            "Streaming failed – all providers exhausted",
            extra={
                "request_id": request_id,
                "team_id": team_id,
                "error": str(exc),
            },
        )
        error_event = (
            'data: {"error": {"type": "service_unavailable", '
            '"message": "All upstream providers are currently unavailable"}}\n\n'
        )
        yield error_event
    finally:
        yield "data: [DONE]\n\n"
