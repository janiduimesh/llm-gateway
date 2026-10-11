from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.core.config import get_settings
from app.services.circuit_breaker import CircuitBreaker
from app.services.guardrails import Guardrails
from app.services.llm_router import LLMRouter
from app.services.rate_limiter import RateLimiter
from app.services.semantic_cache import SemanticCache
from app.utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "Starting LLM Gateway",
        extra={"version": settings.app_version, "environment": settings.environment, "debug": settings.debug},
    )

    circuit_breaker = CircuitBreaker(settings)
    guardrails = Guardrails(settings)
    rate_limiter = RateLimiter(settings)
    semantic_cache = SemanticCache(settings)
    llm_router = LLMRouter(settings, circuit_breaker)

    await rate_limiter.startup()
    await semantic_cache.startup()
    await llm_router.startup()

    app.state.settings = settings
    app.state.circuit_breaker = circuit_breaker
    app.state.guardrails = guardrails
    app.state.rate_limiter = rate_limiter
    app.state.semantic_cache = semantic_cache
    app.state.llm_router = llm_router
    app.state.start_time = time.time()

    log.info("LLM Gateway startup complete – ready to serve requests")

    yield

    log.info("Shutting down LLM Gateway…")
    await llm_router.shutdown()
    await semantic_cache.shutdown()
    await rate_limiter.shutdown()
    log.info("LLM Gateway shut down cleanly")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "**Enterprise LLM Gateway** – A production-grade API gateway "
            "that sits between internal teams and external AI providers.\n\n"
            "### Features\n"
            "- 🔒 **Authentication** – per-team gateway API keys\n"
            "- ⚡ **Rate Limiting** – atomic sliding-window via Redis Lua\n"
            "- 🛡️ **Guardrails** – PII redaction & jailbreak detection\n"
            "- 🧠 **Semantic Cache** – vector-similarity prompt caching\n"
            "- 🔄 **Failover** – Groq primary → OpenAI fallback\n"
            "- 🚦 **Circuit Breaker** – three-state fault isolation\n"
            "- 📊 **Observability** – structured JSON logging throughout\n"
        ),
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    )

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        t0 = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        log.error(
            "Unhandled exception",
            extra={
                "path": request.url.path,
                "method": request.method,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Please try again later.",
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    app.include_router(chat_router)

    @app.get("/health", tags=["Ops"], summary="Liveness probe")
    async def health() -> dict:
        return {"status": "ok", "service": settings.app_name}

    @app.get("/status", tags=["Ops"], summary="Detailed readiness status")
    async def status(request: Request) -> dict:
        uptime = time.time() - getattr(request.app.state, "start_time", time.time())
        cb: CircuitBreaker = request.app.state.circuit_breaker
        cache: SemanticCache = request.app.state.semantic_cache
        llm_router: LLMRouter = request.app.state.llm_router

        cache_stats = await cache.stats()
        return {
            "status": "ready",
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "uptime_seconds": round(uptime, 2),
            "providers": llm_router.get_provider_status(),
            "circuit_breaker": cb.get_status(),
            "cache": cache_stats,
        }

    @app.get("/admin/cache/flush", tags=["Admin"], summary="Flush semantic cache")
    async def flush_cache(request: Request) -> dict:
        cache: SemanticCache = request.app.state.semantic_cache
        removed = await cache.flush()
        return {"flushed": removed}

    @app.get("/admin/circuit/{provider}/reset", tags=["Admin"], summary="Reset circuit breaker")
    async def reset_circuit(provider: str, request: Request) -> dict:
        cb: CircuitBreaker = request.app.state.circuit_breaker
        cb.reset(provider)
        return {"provider": provider, "state": "CLOSED"}

    @app.get("/admin/rate-limit/{team_id}/reset", tags=["Admin"], summary="Reset rate limit")
    async def reset_rate_limit(team_id: str, request: Request) -> dict:
        rl: RateLimiter = request.app.state.rate_limiter
        deleted = await rl.reset(team_id)
        return {"team_id": team_id, "deleted": deleted}

    return app


app = create_app()
