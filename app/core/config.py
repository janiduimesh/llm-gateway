"""
Global Application Configuration
----------------------------------
All settings are loaded from environment variables / .env file via
pydantic-settings.  A single Settings singleton is exported as `settings`.

Teams are modelled as a JSON dict mapping API-key → team metadata, making it
trivial to add/remove teams without code changes.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Team Registry Model
# ---------------------------------------------------------------------------

class TeamConfig:
    """Lightweight container for per-team settings parsed from JSON."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.name: str = raw.get("name", "unknown")
        self.rate_limit_rpm: int = raw.get("rate_limit_rpm", 60)
        self.max_tokens_per_request: int = raw.get("max_tokens_per_request", 4096)
        self.allowed_models: list[str] = raw.get("allowed_models", [])
        self.cost_budget_usd: float = raw.get("cost_budget_usd", 10.0)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TeamConfig(name={self.name!r}, rpm={self.rate_limit_rpm}, "
            f"max_tokens={self.max_tokens_per_request})"
        )


# ---------------------------------------------------------------------------
# Application Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Central configuration loaded from environment variables / .env file.

    All fields have sensible defaults so the application can start in
    development mode without a fully configured environment.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application meta
    # ------------------------------------------------------------------
    app_name: str = Field(default="LLM Gateway", description="Human-readable service name")
    app_version: str = Field(default="1.0.0", description="Semantic version")
    debug: bool = Field(default=False, description="Enable debug mode / verbose logging")
    log_level: str = Field(default="INFO", description="Root log level")
    environment: str = Field(default="production", description="deployment environment")

    # ------------------------------------------------------------------
    # LLM Provider credentials
    # ------------------------------------------------------------------
    groq_api_key: str = Field(default="", description="Groq API key (primary provider)")
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Groq OpenAI-compatible endpoint",
    )
    groq_default_model: str = Field(
        default="llama3-8b-8192",
        description="Default Groq model",
    )

    openai_api_key: str = Field(default="", description="OpenAI API key (fallback provider)")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL",
    )
    openai_default_model: str = Field(
        default="gpt-4o-mini",
        description="Default OpenAI model used on fallback",
    )

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )
    redis_max_connections: int = Field(default=20, description="Redis connection pool size")
    redis_socket_timeout: float = Field(default=2.0, description="Redis socket timeout (s)")
    redis_connect_timeout: float = Field(default=2.0, description="Redis connect timeout (s)")

    # ------------------------------------------------------------------
    # Rate Limiting
    # ------------------------------------------------------------------
    default_rate_limit_rpm: int = Field(
        default=60,
        description="Requests-per-minute cap applied to teams not in the registry",
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        description="Sliding window size for rate-limit counters",
    )

    # ------------------------------------------------------------------
    # Semantic Cache
    # ------------------------------------------------------------------
    cache_enabled: bool = Field(default=True, description="Master switch for semantic cache")
    cache_similarity_threshold: float = Field(
        default=0.92,
        description="Cosine-similarity threshold [0,1] for cache hits",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="TTL for cached responses (seconds)",
    )
    cache_max_entries: int = Field(
        default=10_000,
        description="Maximum number of cache entries stored in Redis",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model for semantic embeddings",
    )

    # ------------------------------------------------------------------
    # Circuit Breaker
    # ------------------------------------------------------------------
    cb_failure_threshold: int = Field(
        default=5,
        description="Consecutive failures before a provider is opened (tripped)",
    )
    cb_recovery_timeout: int = Field(
        default=30,
        description="Seconds a tripped provider stays in OPEN state before HALF-OPEN",
    )
    cb_success_threshold: int = Field(
        default=2,
        description="Consecutive successes in HALF-OPEN required to close the circuit",
    )

    # ------------------------------------------------------------------
    # LLM Router
    # ------------------------------------------------------------------
    llm_request_timeout: float = Field(
        default=30.0,
        description="Per-request HTTP timeout for upstream LLM calls (seconds)",
    )
    llm_max_retries: int = Field(
        default=2,
        description="Retry attempts per provider before failover",
    )

    # ------------------------------------------------------------------
    # Guardrails
    # ------------------------------------------------------------------
    guardrails_enabled: bool = Field(default=True, description="Enable input/output guardrails")
    max_prompt_length: int = Field(
        default=32_768,
        description="Maximum total character length across all messages in a request",
    )
    pii_redaction_enabled: bool = Field(
        default=True,
        description="Replace detected PII tokens with [REDACTED] placeholders",
    )

    # ------------------------------------------------------------------
    # Team API Key Registry
    # Stored as a JSON string:
    #   '{"sk-team-abc": {"name": "eng-team", "rate_limit_rpm": 120, ...}}'
    # ------------------------------------------------------------------
    team_api_keys: str = Field(
        default="{}",
        description=(
            "JSON object mapping gateway API keys → team config dicts. "
            "Set as a single-line JSON string in the environment."
        ),
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return upper

    @field_validator("cache_similarity_threshold")
    @classmethod
    def _validate_similarity(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("cache_similarity_threshold must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def _warn_missing_keys(self) -> "Settings":
        import logging

        _log = logging.getLogger("llm_gateway.config")
        if not self.groq_api_key:
            _log.warning("GROQ_API_KEY is not set – Groq provider will be unavailable")
        if not self.openai_api_key:
            _log.warning("OPENAI_API_KEY is not set – OpenAI fallback will be unavailable")
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_team_registry(self) -> dict[str, TeamConfig]:
        """
        Parse TEAM_API_KEYS JSON string into a mapping of
        ``api_key → TeamConfig``.

        Returns an empty dict when the JSON is absent or malformed.
        """
        try:
            raw: dict[str, Any] = json.loads(self.team_api_keys)
            return {key: TeamConfig(cfg) for key, cfg in raw.items()}
        except (json.JSONDecodeError, TypeError):
            return {}

    def get_team(self, api_key: str) -> TeamConfig | None:
        """Look up a team by its gateway API key."""
        return self.get_team_registry().get(api_key)


# ---------------------------------------------------------------------------
# Singleton accessor (cached for the lifetime of the process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    return Settings()


# Convenience alias used by the rest of the application
settings: Settings = get_settings()
