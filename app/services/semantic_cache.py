"""
Semantic Prompt Cache
----------------------
Avoids redundant (and expensive) LLM calls by caching responses and
retrieving them when a semantically similar prompt is detected.

Architecture
────────────
  • Prompt embeddings are computed locally using ``sentence-transformers``.
  • Each embedding + response is serialised to JSON and stored in Redis
    with a configurable TTL.
  • Cache lookup computes cosine-similarity against all stored embeddings
    using a linear scan.  For production deployments with large caches
    this should be replaced with a vector DB (e.g. Redis Stack's
    ``HNSW`` index via ``redis-py[hnsw]``), but the interface remains
    identical.
  • The cache is keyed on:  ``sc:<model>:<sha256(canonical_prompt)>``
    for exact hits, and a bloom-style embedding scan for fuzzy hits.

Usage::

    cache = SemanticCache(settings)
    await cache.startup()

    hit = await cache.get(messages, model="llama3-8b-8192")
    if hit:
        return hit           # dict matching OpenAI response shape

    response = await call_llm(messages)
    await cache.set(messages, model="llama3-8b-8192", response=response)
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool

from app.core.config import Settings
from app.utils.logger import get_logger

log = get_logger(__name__)

# Lazy import so the service starts even without the ML dependency installed
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    log.warning(
        "sentence-transformers not installed – semantic cache will operate in "
        "exact-match-only mode",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity (avoids numpy import at module level)."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _canonical_prompt(messages: List[dict]) -> str:
    """
    Produce a deterministic string representation of a message list for
    hashing and embedding (strips whitespace, lower-cases role names).
    """
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        parts.append(f"{role}:{content}")
    return " | ".join(parts)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Semantic Cache Service
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Redis-backed semantic cache with optional embedding-based fuzzy matching.

    Args:
        settings: Application settings.
    """

    # Redis key prefixes
    _ENTRY_PREFIX = "sc:entry:"   # sc:entry:<sha256>   → serialised CacheEntry
    _INDEX_KEY = "sc:index"       # Redis Set of all entry SHA keys (for scanning)

    def __init__(self, settings: Settings) -> None:
        self._enabled: bool = settings.cache_enabled
        self._threshold: float = settings.cache_similarity_threshold
        self._ttl: int = settings.cache_ttl_seconds
        self._max_entries: int = settings.cache_max_entries
        self._embedding_model_name: str = settings.embedding_model
        self._settings = settings

        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[aioredis.Redis] = None
        self._encoder: Optional[Any] = None   # SentenceTransformer instance

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Initialise Redis connection and load the embedding model."""
        self._pool = ConnectionPool.from_url(
            self._settings.redis_url,
            max_connections=self._settings.redis_max_connections,
            socket_timeout=self._settings.redis_socket_timeout,
            socket_connect_timeout=self._settings.redis_connect_timeout,
            decode_responses=True,
        )
        self._client = aioredis.Redis(connection_pool=self._pool)
        await self._client.ping()

        if _ST_AVAILABLE and self._enabled:
            try:
                # Model is downloaded on first run and then cached to disk
                self._encoder = SentenceTransformer(self._embedding_model_name)
                log.info(
                    "SemanticCache: embedding model loaded",
                    extra={"model": self._embedding_model_name},
                )
            except Exception as exc:
                log.warning(
                    "SemanticCache: failed to load embedding model – "
                    "falling back to exact-match only",
                    extra={"error": str(exc)},
                )
                self._encoder = None
        else:
            self._encoder = None

        log.info(
            "SemanticCache connected to Redis",
            extra={
                "redis_url": self._settings.redis_url,
                "ttl_s": self._ttl,
                "threshold": self._threshold,
                "semantic": self._encoder is not None,
            },
        )

    async def shutdown(self) -> None:
        """Gracefully close the Redis connection pool."""
        if self._client:
            await self._client.aclose()
        if self._pool:
            await self._pool.aclose()
        log.info("SemanticCache disconnected from Redis")

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text*, or an empty list."""
        if self._encoder is None:
            return []
        vector = self._encoder.encode(text, normalize_embeddings=True)
        return vector.tolist()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(
        self,
        messages: List[dict],
        model: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Look up a cached response for the given *messages* + *model* pair.

        Strategy:
          1. Exact SHA-256 lookup (O(1))
          2. Semantic similarity scan (O(n)) over index if exact misses

        Args:
            messages: Chat message list.
            model:    LLM model identifier (partitions the cache namespace).

        Returns:
            The cached OpenAI-format response dict, or *None* on a miss.
        """
        if not self._enabled or self._client is None:
            return None

        canonical = _canonical_prompt(messages)
        sha = _sha256(f"{model}:{canonical}")
        entry_key = f"{self._ENTRY_PREFIX}{sha}"

        # 1. Exact hit
        raw = await self._client.get(entry_key)
        if raw:
            entry = json.loads(raw)
            log.info(
                "SemanticCache: exact hit",
                extra={"sha": sha[:12], "model": model},
            )
            await self._client.expire(entry_key, self._ttl)  # refresh TTL
            return entry["response"]

        # 2. Semantic scan (only when embeddings are available)
        if self._encoder is None:
            return None

        query_embedding = self._embed(canonical)
        if not query_embedding:
            return None

        all_keys: list[str] = await self._client.smembers(self._INDEX_KEY)  # type: ignore
        best_similarity = 0.0
        best_response: Optional[dict] = None

        for key in all_keys:
            raw_entry = await self._client.get(f"{self._ENTRY_PREFIX}{key}")
            if not raw_entry:
                # Entry expired – clean up index
                await self._client.srem(self._INDEX_KEY, key)
                continue

            entry = json.loads(raw_entry)
            if entry.get("model") != model:
                continue

            stored_embedding = entry.get("embedding", [])
            if not stored_embedding:
                continue

            similarity = _cosine_similarity(query_embedding, stored_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_response = entry["response"]

        if best_similarity >= self._threshold and best_response is not None:
            log.info(
                "SemanticCache: semantic hit",
                extra={
                    "similarity": round(best_similarity, 4),
                    "threshold": self._threshold,
                    "model": model,
                },
            )
            return best_response

        log.debug(
            "SemanticCache: miss",
            extra={
                "best_similarity": round(best_similarity, 4),
                "threshold": self._threshold,
                "model": model,
            },
        )
        return None

    async def set(
        self,
        messages: List[dict],
        model: str,
        response: Dict[str, Any],
    ) -> None:
        """
        Store *response* in the cache for the given *messages* + *model* pair.

        Args:
            messages: Chat message list (used to derive the cache key).
            model:    LLM model identifier.
            response: OpenAI-format response dict to cache.
        """
        if not self._enabled or self._client is None:
            return

        canonical = _canonical_prompt(messages)
        sha = _sha256(f"{model}:{canonical}")
        entry_key = f"{self._ENTRY_PREFIX}{sha}"

        embedding = self._embed(canonical) if self._encoder else []

        entry = {
            "model": model,
            "canonical_prompt": canonical,
            "embedding": embedding,
            "response": response,
            "cached_at": time.time(),
        }

        await self._client.set(entry_key, json.dumps(entry), ex=self._ttl)
        await self._client.sadd(self._INDEX_KEY, sha)

        # Evict oldest entries if the index exceeds max_entries
        index_size = await self._client.scard(self._INDEX_KEY)
        if index_size > self._max_entries:
            # Remove a random sample (simple eviction – replace with LRU if needed)
            to_remove = await self._client.srandmember(self._INDEX_KEY, count=index_size - self._max_entries)  # type: ignore
            for old_sha in to_remove:
                await self._client.delete(f"{self._ENTRY_PREFIX}{old_sha}")
                await self._client.srem(self._INDEX_KEY, old_sha)

        log.info(
            "SemanticCache: entry stored",
            extra={
                "sha": sha[:12],
                "model": model,
                "ttl_s": self._ttl,
                "has_embedding": bool(embedding),
            },
        )

    # ------------------------------------------------------------------
    # Admin helpers
    # ------------------------------------------------------------------

    async def flush(self) -> int:
        """Delete all cache entries.  Returns the number of entries removed."""
        if self._client is None:
            return 0
        keys: list[str] = await self._client.smembers(self._INDEX_KEY)  # type: ignore
        pipeline = self._client.pipeline()
        for sha in keys:
            pipeline.delete(f"{self._ENTRY_PREFIX}{sha}")
        pipeline.delete(self._INDEX_KEY)
        await pipeline.execute()
        log.info("SemanticCache flushed", extra={"entries_removed": len(keys)})
        return len(keys)

    async def stats(self) -> dict:
        """Return current cache statistics."""
        if self._client is None:
            return {"error": "Redis unavailable"}
        index_size = await self._client.scard(self._INDEX_KEY)
        return {
            "total_entries": index_size,
            "max_entries": self._max_entries,
            "ttl_seconds": self._ttl,
            "similarity_threshold": self._threshold,
            "semantic_enabled": self._encoder is not None,
            "embedding_model": self._embedding_model_name if self._encoder else None,
        }
