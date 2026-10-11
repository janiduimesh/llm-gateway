"""
Redis-Backed Atomic Rate Limiter
----------------------------------
Implements a **sliding window counter** using a Lua script that is evaluated
atomically on Redis.  This guarantees correctness even when multiple gateway
replicas share the same Redis instance.

Algorithm
─────────
  1. Each request increments a per-team-per-window counter stored in Redis.
  2. The counter key auto-expires after the window elapses (no background jobs).
  3. If the counter exceeds the team's RPM limit the request is rejected with
     a structured :class:`RateLimitResult` that includes retry-after metadata.

The Lua script is EVALSHA-cached after first load to avoid repeated network
round-trips.

Usage::

    limiter = RateLimiter(settings)
    result = await limiter.check("team-api-key-abc", limit_rpm=120)
    if not result.allowed:
        raise HTTPException(429, detail=result.to_dict())
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Optional

import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool

from app.core.config import Settings
from app.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lua Sliding-Window Script
# ---------------------------------------------------------------------------
# Keys[1] = rate limit counter key
# Keys[2] = TTL key (shadows the counter for expiry tracking)
# Argv[1] = window size in seconds
# Argv[2] = requests-per-window limit
# Argv[3] = current Unix timestamp (integer seconds)
#
# Returns: {current_count, limit, remaining, reset_at}

_LUA_SCRIPT = """
local key         = KEYS[1]
local window      = tonumber(ARGV[1])
local limit       = tonumber(ARGV[2])
local now         = tonumber(ARGV[3])

-- Increment and set expiry atomically
local count = redis.call('INCR', key)
if count == 1 then
    redis.call('EXPIRE', key, window)
end

-- Compute TTL for reset header
local ttl = redis.call('TTL', key)
local reset_at = now + ttl

return {count, limit, math.max(0, limit - count), reset_at}
"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RateLimitResult:
    """Outcome of a single rate-limit check."""

    allowed: bool
    current_count: int
    limit: int
    remaining: int
    reset_at: int        # Unix epoch seconds when the window resets
    team_id: str

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "current_count": self.current_count,
            "limit": self.limit,
            "remaining": self.remaining,
            "reset_at": self.reset_at,
            "team_id": self.team_id,
        }

    def retry_after_seconds(self) -> int:
        """Seconds until the window resets (used in Retry-After header)."""
        return max(0, self.reset_at - int(time.time()))


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Async Redis-backed sliding-window rate limiter.

    A single instance is intended to be shared across the entire FastAPI
    application (stored on ``app.state``).

    Args:
        settings: Application settings with Redis URL and default limits.
    """

    def __init__(self, settings: Settings) -> None:
        self._window: int = settings.rate_limit_window_seconds
        self._default_limit: int = settings.default_rate_limit_rpm
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[aioredis.Redis] = None
        self._script_sha: Optional[str] = None
        self._settings = settings

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Create the Redis connection pool and pre-load the Lua script."""
        self._pool = ConnectionPool.from_url(
            self._settings.redis_url,
            max_connections=self._settings.redis_max_connections,
            socket_timeout=self._settings.redis_socket_timeout,
            socket_connect_timeout=self._settings.redis_connect_timeout,
            decode_responses=False,
        )
        self._client = aioredis.Redis(connection_pool=self._pool)

        # Verify connectivity
        await self._client.ping()

        # Pre-register the Lua script
        self._script_sha = await self._client.script_load(_LUA_SCRIPT)

        log.info(
            "RateLimiter connected to Redis",
            extra={
                "redis_url": self._settings.redis_url,
                "window_s": self._window,
                "default_limit_rpm": self._default_limit,
                "lua_sha": self._script_sha,
            },
        )

    async def shutdown(self) -> None:
        """Gracefully close the Redis connection pool."""
        if self._client:
            await self._client.aclose()
        if self._pool:
            await self._pool.aclose()
        log.info("RateLimiter disconnected from Redis")

    # ------------------------------------------------------------------
    # Core check
    # ------------------------------------------------------------------

    async def check(
        self,
        team_id: str,
        limit_rpm: Optional[int] = None,
    ) -> RateLimitResult:
        """
        Atomically increment the request counter for *team_id* and check
        whether the request is within the allowed rate.

        Args:
            team_id:   The team's API key (used as part of the Redis key).
            limit_rpm: RPM limit to enforce; falls back to the default if None.

        Returns:
            A :class:`RateLimitResult` with ``allowed=True`` when the request
            may proceed, or ``allowed=False`` when it should be rejected.
        """
        effective_limit = limit_rpm if limit_rpm is not None else self._default_limit
        redis_key = f"rl:{team_id}:{self._window}"
        now = int(time.time())

        if self._client is None:
            # Fallback: allow if Redis is unavailable (fail-open for availability)
            log.warning(
                "RateLimiter: Redis client not initialised – failing open",
                extra={"team_id": team_id},
            )
            return RateLimitResult(
                allowed=True,
                current_count=0,
                limit=effective_limit,
                remaining=effective_limit,
                reset_at=now + self._window,
                team_id=team_id,
            )

        try:
            result = await self._client.evalsha(
                self._script_sha,
                1,                       # number of KEYS
                redis_key,               # KEYS[1]
                str(self._window),       # ARGV[1]
                str(effective_limit),    # ARGV[2]
                str(now),                # ARGV[3]
            )

            count, limit, remaining, reset_at = (int(x) for x in result)
            allowed = count <= effective_limit

            if not allowed:
                log.warning(
                    "Rate limit exceeded",
                    extra={
                        "team_id": team_id,
                        "count": count,
                        "limit": limit,
                        "reset_at": reset_at,
                    },
                )

            return RateLimitResult(
                allowed=allowed,
                current_count=count,
                limit=limit,
                remaining=remaining,
                reset_at=reset_at,
                team_id=team_id,
            )

        except aioredis.NoScriptError:
            # Script was flushed (e.g. Redis restart); reload and retry once
            log.warning("Lua script evicted from Redis; reloading", extra={"team_id": team_id})
            self._script_sha = await self._client.script_load(_LUA_SCRIPT)
            return await self.check(team_id, limit_rpm)

        except aioredis.RedisError as exc:
            # Fail open to maintain availability
            log.error(
                "RateLimiter Redis error – failing open",
                extra={"team_id": team_id, "error": str(exc)},
            )
            return RateLimitResult(
                allowed=True,
                current_count=0,
                limit=effective_limit,
                remaining=effective_limit,
                reset_at=now + self._window,
                team_id=team_id,
            )

    # ------------------------------------------------------------------
    # Admin helpers
    # ------------------------------------------------------------------

    async def reset(self, team_id: str) -> bool:
        """
        Delete the rate-limit counter for *team_id* (admin/testing use).

        Returns:
            True if the key existed and was deleted, False otherwise.
        """
        if self._client is None:
            return False
        redis_key = f"rl:{team_id}:{self._window}"
        deleted = await self._client.delete(redis_key)
        log.info("Rate limit counter reset", extra={"team_id": team_id, "deleted": bool(deleted)})
        return bool(deleted)

    async def get_status(self, team_id: str) -> dict:
        """Return the current counter value for *team_id* without incrementing."""
        if self._client is None:
            return {"team_id": team_id, "count": None, "error": "Redis unavailable"}
        redis_key = f"rl:{team_id}:{self._window}"
        count = await self._client.get(redis_key)
        ttl = await self._client.ttl(redis_key)
        return {
            "team_id": team_id,
            "current_count": int(count) if count else 0,
            "ttl_seconds": ttl,
            "window_seconds": self._window,
        }
