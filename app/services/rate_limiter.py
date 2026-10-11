from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool

from app.core.config import Settings
from app.utils.logger import get_logger

log = get_logger(__name__)


_LUA_SCRIPT = """
local key         = KEYS[1]
local window      = tonumber(ARGV[1])
local limit       = tonumber(ARGV[2])
local now         = tonumber(ARGV[3])

local count = redis.call('INCR', key)
if count == 1 then
    redis.call('EXPIRE', key, window)
end

local ttl = redis.call('TTL', key)
local reset_at = now + ttl

return {count, limit, math.max(0, limit - count), reset_at}
"""


@dataclass
class RateLimitResult:
    allowed: bool
    current_count: int
    limit: int
    remaining: int
    reset_at: int
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
        return max(0, self.reset_at - int(time.time()))


class RateLimiter:
    def __init__(self, settings: Settings) -> None:
        self._window: int = settings.rate_limit_window_seconds
        self._default_limit: int = settings.default_rate_limit_rpm
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[aioredis.Redis] = None
        self._script_sha: Optional[str] = None
        self._settings = settings

    async def startup(self) -> None:
        self._pool = ConnectionPool.from_url(
            self._settings.redis_url,
            max_connections=self._settings.redis_max_connections,
            socket_timeout=self._settings.redis_socket_timeout,
            socket_connect_timeout=self._settings.redis_connect_timeout,
            decode_responses=False,
        )
        self._client = aioredis.Redis(connection_pool=self._pool)
        await self._client.ping()
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
        if self._client:
            await self._client.aclose()
        if self._pool:
            await self._pool.aclose()
        log.info("RateLimiter disconnected from Redis")

    async def check(self, team_id: str, limit_rpm: Optional[int] = None) -> RateLimitResult:
        effective_limit = limit_rpm if limit_rpm is not None else self._default_limit
        redis_key = f"rl:{team_id}:{self._window}"
        now = int(time.time())

        if self._client is None:
            log.warning("RateLimiter: Redis client not initialised – failing open", extra={"team_id": team_id})
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
                1,
                redis_key,
                str(self._window),
                str(effective_limit),
                str(now),
            )

            count, limit, remaining, reset_at = (int(x) for x in result)
            allowed = count <= effective_limit

            if not allowed:
                log.warning(
                    "Rate limit exceeded",
                    extra={"team_id": team_id, "count": count, "limit": limit, "reset_at": reset_at},
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
            log.warning("Lua script evicted from Redis; reloading", extra={"team_id": team_id})
            self._script_sha = await self._client.script_load(_LUA_SCRIPT)
            return await self.check(team_id, limit_rpm)

        except aioredis.RedisError as exc:
            log.error("RateLimiter Redis error – failing open", extra={"team_id": team_id, "error": str(exc)})
            return RateLimitResult(
                allowed=True,
                current_count=0,
                limit=effective_limit,
                remaining=effective_limit,
                reset_at=now + self._window,
                team_id=team_id,
            )

    async def reset(self, team_id: str) -> bool:
        if self._client is None:
            return False
        redis_key = f"rl:{team_id}:{self._window}"
        deleted = await self._client.delete(redis_key)
        log.info("Rate limit counter reset", extra={"team_id": team_id, "deleted": bool(deleted)})
        return bool(deleted)

    async def get_status(self, team_id: str) -> dict:
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
