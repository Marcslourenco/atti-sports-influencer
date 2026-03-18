"""
Base Sports Data Adapter — Abstract interface for all data sources.
All concrete adapters must implement this interface.
Includes built-in cache layer and rate limit protection.
"""
import time
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from .sports_schema import SportsDataResponse, Match, SportType

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter to protect API quotas"""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: List[float] = []

    def can_request(self) -> bool:
        now = time.time()
        self._requests = [t for t in self._requests if now - t < self.window_seconds]
        return len(self._requests) < self.max_requests

    def record_request(self):
        self._requests.append(time.time())

    def wait_time(self) -> float:
        if self.can_request():
            return 0.0
        oldest = min(self._requests)
        return self.window_seconds - (time.time() - oldest)


class CacheEntry:
    """Cache entry with TTL"""

    def __init__(self, data: Any, ttl: int = 300):
        self.data = data
        self.created_at = time.time()
        self.ttl = ttl

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl


class BaseSportsAdapter(ABC):
    """
    Abstract base class for sports data adapters.
    Provides built-in caching and rate limiting.
    """

    def __init__(
        self,
        source_name: str,
        sport: SportType = SportType.FOOTBALL,
        cache_ttl: int = 300,
        rate_limit: int = 60,
        rate_window: int = 60,
    ):
        self.source_name = source_name
        self.sport = sport
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._rate_limiter = RateLimiter(rate_limit, rate_window)
        self._initialized = False

    def _cache_key(self, method: str, **kwargs) -> str:
        raw = f"{self.source_name}:{method}:{json.dumps(kwargs, sort_keys=True, default=str)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry and not entry.is_expired:
            logger.debug(f"Cache HIT: {key}")
            return entry.data
        if entry:
            del self._cache[key]
        return None

    def _set_cached(self, key: str, data: Any, ttl: Optional[int] = None):
        self._cache[key] = CacheEntry(data, ttl or self.cache_ttl)

    async def _check_rate_limit(self):
        if not self._rate_limiter.can_request():
            wait = self._rate_limiter.wait_time()
            logger.warning(f"Rate limit hit for {self.source_name}, waiting {wait:.1f}s")
            import asyncio
            await asyncio.sleep(wait)
        self._rate_limiter.record_request()

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the adapter (check API availability, etc)"""
        ...

    @abstractmethod
    async def fetch_live_matches(self) -> List[Match]:
        """Fetch currently live matches"""
        ...

    @abstractmethod
    async def fetch_today_matches(self) -> List[Match]:
        """Fetch all matches scheduled for today"""
        ...

    @abstractmethod
    async def fetch_match_events(self, match_id: str) -> List[Dict[str, Any]]:
        """Fetch events for a specific match"""
        ...

    @abstractmethod
    async def fetch_standings(self, competition_id: str) -> List[Dict[str, Any]]:
        """Fetch standings for a competition"""
        ...

    async def get_live_matches(self) -> List[Match]:
        """Get live matches with caching and rate limiting"""
        cache_key = self._cache_key("live_matches")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        await self._check_rate_limit()
        result = await self.fetch_live_matches()
        self._set_cached(cache_key, result, ttl=30)  # Short TTL for live data
        return result

    async def get_today_matches(self) -> List[Match]:
        """Get today's matches with caching and rate limiting"""
        cache_key = self._cache_key("today_matches")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        await self._check_rate_limit()
        result = await self.fetch_today_matches()
        self._set_cached(cache_key, result, ttl=300)
        return result

    def get_adapter_status(self) -> Dict[str, Any]:
        """Return adapter health status"""
        return {
            "source": self.source_name,
            "sport": self.sport.value,
            "initialized": self._initialized,
            "cache_entries": len(self._cache),
            "rate_limit_available": self._rate_limiter.can_request(),
        }
