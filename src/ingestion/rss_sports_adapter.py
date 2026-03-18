"""
RSS Sports Feed Adapter — Zero-cost sports news ingestion from public RSS feeds.
Sources: ESPN, GloboEsporte, UOL Esporte, Lance, etc.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import feedparser
import httpx

from .base_adapter import BaseSportsAdapter
from .sports_schema import Match, SportType, SportsDataResponse

logger = logging.getLogger(__name__)

# Public RSS feeds for sports news (zero cost)
DEFAULT_FEEDS = {
    "globoesporte": {
        "url": "https://ge.globo.com/rss/futebol/",
        "language": "pt-BR",
        "sport": "football",
    },
    "uol_esporte": {
        "url": "https://esporte.uol.com.br/futebol/rss.xml",
        "language": "pt-BR",
        "sport": "football",
    },
    "espn_football": {
        "url": "https://www.espn.com/espn/rss/soccer/news",
        "language": "en",
        "sport": "football",
    },
    "espn_nba": {
        "url": "https://www.espn.com/espn/rss/nba/news",
        "language": "en",
        "sport": "basketball",
    },
    "bbc_football": {
        "url": "https://feeds.bbci.co.uk/sport/football/rss.xml",
        "language": "en",
        "sport": "football",
    },
}


class RSSFeedEntry:
    """Parsed RSS feed entry"""

    def __init__(self, entry: Dict, source: str, language: str):
        self.title = entry.get("title", "")
        self.summary = entry.get("summary", entry.get("description", ""))
        self.link = entry.get("link", "")
        self.published = entry.get("published", "")
        self.source = source
        self.language = language
        self.tags = [t.get("term", "") for t in entry.get("tags", [])]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "link": self.link,
            "published": self.published,
            "source": self.source,
            "language": self.language,
            "tags": self.tags,
        }


class RSSSportsAdapter(BaseSportsAdapter):
    """
    RSS feed adapter for sports news ingestion.
    Zero cost, no API key required.
    """

    def __init__(self, feeds: Optional[Dict] = None):
        super().__init__(
            source_name="rss-feeds",
            sport=SportType.FOOTBALL,
            cache_ttl=600,  # 10 min cache for news
            rate_limit=30,
            rate_window=60,
        )
        self.feeds = feeds or DEFAULT_FEEDS

    async def initialize(self) -> bool:
        """Verify at least one feed is accessible"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                for name, config in list(self.feeds.items())[:2]:
                    try:
                        resp = await client.get(config["url"])
                        if resp.status_code == 200:
                            self._initialized = True
                            logger.info(f"RSS adapter initialized (tested: {name})")
                            return True
                    except Exception:
                        continue
            logger.warning("No RSS feeds accessible")
            self._initialized = False
            return False
        except Exception as e:
            logger.error(f"RSS adapter init failed: {e}")
            return False

    async def fetch_feed(self, feed_name: str) -> List[RSSFeedEntry]:
        """Fetch and parse a single RSS feed"""
        config = self.feeds.get(feed_name)
        if not config:
            logger.warning(f"Unknown feed: {feed_name}")
            return []

        cache_key = self._cache_key("feed", name=feed_name)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            await self._check_rate_limit()
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(config["url"])
                if resp.status_code != 200:
                    return []

            parsed = feedparser.parse(resp.text)
            entries = [
                RSSFeedEntry(e, feed_name, config.get("language", "en"))
                for e in parsed.entries[:20]
            ]
            self._set_cached(cache_key, entries)
            return entries
        except Exception as e:
            logger.error(f"Failed to fetch {feed_name}: {e}")
            return []

    async def fetch_all_feeds(self, sport_filter: Optional[str] = None) -> List[RSSFeedEntry]:
        """Fetch all configured feeds, optionally filtered by sport"""
        all_entries = []
        for name, config in self.feeds.items():
            if sport_filter and config.get("sport") != sport_filter:
                continue
            entries = await self.fetch_feed(name)
            all_entries.extend(entries)
        return all_entries

    async def fetch_live_matches(self) -> List[Match]:
        """RSS feeds don't provide live match data — returns empty"""
        return []

    async def fetch_today_matches(self) -> List[Match]:
        """RSS feeds don't provide match schedules — returns empty"""
        return []

    async def fetch_match_events(self, match_id: str) -> List[Dict[str, Any]]:
        """RSS feeds don't provide match events — returns empty"""
        return []

    async def fetch_standings(self, competition_id: str) -> List[Dict[str, Any]]:
        """RSS feeds don't provide standings — returns empty"""
        return []

    async def search_news(self, keywords: List[str], sport: str = "football") -> List[RSSFeedEntry]:
        """Search across all feeds for entries matching keywords"""
        all_entries = await self.fetch_all_feeds(sport_filter=sport)
        keywords_lower = [k.lower() for k in keywords]

        matched = []
        for entry in all_entries:
            text = f"{entry.title} {entry.summary}".lower()
            if any(kw in text for kw in keywords_lower):
                matched.append(entry)
        return matched

    def get_sports_news_response(self, entries: List[RSSFeedEntry]) -> SportsDataResponse:
        """Convert entries to standardized response"""
        return SportsDataResponse(
            source=self.source_name,
            sport=self.sport,
            data_type="news",
            items=[e.to_dict() for e in entries],
            cache_ttl=self.cache_ttl,
        )
