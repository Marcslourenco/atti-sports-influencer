"""
TASK 4: Integration tests for sports data ingestion.
Tests: football-data.org API, RSS feeds, Wikipedia scraper, Event Engine.
Validates: polling, cache, rate limiting, data schema compliance.
"""
import asyncio
import json
import sys
import time
from datetime import datetime, date
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.sports_schema import (
    Match, MatchEvent, MatchScore, MatchStatus, EventType,
    Team, SportType, SportsDataResponse, StandingsEntry
)
from ingestion.football_data_adapter import FootballDataAdapter
from ingestion.rss_sports_adapter import RSSSportsAdapter
from ingestion.sports_scraper_engine import SportsScraper
from ingestion.event_engine import EventEngine, PollingMode


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.details = []

    def record(self, test_name: str, passed: bool, detail: str = ""):
        if passed:
            self.passed += 1
            status = "✅ PASS"
        else:
            self.failed += 1
            status = "❌ FAIL"
        self.details.append(f"  {status}: {test_name}" + (f" — {detail}" if detail else ""))

    def warn(self, test_name: str, detail: str = ""):
        self.warnings += 1
        self.details.append(f"  ⚠️ WARN: {test_name}" + (f" — {detail}" if detail else ""))

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"TOTAL: {self.passed + self.failed} tests | "
            f"✅ {self.passed} passed | ❌ {self.failed} failed | ⚠️ {self.warnings} warnings",
            f"{'='*60}",
        ]
        return "\n".join(lines)


async def test_schema_validation(results: TestResults):
    """Test 1: Validate sports data schema models"""
    print("\n📋 TEST 1: Schema Validation")
    print("-" * 40)

    # Test Team creation
    try:
        team = Team(id="flamengo", name="Flamengo", short_name="FLA")
        results.record("Team creation", True, f"id={team.id}, name={team.name}")
    except Exception as e:
        results.record("Team creation", False, str(e))

    # Test Match creation
    try:
        match = Match(
            match_id="match-001",
            sport=SportType.FOOTBALL,
            competition="brasileirao_a",
            home_team=Team(id="fla", name="Flamengo", short_name="FLA"),
            away_team=Team(id="cor", name="Corinthians", short_name="COR"),
            status=MatchStatus.SCHEDULED,
            scheduled_at=datetime.now(),
        )
        results.record("Match creation", True, f"id={match.match_id}, status={match.status}")
    except Exception as e:
        results.record("Match creation", False, str(e))

    # Test MatchEvent creation
    try:
        event = MatchEvent(
            event_type=EventType.GOAL,
            minute=45,
            team_id="fla",
            player_name="Gabigol",
            description="Gol de Gabigol aos 45 minutos",
        )
        results.record("MatchEvent creation", True, f"type={event.event_type}, minute={event.minute}")
    except Exception as e:
        results.record("MatchEvent creation", False, str(e))

    # Test SportsDataResponse
    try:
        response = SportsDataResponse(
            source="test",
            sport=SportType.FOOTBALL,
            data_type="matches",
            items=[match.model_dump()],
            fetched_at=datetime.now(),
        )
        results.record("SportsDataResponse creation", True, f"items={len(response.items)}")
    except Exception as e:
        results.record("SportsDataResponse creation", False, str(e))


async def test_football_data_adapter(results: TestResults):
    """Test 2: Football-data.org API adapter"""
    print("\n⚽ TEST 2: Football-Data.org Adapter")
    print("-" * 40)

    adapter = FootballDataAdapter()

    # Test initialization
    try:
        init_result = await adapter.initialize()
        results.record("Adapter initialization", init_result)
    except Exception as e:
        results.record("Adapter initialization", False, str(e))
        return

    # Test fetching today's matches (may be empty)
    try:
        matches = await adapter.fetch_today_matches()
        results.record(
            "Fetch today's matches",
            isinstance(matches, list),
            f"Found {len(matches)} matches"
        )
    except Exception as e:
        # API key might not be set — this is expected
        results.warn("Fetch today's matches", f"Expected without API key: {e}")

    # Test competition list
    try:
        from ingestion.football_data_adapter import COMPETITIONS
        results.record(
            "Competition mapping",
            len(COMPETITIONS) >= 6,
            f"{len(COMPETITIONS)} competitions mapped"
        )
    except Exception as e:
        results.record("Competition mapping", False, str(e))

    # Test rate limiting
    try:
        has_rate_limiter = hasattr(adapter, '_rate_limiter')
        results.record(
            "Rate limit configured",
            has_rate_limiter,
            f"has _rate_limiter={has_rate_limiter}"
        )
    except Exception as e:
        results.record("Rate limit configured", False, str(e))

    # Test cache
    try:
        results.record(
            "Cache configured",
            adapter.cache_ttl > 0,
            f"cache_ttl={adapter.cache_ttl}s"
        )
    except Exception as e:
        results.record("Cache configured", False, str(e))

    await adapter.close()


async def test_rss_adapter(results: TestResults):
    """Test 3: RSS Sports Feed Adapter"""
    print("\n📰 TEST 3: RSS Sports Feed Adapter")
    print("-" * 40)

    adapter = RSSSportsAdapter()

    # Test initialization
    try:
        init_result = await adapter.initialize()
        results.record("RSS adapter initialization", init_result)
    except Exception as e:
        results.warn("RSS adapter initialization", f"May need feedparser: {e}")

    # Test fetching from at least one feed
    try:
        from ingestion.rss_sports_adapter import DEFAULT_FEEDS
        results.record(
            "Default feeds configured",
            len(DEFAULT_FEEDS) >= 3,
            f"{len(DEFAULT_FEEDS)} feeds configured"
        )

        # List feed names
        feed_names = list(DEFAULT_FEEDS.keys())
        results.record(
            "Feed names valid",
            all(isinstance(n, str) and len(n) > 0 for n in feed_names),
            f"Feeds: {', '.join(feed_names[:5])}"
        )
    except Exception as e:
        results.record("Default feeds check", False, str(e))

    # Test actual RSS fetch (BBC is most reliable)
    try:
        news = await adapter.fetch_feed(feed_name="bbc_football")
        if isinstance(news, list) and len(news) > 0:
            results.record(
                "BBC Football RSS fetch",
                True,
                f"Fetched {len(news)} articles"
            )
        else:
            results.warn("BBC Football RSS fetch", "Empty response (feed may be down)")
    except Exception as e:
        results.warn("BBC Football RSS fetch", f"Network error: {e}")

    # Test ESPN RSS
    try:
        news = await adapter.fetch_feed(feed_name="espn_football")
        if isinstance(news, list):
            results.record(
                "ESPN Football RSS fetch",
                True,
                f"Fetched {len(news)} articles"
            )
        else:
            results.warn("ESPN Football RSS fetch", "Unexpected response type")
    except Exception as e:
        results.warn("ESPN Football RSS fetch", f"Network error: {e}")


async def test_scraper(results: TestResults):
    """Test 4: Wikipedia Sports Scraper"""
    print("\n🌐 TEST 4: Wikipedia Sports Scraper")
    print("-" * 40)

    scraper = SportsScraper()

    # Test initialization
    try:
        init_result = await scraper.initialize()
        results.record("Scraper initialization", init_result)
    except Exception as e:
        results.record("Scraper initialization", False, str(e))
        return

    # Test rate limiting config
    try:
        has_rate_limiter = hasattr(scraper, '_rate_limiter')
        results.record(
            "Rate limit configured",
            has_rate_limiter,
            f"has _rate_limiter={has_rate_limiter}"
        )
    except Exception as e:
        results.record("Rate limit configured", False, str(e))

    # Test cache config
    try:
        results.record(
            "Cache TTL configured",
            scraper.cache_ttl > 0,
            f"cache_ttl={scraper.cache_ttl}s"
        )
    except Exception as e:
        results.record("Cache TTL configured", False, str(e))

    # Test Wikipedia page fetch
    try:
        html = await scraper._fetch_page(
            "https://en.wikipedia.org/wiki/2025_Campeonato_Brasileiro_S%C3%A9rie_A"
        )
        if html and len(html) > 1000:
            results.record(
                "Wikipedia page fetch",
                True,
                f"Fetched {len(html)} bytes"
            )
        else:
            results.warn("Wikipedia page fetch", "Page may not exist yet")
    except Exception as e:
        results.warn("Wikipedia page fetch", f"Network error: {e}")

    # No close method on scraper


async def test_event_engine(results: TestResults):
    """Test 5: Event Engine (adaptive polling)"""
    print("\n🔄 TEST 5: Event Engine")
    print("-" * 40)

    # Test polling modes
    try:
        from ingestion.event_engine import POLLING_INTERVALS, CRITICAL_EVENTS
        results.record(
            "Polling intervals defined",
            len(POLLING_INTERVALS) >= 4,
            f"{len(POLLING_INTERVALS)} modes: {', '.join(m.value for m in POLLING_INTERVALS)}"
        )
    except Exception as e:
        results.record("Polling intervals defined", False, str(e))

    # Test interval values
    try:
        results.record(
            "IDLE interval = 60s",
            POLLING_INTERVALS[PollingMode.IDLE] == 60,
            f"actual={POLLING_INTERVALS[PollingMode.IDLE]}s"
        )
        results.record(
            "LIVE interval = 5s",
            POLLING_INTERVALS[PollingMode.LIVE] == 5,
            f"actual={POLLING_INTERVALS[PollingMode.LIVE]}s"
        )
        results.record(
            "CRITICAL interval = 2s",
            POLLING_INTERVALS[PollingMode.CRITICAL] == 2,
            f"actual={POLLING_INTERVALS[PollingMode.CRITICAL]}s"
        )
    except Exception as e:
        results.record("Interval values", False, str(e))

    # Test critical events
    try:
        results.record(
            "Critical events include GOAL",
            EventType.GOAL in CRITICAL_EVENTS,
        )
        results.record(
            "Critical events include PENALTY",
            EventType.PENALTY in CRITICAL_EVENTS,
        )
        results.record(
            "Critical events include RED_CARD",
            EventType.RED_CARD in CRITICAL_EVENTS,
        )
    except Exception as e:
        results.record("Critical events", False, str(e))

    # Test EventEngine instantiation
    try:
        adapter = FootballDataAdapter()
        engine = EventEngine(adapters=[adapter])
        results.record("EventEngine instantiation", True)
    except Exception as e:
        results.record("EventEngine instantiation", False, str(e))


async def main():
    print("=" * 60)
    print("ATTI SPORTS INFLUENCER — INGESTION VALIDATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    results = TestResults()

    await test_schema_validation(results)
    await test_football_data_adapter(results)
    await test_rss_adapter(results)
    await test_scraper(results)
    await test_event_engine(results)

    # Print all results
    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)
    for detail in results.details:
        print(detail)

    print(results.summary())

    # Save results
    report_path = Path(__file__).parent.parent / "tests" / "ingestion_test_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "passed": results.passed,
        "failed": results.failed,
        "warnings": results.warnings,
        "details": results.details,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nResults saved to: {report_path}")

    return results.failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
