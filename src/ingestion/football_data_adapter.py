"""
Football-Data.org Adapter — Free tier sports data ingestion.
API: https://www.football-data.org/documentation/api
Free tier: 10 requests/minute, major European leagues + Brasileirão.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import httpx

from .base_adapter import BaseSportsAdapter
from .sports_schema import (
    Match, MatchEvent, MatchScore, MatchStatus, EventType,
    Team, Player, SportType
)

logger = logging.getLogger(__name__)

# Competition IDs for football-data.org free tier
COMPETITIONS = {
    "brasileirao_a": "BSA",
    "premier_league": "PL",
    "la_liga": "PD",
    "bundesliga": "BL1",
    "serie_a": "SA",
    "ligue_1": "FL1",
    "champions_league": "CL",
    "copa_libertadores": "CLI",
}

STATUS_MAP = {
    "SCHEDULED": MatchStatus.SCHEDULED,
    "TIMED": MatchStatus.SCHEDULED,
    "IN_PLAY": MatchStatus.LIVE,
    "PAUSED": MatchStatus.HALFTIME,
    "FINISHED": MatchStatus.FINISHED,
    "POSTPONED": MatchStatus.POSTPONED,
    "CANCELLED": MatchStatus.CANCELLED,
    "SUSPENDED": MatchStatus.POSTPONED,
}


class FootballDataAdapter(BaseSportsAdapter):
    """
    Adapter for football-data.org API (free tier).
    Rate limit: 10 requests/minute.
    """

    BASE_URL = "https://api.football-data.org/v4"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            source_name="football-data.org",
            sport=SportType.FOOTBALL,
            cache_ttl=300,
            rate_limit=10,
            rate_window=60,
        )
        self.api_key = api_key or ""
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> bool:
        """Initialize HTTP client and verify API access"""
        try:
            headers = {"X-Auth-Token": self.api_key} if self.api_key else {}
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers=headers,
                timeout=15.0,
            )
            # Test connectivity
            resp = await self._client.get("/competitions")
            self._initialized = resp.status_code == 200
            if self._initialized:
                logger.info("football-data.org adapter initialized successfully")
            else:
                logger.warning(f"football-data.org returned status {resp.status_code}")
            return self._initialized
        except Exception as e:
            logger.error(f"Failed to initialize football-data.org adapter: {e}")
            self._initialized = False
            return False

    def _parse_team(self, data: Dict) -> Team:
        return Team(
            id=str(data.get("id", "")),
            name=data.get("name", data.get("shortName", "Unknown")),
            short_name=data.get("tla", data.get("shortName", "")),
            logo_url=data.get("crest", ""),
        )

    def _parse_match(self, data: Dict) -> Match:
        home = data.get("homeTeam", {})
        away = data.get("awayTeam", {})
        score_data = data.get("score", {})
        ft = score_data.get("fullTime", {})
        ht = score_data.get("halfTime", {})

        status_str = data.get("status", "SCHEDULED")
        status = STATUS_MAP.get(status_str, MatchStatus.SCHEDULED)

        return Match(
            match_id=str(data.get("id", "")),
            sport=SportType.FOOTBALL,
            competition=data.get("competition", {}).get("name", ""),
            season=str(data.get("season", {}).get("id", "")),
            matchday=data.get("matchday"),
            home_team=self._parse_team(home),
            away_team=self._parse_team(away),
            status=status,
            score=MatchScore(
                home=ft.get("home") or 0,
                away=ft.get("away") or 0,
                half_time_home=ht.get("home"),
                half_time_away=ht.get("away"),
            ),
            start_time=datetime.fromisoformat(data["utcDate"].replace("Z", "+00:00"))
            if data.get("utcDate") else None,
            venue=data.get("venue", ""),
            referee=data.get("referees", [{}])[0].get("name", "") if data.get("referees") else "",
            source=self.source_name,
        )

    async def _get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        if not self._client:
            await self.initialize()
        try:
            await self._check_rate_limit()
            resp = await self._client.get(endpoint, params=params)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                logger.warning("Rate limited by football-data.org")
                return None
            else:
                logger.error(f"API error {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    async def fetch_live_matches(self) -> List[Match]:
        """Fetch currently live matches across all competitions"""
        data = await self._get("/matches", params={"status": "LIVE"})
        if not data or "matches" not in data:
            return []
        return [self._parse_match(m) for m in data["matches"]]

    async def fetch_today_matches(self) -> List[Match]:
        """Fetch all matches scheduled for today"""
        today = date.today().isoformat()
        data = await self._get("/matches", params={"dateFrom": today, "dateTo": today})
        if not data or "matches" not in data:
            return []
        return [self._parse_match(m) for m in data["matches"]]

    async def fetch_match_events(self, match_id: str) -> List[Dict[str, Any]]:
        """Fetch events for a specific match"""
        data = await self._get(f"/matches/{match_id}")
        if not data:
            return []
        # football-data.org doesn't provide granular events on free tier
        # Return goals from score data
        events = []
        goals = data.get("goals", [])
        for g in goals:
            events.append({
                "type": "goal",
                "minute": g.get("minute"),
                "scorer": g.get("scorer", {}).get("name", ""),
                "assist": g.get("assist", {}).get("name", ""),
                "team": g.get("team", {}).get("name", ""),
            })
        return events

    async def fetch_standings(self, competition_id: str = "BSA") -> List[Dict[str, Any]]:
        """Fetch standings for a competition"""
        data = await self._get(f"/competitions/{competition_id}/standings")
        if not data or "standings" not in data:
            return []
        standings = []
        for table in data["standings"]:
            if table.get("type") == "TOTAL":
                for entry in table.get("table", []):
                    standings.append({
                        "position": entry.get("position"),
                        "team": entry.get("team", {}).get("name", ""),
                        "played": entry.get("playedGames", 0),
                        "won": entry.get("won", 0),
                        "drawn": entry.get("draw", 0),
                        "lost": entry.get("lost", 0),
                        "goals_for": entry.get("goalsFor", 0),
                        "goals_against": entry.get("goalsAgainst", 0),
                        "goal_difference": entry.get("goalDifference", 0),
                        "points": entry.get("points", 0),
                    })
        return standings

    async def fetch_competition_matches(
        self, competition_id: str = "BSA", matchday: Optional[int] = None
    ) -> List[Match]:
        """Fetch matches for a specific competition"""
        params = {}
        if matchday:
            params["matchday"] = matchday
        data = await self._get(f"/competitions/{competition_id}/matches", params=params)
        if not data or "matches" not in data:
            return []
        return [self._parse_match(m) for m in data["matches"]]

    async def close(self):
        if self._client:
            await self._client.aclose()
