"""
Event Engine — Adaptive polling for sports events with variable intervals.
Normal: 60s | During match: 5s | Critical moments: 2s
Publishes events to Redis Streams for downstream consumption.
"""
import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

from .sports_schema import Match, MatchEvent, MatchStatus, EventType
from .base_adapter import BaseSportsAdapter

logger = logging.getLogger(__name__)


class PollingMode(str, Enum):
    IDLE = "idle"           # No matches: 60s interval
    PRE_MATCH = "pre_match" # Match starting soon: 30s
    LIVE = "live"           # Match in progress: 5s
    CRITICAL = "critical"   # Goal/penalty/red card: 2s
    POST_MATCH = "post_match"  # Match just ended: 15s


POLLING_INTERVALS = {
    PollingMode.IDLE: 60,
    PollingMode.PRE_MATCH: 30,
    PollingMode.LIVE: 5,
    PollingMode.CRITICAL: 2,
    PollingMode.POST_MATCH: 15,
}

# Events that trigger CRITICAL mode
CRITICAL_EVENTS = {
    EventType.GOAL, EventType.PENALTY, EventType.RED_CARD,
    EventType.VAR_DECISION,
}


class EventEngine:
    """
    Adaptive polling engine for sports events.
    Adjusts polling frequency based on match state.
    Publishes events to a callback or Redis stream.
    """

    def __init__(
        self,
        adapters: List[BaseSportsAdapter],
        redis_client: Optional[Any] = None,
        stream_name: str = "sports:events",
    ):
        self.adapters = adapters
        self.redis = redis_client
        self.stream_name = stream_name
        self._mode = PollingMode.IDLE
        self._running = False
        self._tracked_matches: Dict[str, Match] = {}
        self._event_callbacks: List[Any] = []
        self._last_events: Dict[str, set] = {}  # match_id -> set of event_ids
        self._critical_cooldown = 0

    @property
    def polling_interval(self) -> int:
        return POLLING_INTERVALS[self._mode]

    @property
    def mode(self) -> PollingMode:
        return self._mode

    def on_event(self, callback):
        """Register event callback"""
        self._event_callbacks.append(callback)

    def _determine_mode(self) -> PollingMode:
        """Determine polling mode based on tracked matches"""
        if self._critical_cooldown > 0:
            self._critical_cooldown -= 1
            return PollingMode.CRITICAL

        has_live = False
        has_pre = False
        has_post = False

        for match in self._tracked_matches.values():
            if match.status == MatchStatus.LIVE:
                has_live = True
            elif match.status == MatchStatus.HALFTIME:
                has_live = True
            elif match.status == MatchStatus.SCHEDULED:
                has_pre = True
            elif match.status == MatchStatus.FINISHED:
                has_post = True

        if has_live:
            return PollingMode.LIVE
        if has_post:
            return PollingMode.POST_MATCH
        if has_pre:
            return PollingMode.PRE_MATCH
        return PollingMode.IDLE

    async def _publish_event(self, event: Dict[str, Any]):
        """Publish event to Redis stream and callbacks"""
        # Publish to Redis if available
        if self.redis:
            try:
                await self.redis.xadd(
                    self.stream_name,
                    {"data": json.dumps(event, default=str)},
                    maxlen=10000,
                )
            except Exception as e:
                logger.error(f"Redis publish failed: {e}")

        # Call registered callbacks
        for cb in self._event_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    async def _detect_new_events(self, old_match: Optional[Match], new_match: Match) -> List[Dict]:
        """Compare match states to detect new events"""
        events = []

        if old_match is None:
            # New match being tracked
            events.append({
                "type": "match_tracked",
                "match_id": new_match.match_id,
                "home": new_match.home_team.name,
                "away": new_match.away_team.name,
                "competition": new_match.competition,
                "status": new_match.status.value,
                "timestamp": datetime.utcnow().isoformat(),
            })
            return events

        # Status change
        if old_match.status != new_match.status:
            events.append({
                "type": "status_change",
                "match_id": new_match.match_id,
                "from_status": old_match.status.value,
                "to_status": new_match.status.value,
                "home": new_match.home_team.name,
                "away": new_match.away_team.name,
                "score": f"{new_match.score.home}-{new_match.score.away}",
                "timestamp": datetime.utcnow().isoformat(),
            })

        # Score change (GOAL!)
        if (old_match.score.home != new_match.score.home or
                old_match.score.away != new_match.score.away):
            scoring_team = (
                new_match.home_team.name
                if new_match.score.home > old_match.score.home
                else new_match.away_team.name
            )
            events.append({
                "type": "goal",
                "match_id": new_match.match_id,
                "scoring_team": scoring_team,
                "home": new_match.home_team.name,
                "away": new_match.away_team.name,
                "score": f"{new_match.score.home}-{new_match.score.away}",
                "timestamp": datetime.utcnow().isoformat(),
            })
            self._critical_cooldown = 5  # Stay in critical mode for 5 cycles

        return events

    async def poll_once(self) -> List[Dict]:
        """Execute one polling cycle across all adapters"""
        all_events = []

        for adapter in self.adapters:
            try:
                matches = await adapter.get_live_matches()
                if not matches:
                    matches = await adapter.get_today_matches()

                for match in matches:
                    old = self._tracked_matches.get(match.match_id)
                    events = await self._detect_new_events(old, match)
                    self._tracked_matches[match.match_id] = match

                    for event in events:
                        await self._publish_event(event)
                        all_events.append(event)

            except Exception as e:
                logger.error(f"Polling error for {adapter.source_name}: {e}")

        # Update mode
        self._mode = self._determine_mode()
        return all_events

    async def run(self):
        """Main polling loop with adaptive intervals"""
        self._running = True
        logger.info("Event Engine started")

        while self._running:
            try:
                events = await self.poll_once()
                if events:
                    logger.info(
                        f"Detected {len(events)} events | Mode: {self._mode.value} | "
                        f"Interval: {self.polling_interval}s"
                    )
            except Exception as e:
                logger.error(f"Polling cycle error: {e}")

            await asyncio.sleep(self.polling_interval)

    def stop(self):
        """Stop the polling loop"""
        self._running = False
        logger.info("Event Engine stopped")

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "mode": self._mode.value,
            "interval": self.polling_interval,
            "tracked_matches": len(self._tracked_matches),
            "adapters": [a.get_adapter_status() for a in self.adapters],
        }
