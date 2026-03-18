"""
Sports Data Schema — Standardized data models for all sports data sources.
All adapters must output data conforming to these schemas.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class SportType(str, Enum):
    FOOTBALL = "football"
    BASKETBALL = "basketball"
    TENNIS = "tennis"
    GENERIC = "generic"


class MatchStatus(str, Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    HALFTIME = "halftime"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class EventType(str, Enum):
    GOAL = "goal"
    ASSIST = "assist"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"
    SUBSTITUTION = "substitution"
    PENALTY = "penalty"
    VAR_DECISION = "var_decision"
    HALF_TIME = "half_time"
    FULL_TIME = "full_time"
    KICK_OFF = "kick_off"
    INJURY = "injury"
    CORNER = "corner"
    FREE_KICK = "free_kick"
    OFFSIDE = "offside"
    GENERIC = "generic"


class Team(BaseModel):
    """Standardized team representation"""
    id: str
    name: str
    short_name: Optional[str] = None
    logo_url: Optional[str] = None
    country: Optional[str] = None


class Player(BaseModel):
    """Standardized player representation"""
    id: str
    name: str
    number: Optional[int] = None
    position: Optional[str] = None
    team_id: Optional[str] = None


class MatchEvent(BaseModel):
    """Standardized match event"""
    event_id: str = Field(default_factory=lambda: f"evt_{datetime.utcnow().timestamp()}")
    event_type: EventType
    minute: Optional[int] = None
    extra_time_minute: Optional[int] = None
    player: Optional[Player] = None
    assist_player: Optional[Player] = None
    team: Optional[Team] = None
    description: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MatchScore(BaseModel):
    """Current match score"""
    home: int = 0
    away: int = 0
    half_time_home: Optional[int] = None
    half_time_away: Optional[int] = None


class Match(BaseModel):
    """Standardized match representation"""
    match_id: str
    sport: SportType = SportType.FOOTBALL
    competition: str = ""
    season: Optional[str] = None
    matchday: Optional[int] = None
    home_team: Team
    away_team: Team
    status: MatchStatus = MatchStatus.SCHEDULED
    score: MatchScore = Field(default_factory=MatchScore)
    events: List[MatchEvent] = Field(default_factory=list)
    start_time: Optional[datetime] = None
    venue: Optional[str] = None
    referee: Optional[str] = None
    source: str = ""
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StandingsEntry(BaseModel):
    """League standings entry"""
    position: int
    team: Team
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    goals_for: int = 0
    goals_against: int = 0
    goal_difference: int = 0
    points: int = 0


class SportsDataResponse(BaseModel):
    """Standardized response from any sports data adapter"""
    source: str
    sport: SportType
    data_type: str  # "matches", "standings", "events", "news"
    items: List[Dict[str, Any]]
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    cache_ttl: int = 300  # seconds
    metadata: Dict[str, Any] = Field(default_factory=dict)
