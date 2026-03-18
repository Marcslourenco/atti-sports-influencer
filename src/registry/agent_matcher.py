"""
Agent Matcher — Selects the correct agents based on event context.
Enables automatic routing for large agent ecosystems.
Examples:
  sports topic → sports influencer
  football match → football commentator agent
  NBA news → basketball analyst
"""
import logging
from typing import Dict, Any, List, Optional
from .agent_registry import AgentRegistryLoader
from .agent_schema import AgentConfig, AgentStatus, AgentDomain

logger = logging.getLogger(__name__)


class MatchResult:
    """Result of agent matching with confidence score"""

    def __init__(self, agent: AgentConfig, score: float, reasons: List[str]):
        self.agent = agent
        self.score = score  # 0.0 to 1.0
        self.reasons = reasons

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent.agent_id,
            "name": self.agent.name,
            "score": round(self.score, 3),
            "reasons": self.reasons,
        }


class AgentMatcher:
    """
    Matches events to the correct agents based on:
    - Competition rules
    - Team affinity
    - Domain match
    - Event type compatibility
    - Agent status
    """

    def __init__(self, registry: AgentRegistryLoader):
        self.registry = registry

    def match(
        self,
        event: Dict[str, Any],
        min_score: float = 0.1,
        max_agents: Optional[int] = None,
    ) -> List[MatchResult]:
        """
        Find all agents that should respond to an event.

        Args:
            event: Sports event data
            min_score: Minimum match score (0.0 to 1.0)
            max_agents: Maximum number of agents to return

        Returns:
            List of MatchResult sorted by score (highest first)
        """
        active_agents = self.registry.list_active_agents()
        results = []

        for agent in active_agents:
            score, reasons = self._score_agent(agent, event)
            if score >= min_score:
                results.append(MatchResult(agent, score, reasons))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        if max_agents:
            results = results[:max_agents]

        logger.info(
            f"Matched {len(results)} agents for event "
            f"'{event.get('type', 'unknown')}' "
            f"(min_score={min_score})"
        )
        return results

    def match_ids(
        self,
        event: Dict[str, Any],
        min_score: float = 0.1,
    ) -> List[str]:
        """Return just the agent IDs that match an event"""
        results = self.match(event, min_score)
        return [r.agent.agent_id for r in results]

    def _score_agent(
        self, agent: AgentConfig, event: Dict[str, Any]
    ) -> tuple:
        """
        Score how well an agent matches an event.
        Returns (score, reasons).
        """
        score = 0.0
        reasons = []
        rules = agent.match_rules

        # 1. Domain match (0.2 points)
        event_domain = self._infer_domain(event)
        if event_domain and agent.domain.value == event_domain:
            score += 0.2
            reasons.append(f"domain_match:{event_domain}")
        elif agent.domain.value == "generic_sports":
            score += 0.1
            reasons.append("generic_sports_domain")

        # 2. Competition match (0.3 points)
        event_competition = event.get("competition", "")
        allowed_competitions = rules.get("competitions", [])
        if allowed_competitions:
            if any(c in event_competition for c in allowed_competitions):
                score += 0.3
                reasons.append(f"competition_match:{event_competition}")
        else:
            # No competition filter = matches all
            score += 0.15
            reasons.append("no_competition_filter")

        # 3. Team affinity match (0.3 points)
        event_teams = self._extract_teams(event)
        team_affinity = agent.persona.team_affinity
        allowed_teams = rules.get("teams", [])

        if team_affinity and event_teams:
            if any(team_affinity.lower() in t.lower() for t in event_teams):
                score += 0.3
                reasons.append(f"team_affinity:{team_affinity}")
        elif allowed_teams and event_teams:
            for allowed in allowed_teams:
                if any(allowed.lower() in t.lower() for t in event_teams):
                    score += 0.25
                    reasons.append(f"team_rule_match:{allowed}")
                    break
        elif not team_affinity and not allowed_teams:
            # Neutral agent (e.g., news anchor) — matches all teams
            score += 0.1
            reasons.append("neutral_agent")

        # 4. Event type match (0.2 points)
        event_type = event.get("type", "")
        allowed_events = rules.get("events", [])
        if allowed_events:
            if event_type in allowed_events:
                score += 0.2
                reasons.append(f"event_type_match:{event_type}")
        else:
            score += 0.1
            reasons.append("no_event_filter")

        return (min(score, 1.0), reasons)

    def _infer_domain(self, event: Dict[str, Any]) -> Optional[str]:
        """Infer the sports domain from event data"""
        competition = event.get("competition", "").lower()
        event_type = event.get("type", "").lower()
        text = f"{competition} {event_type} {event.get('description', '')}".lower()

        if any(kw in text for kw in ["futebol", "football", "soccer", "brasileir", "premier", "liga", "champions"]):
            return "football"
        if any(kw in text for kw in ["nba", "basketball", "basquete"]):
            return "basketball"
        if any(kw in text for kw in ["tennis", "tênis", "wimbledon", "roland"]):
            return "tennis"
        return None

    def _extract_teams(self, event: Dict[str, Any]) -> List[str]:
        """Extract team names from event data"""
        teams = []
        if event.get("home"):
            teams.append(event["home"])
        if event.get("away"):
            teams.append(event["away"])
        if event.get("team"):
            teams.append(event["team"])
        if event.get("scoring_team"):
            teams.append(event["scoring_team"])
        return teams

    def explain_match(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Explain why each agent matched or didn't match"""
        all_agents = self.registry.list_agents()
        explanations = []

        for agent in all_agents:
            score, reasons = self._score_agent(agent, event)
            explanations.append({
                "agent_id": agent.agent_id,
                "name": agent.name,
                "status": agent.status.value,
                "score": round(score, 3),
                "matched": score >= 0.1,
                "reasons": reasons,
            })

        return explanations
