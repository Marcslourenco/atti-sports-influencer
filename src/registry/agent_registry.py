"""
Agent Registry — Dynamic loader for agent configurations.
Loads agents from YAML/JSON files in agents/ directory.
Supports hot reload, validation, and status management.
"""
import logging
import os
import json
import yaml
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from watchdog.events import FileSystemEventHandler
from .agent_schema import AgentConfig, AgentStatus

logger = logging.getLogger(__name__)


class AgentRegistryLoader:
    """
    Dynamic agent configuration loader.
    Reads YAML/JSON agent configs from a directory and validates them.
    Supports runtime registration and hot reload.
    """

    def __init__(self, agents_dir: Optional[str] = None):
        self.agents_dir = Path(agents_dir or "agents")
        self._agents: Dict[str, AgentConfig] = {}
        self._load_timestamps: Dict[str, float] = {}
        self._errors: Dict[str, str] = {}

    def load_all(self) -> int:
        """Load all agent configurations from directory"""
        if not self.agents_dir.exists():
            logger.warning(f"Agents directory not found: {self.agents_dir}")
            self.agents_dir.mkdir(parents=True, exist_ok=True)
            return 0

        loaded = 0
        for file_path in self.agents_dir.iterdir():
            if file_path.suffix in (".yaml", ".yml", ".json"):
                if self._load_agent_file(file_path):
                    loaded += 1

        logger.info(f"Agent Registry: loaded {loaded} agents from {self.agents_dir}")
        return loaded

    def _load_agent_file(self, file_path: Path) -> bool:
        """Load and validate a single agent config file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.suffix == ".json":
                    raw = json.load(f)
                else:
                    raw = yaml.safe_load(f)

            config = AgentConfig(**raw)
            self._agents[config.agent_id] = config
            self._load_timestamps[config.agent_id] = time.time()

            if config.agent_id in self._errors:
                del self._errors[config.agent_id]

            logger.info(f"Loaded agent: {config.agent_id} ({config.name})")
            return True

        except Exception as e:
            agent_id = file_path.stem
            self._errors[agent_id] = str(e)
            logger.error(f"Failed to load agent {file_path.name}: {e}")
            return False

    def get(self, agent_id: str) -> Optional[AgentConfig]:
        """Get agent configuration by ID"""
        return self._agents.get(agent_id)

    def list_agents(
        self,
        status: Optional[AgentStatus] = None,
        domain: Optional[str] = None,
    ) -> List[AgentConfig]:
        """List agents with optional filtering"""
        agents = list(self._agents.values())

        if status:
            agents = [a for a in agents if a.status == status]
        if domain:
            agents = [a for a in agents if a.domain.value == domain]

        return agents

    def list_active_agents(self) -> List[AgentConfig]:
        """List only active agents"""
        return self.list_agents(status=AgentStatus.ACTIVE)

    def register(self, config: Dict[str, Any]) -> AgentConfig:
        """Register a new agent at runtime"""
        agent = AgentConfig(**config)
        self._agents[agent.agent_id] = agent
        self._load_timestamps[agent.agent_id] = time.time()
        logger.info(f"Registered agent: {agent.agent_id}")
        return agent

    def unregister(self, agent_id: str) -> bool:
        """Remove an agent from the registry"""
        if agent_id in self._agents:
            del self._agents[agent_id]
            if agent_id in self._load_timestamps:
                del self._load_timestamps[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
            return True
        return False

    def update_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Update agent status"""
        agent = self._agents.get(agent_id)
        if agent:
            agent.status = status
            logger.info(f"Agent {agent_id} status → {status.value}")
            return True
        return False

    def reload(self) -> Dict[str, Any]:
        """Reload all agent configs (hot reload)"""
        old_count = len(self._agents)
        self._agents.clear()
        self._errors.clear()
        new_count = self.load_all()

        return {
            "previous_count": old_count,
            "new_count": new_count,
            "errors": dict(self._errors),
        }

    def check_for_changes(self) -> List[str]:
        """Check if any config files have been modified since last load"""
        changed = []
        if not self.agents_dir.exists():
            return changed

        for file_path in self.agents_dir.iterdir():
            if file_path.suffix not in (".yaml", ".yml", ".json"):
                continue
            mtime = file_path.stat().st_mtime
            agent_id = file_path.stem
            last_load = self._load_timestamps.get(agent_id, 0)
            if mtime > last_load:
                changed.append(str(file_path))

        return changed

    def get_status(self) -> Dict[str, Any]:
        """Get registry status"""
        return {
            "total_agents": len(self._agents),
            "active_agents": len(self.list_active_agents()),
            "agents_dir": str(self.agents_dir),
            "errors": dict(self._errors),
            "agents": {
                aid: {
                    "name": a.name,
                    "status": a.status.value,
                    "domain": a.domain.value,
                    "team": a.persona.team_affinity or "none",
                }
                for aid, a in self._agents.items()
            },
        }

    def save_agent(self, agent_id: str, format: str = "yaml") -> bool:
        """Save agent config to file"""
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        self.agents_dir.mkdir(parents=True, exist_ok=True)
        ext = ".yaml" if format == "yaml" else ".json"
        file_path = self.agents_dir / f"{agent_id}{ext}"

        try:
            data = agent.model_dump()
            with open(file_path, "w", encoding="utf-8") as f:
                if format == "yaml":
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved agent config: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save agent {agent_id}: {e}")
            return False
