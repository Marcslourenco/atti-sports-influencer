"""
ATTI Persona Adapter — Real integration with PersonaEngine from atti_avatar_engine.
Connects to: modules/atti_avatar_engine/persona_layer/persona_engine.py

Features:
- Load ATTI persona configs (SPFC, Corinthians, custom)
- Extend PersonaEngine for sports influencer domain
- Emotional mode detection and switching
- Response pipeline: 20% gancho + 55% factual + 15% cultural + 10% motivador
- Hot reload of persona configs from YAML/JSON
- Compatible with Agent Registry agent configs
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml

logger = logging.getLogger(__name__)


class SportsPersona:
    """Sports persona with emotional modes and response pipeline"""

    def __init__(self, config: Dict[str, Any]):
        self.persona_id = config.get("persona_id", config.get("id", config.get("agent_id", "")))
        self.name = config.get("name", "Comentarista")
        self.role = config.get("role", "Influenciador Esportivo")
        self.expertise = config.get("expertise", ["futebol"])
        self.tone = config.get("tone", "apaixonado")
        self.team_affinity = config.get("team_affinity", "")
        self.language = config.get("language", "pt-BR")
        self.rivalry_targets = config.get("rivalry_targets", [])
        self.emotional_modes = config.get("emotional_modes", {})
        self.response_template = config.get("response_template", "{commentary}")
        self.content_style = config.get("content_style", {})
        self._raw = config

        # ATTI Response Pipeline weights
        self.pipeline_weights = config.get("pipeline_weights", {
            "gancho_emocional": 0.20,
            "factual": 0.55,
            "cultural": 0.15,
            "motivacional": 0.10,
        })

        # ATTI Avatar Engine compatibility
        self.avatar_type = config.get("avatar_type", "CUSTOM")
        self.voice_config = config.get("voice_config", {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "name": self.name,
            "role": self.role,
            "expertise": self.expertise,
            "tone": self.tone,
            "team_affinity": self.team_affinity,
            "language": self.language,
            "rivalry_targets": self.rivalry_targets,
            "emotional_modes": self.emotional_modes,
            "response_template": self.response_template,
            "pipeline_weights": self.pipeline_weights,
            "avatar_type": self.avatar_type,
        }

    def to_atti_persona(self) -> Dict[str, Any]:
        """Convert to ATTI PersonaEngine format (persona_layer compatible)"""
        return {
            "id": self.persona_id,
            "nome": self.name,
            "papel": self.role,
            "tom": self.tone,
            "especialidades": self.expertise,
            "time_afiliado": self.team_affinity,
            "idioma": self.language,
            "rivais": self.rivalry_targets,
            "modos_emocionais": self.emotional_modes,
            "pipeline_resposta": self.pipeline_weights,
            "tipo_avatar": self.avatar_type,
            # Backward compat with old format
            "name": self.name,
            "role": self.role,
            "expertise": self.expertise,
            "tone": self.tone,
            "response_template": self.response_template,
            "team_affinity": self.team_affinity,
            "emotional_modes": self.emotional_modes,
            "rivalry_targets": self.rivalry_targets,
            "content_style": self.content_style,
        }

    def get_emotional_mode(self, context: str = "neutral") -> Dict[str, Any]:
        """Get tone/speed adjustments based on emotional context"""
        return self.emotional_modes.get(context, {
            "tone": self.tone,
            "speed": "normal",
        })

    def detect_emotional_mode(self, event: Dict[str, Any]) -> str:
        """Detect emotional mode based on event context"""
        event_type = event.get("type", "")
        scoring_team = event.get("scoring_team", "")
        home = event.get("home", "")
        away = event.get("away", "")

        my_team = self.team_affinity.lower() if self.team_affinity else ""

        if event_type == "goal":
            if my_team and my_team in scoring_team.lower():
                return "victory"
            elif my_team and (my_team in home.lower() or my_team in away.lower()):
                return "defeat"

        if event_type == "match_end":
            home_score = event.get("home_score", 0)
            away_score = event.get("away_score", 0)
            if my_team in home.lower():
                return "victory" if home_score > away_score else "defeat" if home_score < away_score else "neutral"
            elif my_team in away.lower():
                return "victory" if away_score > home_score else "defeat" if away_score < home_score else "neutral"

        # Check rivalry
        rivals_lower = [r.lower() for r in self.rivalry_targets]
        if any(r in home.lower() or r in away.lower() for r in rivals_lower):
            return "rivalry"

        return "neutral"


class ATTIPersonaAdapter:
    """
    Real adapter for ATTI's PersonaEngine.

    Integration points:
    1. Loads ATTI persona configs from persona_layer/ (SPFC, Corinthians)
    2. Loads sports agent configs from Agent Registry YAML
    3. Extends PersonaEngine with sports-specific emotional modes
    4. Provides response pipeline compatible with ATTI format
    """

    def __init__(
        self,
        atti_persona_dir: Optional[str] = None,
        sports_persona_dir: Optional[str] = None,
    ):
        self.atti_persona_dir = Path(
            atti_persona_dir
            or os.getenv(
                "ATTI_PERSONA_DIR",
                "/home/ubuntu/atti-agent-template/modules/atti_avatar_engine/persona_layer"
            )
        )
        self.sports_persona_dir = Path(
            sports_persona_dir
            or os.getenv(
                "SPORTS_PERSONA_DIR",
                "/home/ubuntu/atti-sports-influencer/agents"
            )
        )
        self._personas: Dict[str, SportsPersona] = {}
        self._atti_personas: Dict[str, Dict] = {}
        self._initialized = False

    def initialize(self) -> bool:
        """Load all personas from ATTI and sports directories"""
        try:
            self._load_atti_personas()
            self._load_sports_personas()
            self._initialized = True
            logger.info(
                f"Persona adapter initialized: "
                f"{len(self._atti_personas)} ATTI personas, "
                f"{len(self._personas)} total personas"
            )
            return True
        except Exception as e:
            logger.error(f"Persona initialization failed: {e}")
            return False

    def _load_atti_personas(self):
        """Load existing ATTI persona configs (SPFC, Corinthians, etc.)"""
        if not self.atti_persona_dir.exists():
            logger.warning(f"ATTI persona dir not found: {self.atti_persona_dir}")
            return

        for json_file in self.atti_persona_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                persona_id = data.get("id", data.get("persona_id", json_file.stem.lower()))
                self._atti_personas[persona_id] = data

                # Create SportsPersona wrapper for ATTI personas
                sports_config = self._convert_atti_to_sports(data, persona_id)
                self._personas[persona_id] = SportsPersona(sports_config)

                logger.info(f"Loaded ATTI persona: {persona_id} from {json_file.name}")
            except Exception as e:
                logger.error(f"Error loading ATTI persona {json_file}: {e}")

    def _load_sports_personas(self):
        """Load sports agent personas from YAML configs"""
        if not self.sports_persona_dir.exists():
            logger.warning(f"Sports persona dir not found: {self.sports_persona_dir}")
            return

        for yaml_file in self.sports_persona_dir.glob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                agent_id = data.get("agent_id", yaml_file.stem)
                persona_data = data.get("persona", {})

                config = {
                    "persona_id": agent_id,
                    "name": persona_data.get("name", agent_id),
                    "role": persona_data.get("role", "Influenciador Esportivo"),
                    "expertise": persona_data.get("expertise", ["futebol"]),
                    "tone": persona_data.get("tone", "apaixonado"),
                    "team_affinity": persona_data.get("team_affinity", ""),
                    "language": persona_data.get("language", "pt-BR"),
                    "rivalry_targets": persona_data.get("rivalry_targets", []),
                    "emotional_modes": persona_data.get("emotional_modes", {}),
                    "response_template": persona_data.get("response_template", "{commentary}"),
                }

                self._personas[agent_id] = SportsPersona(config)
                logger.info(f"Loaded sports persona: {agent_id} from {yaml_file.name}")
            except Exception as e:
                logger.error(f"Error loading sports persona {yaml_file}: {e}")

        for json_file in self.sports_persona_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                agent_id = data.get("agent_id", json_file.stem)
                persona_data = data.get("persona", data)
                config = {"persona_id": agent_id, **persona_data}
                self._personas[agent_id] = SportsPersona(config)
                logger.info(f"Loaded sports persona: {agent_id} from {json_file.name}")
            except Exception as e:
                logger.error(f"Error loading sports persona {json_file}: {e}")

    def _convert_atti_to_sports(self, atti_data: Dict, persona_id: str) -> Dict[str, Any]:
        """Convert ATTI persona format to sports persona format"""
        return {
            "persona_id": persona_id,
            "name": atti_data.get("nome", atti_data.get("name", persona_id)),
            "role": atti_data.get("papel", atti_data.get("role", "Avatar ATTI")),
            "expertise": atti_data.get("especialidades", atti_data.get("expertise", [])),
            "tone": atti_data.get("tom", atti_data.get("tone", "profissional")),
            "team_affinity": atti_data.get("time_afiliado", atti_data.get("team", "")),
            "language": atti_data.get("idioma", "pt-BR"),
            "rivalry_targets": atti_data.get("rivais", []),
            "emotional_modes": atti_data.get("modos_emocionais", {}),
            "avatar_type": atti_data.get("tipo_avatar", "ATTI"),
        }

    def get_persona(self, persona_id: str) -> Optional[SportsPersona]:
        """Get persona by ID"""
        return self._personas.get(persona_id)

    def register_persona(self, persona_id: str, config: Dict[str, Any]) -> bool:
        """Register a new persona at runtime"""
        try:
            persona = SportsPersona({"persona_id": persona_id, **config})
            self._personas[persona_id] = persona
            logger.info(f"Registered persona: {persona_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register persona: {e}")
            return False

    def detect_emotional_mode(self, persona_id: str, event: Dict[str, Any]) -> str:
        """Detect emotional mode for a persona given an event"""
        persona = self._personas.get(persona_id)
        if persona:
            return persona.detect_emotional_mode(event)
        return "neutral"

    def get_personas_for_team(self, team_name: str) -> List[SportsPersona]:
        """Find all personas affiliated with a team"""
        return [
            p for p in self._personas.values()
            if p.team_affinity and team_name.lower() in p.team_affinity.lower()
        ]

    def list_personas(self) -> List[Dict[str, Any]]:
        """List all available personas"""
        return [
            {
                "persona_id": p.persona_id,
                "name": p.name,
                "team_affinity": p.team_affinity,
                "tone": p.tone,
                "source": "atti" if p.persona_id in self._atti_personas else "sports",
            }
            for p in self._personas.values()
        ]

    def get_all_atti_personas(self) -> Dict[str, Dict[str, Any]]:
        """Get all personas in ATTI PersonaManager format"""
        return {pid: p.to_atti_persona() for pid, p in self._personas.items()}

    def reload(self):
        """Hot reload all persona configs"""
        self._personas.clear()
        self._atti_personas.clear()
        self.initialize()
        logger.info("Personas reloaded")

    def get_status(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "total_personas": len(self._personas),
            "atti_personas": len(self._atti_personas),
            "sports_personas": len(self._personas) - len(self._atti_personas),
            "atti_dir": str(self.atti_persona_dir),
            "sports_dir": str(self.sports_persona_dir),
            "persona_ids": list(self._personas.keys()),
        }
