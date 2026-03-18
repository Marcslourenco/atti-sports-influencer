"""
Agent Config Schema — Pydantic models for agent configuration validation.
Ensures all agent YAML/JSON configs conform to the standard format.
Compatible with ATTI PersonaManager and Knowledge Package formats.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from enum import Enum


class AgentStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    TESTING = "testing"


class AgentDomain(str, Enum):
    FOOTBALL = "football"
    BASKETBALL = "basketball"
    TENNIS = "tennis"
    ESPORTS = "esports"
    GENERIC_SPORTS = "generic_sports"


class EmotionalMode(BaseModel):
    """Emotional mode configuration for persona"""
    tone: str = "neutral"
    speed: str = "normal"
    vocabulary: List[str] = Field(default_factory=list)
    emoji_level: str = "moderate"  # none, low, moderate, high


class PersonaProfile(BaseModel):
    """Persona configuration — compatible with ATTI PersonaManager"""
    name: str
    role: str = "Influenciador Esportivo"
    expertise: List[str] = Field(default_factory=lambda: ["futebol"])
    tone: str = "apaixonado"
    team_affinity: Optional[str] = None
    language: str = "pt-BR"
    response_template: str = "{commentary}"
    emotional_modes: Dict[str, EmotionalMode] = Field(default_factory=dict)
    rivalry_targets: List[str] = Field(default_factory=list)


class ContentStyle(BaseModel):
    """Content generation style preferences"""
    max_length: int = 280  # Twitter-compatible default
    hashtag_strategy: str = "moderate"  # none, minimal, moderate, aggressive
    mention_strategy: str = "contextual"
    media_preference: str = "text_first"  # text_only, text_first, media_first, video_first
    posting_frequency: str = "event_driven"  # scheduled, event_driven, hybrid


class TTSConfig(BaseModel):
    """Text-to-Speech configuration"""
    engine: str = "xtts_v2"
    voice_clone_sample: Optional[str] = None
    language: str = "pt-BR"
    speed: float = 1.0


class AvatarConfig(BaseModel):
    """Avatar video generation configuration"""
    engine: str = "liveportrait"
    base_image: Optional[str] = None
    style: str = "realistic"


class PlatformConfig(BaseModel):
    """Social media platform configuration"""
    enabled: bool = False
    account_id: Optional[str] = None
    content_types: List[str] = Field(default_factory=lambda: ["text"])
    webhook_url: Optional[str] = None
    api_key_env: Optional[str] = None  # Environment variable name for API key


class ContentPipeline(BaseModel):
    """Content pipeline configuration"""
    commentary: bool = True
    tts: bool = False
    avatar_video: bool = False
    image_gen: bool = False
    tts_config: Optional[TTSConfig] = None
    avatar_config: Optional[AvatarConfig] = None


class KnowledgeConfig(BaseModel):
    """Knowledge base configuration — compatible with ATTI Knowledge Packages"""
    package: Optional[str] = None  # Reference to knowledge package file
    vector_index: Optional[str] = None  # FAISS index path
    embedding_dim: int = 384  # ATTI standard
    categories: List[str] = Field(default_factory=list)


class SchedulingConfig(BaseModel):
    """Agent scheduling configuration"""
    match_days: bool = True
    pre_match_hours: int = 2
    post_match_hours: int = 4
    daily_content: bool = False
    daily_content_times: List[str] = Field(default_factory=list)  # ["09:00", "18:00"]


class QuotaConfig(BaseModel):
    """Agent quota configuration — compatible with ATTI QuotaManager"""
    max_posts_per_day: int = 20
    max_video_per_day: int = 5
    max_api_calls_per_hour: int = 100
    max_storage_mb: int = 500


class AgentConfig(BaseModel):
    """
    Complete agent configuration.
    This is the schema for agent YAML/JSON files in agents/ directory.
    """
    agent_id: str
    name: str
    version: str = "1.0.0"
    status: AgentStatus = AgentStatus.ACTIVE
    domain: AgentDomain = AgentDomain.FOOTBALL
    description: Optional[str] = None

    persona: PersonaProfile
    content_style: ContentStyle = Field(default_factory=ContentStyle)
    content_pipeline: ContentPipeline = Field(default_factory=ContentPipeline)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    scheduling: SchedulingConfig = Field(default_factory=SchedulingConfig)
    quotas: QuotaConfig = Field(default_factory=QuotaConfig)

    platforms: Dict[str, PlatformConfig] = Field(default_factory=dict)

    # Matching rules: which events this agent responds to
    match_rules: Dict[str, Any] = Field(default_factory=dict)
    # e.g., {"competitions": ["BSA"], "teams": ["São Paulo FC"]}

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, v: str) -> str:
        if not v or len(v) < 3:
            raise ValueError("agent_id must be at least 3 characters")
        return v

    def to_tenant_id(self) -> str:
        """Map agent to ATTI multi-tenant ID"""
        return f"sports_agent_{self.agent_id}"

    def to_atti_persona(self) -> Dict[str, Any]:
        """Convert to ATTI PersonaManager format"""
        return {
            "name": self.persona.name,
            "role": self.persona.role,
            "expertise": self.persona.expertise,
            "tone": self.persona.tone,
            "response_template": self.persona.response_template,
        }
