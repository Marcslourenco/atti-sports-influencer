"""
Media Pipeline — Orchestrates the full media generation chain.
Pipeline: Commentary Text -> TTS -> Avatar Animation -> Final Video

Features:
- End-to-end media generation from text
- Parallel processing when possible
- Quality validation at each step
- Media worker integration for stateless processing
- Output ready for social media publishing
"""
import asyncio
import logging
import time
from typing import Dict, Any, Optional

from .tts_engine import TTSEngine
from .avatar_engine import AvatarEngine

logger = logging.getLogger(__name__)


class MediaPipeline:
    """
    Orchestrates: Text -> Voice -> Avatar -> Video

    Steps:
    1. TTS: Convert commentary text to speech audio
    2. Avatar: Animate persona avatar with audio
    3. Compose: Add overlays, branding, subtitles
    4. Export: Final MP4 ready for publishing
    """

    def __init__(self):
        self.tts = TTSEngine()
        self.avatar = AvatarEngine()
        self._metrics = {
            "total_pipelines": 0,
            "successful_pipelines": 0,
            "failed_pipelines": 0,
            "total_latency_ms": 0.0,
        }

    async def initialize(self) -> Dict[str, Any]:
        """Initialize all media engines"""
        await self.tts.initialize()
        await self.avatar.initialize()
        return await self.health_check()

    async def generate_commentary_video(
        self,
        commentary_text: str,
        persona_id: str,
        voice_id: Optional[str] = None,
        language: str = "pt",
        emotional_mode: str = "neutral",
        overlay_text: Optional[str] = None,
        event_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: Text -> Voice -> Avatar -> Video

        Args:
            commentary_text: Generated sports commentary
            persona_id: Agent persona ID
            voice_id: Voice profile (defaults to persona_id)
            language: Language code
            emotional_mode: Emotional tone for voice and expression
            overlay_text: Score/team overlay text
            event_data: Original event data for metadata

        Returns:
            Dict with video_path, audio_path, duration, engine info
        """
        self._metrics["total_pipelines"] += 1
        start = time.time()
        voice = voice_id or persona_id

        # Step 1: TTS
        logger.info(f"[MediaPipeline] Step 1/3: TTS for persona={persona_id}")
        tts_result = await self.tts.generate_speech(
            text=commentary_text,
            voice_id=voice,
            language=language,
            emotional_mode=emotional_mode,
        )

        if not tts_result.get("success"):
            self._metrics["failed_pipelines"] += 1
            return {
                "success": False,
                "step_failed": "tts",
                "error": tts_result.get("error", "TTS generation failed"),
            }

        audio_path = tts_result["audio_path"]

        # Step 2: Avatar Animation
        logger.info(f"[MediaPipeline] Step 2/3: Avatar animation for persona={persona_id}")
        avatar_result = await self.avatar.generate_avatar_video(
            audio_path=audio_path,
            persona_id=persona_id,
            emotional_mode=emotional_mode,
            overlay_text=overlay_text,
            duration_s=tts_result.get("duration_s"),
        )

        if not avatar_result.get("success"):
            # Partial success: audio generated but video failed
            self._metrics["failed_pipelines"] += 1
            elapsed_ms = (time.time() - start) * 1000
            return {
                "success": False,
                "partial": True,
                "step_failed": "avatar",
                "audio_path": audio_path,
                "audio_duration_s": tts_result.get("duration_s"),
                "tts_engine": tts_result.get("engine"),
                "error": avatar_result.get("error", "Avatar generation failed"),
                "latency_ms": round(elapsed_ms, 1),
            }

        # Step 3: Success
        elapsed_ms = (time.time() - start) * 1000
        self._metrics["successful_pipelines"] += 1
        self._metrics["total_latency_ms"] += elapsed_ms

        logger.info(
            f"[MediaPipeline] Step 3/3: Complete! "
            f"TTS={tts_result.get('engine')}, "
            f"Avatar={avatar_result.get('engine')}, "
            f"Latency={elapsed_ms:.0f}ms"
        )

        return {
            "success": True,
            "video_path": avatar_result["video_path"],
            "audio_path": audio_path,
            "audio_duration_s": tts_result.get("duration_s"),
            "tts_engine": tts_result.get("engine"),
            "avatar_engine": avatar_result.get("engine"),
            "format": "mp4",
            "persona_id": persona_id,
            "emotional_mode": emotional_mode,
            "latency_ms": round(elapsed_ms, 1),
            "event": event_data,
        }

    async def generate_audio_only(
        self,
        commentary_text: str,
        persona_id: str,
        voice_id: Optional[str] = None,
        language: str = "pt",
        emotional_mode: str = "neutral",
    ) -> Dict[str, Any]:
        """Generate audio-only commentary (for Telegram voice, podcasts)"""
        voice = voice_id or persona_id
        return await self.tts.generate_speech(
            text=commentary_text,
            voice_id=voice,
            language=language,
            emotional_mode=emotional_mode,
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check all media pipeline components"""
        tts_health = await self.tts.health_check()
        avatar_health = await self.avatar.health_check()

        overall = "healthy"
        if tts_health.get("pyttsx3") != "available":
            overall = "degraded"

        return {
            "overall": overall,
            "tts": tts_health,
            "avatar": avatar_health,
        }

    def get_metrics(self) -> Dict[str, Any]:
        total = self._metrics["total_pipelines"]
        return {
            **self._metrics,
            "success_rate": (
                self._metrics["successful_pipelines"] / total if total > 0 else 0
            ),
            "avg_latency_ms": (
                self._metrics["total_latency_ms"] / max(self._metrics["successful_pipelines"], 1)
            ),
            "tts_metrics": self.tts.get_metrics(),
            "avatar_metrics": self.avatar.get_metrics(),
        }

    async def close(self):
        await asyncio.gather(
            self.tts.close(),
            self.avatar.close(),
            return_exceptions=True,
        )
