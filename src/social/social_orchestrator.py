"""
Social Orchestrator — Coordinates publishing across all social platforms.

Pipeline: Event -> Commentary -> [Telegram, Instagram, n8n] -> Analytics

Features:
- Multi-platform simultaneous publishing
- Platform-specific content adaptation
- Publish queue with priority
- Failure recovery and retry
- Analytics aggregation
"""
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List

from .telegram_publisher import TelegramPublisher
from .instagram_publisher import InstagramPublisher
from .n8n_webhook_adapter import N8NWebhookAdapter

logger = logging.getLogger(__name__)


class SocialOrchestrator:
    """
    Orchestrates content publishing across all social platforms.

    Supports:
    - Direct publishing: Telegram (text, photo, video, voice)
    - API publishing: Instagram (Reels, Stories, Posts)
    - Workflow publishing: n8n (multi-platform, scheduled)
    """

    def __init__(self):
        self.telegram = TelegramPublisher()
        self.instagram = InstagramPublisher()
        self.n8n = N8NWebhookAdapter()
        self._publish_history: List[Dict[str, Any]] = []

    async def publish_commentary(
        self,
        commentary: str,
        event: Dict[str, Any],
        agent_id: str,
        platforms: Optional[List[str]] = None,
        media: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Publish commentary to specified platforms.

        Args:
            commentary: Generated commentary text
            event: Original sports event data
            agent_id: Agent that generated the commentary
            platforms: Target platforms (default: all configured)
            media: Optional media attachments (video_path, audio_path, image_path)

        Returns:
            Dict with per-platform results
        """
        platforms = platforms or ["telegram"]
        results = {}
        start = time.time()

        tasks = []

        if "telegram" in platforms:
            tasks.append(self._publish_telegram(commentary, event, media))

        if "instagram" in platforms and media:
            tasks.append(self._publish_instagram(commentary, event, media, agent_id))

        if "n8n" in platforms:
            tasks.append(self._publish_n8n(commentary, event, agent_id, platforms, media))

        # Execute all platform publishes concurrently
        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            platform_names = []
            if "telegram" in platforms:
                platform_names.append("telegram")
            if "instagram" in platforms and media:
                platform_names.append("instagram")
            if "n8n" in platforms:
                platform_names.append("n8n")

            for name, result in zip(platform_names, task_results):
                if isinstance(result, Exception):
                    results[name] = {"success": False, "error": str(result)}
                else:
                    results[name] = result

        elapsed_ms = (time.time() - start) * 1000

        publish_record = {
            "agent_id": agent_id,
            "event_type": event.get("type", ""),
            "platforms": platforms,
            "results": results,
            "latency_ms": round(elapsed_ms, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._publish_history.append(publish_record)

        return {
            "results": results,
            "total_platforms": len(platforms),
            "successful": sum(1 for r in results.values() if r.get("success")),
            "latency_ms": round(elapsed_ms, 1),
        }

    async def _publish_telegram(
        self,
        commentary: str,
        event: Dict[str, Any],
        media: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Publish to Telegram with appropriate media type"""
        event_type = event.get("type", "")

        # Send text commentary
        keyboard = self.telegram.build_engagement_keyboard(event_type)
        result = await self.telegram.send_commentary(
            text=commentary,
            reply_markup=keyboard,
        )

        # Send media if available
        if media:
            if media.get("video_path"):
                await self.telegram.send_video(
                    video_path=media["video_path"],
                    caption=f"🎬 {commentary[:200]}",
                )
            elif media.get("audio_path"):
                await self.telegram.send_voice(
                    voice_path=media["audio_path"],
                    caption=commentary[:200],
                )
            elif media.get("image_path"):
                await self.telegram.send_photo(
                    photo_path=media["image_path"],
                    caption=commentary[:1024],
                )

        return result

    async def _publish_instagram(
        self,
        commentary: str,
        event: Dict[str, Any],
        media: Dict[str, Any],
        agent_id: str,
    ) -> Dict[str, Any]:
        """Publish to Instagram (Reels for video, Posts for images)"""
        sport = event.get("sport", "futebol")
        competition = event.get("competition", "")
        teams = [event.get("home", ""), event.get("away", "")]
        teams = [t for t in teams if t]

        hashtags = InstagramPublisher.build_sports_hashtags(sport, competition, teams)

        if media.get("video_url"):
            return await self.instagram.publish_reel(
                video_url=media["video_url"],
                caption=commentary,
                hashtags=hashtags,
            )
        elif media.get("image_url"):
            return await self.instagram.publish_image(
                image_url=media["image_url"],
                caption=commentary,
                hashtags=hashtags,
            )

        return {"success": False, "error": "No media URL for Instagram", "platform": "instagram"}

    async def _publish_n8n(
        self,
        commentary: str,
        event: Dict[str, Any],
        agent_id: str,
        platforms: List[str],
        media: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Trigger n8n workflow for multi-platform publishing"""
        content = {
            "text": commentary,
            "event": event,
            "media": media or {},
        }

        return await self.n8n.trigger_publish_workflow(
            content=content,
            platforms=platforms,
            agent_id=agent_id,
        )

    async def health_check_all(self) -> Dict[str, Any]:
        """Check health of all social platforms"""
        checks = await asyncio.gather(
            self.telegram.health_check(),
            self.instagram.health_check(),
            self.n8n.health_check(),
            return_exceptions=True,
        )

        return {
            "telegram": checks[0] if not isinstance(checks[0], Exception) else {"status": "error"},
            "instagram": checks[1] if not isinstance(checks[1], Exception) else {"status": "error"},
            "n8n": checks[2] if not isinstance(checks[2], Exception) else {"status": "error"},
        }

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "telegram": self.telegram.get_metrics(),
            "instagram": self.instagram.get_metrics(),
            "n8n": self.n8n.get_metrics(),
            "total_publishes": len(self._publish_history),
        }

    def get_publish_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._publish_history[-limit:]

    async def close(self):
        await asyncio.gather(
            self.telegram.close(),
            self.instagram.close(),
            self.n8n.close(),
            return_exceptions=True,
        )
