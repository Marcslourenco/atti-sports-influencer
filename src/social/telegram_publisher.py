"""
Telegram Bot Publisher — Publishes sports commentary to Telegram channels/groups.

Features:
- Text messages with Markdown formatting
- Photo + caption for match graphics
- Video messages for avatar clips
- Voice messages for TTS audio
- Channel and group support
- Rate limiting (30 messages/second per bot)
- Inline keyboard for engagement (reactions, share)
- Message editing for live score updates
"""
import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import httpx

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}"


class TelegramPublisher:
    """
    Telegram Bot API publisher for sports commentary.

    Supports:
    - sendMessage: Text commentary
    - sendPhoto: Match graphics with caption
    - sendVideo: Avatar video clips
    - sendVoice: TTS audio commentary
    - editMessageText: Live score updates
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        default_chat_id: Optional[str] = None,
        rate_limit: float = 0.05,  # 20 msgs/sec (below 30/sec limit)
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.default_chat_id = default_chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.rate_limit = rate_limit
        self._api_base = TELEGRAM_API_BASE.format(token=self.bot_token)
        self._client: Optional[httpx.AsyncClient] = None
        self._last_send_time = 0.0
        self._metrics = {
            "messages_sent": 0,
            "messages_failed": 0,
            "photos_sent": 0,
            "videos_sent": 0,
            "voices_sent": 0,
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if not self._client:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _rate_limit_wait(self):
        """Enforce rate limiting"""
        now = time.time()
        elapsed = now - self._last_send_time
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self._last_send_time = time.time()

    async def send_commentary(
        self,
        text: str,
        chat_id: Optional[str] = None,
        parse_mode: str = "Markdown",
        reply_markup: Optional[Dict] = None,
        disable_notification: bool = False,
    ) -> Dict[str, Any]:
        """Send text commentary to Telegram"""
        await self._rate_limit_wait()
        target = chat_id or self.default_chat_id

        payload = {
            "chat_id": target,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self._api_base}/sendMessage",
                json=payload,
            )

            if response.status_code == 200:
                data = response.json()
                self._metrics["messages_sent"] += 1
                return {
                    "success": True,
                    "message_id": data.get("result", {}).get("message_id"),
                    "chat_id": target,
                    "platform": "telegram",
                }
            else:
                self._metrics["messages_failed"] += 1
                error_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                return {
                    "success": False,
                    "error": error_data.get("description", f"HTTP {response.status_code}"),
                    "platform": "telegram",
                }

        except Exception as e:
            self._metrics["messages_failed"] += 1
            return {"success": False, "error": str(e), "platform": "telegram"}

    async def send_photo(
        self,
        photo_path: str,
        caption: str = "",
        chat_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send photo with caption (match graphics, stats)"""
        await self._rate_limit_wait()
        target = chat_id or self.default_chat_id

        try:
            client = await self._get_client()
            files = {"photo": open(photo_path, "rb")}
            data = {
                "chat_id": target,
                "caption": caption,
                "parse_mode": "Markdown",
            }

            response = await client.post(
                f"{self._api_base}/sendPhoto",
                data=data,
                files=files,
            )

            if response.status_code == 200:
                self._metrics["photos_sent"] += 1
                result = response.json()
                return {
                    "success": True,
                    "message_id": result.get("result", {}).get("message_id"),
                    "platform": "telegram",
                    "type": "photo",
                }
            else:
                self._metrics["messages_failed"] += 1
                return {"success": False, "error": f"HTTP {response.status_code}", "platform": "telegram"}

        except Exception as e:
            self._metrics["messages_failed"] += 1
            return {"success": False, "error": str(e), "platform": "telegram"}

    async def send_video(
        self,
        video_path: str,
        caption: str = "",
        chat_id: Optional[str] = None,
        duration: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send video (avatar clips)"""
        await self._rate_limit_wait()
        target = chat_id or self.default_chat_id

        try:
            client = await self._get_client()
            files = {"video": open(video_path, "rb")}
            data = {
                "chat_id": target,
                "caption": caption,
                "parse_mode": "Markdown",
            }
            if duration:
                data["duration"] = str(duration)

            response = await client.post(
                f"{self._api_base}/sendVideo",
                data=data,
                files=files,
            )

            if response.status_code == 200:
                self._metrics["videos_sent"] += 1
                result = response.json()
                return {
                    "success": True,
                    "message_id": result.get("result", {}).get("message_id"),
                    "platform": "telegram",
                    "type": "video",
                }
            else:
                self._metrics["messages_failed"] += 1
                return {"success": False, "error": f"HTTP {response.status_code}", "platform": "telegram"}

        except Exception as e:
            self._metrics["messages_failed"] += 1
            return {"success": False, "error": str(e), "platform": "telegram"}

    async def send_voice(
        self,
        voice_path: str,
        caption: str = "",
        chat_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send voice message (TTS commentary)"""
        await self._rate_limit_wait()
        target = chat_id or self.default_chat_id

        try:
            client = await self._get_client()
            files = {"voice": open(voice_path, "rb")}
            data = {"chat_id": target, "caption": caption}

            response = await client.post(
                f"{self._api_base}/sendVoice",
                data=data,
                files=files,
            )

            if response.status_code == 200:
                self._metrics["voices_sent"] += 1
                result = response.json()
                return {
                    "success": True,
                    "message_id": result.get("result", {}).get("message_id"),
                    "platform": "telegram",
                    "type": "voice",
                }
            else:
                self._metrics["messages_failed"] += 1
                return {"success": False, "error": f"HTTP {response.status_code}", "platform": "telegram"}

        except Exception as e:
            self._metrics["messages_failed"] += 1
            return {"success": False, "error": str(e), "platform": "telegram"}

    async def update_score(
        self,
        message_id: int,
        new_text: str,
        chat_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Edit existing message (live score updates)"""
        await self._rate_limit_wait()
        target = chat_id or self.default_chat_id

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self._api_base}/editMessageText",
                json={
                    "chat_id": target,
                    "message_id": message_id,
                    "text": new_text,
                    "parse_mode": "Markdown",
                },
            )
            return {
                "success": response.status_code == 200,
                "message_id": message_id,
                "platform": "telegram",
                "type": "edit",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "platform": "telegram"}

    def build_engagement_keyboard(
        self, event_type: str = "goal"
    ) -> Dict[str, Any]:
        """Build inline keyboard for engagement"""
        buttons = []
        if event_type == "goal":
            buttons = [
                [
                    {"text": "🔥 Golaço!", "callback_data": "react_fire"},
                    {"text": "⚽ Gol!", "callback_data": "react_goal"},
                ],
                [
                    {"text": "📊 Estatísticas", "callback_data": "stats"},
                    {"text": "📢 Compartilhar", "callback_data": "share"},
                ],
            ]
        elif event_type in ("match_start", "match_end"):
            buttons = [
                [
                    {"text": "📊 Escalação", "callback_data": "lineup"},
                    {"text": "📈 Odds", "callback_data": "odds"},
                ],
            ]

        return {"inline_keyboard": buttons}

    async def health_check(self) -> Dict[str, Any]:
        """Check Telegram Bot API health"""
        if not self.bot_token:
            return {"status": "not_configured", "platform": "telegram"}

        try:
            client = await self._get_client()
            response = await client.get(f"{self._api_base}/getMe")
            if response.status_code == 200:
                data = response.json()
                bot_info = data.get("result", {})
                return {
                    "status": "healthy",
                    "platform": "telegram",
                    "bot_name": bot_info.get("username", ""),
                    "bot_id": bot_info.get("id", ""),
                }
            return {"status": "degraded", "platform": "telegram"}
        except Exception as e:
            return {"status": "unreachable", "platform": "telegram", "error": str(e)}

    def get_metrics(self) -> Dict[str, Any]:
        return {**self._metrics, "platform": "telegram"}

    async def close(self):
        if self._client:
            await self._client.aclose()
