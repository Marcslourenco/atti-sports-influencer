"""
Instagram Graph API Publisher — Publishes sports content as Reels, Stories, and Posts.

Features:
- Instagram Reels (video + caption)
- Instagram Stories (image/video)
- Instagram Feed Posts (carousel, single image)
- Hashtag management for sports content
- Rate limiting (200 API calls/hour)
- Media upload via container creation flow
- Caption formatting with emoji and hashtags
"""
import asyncio
import logging
import os
import time
from typing import Dict, Any, Optional, List

import httpx

logger = logging.getLogger(__name__)

GRAPH_API_BASE = "https://graph.facebook.com/v19.0"


class InstagramPublisher:
    """
    Instagram Graph API publisher for sports content.

    Flow for publishing:
    1. Create media container (POST /{ig-user-id}/media)
    2. Wait for container to be ready (GET /{container-id}?fields=status_code)
    3. Publish container (POST /{ig-user-id}/media_publish)

    Supported content types:
    - REELS: Video content (avatar commentary clips)
    - IMAGE: Single image posts (match graphics)
    - CAROUSEL: Multiple images (match highlights)
    - STORIES: Ephemeral content (live updates)
    """

    def __init__(
        self,
        access_token: Optional[str] = None,
        ig_user_id: Optional[str] = None,
        rate_limit_per_hour: int = 200,
    ):
        self.access_token = access_token or os.getenv("INSTAGRAM_ACCESS_TOKEN", "")
        self.ig_user_id = ig_user_id or os.getenv("INSTAGRAM_USER_ID", "")
        self.rate_limit_per_hour = rate_limit_per_hour
        self._client: Optional[httpx.AsyncClient] = None
        self._request_timestamps: List[float] = []
        self._metrics = {
            "reels_published": 0,
            "posts_published": 0,
            "stories_published": 0,
            "publish_failed": 0,
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if not self._client:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def _rate_limit_check(self):
        """Enforce Instagram API rate limits"""
        now = time.time()
        # Remove timestamps older than 1 hour
        self._request_timestamps = [
            t for t in self._request_timestamps if now - t < 3600
        ]
        if len(self._request_timestamps) >= self.rate_limit_per_hour:
            wait_time = 3600 - (now - self._request_timestamps[0])
            logger.warning(f"Instagram rate limit reached, waiting {wait_time:.0f}s")
            await asyncio.sleep(wait_time)
        self._request_timestamps.append(time.time())

    async def publish_reel(
        self,
        video_url: str,
        caption: str,
        cover_url: Optional[str] = None,
        share_to_feed: bool = True,
        hashtags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Publish a Reel (video) to Instagram.

        Args:
            video_url: Public URL of the video file (must be accessible by Instagram)
            caption: Reel caption text
            cover_url: Optional cover image URL
            share_to_feed: Whether to share to feed
            hashtags: Optional list of hashtags to append

        Returns:
            Dict with success status and media ID
        """
        await self._rate_limit_check()

        # Append hashtags to caption
        if hashtags:
            caption = f"{caption}\n\n{' '.join('#' + h for h in hashtags)}"

        # Step 1: Create media container
        container_params = {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": str(share_to_feed).lower(),
            "access_token": self.access_token,
        }
        if cover_url:
            container_params["cover_url"] = cover_url

        try:
            client = await self._get_client()

            # Create container
            response = await client.post(
                f"{GRAPH_API_BASE}/{self.ig_user_id}/media",
                data=container_params,
            )

            if response.status_code != 200:
                self._metrics["publish_failed"] += 1
                error = response.json().get("error", {})
                return {
                    "success": False,
                    "error": error.get("message", f"HTTP {response.status_code}"),
                    "platform": "instagram",
                    "type": "reel",
                }

            container_id = response.json().get("id")

            # Step 2: Wait for container to be ready
            ready = await self._wait_for_container(container_id)
            if not ready:
                self._metrics["publish_failed"] += 1
                return {
                    "success": False,
                    "error": "Container processing timeout",
                    "platform": "instagram",
                    "type": "reel",
                }

            # Step 3: Publish
            publish_response = await client.post(
                f"{GRAPH_API_BASE}/{self.ig_user_id}/media_publish",
                data={
                    "creation_id": container_id,
                    "access_token": self.access_token,
                },
            )

            if publish_response.status_code == 200:
                media_id = publish_response.json().get("id")
                self._metrics["reels_published"] += 1
                return {
                    "success": True,
                    "media_id": media_id,
                    "container_id": container_id,
                    "platform": "instagram",
                    "type": "reel",
                }
            else:
                self._metrics["publish_failed"] += 1
                error = publish_response.json().get("error", {})
                return {
                    "success": False,
                    "error": error.get("message", f"Publish failed: HTTP {publish_response.status_code}"),
                    "platform": "instagram",
                    "type": "reel",
                }

        except Exception as e:
            self._metrics["publish_failed"] += 1
            return {"success": False, "error": str(e), "platform": "instagram", "type": "reel"}

    async def publish_image(
        self,
        image_url: str,
        caption: str,
        hashtags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Publish a single image post"""
        await self._rate_limit_check()

        if hashtags:
            caption = f"{caption}\n\n{' '.join('#' + h for h in hashtags)}"

        try:
            client = await self._get_client()

            # Create container
            response = await client.post(
                f"{GRAPH_API_BASE}/{self.ig_user_id}/media",
                data={
                    "image_url": image_url,
                    "caption": caption,
                    "access_token": self.access_token,
                },
            )

            if response.status_code != 200:
                self._metrics["publish_failed"] += 1
                return {"success": False, "error": f"HTTP {response.status_code}", "platform": "instagram"}

            container_id = response.json().get("id")

            # Publish
            publish_response = await client.post(
                f"{GRAPH_API_BASE}/{self.ig_user_id}/media_publish",
                data={
                    "creation_id": container_id,
                    "access_token": self.access_token,
                },
            )

            if publish_response.status_code == 200:
                self._metrics["posts_published"] += 1
                return {
                    "success": True,
                    "media_id": publish_response.json().get("id"),
                    "platform": "instagram",
                    "type": "image",
                }
            else:
                self._metrics["publish_failed"] += 1
                return {"success": False, "error": "Publish failed", "platform": "instagram"}

        except Exception as e:
            self._metrics["publish_failed"] += 1
            return {"success": False, "error": str(e), "platform": "instagram"}

    async def publish_story(
        self,
        media_url: str,
        media_type: str = "IMAGE",
    ) -> Dict[str, Any]:
        """Publish a Story (image or video)"""
        await self._rate_limit_check()

        try:
            client = await self._get_client()

            params: Dict[str, Any] = {
                "access_token": self.access_token,
            }
            if media_type == "VIDEO":
                params["media_type"] = "STORIES"
                params["video_url"] = media_url
            else:
                params["media_type"] = "STORIES"
                params["image_url"] = media_url

            response = await client.post(
                f"{GRAPH_API_BASE}/{self.ig_user_id}/media",
                data=params,
            )

            if response.status_code != 200:
                self._metrics["publish_failed"] += 1
                return {"success": False, "error": f"HTTP {response.status_code}", "platform": "instagram"}

            container_id = response.json().get("id")

            if media_type == "VIDEO":
                await self._wait_for_container(container_id)

            publish_response = await client.post(
                f"{GRAPH_API_BASE}/{self.ig_user_id}/media_publish",
                data={
                    "creation_id": container_id,
                    "access_token": self.access_token,
                },
            )

            if publish_response.status_code == 200:
                self._metrics["stories_published"] += 1
                return {
                    "success": True,
                    "media_id": publish_response.json().get("id"),
                    "platform": "instagram",
                    "type": "story",
                }
            else:
                self._metrics["publish_failed"] += 1
                return {"success": False, "error": "Story publish failed", "platform": "instagram"}

        except Exception as e:
            self._metrics["publish_failed"] += 1
            return {"success": False, "error": str(e), "platform": "instagram"}

    async def _wait_for_container(
        self, container_id: str, max_wait: float = 120.0, poll_interval: float = 5.0
    ) -> bool:
        """Wait for media container to finish processing"""
        start = time.time()
        client = await self._get_client()

        while (time.time() - start) < max_wait:
            try:
                response = await client.get(
                    f"{GRAPH_API_BASE}/{container_id}",
                    params={
                        "fields": "status_code",
                        "access_token": self.access_token,
                    },
                )
                if response.status_code == 200:
                    status = response.json().get("status_code", "")
                    if status == "FINISHED":
                        return True
                    elif status == "ERROR":
                        logger.error(f"Container {container_id} processing error")
                        return False
            except Exception as e:
                logger.warning(f"Container status check error: {e}")

            await asyncio.sleep(poll_interval)

        return False

    @staticmethod
    def build_sports_hashtags(
        sport: str = "futebol",
        competition: str = "",
        teams: Optional[List[str]] = None,
    ) -> List[str]:
        """Build relevant sports hashtags"""
        tags = [sport]

        if competition:
            tag = competition.replace(" ", "").replace("-", "")
            tags.append(tag)

        if teams:
            for team in teams:
                tag = team.replace(" ", "").replace("-", "")
                tags.append(tag)

        # Common sports hashtags
        sport_tags = {
            "futebol": ["Futebol", "FutebolBrasileiro", "Gol", "BrasileiraoAssai"],
            "basketball": ["NBA", "Basketball", "Basquete"],
            "tennis": ["Tennis", "ATPTour", "WTA"],
        }
        tags.extend(sport_tags.get(sport, []))

        return tags[:30]  # Instagram limit: 30 hashtags

    async def health_check(self) -> Dict[str, Any]:
        """Check Instagram Graph API health"""
        if not self.access_token or not self.ig_user_id:
            return {"status": "not_configured", "platform": "instagram"}

        try:
            client = await self._get_client()
            response = await client.get(
                f"{GRAPH_API_BASE}/{self.ig_user_id}",
                params={
                    "fields": "id,username,media_count",
                    "access_token": self.access_token,
                },
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "platform": "instagram",
                    "username": data.get("username", ""),
                    "media_count": data.get("media_count", 0),
                }
            return {"status": "degraded", "platform": "instagram", "code": response.status_code}
        except Exception as e:
            return {"status": "unreachable", "platform": "instagram", "error": str(e)}

    def get_metrics(self) -> Dict[str, Any]:
        return {**self._metrics, "platform": "instagram"}

    async def close(self):
        if self._client:
            await self._client.aclose()
