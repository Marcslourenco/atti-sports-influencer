"""
n8n Webhook Adapter — Triggers n8n workflows for multi-platform publishing.

Features:
- Webhook trigger for n8n workflows
- Payload formatting for different content types
- Multi-platform orchestration (Telegram + Instagram + Discord + Twitter)
- Retry logic with exponential backoff
- Workflow status tracking
- Support for n8n cloud and self-hosted instances
"""
import asyncio
import logging
import os
import time
from typing import Dict, Any, Optional, List

import httpx

logger = logging.getLogger(__name__)


class N8NWebhookAdapter:
    """
    n8n Webhook adapter for multi-platform content distribution.

    Triggers n8n workflows that handle:
    1. Content formatting per platform
    2. Media transcoding (video sizes, image crops)
    3. Scheduling (optimal posting times)
    4. Cross-posting to multiple platforms
    5. Analytics collection
    """

    def __init__(
        self,
        webhook_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self.webhook_base_url = webhook_base_url or os.getenv(
            "N8N_WEBHOOK_URL",
            "http://localhost:5678/webhook"
        )
        self.api_key = api_key or os.getenv("N8N_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._metrics = {
            "webhooks_triggered": 0,
            "webhooks_failed": 0,
            "total_latency_ms": 0.0,
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if not self._client:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["X-N8N-API-KEY"] = self.api_key
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=headers,
            )
        return self._client

    async def trigger_publish_workflow(
        self,
        content: Dict[str, Any],
        platforms: List[str],
        agent_id: str = "",
        schedule_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Trigger n8n multi-platform publish workflow.

        Args:
            content: Content payload (text, media_url, caption, etc.)
            platforms: Target platforms ["telegram", "instagram", "discord", "twitter"]
            agent_id: Agent that generated the content
            schedule_at: Optional ISO datetime for scheduled posting

        Returns:
            Dict with workflow execution status
        """
        payload = {
            "event": "sports_commentary_publish",
            "content": content,
            "platforms": platforms,
            "agent_id": agent_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": {
                "source": "atti-sports-influencer",
                "version": "2.0.0",
            },
        }
        if schedule_at:
            payload["schedule_at"] = schedule_at

        return await self._trigger_webhook("publish-sports-content", payload)

    async def trigger_event_workflow(
        self,
        event: Dict[str, Any],
        matched_agents: List[str],
    ) -> Dict[str, Any]:
        """
        Trigger n8n event processing workflow.
        Used when a sports event is detected and needs to be processed.
        """
        payload = {
            "event": "sports_event_detected",
            "event_data": event,
            "matched_agents": matched_agents,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        return await self._trigger_webhook("process-sports-event", payload)

    async def trigger_media_workflow(
        self,
        media_type: str,
        source_url: str,
        target_platforms: List[str],
        transformations: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Trigger n8n media processing workflow.
        Handles transcoding, resizing, and platform-specific formatting.
        """
        payload = {
            "event": "media_processing",
            "media_type": media_type,
            "source_url": source_url,
            "target_platforms": target_platforms,
            "transformations": transformations or {},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        return await self._trigger_webhook("process-media", payload)

    async def trigger_analytics_workflow(
        self,
        agent_id: str,
        platform: str,
        post_id: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Trigger n8n analytics collection workflow.
        Collects engagement metrics from published content.
        """
        payload = {
            "event": "collect_analytics",
            "agent_id": agent_id,
            "platform": platform,
            "post_id": post_id,
            "metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        return await self._trigger_webhook("collect-analytics", payload)

    async def _trigger_webhook(
        self, workflow_path: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Trigger a specific n8n webhook with retry logic"""
        self._metrics["webhooks_triggered"] += 1
        start = time.time()
        url = f"{self.webhook_base_url}/{workflow_path}"

        for attempt in range(self.max_retries + 1):
            try:
                client = await self._get_client()
                response = await client.post(url, json=payload)

                if response.status_code in (200, 201, 202):
                    elapsed_ms = (time.time() - start) * 1000
                    self._metrics["total_latency_ms"] += elapsed_ms

                    # n8n returns execution data
                    data = {}
                    try:
                        data = response.json()
                    except Exception:
                        pass

                    return {
                        "success": True,
                        "workflow": workflow_path,
                        "execution_id": data.get("executionId", data.get("id", "")),
                        "status": data.get("status", "triggered"),
                        "latency_ms": round(elapsed_ms, 1),
                        "attempt": attempt + 1,
                    }
                else:
                    logger.warning(
                        f"n8n webhook {workflow_path} attempt {attempt + 1}: "
                        f"HTTP {response.status_code}"
                    )

            except Exception as e:
                logger.warning(f"n8n webhook {workflow_path} attempt {attempt + 1}: {e}")

            if attempt < self.max_retries:
                await asyncio.sleep(2 ** attempt)

        self._metrics["webhooks_failed"] += 1
        return {
            "success": False,
            "workflow": workflow_path,
            "error": f"Failed after {self.max_retries + 1} attempts",
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check n8n instance health"""
        if not self.webhook_base_url:
            return {"status": "not_configured", "platform": "n8n"}

        try:
            client = await self._get_client()
            # Try to reach the n8n health endpoint
            base = self.webhook_base_url.rsplit("/webhook", 1)[0]
            response = await client.get(f"{base}/healthz")
            return {
                "status": "healthy" if response.status_code == 200 else "degraded",
                "platform": "n8n",
                "base_url": self.webhook_base_url,
            }
        except Exception as e:
            return {
                "status": "unreachable",
                "platform": "n8n",
                "base_url": self.webhook_base_url,
                "error": str(e),
            }

    def get_metrics(self) -> Dict[str, Any]:
        total = self._metrics["webhooks_triggered"]
        return {
            **self._metrics,
            "success_rate": (
                (total - self._metrics["webhooks_failed"]) / total if total > 0 else 0
            ),
            "avg_latency_ms": (
                self._metrics["total_latency_ms"] / max(total - self._metrics["webhooks_failed"], 1)
            ),
            "platform": "n8n",
        }

    async def close(self):
        if self._client:
            await self._client.aclose()
