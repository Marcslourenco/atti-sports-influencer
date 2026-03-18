"""
Media Worker — Stateless worker for video/audio generation tasks.
Designed to run in the Worker Pool alongside text generation workers.

Features:
- Stateless: loads persona config per-task from Agent Registry
- Handles: tts_generation, avatar_generation, full_media_pipeline
- Integrates with Redis broker for task queue
- Reports metrics back to worker pool
"""
import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional

from .media_pipeline import MediaPipeline

logger = logging.getLogger(__name__)


class MediaWorker:
    """
    Stateless media worker for the Worker Pool.

    Task types:
    - tts_generation: Text -> Audio
    - avatar_generation: Audio -> Video
    - full_media_pipeline: Text -> Audio -> Video
    """

    def __init__(self, worker_id: str = "media-worker-0"):
        self.worker_id = worker_id
        self.pipeline = MediaPipeline()
        self._running = False
        self._metrics = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "total_latency_ms": 0.0,
        }

    async def initialize(self):
        """Initialize media pipeline"""
        await self.pipeline.initialize()
        self._running = True
        logger.info(f"MediaWorker {self.worker_id} initialized")

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single media task.

        Args:
            task: Task dict with task_type and payload

        Returns:
            Dict with result or error
        """
        start = time.time()
        task_type = task.get("task_type", "")
        payload = task.get("payload", {})
        task_id = task.get("task_id", "unknown")

        logger.info(f"[{self.worker_id}] Processing {task_type} (task_id={task_id})")

        try:
            if task_type == "tts_generation":
                result = await self._handle_tts(payload)
            elif task_type == "avatar_generation":
                result = await self._handle_avatar(payload)
            elif task_type == "full_media_pipeline":
                result = await self._handle_full_pipeline(payload)
            else:
                result = {"success": False, "error": f"Unknown task type: {task_type}"}

            elapsed_ms = (time.time() - start) * 1000
            self._metrics["tasks_processed"] += 1
            self._metrics["total_latency_ms"] += elapsed_ms

            if not result.get("success"):
                self._metrics["tasks_failed"] += 1

            result["worker_id"] = self.worker_id
            result["task_id"] = task_id
            result["processing_ms"] = round(elapsed_ms, 1)
            return result

        except Exception as e:
            self._metrics["tasks_processed"] += 1
            self._metrics["tasks_failed"] += 1
            elapsed_ms = (time.time() - start) * 1000
            self._metrics["total_latency_ms"] += elapsed_ms
            return {
                "success": False,
                "error": str(e),
                "worker_id": self.worker_id,
                "task_id": task_id,
                "processing_ms": round(elapsed_ms, 1),
            }

    async def _handle_tts(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle TTS generation task"""
        return await self.pipeline.generate_audio_only(
            commentary_text=payload.get("text", ""),
            persona_id=payload.get("persona_id", "default"),
            voice_id=payload.get("voice_id"),
            language=payload.get("language", "pt"),
            emotional_mode=payload.get("emotional_mode", "neutral"),
        )

    async def _handle_avatar(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle avatar generation task"""
        return await self.pipeline.avatar.generate_avatar_video(
            audio_path=payload.get("audio_path", ""),
            persona_id=payload.get("persona_id", "default"),
            emotional_mode=payload.get("emotional_mode", "neutral"),
            overlay_text=payload.get("overlay_text"),
        )

    async def _handle_full_pipeline(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle full media pipeline task"""
        return await self.pipeline.generate_commentary_video(
            commentary_text=payload.get("text", ""),
            persona_id=payload.get("persona_id", "default"),
            voice_id=payload.get("voice_id"),
            language=payload.get("language", "pt"),
            emotional_mode=payload.get("emotional_mode", "neutral"),
            overlay_text=payload.get("overlay_text"),
            event_data=payload.get("event_data"),
        )

    def get_metrics(self) -> Dict[str, Any]:
        total = self._metrics["tasks_processed"]
        return {
            **self._metrics,
            "worker_id": self.worker_id,
            "success_rate": (
                (total - self._metrics["tasks_failed"]) / total if total > 0 else 0
            ),
            "avg_latency_ms": (
                self._metrics["total_latency_ms"] / total if total > 0 else 0
            ),
            "pipeline_metrics": self.pipeline.get_metrics(),
        }

    async def shutdown(self):
        """Graceful shutdown"""
        self._running = False
        await self.pipeline.close()
        logger.info(f"MediaWorker {self.worker_id} shut down")
