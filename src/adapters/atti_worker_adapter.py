"""
ATTI Worker Adapter — Real integration with digital-worker-platform.
Connects to: atti-digital-worker-platform WorkerRouter and TaskQueue.

Features:
- HTTP-based task submission to ATTI worker platform
- Task status tracking and result retrieval
- Worker health monitoring
- Compatible with ATTI WorkerRouter task format
- Retry logic with exponential backoff
- Batch task submission for parallel processing
"""
import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, Optional, List
from enum import Enum

import httpx

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskResult:
    """Result from an ATTI worker task"""

    def __init__(self, data: Dict[str, Any]):
        self.task_id = data.get("task_id", "")
        self.status = TaskStatus(data.get("status", "pending"))
        self.result = data.get("result", None)
        self.error = data.get("error", None)
        self.worker_id = data.get("worker_id", "")
        self.started_at = data.get("started_at", "")
        self.completed_at = data.get("completed_at", "")
        self.duration_ms = data.get("duration_ms", 0)

    def is_success(self) -> bool:
        return self.status == TaskStatus.COMPLETED and self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "worker_id": self.worker_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
        }


class ATTIWorkerAdapter:
    """
    Real adapter for ATTI's digital-worker-platform.

    Connection chain:
    ATTIWorkerAdapter -> HTTP -> WorkerRouter -> TaskQueue -> Worker -> Result

    Task types supported:
    - text_generation: Generate text content via LLM
    - tts_generation: Generate speech audio
    - avatar_generation: Generate avatar video
    - social_publish: Publish to social platforms
    - data_ingestion: Ingest and process sports data
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.endpoint = endpoint or os.getenv(
            "ATTI_WORKER_ENDPOINT",
            "http://localhost:8001/api/workers"
        )
        self.api_key = api_key or os.getenv("ATTI_WORKER_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._metrics = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_latency_ms": 0.0,
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if not self._client:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=headers,
            )
        return self._client

    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 5,
        agent_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        callback_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a task to ATTI worker platform.

        Args:
            task_type: Type of task (text_generation, tts_generation, etc.)
            payload: Task-specific payload
            priority: Task priority (1=highest, 10=lowest)
            agent_id: Optional agent ID for tracking
            tenant_id: Optional tenant ID for multi-tenant isolation
            callback_url: Optional webhook URL for completion notification

        Returns:
            Dict with task_id and initial status
        """
        self._metrics["tasks_submitted"] += 1
        start = time.time()

        task_data = {
            "task_type": task_type,
            "payload": payload,
            "priority": priority,
            "metadata": {
                "source": "atti-sports-influencer",
                "agent_id": agent_id or "unknown",
                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        }
        if callback_url:
            task_data["callback_url"] = callback_url

        extra_headers = {}
        if tenant_id:
            extra_headers["X-Tenant-ID"] = tenant_id

        for attempt in range(self.max_retries + 1):
            try:
                client = await self._get_client()
                response = await client.post(
                    f"{self.endpoint}/tasks",
                    json=task_data,
                    headers=extra_headers,
                )

                if response.status_code in (200, 201, 202):
                    data = response.json()
                    elapsed_ms = (time.time() - start) * 1000
                    return {
                        "task_id": data.get("task_id", data.get("id", "")),
                        "status": data.get("status", "pending"),
                        "queue_position": data.get("queue_position", -1),
                        "estimated_wait_ms": data.get("estimated_wait_ms", 0),
                        "latency_ms": round(elapsed_ms, 1),
                    }
                else:
                    logger.warning(
                        f"Worker submit attempt {attempt + 1} failed: "
                        f"status={response.status_code}"
                    )

            except Exception as e:
                logger.warning(f"Worker submit attempt {attempt + 1} error: {e}")

            if attempt < self.max_retries:
                await asyncio.sleep(2 ** attempt)

        self._metrics["tasks_failed"] += 1
        return {
            "task_id": "",
            "status": "submission_failed",
            "error": f"Failed after {self.max_retries + 1} attempts",
        }

    async def process_query(
        self,
        query: str,
        persona_type: str = "sports",
        tenant_id: Optional[str] = None,
        max_context_chunks: int = 5,
    ) -> Dict[str, Any]:
        """
        Route a query through ATTI's existing QueryOrchestrator.
        Maps to: POST /api/v1/worker/query
        """
        try:
            client = await self._get_client()
            headers = {}
            if tenant_id:
                headers["X-Tenant-ID"] = tenant_id

            response = await client.post(
                f"{self.endpoint}/query",
                json={
                    "query": query,
                    "persona_type": persona_type,
                    "max_context_chunks": max_context_chunks,
                },
                headers=headers,
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Worker query failed: {response.status_code}")
                return {"error": f"Status {response.status_code}"}

        except Exception as e:
            logger.error(f"Worker adapter error: {e}")
            return {"error": str(e)}

    async def get_task_status(self, task_id: str) -> TaskResult:
        """Get the status of a submitted task"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.endpoint}/tasks/{task_id}")

            if response.status_code == 200:
                return TaskResult(response.json())
            else:
                return TaskResult({
                    "task_id": task_id,
                    "status": "failed",
                    "error": f"Status check failed: {response.status_code}",
                })
        except Exception as e:
            return TaskResult({
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
            })

    async def wait_for_task(
        self,
        task_id: str,
        poll_interval: float = 2.0,
        max_wait: float = 120.0,
    ) -> TaskResult:
        """Wait for a task to complete with polling"""
        start = time.time()

        while (time.time() - start) < max_wait:
            result = await self.get_task_status(task_id)

            if result.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                elapsed_ms = (time.time() - start) * 1000
                self._metrics["total_latency_ms"] += elapsed_ms
                if result.is_success():
                    self._metrics["tasks_completed"] += 1
                else:
                    self._metrics["tasks_failed"] += 1
                return result

            await asyncio.sleep(poll_interval)

        self._metrics["tasks_failed"] += 1
        return TaskResult({
            "task_id": task_id,
            "status": "failed",
            "error": f"Timeout after {max_wait}s",
        })

    async def submit_and_wait(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 5,
        agent_id: Optional[str] = None,
        max_wait: float = 120.0,
    ) -> TaskResult:
        """Submit a task and wait for completion"""
        submission = await self.submit_task(task_type, payload, priority, agent_id)

        if submission.get("status") == "submission_failed":
            return TaskResult({
                "task_id": "",
                "status": "failed",
                "error": submission.get("error", "Submission failed"),
            })

        task_id = submission["task_id"]
        return await self.wait_for_task(task_id, max_wait=max_wait)

    async def submit_batch(
        self,
        tasks: List[Dict[str, Any]],
        max_concurrent: int = 10,
    ) -> List[Dict[str, Any]]:
        """Submit multiple tasks with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []

        async def _submit_one(task: Dict) -> Dict:
            async with semaphore:
                return await self.submit_task(
                    task_type=task["task_type"],
                    payload=task["payload"],
                    priority=task.get("priority", 5),
                    agent_id=task.get("agent_id"),
                )

        tasks_coros = [_submit_one(t) for t in tasks]
        results = await asyncio.gather(*tasks_coros, return_exceptions=True)

        return [
            r if isinstance(r, dict) else {"status": "error", "error": str(r)}
            for r in results
        ]

    async def register_sports_persona(
        self,
        persona_id: str,
        persona_config: Dict[str, Any],
        tenant_id: Optional[str] = None,
    ) -> bool:
        """Register a sports persona in ATTI's PersonaManager via API"""
        try:
            client = await self._get_client()
            headers = {}
            if tenant_id:
                headers["X-Tenant-ID"] = tenant_id

            response = await client.post(
                f"{self.endpoint}/personas",
                json={"persona_id": persona_id, "config": persona_config},
                headers=headers,
            )
            return response.status_code in (200, 201)
        except Exception as e:
            logger.error(f"Register persona error: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Check worker platform health"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.endpoint}/health")
            return {
                "status": "healthy" if response.status_code == 200 else "degraded",
                "endpoint": self.endpoint,
                "status_code": response.status_code,
            }
        except Exception as e:
            return {
                "status": "unreachable",
                "endpoint": self.endpoint,
                "error": str(e),
            }

    def get_metrics(self) -> Dict[str, Any]:
        total = self._metrics["tasks_submitted"]
        return {
            **self._metrics,
            "success_rate": (
                self._metrics["tasks_completed"] / total if total > 0 else 0
            ),
            "avg_latency_ms": (
                self._metrics["total_latency_ms"] / max(self._metrics["tasks_completed"], 1)
            ),
        }

    async def close(self):
        if self._client:
            await self._client.aclose()
