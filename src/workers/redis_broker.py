"""
Redis Message Broker — Async job routing, queue scaling, worker distribution.
Uses Redis Streams for reliable message delivery.
Loosely coupled: Redis is infrastructure, not tightly bound to agent logic.
"""
import asyncio
import json
import logging
import os
import time
import uuid
from typing import Dict, Any, Optional, List, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class TaskPriority(int, Enum):
    LOW = 0
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


class Task:
    """Represents a work item in the queue"""

    def __init__(
        self,
        task_type: str,
        payload: Dict[str, Any],
        agent_id: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        task_id: Optional[str] = None,
    ):
        self.task_id = task_id or f"task_{uuid.uuid4().hex[:12]}"
        self.task_type = task_type
        self.payload = payload
        self.agent_id = agent_id
        self.priority = priority
        self.status = TaskStatus.QUEUED
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "payload": json.dumps(self.payload, default=str),
            "agent_id": self.agent_id,
            "priority": str(self.priority.value),
            "status": self.status.value,
            "created_at": str(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        payload = data.get("payload", "{}")
        if isinstance(payload, str):
            payload = json.loads(payload)
        task = cls(
            task_type=data.get("task_type", "unknown"),
            payload=payload,
            agent_id=data.get("agent_id", ""),
            priority=TaskPriority(int(data.get("priority", 5))),
            task_id=data.get("task_id"),
        )
        task.status = TaskStatus(data.get("status", "queued"))
        return task


class RedisBroker:
    """
    Redis-based message broker for async task distribution.
    Uses Redis Streams for reliable delivery.
    Falls back to in-memory queue if Redis is unavailable.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        stream_prefix: str = "atti:sports:",
        consumer_group: str = "workers",
    ):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.stream_prefix = stream_prefix
        self.consumer_group = consumer_group
        self._redis = None
        self._fallback_queue: asyncio.Queue = asyncio.Queue()
        self._task_store: Dict[str, Task] = {}
        self._use_redis = False

    async def connect(self) -> bool:
        """Connect to Redis"""
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5,
            )
            await self._redis.ping()
            self._use_redis = True
            logger.info(f"Redis broker connected: {self.redis_url}")

            # Create consumer groups
            for stream in ["text_tasks", "media_tasks", "publish_tasks"]:
                stream_name = f"{self.stream_prefix}{stream}"
                try:
                    await self._redis.xgroup_create(
                        stream_name, self.consumer_group, id="0", mkstream=True
                    )
                except Exception:
                    pass  # Group already exists

            return True
        except ImportError:
            logger.warning("redis package not installed, using in-memory queue")
            self._use_redis = False
            return False
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}), using in-memory queue")
            self._use_redis = False
            return False

    def _stream_for_type(self, task_type: str) -> str:
        """Map task type to Redis stream"""
        if task_type in ("commentary", "text_generation"):
            return f"{self.stream_prefix}text_tasks"
        elif task_type in ("tts", "avatar_video", "image_gen"):
            return f"{self.stream_prefix}media_tasks"
        elif task_type in ("publish", "social_post"):
            return f"{self.stream_prefix}publish_tasks"
        return f"{self.stream_prefix}text_tasks"

    async def enqueue(self, task: Task) -> str:
        """Add task to queue"""
        self._task_store[task.task_id] = task

        if self._use_redis and self._redis:
            stream = self._stream_for_type(task.task_type)
            try:
                await self._redis.xadd(
                    stream,
                    task.to_dict(),
                    maxlen=50000,
                )
                logger.debug(f"Task {task.task_id} enqueued to {stream}")
            except Exception as e:
                logger.error(f"Redis enqueue failed: {e}")
                await self._fallback_queue.put(task)
        else:
            await self._fallback_queue.put(task)

        return task.task_id

    async def dequeue(
        self,
        task_type: str = "text",
        consumer_name: str = "worker_0",
        timeout: int = 5000,
    ) -> Optional[Task]:
        """Get next task from queue"""
        if self._use_redis and self._redis:
            stream = self._stream_for_type(task_type)
            try:
                messages = await self._redis.xreadgroup(
                    self.consumer_group,
                    consumer_name,
                    {stream: ">"},
                    count=1,
                    block=timeout,
                )
                if messages:
                    stream_name, entries = messages[0]
                    msg_id, data = entries[0]
                    task = Task.from_dict(data)
                    task.status = TaskStatus.PROCESSING
                    task.started_at = time.time()
                    self._task_store[task.task_id] = task
                    # Acknowledge
                    await self._redis.xack(stream_name, self.consumer_group, msg_id)
                    return task
                return None
            except Exception as e:
                logger.error(f"Redis dequeue failed: {e}")

        # Fallback
        try:
            task = self._fallback_queue.get_nowait()
            task.status = TaskStatus.PROCESSING
            task.started_at = time.time()
            return task
        except asyncio.QueueEmpty:
            return None

    async def complete_task(self, task_id: str, result: Dict[str, Any]):
        """Mark task as completed"""
        task = self._task_store.get(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.result = result

        if self._use_redis and self._redis:
            try:
                await self._redis.hset(
                    f"{self.stream_prefix}results:{task_id}",
                    mapping={
                        "status": "completed",
                        "result": json.dumps(result, default=str),
                        "completed_at": str(time.time()),
                    },
                )
                await self._redis.expire(
                    f"{self.stream_prefix}results:{task_id}", 3600
                )
            except Exception as e:
                logger.error(f"Redis complete failed: {e}")

    async def fail_task(self, task_id: str, error: str):
        """Mark task as failed"""
        task = self._task_store.get(task_id)
        if task:
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            task.error = error

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        task = self._task_store.get(task_id)
        if task:
            return {
                "task_id": task.task_id,
                "status": task.status.value,
                "task_type": task.task_type,
                "agent_id": task.agent_id,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "error": task.error,
            }
        return None

    async def get_queue_metrics(self) -> Dict[str, Any]:
        """Get queue metrics"""
        metrics = {
            "use_redis": self._use_redis,
            "total_tasks": len(self._task_store),
            "fallback_queue_size": self._fallback_queue.qsize(),
        }

        if self._use_redis and self._redis:
            try:
                for stream_type in ["text_tasks", "media_tasks", "publish_tasks"]:
                    stream = f"{self.stream_prefix}{stream_type}"
                    info = await self._redis.xinfo_stream(stream)
                    metrics[stream_type] = {
                        "length": info.get("length", 0),
                        "groups": info.get("groups", 0),
                    }
            except Exception:
                pass

        # Count by status
        for status in TaskStatus:
            count = sum(1 for t in self._task_store.values() if t.status == status)
            metrics[f"tasks_{status.value}"] = count

        return metrics

    async def close(self):
        if self._redis:
            await self._redis.close()
