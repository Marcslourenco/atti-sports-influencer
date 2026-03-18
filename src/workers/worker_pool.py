"""
Stateless Worker Pool — Shared workers that process tasks for any agent.
Workers do NOT maintain per-agent state. Agent context is loaded dynamically per task.
Supports text workers (CPU) and media workers (GPU).
"""
import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, List
from enum import Enum

from .redis_broker import RedisBroker, Task, TaskStatus, TaskPriority

logger = logging.getLogger(__name__)


class WorkerType(str, Enum):
    TEXT = "text"
    MEDIA = "media"
    PUBLISH = "publish"


class WorkerStats:
    """Track worker performance metrics"""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.tasks_processed = 0
        self.tasks_failed = 0
        self.total_processing_time = 0.0
        self.last_task_at: Optional[float] = None
        self.started_at = time.time()

    @property
    def avg_processing_time(self) -> float:
        if self.tasks_processed == 0:
            return 0.0
        return self.total_processing_time / self.tasks_processed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "tasks_processed": self.tasks_processed,
            "tasks_failed": self.tasks_failed,
            "avg_processing_time_ms": round(self.avg_processing_time * 1000, 2),
            "uptime_seconds": round(time.time() - self.started_at, 1),
        }


class Worker:
    """
    Stateless worker that processes tasks from the queue.
    Loads agent context dynamically for each task.
    """

    def __init__(
        self,
        worker_id: str,
        worker_type: WorkerType,
        broker: RedisBroker,
        handler: Callable,
    ):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.broker = broker
        self.handler = handler
        self.stats = WorkerStats(worker_id)
        self._running = False

    async def run(self):
        """Main worker loop — pull tasks and process them"""
        self._running = True
        logger.info(f"Worker {self.worker_id} ({self.worker_type.value}) started")

        while self._running:
            try:
                task = await self.broker.dequeue(
                    task_type=self.worker_type.value,
                    consumer_name=self.worker_id,
                    timeout=2000,
                )

                if task is None:
                    await asyncio.sleep(0.1)
                    continue

                start = time.time()
                try:
                    result = await self.handler(task)
                    elapsed = time.time() - start

                    await self.broker.complete_task(task.task_id, result)
                    self.stats.tasks_processed += 1
                    self.stats.total_processing_time += elapsed
                    self.stats.last_task_at = time.time()

                    logger.debug(
                        f"Worker {self.worker_id} completed {task.task_id} "
                        f"in {elapsed:.2f}s"
                    )

                except Exception as e:
                    elapsed = time.time() - start
                    await self.broker.fail_task(task.task_id, str(e))
                    self.stats.tasks_failed += 1
                    logger.error(
                        f"Worker {self.worker_id} failed {task.task_id}: {e}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {self.worker_id} loop error: {e}")
                await asyncio.sleep(1)

        logger.info(f"Worker {self.worker_id} stopped")

    def stop(self):
        self._running = False


class WorkerPool:
    """
    Pool of stateless workers that process tasks for any agent.
    Supports dynamic scaling of text and media workers.
    """

    def __init__(self, broker: RedisBroker):
        self.broker = broker
        self._workers: Dict[str, Worker] = {}
        self._tasks: List[asyncio.Task] = []
        self._handlers: Dict[WorkerType, Callable] = {}

    def register_handler(self, worker_type: WorkerType, handler: Callable):
        """Register a task handler for a worker type"""
        self._handlers[worker_type] = handler
        logger.info(f"Registered handler for {worker_type.value} workers")

    async def start(
        self,
        n_text: int = 3,
        n_media: int = 1,
        n_publish: int = 1,
    ):
        """Start the worker pool with specified worker counts"""
        logger.info(
            f"Starting worker pool: {n_text} text, {n_media} media, {n_publish} publish"
        )

        # Start text workers
        for i in range(n_text):
            await self._add_worker(f"text_worker_{i}", WorkerType.TEXT)

        # Start media workers
        for i in range(n_media):
            await self._add_worker(f"media_worker_{i}", WorkerType.MEDIA)

        # Start publish workers
        for i in range(n_publish):
            await self._add_worker(f"publish_worker_{i}", WorkerType.PUBLISH)

    async def _add_worker(self, worker_id: str, worker_type: WorkerType):
        """Add a single worker to the pool"""
        handler = self._handlers.get(worker_type)
        if not handler:
            logger.warning(f"No handler for {worker_type.value}, using default")
            handler = self._default_handler

        worker = Worker(worker_id, worker_type, self.broker, handler)
        self._workers[worker_id] = worker
        task = asyncio.create_task(worker.run())
        self._tasks.append(task)

    async def _default_handler(self, task: Task) -> Dict[str, Any]:
        """Default handler that logs and returns the task"""
        logger.warning(f"Default handler processing task {task.task_id}")
        return {"status": "processed_by_default", "task_type": task.task_type}

    async def scale(self, worker_type: WorkerType, target_count: int):
        """Scale workers of a specific type to target count"""
        prefix = f"{worker_type.value}_worker_"
        current = [w for wid, w in self._workers.items() if wid.startswith(prefix)]
        current_count = len(current)

        if target_count > current_count:
            # Scale up
            for i in range(current_count, target_count):
                await self._add_worker(f"{prefix}{i}", worker_type)
            logger.info(f"Scaled {worker_type.value} workers: {current_count} → {target_count}")
        elif target_count < current_count:
            # Scale down
            to_remove = current[target_count:]
            for worker in to_remove:
                worker.stop()
                del self._workers[worker.worker_id]
            logger.info(f"Scaled {worker_type.value} workers: {current_count} → {target_count}")

    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        agent_id: str,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> str:
        """Submit a task to the pool"""
        task = Task(
            task_type=task_type,
            payload=payload,
            agent_id=agent_id,
            priority=priority,
        )
        return await self.broker.enqueue(task)

    def get_pool_metrics(self) -> Dict[str, Any]:
        """Get pool-wide metrics"""
        metrics = {
            "total_workers": len(self._workers),
            "workers_by_type": {},
            "worker_stats": [],
        }

        for wtype in WorkerType:
            prefix = f"{wtype.value}_worker_"
            count = sum(1 for wid in self._workers if wid.startswith(prefix))
            metrics["workers_by_type"][wtype.value] = count

        for worker in self._workers.values():
            metrics["worker_stats"].append(worker.stats.to_dict())

        return metrics

    async def stop(self):
        """Stop all workers"""
        for worker in self._workers.values():
            worker.stop()

        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._workers.clear()
        self._tasks.clear()
        logger.info("Worker pool stopped")
