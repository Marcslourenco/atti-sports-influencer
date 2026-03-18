"""
Sports Worker Router — Extends ATTI WorkerRouter for sports agent tasks.
Backward compatible: does not modify existing ATTI WorkerRouter.
Adds: dynamic agent config loading, persona attachment, parallel pipeline.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List

from ..registry.agent_registry import AgentRegistryLoader
from ..registry.agent_schema import AgentConfig
from ..adapters.atti_llm_adapter import ATTILLMAdapter
from ..adapters.atti_rag_adapter import ATTIRAGAdapter
from ..adapters.atti_persona_adapter import ATTIPersonaAdapter
from ..workers.worker_pool import WorkerPool, WorkerType
from ..workers.redis_broker import Task, TaskPriority

logger = logging.getLogger(__name__)


class SportsWorkerRouter:
    """
    Extension of ATTI WorkerRouter for sports influencer tasks.
    
    Capabilities:
    - Receives tasks for any agent
    - Fetches agent configuration from Agent Registry
    - Attaches persona and knowledge context dynamically
    - Routes to appropriate worker pool
    - Supports parallel pipeline execution
    
    Backward compatible with existing ATTI workflows.
    """

    def __init__(
        self,
        registry: AgentRegistryLoader,
        worker_pool: WorkerPool,
        llm_adapter: ATTILLMAdapter,
        rag_adapter: ATTIRAGAdapter,
        persona_adapter: ATTIPersonaAdapter,
    ):
        self.registry = registry
        self.pool = worker_pool
        self.llm = llm_adapter
        self.rag = rag_adapter
        self.personas = persona_adapter

    async def route_event(
        self,
        event: Dict[str, Any],
        agent_id: str,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> Dict[str, Any]:
        """
        Route a sports event to the appropriate pipeline for an agent.
        
        Flow:
        1. Load agent config from Registry
        2. Attach persona context
        3. Build knowledge context from RAG
        4. Create pipeline tasks (parallel where possible)
        5. Submit to worker pool
        """
        # Step 1: Load agent config
        agent = self.registry.get(agent_id)
        if not agent:
            logger.error(f"Agent not found: {agent_id}")
            return {"error": f"Agent {agent_id} not found"}

        if agent.status.value != "active":
            logger.info(f"Agent {agent_id} is {agent.status.value}, skipping")
            return {"skipped": True, "reason": f"Agent status: {agent.status.value}"}

        # Step 2: Get persona
        persona = self.personas.get_persona(agent_id)
        persona_dict = persona.to_dict() if persona else agent.to_atti_persona()

        # Step 3: Build RAG context
        query = self._build_rag_query(event, agent)
        context = self.rag.build_context(query, max_chunks=3)

        # Step 4: Create pipeline tasks based on agent config
        task_ids = await self._create_pipeline_tasks(
            event=event,
            agent=agent,
            persona=persona_dict,
            context=context,
            priority=priority,
        )

        return {
            "agent_id": agent_id,
            "event_type": event.get("type", "unknown"),
            "tasks_created": task_ids,
            "pipeline": {
                "commentary": agent.content_pipeline.commentary,
                "tts": agent.content_pipeline.tts,
                "avatar_video": agent.content_pipeline.avatar_video,
                "image_gen": agent.content_pipeline.image_gen,
            },
        }

    async def _create_pipeline_tasks(
        self,
        event: Dict[str, Any],
        agent: AgentConfig,
        persona: Dict[str, Any],
        context: Dict[str, Any],
        priority: TaskPriority,
    ) -> List[str]:
        """
        Create pipeline tasks based on agent's content_pipeline config.
        Uses PARALLEL execution where possible (not sequential).
        """
        task_ids = []
        pipeline = agent.content_pipeline

        # Commentary is always first (other tasks depend on it)
        if pipeline.commentary:
            commentary_task_id = await self.pool.submit_task(
                task_type="commentary",
                payload={
                    "event": event,
                    "persona": persona,
                    "context": context.get("context", ""),
                    "content_style": agent.content_style.model_dump(),
                    "agent_id": agent.agent_id,
                },
                agent_id=agent.agent_id,
                priority=priority,
            )
            task_ids.append(commentary_task_id)

        # TTS and Image Gen can run in parallel (after commentary)
        parallel_tasks = []

        if pipeline.tts:
            tts_config = pipeline.tts_config.model_dump() if pipeline.tts_config else {}
            parallel_tasks.append(
                self.pool.submit_task(
                    task_type="tts",
                    payload={
                        "depends_on": task_ids[0] if task_ids else None,
                        "tts_config": tts_config,
                        "agent_id": agent.agent_id,
                    },
                    agent_id=agent.agent_id,
                    priority=priority,
                )
            )

        if pipeline.image_gen:
            parallel_tasks.append(
                self.pool.submit_task(
                    task_type="image_gen",
                    payload={
                        "event": event,
                        "persona": persona,
                        "agent_id": agent.agent_id,
                    },
                    agent_id=agent.agent_id,
                    priority=priority,
                )
            )

        if parallel_tasks:
            results = await asyncio.gather(*parallel_tasks)
            task_ids.extend(results)

        # Avatar video depends on TTS output
        if pipeline.avatar_video:
            avatar_config = pipeline.avatar_config.model_dump() if pipeline.avatar_config else {}
            avatar_task_id = await self.pool.submit_task(
                task_type="avatar_video",
                payload={
                    "depends_on": task_ids[-1] if task_ids else None,
                    "avatar_config": avatar_config,
                    "agent_id": agent.agent_id,
                },
                agent_id=agent.agent_id,
                priority=priority,
            )
            task_ids.append(avatar_task_id)

        # Publishing task (after all content is generated)
        if any(p.enabled for p in agent.platforms.values()):
            publish_task_id = await self.pool.submit_task(
                task_type="publish",
                payload={
                    "depends_on": task_ids,
                    "platforms": {
                        k: v.model_dump()
                        for k, v in agent.platforms.items()
                        if v.enabled
                    },
                    "agent_id": agent.agent_id,
                },
                agent_id=agent.agent_id,
                priority=TaskPriority.LOW,
            )
            task_ids.append(publish_task_id)

        return task_ids

    def _build_rag_query(self, event: Dict[str, Any], agent: AgentConfig) -> str:
        """Build a RAG query from event and agent context"""
        parts = []

        event_type = event.get("type", "")
        if event_type:
            parts.append(event_type)

        home = event.get("home", "")
        away = event.get("away", "")
        if home and away:
            parts.append(f"{home} vs {away}")

        team = agent.persona.team_affinity
        if team:
            parts.append(team)

        competition = event.get("competition", "")
        if competition:
            parts.append(competition)

        return " ".join(parts)

    async def route_batch(
        self,
        event: Dict[str, Any],
        agent_ids: List[str],
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> Dict[str, Any]:
        """Route an event to multiple agents in parallel"""
        results = await asyncio.gather(
            *[self.route_event(event, aid, priority) for aid in agent_ids],
            return_exceptions=True,
        )

        return {
            "event": event.get("type", "unknown"),
            "agents_processed": len(agent_ids),
            "results": [
                r if not isinstance(r, Exception) else {"error": str(r)}
                for r in results
            ],
        }

    def get_router_status(self) -> Dict[str, Any]:
        return {
            "registry": self.registry.get_status(),
            "pool": self.pool.get_pool_metrics(),
            "llm": {"endpoint": self.llm.endpoint},
            "rag": self.rag.get_status(),
            "personas": self.personas.list_personas(),
        }
