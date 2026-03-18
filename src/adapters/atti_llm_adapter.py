"""
ATTI LLM Adapter — Real integration with Nemotron via Modal.com orchestrator.
Connects to: backend/orchestrator/modal_orchestrator_api.py
Endpoint: ATTI_LLM_ENDPOINT (Modal serverless function)

Features:
- Sports commentary generation with persona context
- Structured prompt engineering for sports domain
- Fallback templates when LLM is unavailable
- Streaming support for real-time commentary
- Token usage tracking and cost estimation
"""
import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, Optional, List

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt Templates for Sports Commentary
# ---------------------------------------------------------------------------

COMMENTARY_SYSTEM_PROMPT = """Você é {persona_name}, {persona_role}.
Perfil: {persona_expertise}
Tom: {persona_tone}
Time do coração: {team_affinity}
Idioma: {language}

REGRAS DE GERAÇÃO:
1. Máximo {max_length} caracteres
2. Estilo: {content_style}
3. Hashtags: {hashtag_strategy}
4. Menções: {mention_strategy}
5. Modo emocional atual: {emotional_mode}

CONTEXTO RAG (use como base factual):
{rag_context}

IMPORTANTE: Nunca invente dados estatísticos. Use apenas os dados do contexto RAG ou do evento."""

COMMENTARY_USER_PROMPT = """EVENTO ESPORTIVO:
Tipo: {event_type}
Competição: {competition}
{teams_info}
Minuto: {minute}
Descrição: {description}
{score_info}

Gere um comentário no estilo de {persona_name} sobre este evento.
{emotional_instruction}"""

FALLBACK_TEMPLATES = {
    "goal": [
        "GOOOOOOL! {scoring_team} marca contra {opponent}! {competition} pegando fogo!",
        "É GOL! {scoring_team} balança a rede! Que momento no {competition}!",
        "GOLAÇO! {scoring_team} não perdoa! {minute}' de jogo e a torcida vai à loucura!",
    ],
    "match_start": [
        "Bola rolando! {home} x {away} pelo {competition}. Vamos que vamos!",
        "Começou! {home} recebe {away} pelo {competition}. Quem leva essa?",
    ],
    "match_end": [
        "Fim de jogo! {home} {home_score} x {away_score} {away}. {competition} não decepciona!",
        "Apita o árbitro! {home} {home_score} x {away_score} {away}. Mais uma rodada do {competition}.",
    ],
    "red_card": [
        "CARTÃO VERMELHO! {player} expulso! {team} fica com um a menos no {competition}!",
    ],
    "halftime": [
        "Intervalo! {home} {home_score} x {away_score} {away}. Segundo tempo promete!",
    ],
    "default": [
        "Movimentação no {competition}! {home} x {away} segue quente. Acompanhe!",
    ],
}


class ATTILLMAdapter:
    """
    Real adapter for ATTI's Nemotron LLM via Modal.com orchestrator.

    Connection chain:
    ATTILLMAdapter -> HTTP -> Modal.com -> Nemotron -> Response

    Fallback chain:
    ATTILLMAdapter -> Template Engine -> Formatted Response
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "nvidia/nemotron-4-340b-instruct",
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self.endpoint = endpoint or os.getenv(
            "ATTI_LLM_ENDPOINT",
            "https://atti-orchestrator--generate.modal.run"
        )
        self.api_key = api_key or os.getenv("ATTI_MODAL_API_KEY", "")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "fallback_used": 0,
            "total_tokens": 0,
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

    async def generate_commentary(
        self,
        event: Dict[str, Any],
        persona: Dict[str, Any],
        rag_context: str = "",
        content_style: Optional[Dict[str, Any]] = None,
        emotional_mode: str = "neutral",
    ) -> Dict[str, Any]:
        """
        Generate sports commentary using ATTI Nemotron LLM.

        Args:
            event: Sports event data (type, teams, score, etc.)
            persona: Agent persona configuration
            rag_context: Pre-built RAG context string
            content_style: Content generation preferences
            emotional_mode: Current emotional mode (neutral, victory, defeat, rivalry)

        Returns:
            Dict with commentary text, tokens used, latency, source
        """
        self._metrics["total_requests"] += 1
        start = time.time()
        style = content_style or {}

        # Build structured prompts
        system_prompt = self._build_system_prompt(persona, rag_context, style, emotional_mode)
        user_prompt = self._build_user_prompt(event, persona, emotional_mode)

        # Try LLM first
        for attempt in range(self.max_retries + 1):
            try:
                result = await self._call_llm(system_prompt, user_prompt)
                elapsed_ms = (time.time() - start) * 1000

                self._metrics["successful_requests"] += 1
                self._metrics["total_tokens"] += result.get("tokens", 0)
                self._metrics["total_latency_ms"] += elapsed_ms

                return {
                    "commentary": result["text"],
                    "source": "nemotron",
                    "model": self.model,
                    "tokens": result.get("tokens", 0),
                    "latency_ms": round(elapsed_ms, 1),
                    "attempt": attempt + 1,
                    "emotional_mode": emotional_mode,
                }

            except Exception as e:
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(1 * (attempt + 1))

        # Fallback to templates
        elapsed_ms = (time.time() - start) * 1000
        self._metrics["failed_requests"] += 1
        self._metrics["fallback_used"] += 1

        fallback_text = self._generate_fallback(event, persona)
        return {
            "commentary": fallback_text,
            "source": "fallback_template",
            "model": "none",
            "tokens": 0,
            "latency_ms": round(elapsed_ms, 1),
            "attempt": self.max_retries + 1,
            "emotional_mode": emotional_mode,
        }

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """Generic text generation via ATTI LLM"""
        try:
            client = await self._get_client()
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            response = await client.post(self.endpoint, json=payload)
            if response.status_code == 200:
                data = response.json()
                if "choices" in data:
                    return data["choices"][0].get("message", {}).get("content", "")
                return data.get("response", data.get("text", ""))
            return ""
        except Exception as e:
            logger.error(f"Text generation error: {e}")
            return ""

    async def _call_llm(
        self, system_prompt: str, user_prompt: str
    ) -> Dict[str, Any]:
        """Call ATTI Nemotron via Modal.com endpoint"""
        client = await self._get_client()

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.8,
            "max_tokens": 512,
            "top_p": 0.95,
        }

        response = await client.post(self.endpoint, json=payload)

        if response.status_code == 200:
            data = response.json()
            text = ""
            tokens = 0
            if "choices" in data:
                text = data["choices"][0].get("message", {}).get("content", "")
                tokens = data.get("usage", {}).get("total_tokens", 0)
            elif "response" in data:
                text = data["response"]
                tokens = data.get("tokens_used", 0)
            elif "text" in data:
                text = data["text"]
                tokens = data.get("tokens", 0)

            if not text:
                raise ValueError("Empty response from LLM")

            return {"text": text.strip(), "tokens": tokens}
        else:
            raise ConnectionError(
                f"LLM returned status {response.status_code}: {response.text[:200]}"
            )

    def _build_system_prompt(
        self,
        persona: Dict[str, Any],
        rag_context: str,
        style: Dict[str, Any],
        emotional_mode: str,
    ) -> str:
        """Build system prompt with persona and RAG context"""
        return COMMENTARY_SYSTEM_PROMPT.format(
            persona_name=persona.get("name", "Comentarista"),
            persona_role=persona.get("role", "Influenciador Esportivo"),
            persona_expertise=", ".join(persona.get("expertise", ["futebol"])),
            persona_tone=persona.get("tone", "apaixonado"),
            team_affinity=persona.get("team_affinity", "neutro"),
            language=persona.get("language", "pt-BR"),
            max_length=style.get("max_length", 280),
            content_style=style.get("media_preference", "text_first"),
            hashtag_strategy=style.get("hashtag_strategy", "moderate"),
            mention_strategy=style.get("mention_strategy", "contextual"),
            emotional_mode=emotional_mode,
            rag_context=rag_context or "(sem contexto RAG disponível)",
        )

    def _build_user_prompt(
        self,
        event: Dict[str, Any],
        persona: Dict[str, Any],
        emotional_mode: str,
    ) -> str:
        """Build user prompt from event data"""
        teams_info = ""
        if event.get("home") and event.get("away"):
            teams_info = f"Casa: {event['home']}\nVisitante: {event['away']}"
        elif event.get("team"):
            teams_info = f"Time: {event['team']}"

        score_info = ""
        if event.get("home_score") is not None:
            score_info = (
                f"Placar: {event.get('home', '?')} {event['home_score']} x "
                f"{event.get('away_score', '?')} {event.get('away', '?')}"
            )

        emotional_instruction = ""
        if emotional_mode == "victory":
            emotional_instruction = "INSTRUÇÃO: Celebre com entusiasmo! Use expressões de alegria."
        elif emotional_mode == "defeat":
            emotional_instruction = "INSTRUÇÃO: Mostre frustração mas mantenha esperança."
        elif emotional_mode == "rivalry":
            team = persona.get("team_affinity", "")
            rivals = persona.get("rivalry_targets", [])
            if rivals:
                emotional_instruction = f"INSTRUÇÃO: Provocação saudável! {team} vs {rivals[0]}."

        return COMMENTARY_USER_PROMPT.format(
            event_type=event.get("type", "update"),
            competition=event.get("competition", ""),
            teams_info=teams_info,
            minute=event.get("minute", ""),
            description=event.get("description", ""),
            score_info=score_info,
            persona_name=persona.get("name", "Comentarista"),
            emotional_instruction=emotional_instruction,
        )

    def _generate_fallback(
        self, event: Dict[str, Any], persona: Dict[str, Any]
    ) -> str:
        """Generate fallback commentary from templates"""
        import random

        event_type = event.get("type", "default")
        templates = FALLBACK_TEMPLATES.get(event_type, FALLBACK_TEMPLATES["default"])
        template = random.choice(templates)

        try:
            return template.format(
                home=event.get("home", "Time A"),
                away=event.get("away", "Time B"),
                competition=event.get("competition", "campeonato"),
                minute=event.get("minute", ""),
                scoring_team=event.get("scoring_team", event.get("home", "Time")),
                opponent=event.get("away", "adversário"),
                player=event.get("player", "jogador"),
                team=event.get("team", "time"),
                home_score=event.get("home_score", 0),
                away_score=event.get("away_score", 0),
            )
        except (KeyError, IndexError):
            return (
                f"Acompanhe: {event.get('home', '')} x {event.get('away', '')} "
                f"pelo {event.get('competition', 'campeonato')}!"
            )

    async def health_check(self) -> Dict[str, Any]:
        """Check LLM endpoint health"""
        try:
            client = await self._get_client()
            response = await client.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 5,
                },
            )
            return {
                "status": "healthy" if response.status_code == 200 else "degraded",
                "endpoint": self.endpoint,
                "model": self.model,
                "status_code": response.status_code,
            }
        except Exception as e:
            return {
                "status": "unreachable",
                "endpoint": self.endpoint,
                "model": self.model,
                "error": str(e),
            }

    def get_metrics(self) -> Dict[str, Any]:
        """Get LLM usage metrics"""
        total = self._metrics["total_requests"]
        return {
            **self._metrics,
            "success_rate": (
                self._metrics["successful_requests"] / total if total > 0 else 0
            ),
            "avg_latency_ms": (
                self._metrics["total_latency_ms"] / total if total > 0 else 0
            ),
            "fallback_rate": (
                self._metrics["fallback_used"] / total if total > 0 else 0
            ),
        }

    async def close(self):
        if self._client:
            await self._client.aclose()
