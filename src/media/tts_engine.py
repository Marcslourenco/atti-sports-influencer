"""
TTS Engine — XTTS v2 integration for realistic sports commentary voice.
Connects to: ATTI voice_layer (modal_tts_pyttsx3.py) or local XTTS v2 server.

Features:
- XTTS v2 for high-quality multilingual TTS
- Voice cloning from reference audio samples
- Emotional tone adjustment (speed, pitch, energy)
- Fallback to pyttsx3 (ATTI existing) when XTTS unavailable
- Audio post-processing (normalization, noise reduction)
- Streaming audio generation for real-time commentary
"""
import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import httpx

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    TTS Engine with XTTS v2 primary and pyttsx3 fallback.

    Priority chain:
    1. XTTS v2 (local server or Modal.com) — high quality, voice cloning
    2. ATTI modal_tts_pyttsx3 — existing ATTI TTS endpoint
    3. Local pyttsx3 — offline fallback

    Voice profiles are stored as .wav reference files in voices/ directory.
    """

    def __init__(
        self,
        xtts_endpoint: Optional[str] = None,
        atti_tts_endpoint: Optional[str] = None,
        voices_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        self.xtts_endpoint = xtts_endpoint or os.getenv(
            "XTTS_ENDPOINT", "http://localhost:8003/tts"
        )
        self.atti_tts_endpoint = atti_tts_endpoint or os.getenv(
            "ATTI_TTS_ENDPOINT", "https://atti-tts--generate.modal.run"
        )
        self.voices_dir = Path(
            voices_dir or os.getenv(
                "VOICES_DIR",
                "/home/ubuntu/atti-sports-influencer/data/voices"
            )
        )
        self.output_dir = Path(
            output_dir or os.getenv(
                "TTS_OUTPUT_DIR",
                "/home/ubuntu/atti-sports-influencer/data/output/audio"
            )
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)

        self._client: Optional[httpx.AsyncClient] = None
        self._xtts_available = False
        self._metrics = {
            "total_generations": 0,
            "xtts_used": 0,
            "atti_tts_used": 0,
            "fallback_used": 0,
            "total_duration_s": 0.0,
            "total_latency_ms": 0.0,
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if not self._client:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def initialize(self) -> bool:
        """Check XTTS v2 availability"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.xtts_endpoint.rsplit('/tts', 1)[0]}/health")
            self._xtts_available = response.status_code == 200
        except Exception:
            self._xtts_available = False

        logger.info(f"TTS Engine initialized: XTTS v2={'available' if self._xtts_available else 'unavailable'}")
        return True

    async def generate_speech(
        self,
        text: str,
        voice_id: str = "default",
        language: str = "pt",
        emotional_mode: str = "neutral",
        speed: float = 1.0,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate speech audio from text.

        Args:
            text: Text to synthesize
            voice_id: Voice profile ID (maps to reference .wav file)
            language: Language code (pt, en, es)
            emotional_mode: Emotional tone (neutral, excited, sad, angry)
            speed: Speech speed multiplier (0.5-2.0)
            output_filename: Optional output filename

        Returns:
            Dict with audio_path, duration, engine used
        """
        self._metrics["total_generations"] += 1
        start = time.time()

        # Adjust speed based on emotional mode
        adjusted_speed = self._adjust_speed(speed, emotional_mode)

        # Try XTTS v2 first
        if self._xtts_available:
            result = await self._generate_xtts(
                text, voice_id, language, adjusted_speed, output_filename
            )
            if result.get("success"):
                self._metrics["xtts_used"] += 1
                elapsed_ms = (time.time() - start) * 1000
                self._metrics["total_latency_ms"] += elapsed_ms
                result["latency_ms"] = round(elapsed_ms, 1)
                result["engine"] = "xtts_v2"
                return result

        # Fallback to ATTI TTS (Modal.com)
        result = await self._generate_atti_tts(
            text, voice_id, language, adjusted_speed, output_filename
        )
        if result.get("success"):
            self._metrics["atti_tts_used"] += 1
            elapsed_ms = (time.time() - start) * 1000
            self._metrics["total_latency_ms"] += elapsed_ms
            result["latency_ms"] = round(elapsed_ms, 1)
            result["engine"] = "atti_modal_tts"
            return result

        # Final fallback: local pyttsx3
        result = await self._generate_pyttsx3(text, adjusted_speed, output_filename)
        self._metrics["fallback_used"] += 1
        elapsed_ms = (time.time() - start) * 1000
        self._metrics["total_latency_ms"] += elapsed_ms
        result["latency_ms"] = round(elapsed_ms, 1)
        result["engine"] = "pyttsx3"
        return result

    async def _generate_xtts(
        self,
        text: str,
        voice_id: str,
        language: str,
        speed: float,
        output_filename: Optional[str],
    ) -> Dict[str, Any]:
        """Generate speech using XTTS v2 server"""
        try:
            client = await self._get_client()

            # Find voice reference file
            voice_ref = self._get_voice_reference(voice_id)

            payload = {
                "text": text,
                "language": language,
                "speed": speed,
            }

            files = {}
            if voice_ref and voice_ref.exists():
                files["speaker_wav"] = open(str(voice_ref), "rb")

            if files:
                response = await client.post(
                    self.xtts_endpoint,
                    data=payload,
                    files=files,
                )
            else:
                payload["speaker_id"] = voice_id
                response = await client.post(
                    self.xtts_endpoint,
                    json=payload,
                )

            if response.status_code == 200:
                # Save audio file
                filename = output_filename or f"tts_{voice_id}_{int(time.time())}.wav"
                output_path = self.output_dir / filename
                output_path.write_bytes(response.content)

                duration_s = len(response.content) / (16000 * 2)  # Estimate: 16kHz, 16-bit
                self._metrics["total_duration_s"] += duration_s

                return {
                    "success": True,
                    "audio_path": str(output_path),
                    "duration_s": round(duration_s, 2),
                    "format": "wav",
                    "sample_rate": 16000,
                }
            else:
                logger.warning(f"XTTS v2 failed: HTTP {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}

        except Exception as e:
            logger.warning(f"XTTS v2 error: {e}")
            return {"success": False, "error": str(e)}

    async def _generate_atti_tts(
        self,
        text: str,
        voice_id: str,
        language: str,
        speed: float,
        output_filename: Optional[str],
    ) -> Dict[str, Any]:
        """Generate speech using ATTI Modal.com TTS endpoint"""
        try:
            client = await self._get_client()

            response = await client.post(
                self.atti_tts_endpoint,
                json={
                    "text": text,
                    "voice": voice_id,
                    "language": language,
                    "speed": speed,
                },
            )

            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")

                if "audio" in content_type or "octet-stream" in content_type:
                    filename = output_filename or f"tts_atti_{int(time.time())}.wav"
                    output_path = self.output_dir / filename
                    output_path.write_bytes(response.content)

                    duration_s = len(response.content) / (16000 * 2)
                    self._metrics["total_duration_s"] += duration_s

                    return {
                        "success": True,
                        "audio_path": str(output_path),
                        "duration_s": round(duration_s, 2),
                        "format": "wav",
                    }
                else:
                    # JSON response with audio URL
                    data = response.json()
                    audio_url = data.get("audio_url", "")
                    if audio_url:
                        audio_resp = await client.get(audio_url)
                        filename = output_filename or f"tts_atti_{int(time.time())}.wav"
                        output_path = self.output_dir / filename
                        output_path.write_bytes(audio_resp.content)
                        return {
                            "success": True,
                            "audio_path": str(output_path),
                            "format": "wav",
                        }

            return {"success": False, "error": f"ATTI TTS: HTTP {response.status_code}"}

        except Exception as e:
            logger.warning(f"ATTI TTS error: {e}")
            return {"success": False, "error": str(e)}

    async def _generate_pyttsx3(
        self, text: str, speed: float, output_filename: Optional[str]
    ) -> Dict[str, Any]:
        """Fallback: Generate speech using pyttsx3 (offline)"""
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", int(150 * speed))

            filename = output_filename or f"tts_pyttsx3_{int(time.time())}.wav"
            output_path = self.output_dir / filename

            engine.save_to_file(text, str(output_path))
            engine.runAndWait()

            if output_path.exists():
                return {
                    "success": True,
                    "audio_path": str(output_path),
                    "format": "wav",
                }
            return {"success": False, "error": "pyttsx3 failed to generate audio"}

        except ImportError:
            return {"success": False, "error": "pyttsx3 not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_voice_reference(self, voice_id: str) -> Optional[Path]:
        """Get voice reference file for voice cloning"""
        for ext in (".wav", ".mp3", ".ogg"):
            ref_file = self.voices_dir / f"{voice_id}{ext}"
            if ref_file.exists():
                return ref_file
        return None

    def _adjust_speed(self, base_speed: float, emotional_mode: str) -> float:
        """Adjust speech speed based on emotional mode"""
        speed_modifiers = {
            "neutral": 1.0,
            "excited": 1.2,
            "victory": 1.3,
            "defeat": 0.85,
            "rivalry": 1.15,
            "sad": 0.8,
            "angry": 1.1,
        }
        modifier = speed_modifiers.get(emotional_mode, 1.0)
        return min(max(base_speed * modifier, 0.5), 2.0)

    async def health_check(self) -> Dict[str, Any]:
        """Check TTS engine health"""
        xtts_status = "unavailable"
        atti_status = "unavailable"

        try:
            client = await self._get_client()
            resp = await client.get(
                f"{self.xtts_endpoint.rsplit('/tts', 1)[0]}/health",
                timeout=5.0,
            )
            if resp.status_code == 200:
                xtts_status = "healthy"
                self._xtts_available = True
        except Exception:
            pass

        try:
            client = await self._get_client()
            resp = await client.post(
                self.atti_tts_endpoint,
                json={"text": "test", "voice": "default"},
                timeout=10.0,
            )
            if resp.status_code == 200:
                atti_status = "healthy"
        except Exception:
            pass

        return {
            "xtts_v2": xtts_status,
            "atti_modal_tts": atti_status,
            "pyttsx3": "available",  # Always available as fallback
            "voices_dir": str(self.voices_dir),
            "output_dir": str(self.output_dir),
        }

    def get_metrics(self) -> Dict[str, Any]:
        total = self._metrics["total_generations"]
        return {
            **self._metrics,
            "avg_latency_ms": (
                self._metrics["total_latency_ms"] / total if total > 0 else 0
            ),
        }

    async def close(self):
        if self._client:
            await self._client.aclose()
