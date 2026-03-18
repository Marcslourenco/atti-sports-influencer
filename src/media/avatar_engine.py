"""
Avatar Engine — LivePortrait integration for facial animation.
Connects to: ATTI atti_avatar_engine (lip_sync, persona_layer).

Features:
- LivePortrait for facial animation from audio
- Lip sync from TTS audio output
- Avatar image management (per-persona)
- Video composition (avatar + overlay graphics)
- Fallback to static image with audio overlay
- Output in MP4 format for social media
"""
import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional

import httpx

logger = logging.getLogger(__name__)


class AvatarEngine:
    """
    Avatar Engine with LivePortrait primary and static fallback.

    Pipeline:
    1. Load avatar image for persona
    2. Generate facial animation from audio (LivePortrait)
    3. Compose final video with overlays
    4. Export as MP4

    Fallback:
    1. Load avatar image
    2. Create static video with audio overlay (FFmpeg)
    3. Export as MP4
    """

    def __init__(
        self,
        liveportrait_endpoint: Optional[str] = None,
        atti_avatar_endpoint: Optional[str] = None,
        avatars_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        self.liveportrait_endpoint = liveportrait_endpoint or os.getenv(
            "LIVEPORTRAIT_ENDPOINT", "http://localhost:8004/animate"
        )
        self.atti_avatar_endpoint = atti_avatar_endpoint or os.getenv(
            "ATTI_AVATAR_ENDPOINT",
            "https://atti-avatar--generate.modal.run"
        )
        self.avatars_dir = Path(
            avatars_dir or os.getenv(
                "AVATARS_DIR",
                "/home/ubuntu/atti-sports-influencer/data/avatars"
            )
        )
        self.output_dir = Path(
            output_dir or os.getenv(
                "AVATAR_OUTPUT_DIR",
                "/home/ubuntu/atti-sports-influencer/data/output/video"
            )
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.avatars_dir.mkdir(parents=True, exist_ok=True)

        self._client: Optional[httpx.AsyncClient] = None
        self._liveportrait_available = False
        self._metrics = {
            "total_generations": 0,
            "liveportrait_used": 0,
            "atti_avatar_used": 0,
            "fallback_used": 0,
            "total_duration_s": 0.0,
            "total_latency_ms": 0.0,
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if not self._client:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def initialize(self) -> bool:
        """Check LivePortrait availability"""
        try:
            client = await self._get_client()
            base = self.liveportrait_endpoint.rsplit("/animate", 1)[0]
            response = await client.get(f"{base}/health", timeout=5.0)
            self._liveportrait_available = response.status_code == 200
        except Exception:
            self._liveportrait_available = False

        logger.info(
            f"Avatar Engine initialized: "
            f"LivePortrait={'available' if self._liveportrait_available else 'unavailable'}"
        )
        return True

    async def generate_avatar_video(
        self,
        audio_path: str,
        persona_id: str,
        emotional_mode: str = "neutral",
        overlay_text: Optional[str] = None,
        output_filename: Optional[str] = None,
        duration_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate avatar video from audio.

        Args:
            audio_path: Path to TTS audio file
            persona_id: Persona ID (maps to avatar image)
            emotional_mode: Emotional mode for expression
            overlay_text: Optional text overlay (score, team names)
            output_filename: Optional output filename
            duration_s: Optional duration override

        Returns:
            Dict with video_path, duration, engine used
        """
        self._metrics["total_generations"] += 1
        start = time.time()

        # Get avatar image
        avatar_image = self._get_avatar_image(persona_id)
        if not avatar_image:
            logger.warning(f"No avatar image for persona {persona_id}, using default")
            avatar_image = self._get_default_avatar()

        # Try LivePortrait first
        if self._liveportrait_available and avatar_image:
            result = await self._generate_liveportrait(
                audio_path, str(avatar_image), emotional_mode, output_filename
            )
            if result.get("success"):
                self._metrics["liveportrait_used"] += 1
                elapsed_ms = (time.time() - start) * 1000
                self._metrics["total_latency_ms"] += elapsed_ms
                result["latency_ms"] = round(elapsed_ms, 1)
                result["engine"] = "liveportrait"

                # Add overlay if requested
                if overlay_text and result.get("video_path"):
                    result["video_path"] = await self._add_overlay(
                        result["video_path"], overlay_text
                    )

                return result

        # Try ATTI Avatar Engine
        if avatar_image:
            result = await self._generate_atti_avatar(
                audio_path, str(avatar_image), persona_id, output_filename
            )
            if result.get("success"):
                self._metrics["atti_avatar_used"] += 1
                elapsed_ms = (time.time() - start) * 1000
                self._metrics["total_latency_ms"] += elapsed_ms
                result["latency_ms"] = round(elapsed_ms, 1)
                result["engine"] = "atti_avatar"
                return result

        # Fallback: static image + audio
        result = await self._generate_static_video(
            audio_path, str(avatar_image) if avatar_image else None,
            overlay_text, output_filename, duration_s
        )
        self._metrics["fallback_used"] += 1
        elapsed_ms = (time.time() - start) * 1000
        self._metrics["total_latency_ms"] += elapsed_ms
        result["latency_ms"] = round(elapsed_ms, 1)
        result["engine"] = "ffmpeg_static"
        return result

    async def _generate_liveportrait(
        self,
        audio_path: str,
        avatar_path: str,
        emotional_mode: str,
        output_filename: Optional[str],
    ) -> Dict[str, Any]:
        """Generate animated avatar using LivePortrait"""
        try:
            client = await self._get_client()

            files = {
                "source_image": open(avatar_path, "rb"),
                "driving_audio": open(audio_path, "rb"),
            }
            data = {
                "expression_mode": emotional_mode,
                "output_format": "mp4",
            }

            response = await client.post(
                self.liveportrait_endpoint,
                data=data,
                files=files,
            )

            if response.status_code == 200:
                filename = output_filename or f"avatar_lp_{int(time.time())}.mp4"
                output_path = self.output_dir / filename
                output_path.write_bytes(response.content)

                return {
                    "success": True,
                    "video_path": str(output_path),
                    "format": "mp4",
                }
            else:
                return {"success": False, "error": f"LivePortrait: HTTP {response.status_code}"}

        except Exception as e:
            logger.warning(f"LivePortrait error: {e}")
            return {"success": False, "error": str(e)}

    async def _generate_atti_avatar(
        self,
        audio_path: str,
        avatar_path: str,
        persona_id: str,
        output_filename: Optional[str],
    ) -> Dict[str, Any]:
        """Generate avatar using ATTI Avatar Engine (Modal.com)"""
        try:
            client = await self._get_client()

            files = {
                "avatar_image": open(avatar_path, "rb"),
                "audio": open(audio_path, "rb"),
            }
            data = {
                "persona_id": persona_id,
                "output_format": "mp4",
            }

            response = await client.post(
                self.atti_avatar_endpoint,
                data=data,
                files=files,
            )

            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")

                if "video" in content_type or "octet-stream" in content_type:
                    filename = output_filename or f"avatar_atti_{int(time.time())}.mp4"
                    output_path = self.output_dir / filename
                    output_path.write_bytes(response.content)
                    return {
                        "success": True,
                        "video_path": str(output_path),
                        "format": "mp4",
                    }
                else:
                    data = response.json()
                    video_url = data.get("video_url", "")
                    if video_url:
                        video_resp = await client.get(video_url)
                        filename = output_filename or f"avatar_atti_{int(time.time())}.mp4"
                        output_path = self.output_dir / filename
                        output_path.write_bytes(video_resp.content)
                        return {
                            "success": True,
                            "video_path": str(output_path),
                            "format": "mp4",
                        }

            return {"success": False, "error": f"ATTI Avatar: HTTP {response.status_code}"}

        except Exception as e:
            logger.warning(f"ATTI Avatar error: {e}")
            return {"success": False, "error": str(e)}

    async def _generate_static_video(
        self,
        audio_path: str,
        avatar_path: Optional[str],
        overlay_text: Optional[str],
        output_filename: Optional[str],
        duration_s: Optional[float],
    ) -> Dict[str, Any]:
        """Fallback: Create static video from image + audio using FFmpeg"""
        try:
            filename = output_filename or f"avatar_static_{int(time.time())}.mp4"
            output_path = self.output_dir / filename

            if avatar_path and Path(avatar_path).exists():
                # Static image + audio
                cmd = [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-i", avatar_path,
                    "-i", audio_path,
                    "-c:v", "libx264",
                    "-tune", "stillimage",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-pix_fmt", "yuv420p",
                    "-shortest",
                    "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
                ]

                if overlay_text:
                    # Add text overlay
                    escaped = overlay_text.replace("'", "\\'").replace(":", "\\:")
                    cmd.extend([
                        "-vf",
                        f"scale=1080:1920:force_original_aspect_ratio=decrease,"
                        f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
                        f"drawtext=text='{escaped}':fontsize=48:fontcolor=white:"
                        f"x=(w-text_w)/2:y=h-100:box=1:boxcolor=black@0.7:boxborderw=10",
                    ])

                cmd.append(str(output_path))

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await asyncio.wait_for(process.communicate(), timeout=60)

                if output_path.exists():
                    return {
                        "success": True,
                        "video_path": str(output_path),
                        "format": "mp4",
                    }
                else:
                    return {"success": False, "error": f"FFmpeg failed: {stderr.decode()[:200]}"}
            else:
                # Audio only — create black video with audio
                dur = duration_s or 30
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", f"color=c=black:s=1080x1920:d={dur}",
                    "-i", audio_path,
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-shortest",
                    str(output_path),
                ]

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(process.communicate(), timeout=60)

                if output_path.exists():
                    return {
                        "success": True,
                        "video_path": str(output_path),
                        "format": "mp4",
                    }
                return {"success": False, "error": "FFmpeg black video failed"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _add_overlay(self, video_path: str, text: str) -> str:
        """Add text overlay to existing video"""
        try:
            output_path = video_path.replace(".mp4", "_overlay.mp4")
            escaped = text.replace("'", "\\'").replace(":", "\\:")

            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf",
                f"drawtext=text='{escaped}':fontsize=36:fontcolor=white:"
                f"x=(w-text_w)/2:y=h-80:box=1:boxcolor=black@0.7:boxborderw=8",
                "-c:a", "copy",
                output_path,
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(process.communicate(), timeout=60)

            return output_path if Path(output_path).exists() else video_path
        except Exception:
            return video_path

    def _get_avatar_image(self, persona_id: str) -> Optional[Path]:
        """Get avatar image for a persona"""
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            img = self.avatars_dir / f"{persona_id}{ext}"
            if img.exists():
                return img
        return None

    def _get_default_avatar(self) -> Optional[Path]:
        """Get default avatar image"""
        for ext in (".png", ".jpg", ".jpeg"):
            default = self.avatars_dir / f"default{ext}"
            if default.exists():
                return default
        return None

    async def health_check(self) -> Dict[str, Any]:
        """Check avatar engine health"""
        lp_status = "unavailable"
        atti_status = "unavailable"

        try:
            client = await self._get_client()
            base = self.liveportrait_endpoint.rsplit("/animate", 1)[0]
            resp = await client.get(f"{base}/health", timeout=5.0)
            if resp.status_code == 200:
                lp_status = "healthy"
                self._liveportrait_available = True
        except Exception:
            pass

        # Check FFmpeg
        ffmpeg_status = "unavailable"
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            if proc.returncode == 0:
                ffmpeg_status = "available"
        except Exception:
            pass

        return {
            "liveportrait": lp_status,
            "atti_avatar": atti_status,
            "ffmpeg": ffmpeg_status,
            "avatars_dir": str(self.avatars_dir),
            "output_dir": str(self.output_dir),
            "avatar_count": len(list(self.avatars_dir.glob("*.*"))),
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
