"""Music generation through OpenRouter's audio-output chat models (Lyria 3).

OpenRouter exposes music models as chat models whose output modality includes
``audio``. Audio output requires streaming, and the bytes arrive base64-encoded
in ``choices[].delta.audio.data`` chunks that are concatenated here into one
file. ``google/lyria-3-clip-preview`` and ``google/lyria-3-pro-preview`` are the
routes advertised with audio output on 2026-10-15.

Duration is a prompt target, not a guaranteed length: state it in the prompt and
verify the rendered file with ffprobe.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

from tools import openrouter_client as client
from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)
from tools.openrouter_models import DEFAULT_MUSIC_MODEL, MUSIC_MODELS
from tools.provider_pricing import PriceQuoteRequired

_FORMATS = {"mp3": "mp3", "wav": "wav", "flac": "flac", "opus": "opus", "pcm16": "pcm"}


def audio_duration_seconds(path: str) -> float:
    """Measure what was actually written; 0.0 when ffprobe cannot say.

    Music duration is a prompt target, so the delivered length is the only fact
    worth reporting back to the caller.
    """
    import shutil
    import subprocess

    if not shutil.which("ffprobe"):
        return 0.0
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return float(out.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return 0.0


class OpenRouterMusic(BaseTool):
    name = "openrouter_music"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "music_generation"
    provider = "openrouter"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = ["env:OPENROUTER_API_KEY"]
    install_instructions = (
        "Set OPENROUTER_API_KEY in .env:\n"
        "  OPENROUTER_API_KEY=<your key>\n"
        "One key covers image, video, speech and music routes. "
        "Get it at https://openrouter.ai/keys"
    )
    agent_skills = ["openrouter-media"]

    capabilities = ["generate_music", "generate_instrumental", "soundtrack"]
    supports = {
        "instrumental": True,
        "vocals_in_prompt": True,
        "duration_hint": True,
        "offline": False,
    }
    best_for = [
        "quick instrumental beds on the same key as the rest of the pipeline",
        "underscore and stingers where an exact length is not contractual",
    ]
    not_good_for = [
        "exact-length music (duration is a prompt target)",
        "pre-flight price quotes",
    ]

    _MODELS = dict(MUSIC_MODELS)

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Style, instrumentation, mood and target duration",
            },
            "model": {"type": "string", "default": DEFAULT_MUSIC_MODEL},
            "duration_seconds": {"type": "integer", "minimum": 5, "maximum": 300},
            "format": {"type": "string", "enum": sorted(_FORMATS), "default": "mp3"},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=200, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["prompt", "model", "duration_seconds", "format"]
    side_effects = ["writes an audio file to output_path", "streams audio from OpenRouter"]
    user_visible_verification = ["Listen to the track and check mood, length and clipping"]

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if client.api_key() else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        raise PriceQuoteRequired(
            "OpenRouter does not publish music pricing on the API; the stream carries "
            "no usage envelope — check the OpenRouter activity console"
        )

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 30.0 + float(inputs.get("duration_seconds") or 30)

    def build_request(self, inputs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        prompt = (inputs.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt is required")
        model = inputs.get("model") or DEFAULT_MUSIC_MODEL
        duration = inputs.get("duration_seconds")
        if duration:
            prompt = f"{prompt}\n\nTarget duration: about {int(duration)} seconds."
        fmt = str(inputs.get("format") or "mp3").lower()
        return model, {
            "model": model,
            "modalities": ["text", "audio"],
            "audio": {"format": fmt},
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
        }

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if not client.api_key():
            return ToolResult(
                success=False,
                error="OPENROUTER_API_KEY is not configured. " + self.install_instructions,
            )
        try:
            model, payload = self.build_request(inputs)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))

        fmt = str(inputs.get("format") or "mp3").lower()
        extension = _FORMATS.get(fmt, "mp3")
        start = time.time()
        chunks: list[bytes] = []
        transcript: list[str] = []
        usage: dict[str, Any] | None = None
        try:
            response = client.request(
                "POST", "/chat/completions", payload=payload, timeout=900, stream=True
            )
            with response:
                for raw_line in response.iter_lines(decode_unicode=False):
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="replace")
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if not body or body == "[DONE]":
                        continue
                    try:
                        event = json.loads(body)
                    except ValueError:
                        continue
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        audio = delta.get("audio") or {}
                        if audio.get("data"):
                            chunks.append(base64.b64decode(audio["data"]))
                        if audio.get("transcript"):
                            transcript.append(str(audio["transcript"]))
        except client.OpenRouterError as exc:
            return ToolResult(success=False, error=f"OpenRouter music request failed: {exc}")
        except (ValueError, OSError) as exc:
            return ToolResult(success=False, error=f"OpenRouter music stream failed: {exc}")

        audio = b"".join(chunks)
        if not audio:
            return ToolResult(
                success=False,
                error="OpenRouter returned no audio data for the music request",
            )
        output_path = Path(inputs.get("output_path") or f"{self.name}.{extension}")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio)
        except OSError as exc:
            return ToolResult(success=False, error=f"Could not write music output: {exc}")

        cost = client.usage_cost({"usage": usage}) if usage else None
        delivered = audio_duration_seconds(str(output_path))
        requested = inputs.get("duration_seconds")
        shortfall = bool(
            requested and delivered and delivered < float(requested) * 0.9
        )
        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "hosting_provider": "openrouter.ai",
                "model": model,
                "format": fmt,
                "duration_seconds": requested,
                "delivered_seconds": round(delivered, 2) if delivered else None,
                # The clip model maxes out around 30 s whatever you ask for; say so
                # instead of letting the caller discover it when the bed runs out.
                "duration_shortfall": shortfall,
                "output": str(output_path),
                "bytes": len(audio),
                "chunks": len(chunks),
                "transcript": "".join(transcript).strip() or None,
                "usage": usage,
                "cost_status": "billed" if cost is not None else "unquoted",
            },
            artifacts=[str(output_path)],
            cost_usd=cost,
            duration_seconds=round(time.time() - start, 2),
            model=model,
        )
