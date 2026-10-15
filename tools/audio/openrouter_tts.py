"""Speech synthesis through OpenRouter's TTS endpoint.

``POST /api/v1/audio/speech`` returns a raw audio bytestream (no JSON envelope),
so the adapter writes the bytes and reports the byte count and detected format.
OpenRouter exposes no TTS model-list or price endpoint: the model catalog in
``tools/openrouter_models.SPEECH_MODELS`` is hand-verified, and pricing is
reported as unquoted rather than guessed (check the OpenRouter console for spend).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

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
from tools.openrouter_models import (
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    SPEECH_MODELS,
)
from tools.provider_pricing import PriceQuoteRequired

_FORMAT_EXTENSIONS = {
    "mp3": "mp3",
    "wav": "wav",
    "flac": "flac",
    "opus": "opus",
    "aac": "aac",
    "pcm": "pcm",
}

# What the endpoint actually returns, which is what the file extension must follow.
_CONTENT_TYPE_FORMATS = {
    "audio/mpeg": ("mp3", "mp3"),
    "audio/mp3": ("mp3", "mp3"),
    "audio/wav": ("wav", "wav"),
    "audio/x-wav": ("wav", "wav"),
    "audio/wave": ("wav", "wav"),
    "audio/flac": ("flac", "flac"),
    "audio/ogg": ("opus", "opus"),
    "audio/opus": ("opus", "opus"),
    "audio/aac": ("aac", "aac"),
    "audio/pcm": ("pcm", "pcm"),
}


def format_from_content_type(content_type: str) -> tuple[Optional[str], Optional[str]]:
    """Map a response content type to ``(format, extension)``; None when unknown."""
    return _CONTENT_TYPE_FORMATS.get((content_type or "").split(";")[0].strip().lower(), (None, None))


def pcm_params(content_type: str, *, default_rate: int = 24000, default_channels: int = 1) -> tuple[int, int]:
    """Read ``rate`` and ``channels`` from e.g. ``audio/pcm;rate=24000;channels=1``.

    Raw PCM carries no header, so these values are the only way to interpret the
    bytes; OpenRouter states them on the response, not in the request.
    """
    rate, channels = default_rate, default_channels
    for part in (content_type or "").split(";")[1:]:
        key, _, value = part.strip().partition("=")
        if key == "rate":
            try:
                rate = int(value)
            except ValueError:
                pass
        elif key == "channels":
            try:
                channels = int(value)
            except ValueError:
                pass
    return rate, channels


def wrap_pcm_as_wav(data: bytes, *, rate: int, channels: int) -> bytes:
    """Wrap headerless 16-bit little-endian PCM in a WAV container.

    Some hosts accept only ``mp3`` or ``pcm`` (Voxtral rejects ``wav``), and mp3
    would be a needless first-generation lossy encode before the final mix. Wrapped
    PCM is lossless, playable by ffmpeg/Remotion, and half the bytes of the raw
    stream being misinterpreted downstream.
    """
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data)
    return buffer.getvalue()


class OpenRouterTTS(BaseTool):
    name = "openrouter_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "openrouter"
    stability = ToolStability.BETA
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
    agent_skills = ["openrouter-media", "text-to-speech"]

    capabilities = [
        "text_to_speech",
        "voice_selection",
        "multi_speaker",
        "voice_cloning",
    ]
    supports = {
        "voice_cloning": True,
        "multi_speaker": True,
        "instructions": True,
        "speed": True,
        "offline": False,
        "native_audio": True,
    }
    best_for = [
        "narration on the same key as image and video generation",
        "provider-agnostic voice switching (Voxtral, Fish Audio, ElevenLabs)",
        "two-speaker dialogue in a single request",
    ]
    not_good_for = [
        "pre-flight price quotes (OpenRouter exposes none for speech)",
        "fully offline production",
    ]

    _MODELS = dict(SPEECH_MODELS)

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "description": "Text to synthesize"},
            "model_id": {
                "type": "string",
                "default": DEFAULT_TTS_MODEL,
                "description": "OpenRouter speech model id, e.g. mistralai/voxtral-mini-tts-2603",
            },
            "model": {"type": "string", "description": "Alias for model_id"},
            "voice": {
                "type": "string",
                "description": (
                    "Provider-specific voice. Required by some hosts "
                    f"({DEFAULT_TTS_MODEL} needs one; it defaults to "
                    f"{DEFAULT_TTS_VOICE} when omitted)."
                ),
            },
            "voice_id": {"type": "string", "description": "Alias for voice"},
            "response_format": {
                "type": "string",
                "enum": ["mp3", "pcm"],
                "default": "mp3",
                "description": (
                    "Verified values on this API. 'mp3' is delivery-ready; 'pcm' is "
                    "lossless (Voxtral rejects wav) and is returned wrapped in a WAV "
                    "container, which is what to use before mixing or mastering."
                ),
            },
            "speed": {"type": "number", "minimum": 0.25, "maximum": 4.0},
            "instructions": {
                "type": "string",
                "description": "Delivery guidance (tone, pacing, emotion) where the model supports it",
            },
            "turns": {
                "type": "array",
                "description": "Multi-speaker dialogue: [{text, voice, instructions}]",
                "items": {"type": "object"},
            },
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=50, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = [
        "text",
        "turns",
        "model_id",
        "model",
        "voice",
        "response_format",
        "speed",
        "instructions",
    ]
    side_effects = ["writes an audio file to output_path", "calls the OpenRouter speech endpoint"]
    user_visible_verification = ["Play the audio and check pronunciation and pacing"]

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if client.api_key() else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        raise PriceQuoteRequired(
            "OpenRouter does not publish speech pricing on the API; "
            "check the OpenRouter activity console for the billed amount"
        )

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        characters = len(str(inputs.get("text") or ""))
        return 2.0 + characters / 200.0

    # ---- Request ----

    def build_request(self, inputs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        turns = inputs.get("turns") or []
        text = (inputs.get("text") or "").strip()
        if not turns and not text:
            raise ValueError("text (or turns) is required")
        model = inputs.get("model_id") or inputs.get("model") or DEFAULT_TTS_MODEL
        voice = inputs.get("voice") or inputs.get("voice_id")
        known = SPEECH_MODELS.get(model)
        if known and known.get("requires_voice") and not voice:
            # Some hosts reject a voiceless request. Fall back to the catalog's
            # sample voice so a selector call without an explicit voice still
            # produces audio; the caller can override it any time.
            sample = known.get("sample_voice")
            if not sample:
                raise ValueError(
                    f"{model} requires an explicit voice and the catalog has no sample for it"
                )
            voice = sample
        payload: dict[str, Any] = {"model": model, "input": turns or text}
        if voice:
            payload["voice"] = voice
        # Always send the format: the endpoint defaults to raw PCM when this is
        # omitted, which cannot be written into a .mp3 file.
        payload["response_format"] = str(inputs.get("response_format") or "mp3").lower()
        for field in ("speed", "instructions"):
            if inputs.get(field) is not None:
                payload[field] = inputs[field]
        return model, payload

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

        requested_format = str(payload["response_format"])
        start = time.time()
        try:
            audio, content_type = client.request_media(
                "POST", "/audio/speech", payload=payload, timeout=600
            )
        except client.OpenRouterError as exc:
            return ToolResult(
                success=False, error=f"OpenRouter speech request failed: {exc}"
            )

        # The endpoint is authoritative about what it actually streamed.
        detected_format, detected_extension = format_from_content_type(content_type)
        delivered_format = detected_format or requested_format
        extension = detected_extension or _FORMAT_EXTENSIONS.get(requested_format, "mp3")
        wrapped = False
        pcm_rate: Optional[int] = None
        pcm_channels: Optional[int] = None
        if delivered_format == "pcm":
            # Headerless PCM is useless to ffmpeg/Remotion and easy to misread;
            # wrap it losslessly instead of handing over a raw stream.
            pcm_rate, pcm_channels = pcm_params(content_type)
            audio = wrap_pcm_as_wav(audio, rate=pcm_rate, channels=pcm_channels)
            extension = "wav"
            delivered_format = "wav"
            wrapped = True
        output_path = Path(inputs.get("output_path") or f"{self.name}.{extension}")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio)
        except OSError as exc:
            return ToolResult(success=False, error=f"Could not write audio output: {exc}")

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "hosting_provider": "openrouter.ai",
                "model": model,
                "voice": payload.get("voice"),
                "voice_source": (
                    "explicit"
                    if (inputs.get("voice") or inputs.get("voice_id"))
                    else ("catalog_default" if payload.get("voice") else None)
                ),
                "response_format": requested_format,
                "delivered_format": delivered_format,
                "content_type": content_type or None,
                "format_mismatch": bool(detected_format and detected_format != requested_format),
                "pcm_wrapped": wrapped,
                "sample_rate": pcm_rate,
                "channels": pcm_channels,
                "lossless": wrapped or requested_format in ("wav", "flac"),
                "text_length": len(str(payload["input"])),
                "turn_count": len(payload["input"]) if isinstance(payload["input"], list) else 0,
                "output": str(output_path),
                "bytes": len(audio),
                "head": audio[:4].hex(),
                # OpenRouter returns a raw stream with no usage envelope; the
                # console is the source of truth for what this call cost.
                "cost_status": "unquoted",
            },
            artifacts=[str(output_path)],
            cost_usd=None,
            duration_seconds=round(time.time() - start, 2),
            model=model,
        )
