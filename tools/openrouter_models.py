"""OpenRouter media model catalog used by the OpenRouter generation tools.

``tools/provider_contracts/openrouter_media.json`` is a snapshot of the live
catalogs served by ``GET /api/v1/images/models`` and ``GET /api/v1/videos/models``.
Refresh it with ``scripts/openrouter_models_refresh.py`` (needs a key).

The catalogs are published so the capability selectors can honor an explicit
``model``: ``filter_explicit_route`` only keeps a tool whose ``_MODELS`` contains
the requested model, and exact ids must never be silently substituted.

Speech has no model-list endpoint on OpenRouter, so ``SPEECH_MODELS`` is
hand-maintained. Every entry below was verified against ``POST /api/v1/audio/speech``
on 2026-10-15; the voice catalog is provider-specific and required by some hosts.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

CONTRACT_PATH = Path(__file__).resolve().parent / "provider_contracts" / "openrouter_media.json"

# Cheap-first defaults: overridable per call with an explicit ``model``.
DEFAULT_IMAGE_MODEL = "black-forest-labs/flux.2-klein-4b"
DEFAULT_VIDEO_MODEL = "x-ai/grok-imagine-video-1.5-lite"
DEFAULT_TTS_MODEL = "mistralai/voxtral-mini-tts-2603"
DEFAULT_TTS_VOICE = "en_paul_neutral"
DEFAULT_MUSIC_MODEL = "google/lyria-3-clip-preview"

# Verified on 2026-10-15 against the live endpoint. ``requires_voice`` means a
# request without ``voice`` returns 400 ("An explicit voice is required").
SPEECH_MODELS: dict[str, dict[str, Any]] = {
    "mistralai/voxtral-mini-tts-2603": {
        "name": "Voxtral Mini TTS",
        "formats": ("mp3", "wav", "pcm", "flac", "opus"),
        "requires_voice": True,
        "sample_voice": "en_paul_neutral",
        "notes": "Fast, low-cost narration. Voice ids are Voxtral speaker ids.",
    },
    "fish-audio/s2-pro": {
        "name": "Fish Audio S2 Pro",
        "formats": ("mp3", "wav", "pcm"),
        "requires_voice": False,
        "sample_voice": "alloy",
        "notes": "Accepts a named voice or a cloned reference via input_references.",
    },
    "elevenlabs/eleven-v4-turbo": {
        "name": "ElevenLabs Eleven v4 Turbo",
        "formats": ("mp3", "wav", "pcm"),
        "requires_voice": True,
        "sample_voice": None,
        "notes": "Requires a real ElevenLabs voice id; unknown names are rejected.",
    },
}

MUSIC_MODELS: dict[str, dict[str, Any]] = {
    "google/lyria-3-clip-preview": {"name": "Lyria 3 Clip", "notes": "Short clips; duration is a prompt target."},
    "google/lyria-3-pro-preview": {"name": "Lyria 3 Pro", "notes": "Longer arrangements; duration is a prompt target."},
}

# Per-image prices confirmed against the model's endpoint pricing on 2026-10-15.
# Anything absent from this table is quoted as unknown rather than guessed.
IMAGE_PRICES: dict[str, dict[str, Any]] = {
    "black-forest-labs/flux.2-klein-4b": {"cost_usd": 0.014, "unit": "megapixel"},
    "bytedance-seed/seedream-5-0-lite": {"cost_usd": 0.035, "unit": "image"},
    "bytedance-seed/seedream-4.5": {"cost_usd": 0.05, "unit": "image"},
}


@lru_cache(maxsize=1)
def catalog() -> dict[str, Any]:
    """Load the checked-in snapshot; an empty catalog is not fatal."""
    try:
        with open(CONTRACT_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {"image_models": [], "video_models": [], "retrieved_at": None}


def _index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in rows if row.get("id")}


@lru_cache(maxsize=1)
def image_models() -> dict[str, dict[str, Any]]:
    return _index(catalog().get("image_models") or [])


@lru_cache(maxsize=1)
def video_models() -> dict[str, dict[str, Any]]:
    return _index(catalog().get("video_models") or [])


def model_ids(kind: str) -> dict[str, dict[str, Any]]:
    if kind == "image_generation":
        return image_models()
    if kind == "video_generation":
        return video_models()
    if kind == "tts":
        return SPEECH_MODELS
    if kind == "music_generation":
        return MUSIC_MODELS
    return {}


def video_model(model: str) -> Optional[dict[str, Any]]:
    return video_models().get(model)


def estimate_video_cost(
    model_row: Optional[dict[str, Any]],
    *,
    duration: Optional[float],
    resolution: Optional[str] = None,
    generate_audio: Optional[bool] = None,
) -> Optional[float]:
    """Estimate a video job in USD from the model's advertised pricing SKUs.

    OpenRouter advertises two shapes: cents-per-second keys
    (``cents_per_video_output_second_480p``) and dollar-per-second keys
    (``duration_seconds_with_audio_720p``). Token-priced models
    (``video_tokens``) cannot be estimated without a token count and return None
    so the caller reports an unquoted price instead of a guess.
    """
    if not model_row or not duration:
        return None
    skus = model_row.get("pricing_skus") or {}
    candidates: list[tuple[int, float]] = []
    for key, raw in skus.items():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        if key.startswith(("cents_per_video_output_second", "cents_per_second")):
            per_second = value / 100.0
        elif key.startswith("duration_seconds") or key.startswith("text_to_video_duration_seconds") or key.startswith("image_to_video_duration_seconds"):
            per_second = value
        else:
            continue
        score = 0
        if resolution and resolution.lower() in key:
            score += 2
        if generate_audio is not None:
            if "with_audio" in key:
                score += 3 if generate_audio else -2
            if "without_audio" in key:
                score += 3 if generate_audio is False else -2
        if "continuation" in key:
            score -= 5
        candidates.append((score, per_second))
    if not candidates:
        return None
    best = max(candidates, key=lambda item: item[0])[1]
    return round(best * float(duration), 4)
