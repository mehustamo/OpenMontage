#!/usr/bin/env python3
"""Refresh the OpenRouter media model catalog used by the OpenRouter tools.

OpenRouter publishes generated-media catalogs at:

    GET /api/v1/images/models   — image models + supported parameters
    GET /api/v1/videos/models   — video models + durations, resolutions, pricing SKUs

Speech has no model-list endpoint, so ``tools/openrouter_models.SPEECH_MODELS``
stays hand-maintained and is only annotated by this script.

The snapshot lands in ``tools/provider_contracts/openrouter_media.json`` and is
read back by ``tools/openrouter_models`` at import time (no network at runtime).
Run with OPENROUTER_API_KEY set:

    .venv/bin/python scripts/openrouter_models_refresh.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "tools" / "provider_contracts" / "openrouter_media.json"
BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")

# Fields worth persisting: everything a selector or a cost estimate needs.
IMAGE_KEYS = ("id", "name", "created", "description", "architecture", "supported_parameters", "supports_streaming")
VIDEO_KEYS = (
    "id",
    "canonical_slug",
    "name",
    "created",
    "description",
    "generate_audio",
    "pricing_skus",
    "supported_durations",
    "supported_resolutions",
    "supported_aspect_ratios",
    "supported_frame_images",
    "supported_sizes",
    "allowed_passthrough_parameters",
)


def fetch(path: str, key: str) -> list[dict]:
    import urllib.request

    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read())
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError(f"{path} returned no data list")
    return data


def project(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    return [{key: row.get(key) for key in keys if key in row} for row in rows]


def main() -> int:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2

    images = fetch("/images/models", key)
    videos = fetch("/videos/models", key)

    snapshot = {
        "provider": "openrouter",
        "source": {
            "images": f"{BASE_URL}/images/models",
            "videos": f"{BASE_URL}/videos/models",
            "speech": "hand-maintained; OpenRouter exposes no TTS model list",
        },
        "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "image_models": project(images, IMAGE_KEYS),
        "video_models": project(videos, VIDEO_KEYS),
    }

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    temporary = TARGET.with_name(TARGET.name + ".part")
    temporary.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(TARGET)
    print(f"wrote {TARGET.relative_to(REPO_ROOT)}: {len(images)} image, {len(videos)} video models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
