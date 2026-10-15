"""Image generation and editing through OpenRouter's dedicated Image API.

``POST /api/v1/images`` returns base64 image bytes plus the exact billed cost in
``usage.cost``, so a successful call reports a real price instead of an estimate.
Model discovery lives in ``GET /api/v1/images/models``; the checked-in snapshot
in ``tools/provider_contracts/openrouter_media.json`` feeds the selector-facing
catalog. Refresh it with ``scripts/openrouter_models_refresh.py``.
"""

from __future__ import annotations

import base64
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
from tools.openrouter_models import DEFAULT_IMAGE_MODEL, IMAGE_PRICES, image_models
from tools.provider_pricing import PriceQuoteRequired

_EXTENSIONS = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/svg+xml": "svg",
}
# Assumed megapixels per resolution tier for the few routes priced per megapixel.
_TIER_MEGAPIXELS = {"512": 0.26, "768": 0.59, "1K": 1.0, "1.5K": 2.25, "2K": 4.0, "4K": 16.0}


class OpenRouterImage(BaseTool):
    name = "openrouter_image"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
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
    agent_skills = ["openrouter-media", "ai-video-gen"]

    capabilities = [
        "generate_image",
        "edit_image",
        "text_to_image",
        "image_to_image",
        "multi_image",
    ]
    supports = {
        "negative_prompt": False,
        "seed": True,
        "custom_size": True,
        "image_to_image": True,
        "multiple_outputs": True,
        "transparent_background": True,
    }
    best_for = [
        "single-key image generation across many model families",
        "routes with a per-call exact price in the response",
        "editing an existing image with reference-image models",
    ]
    not_good_for = [
        "offline generation",
        "negative-prompt control (OpenRouter exposes none)",
    ]

    # Selector-facing catalog: filter_explicit_route keeps this tool only when the
    # requested model is listed here, so an unknown id cannot be silently swapped.
    _MODELS = image_models()

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "description": "Image description"},
            "model": {
                "type": "string",
                "default": DEFAULT_IMAGE_MODEL,
                "description": (
                    "OpenRouter image model id, e.g. black-forest-labs/flux.2-klein-4b, "
                    "google/gemini-3.1-flash-image, bytedance-seed/seedream-5-0-pro"
                ),
            },
            "generation_mode": {
                "type": "string",
                "enum": ["text_to_image", "image_to_image"],
                "default": "text_to_image",
            },
            "aspect_ratio": {
                "type": "string",
                "description": "Normalized ratio: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3 or auto",
            },
            "resolution": {
                "type": "string",
                "description": "Normalized tier: 512, 768, 1K, 1.5K, 2K, 4K",
            },
            "size": {
                "type": "string",
                "description": "Explicit pixels, e.g. 2048x2048; authoritative over resolution/aspect_ratio",
            },
            "width": {
                "type": "integer",
                "description": "Pixel width; combined with height into an explicit size",
            },
            "height": {
                "type": "integer",
                "description": "Pixel height; combined with width into an explicit size",
            },
            "quality": {"type": "string", "enum": ["auto", "low", "medium", "high"]},
            "background": {"type": "string", "enum": ["auto", "transparent", "opaque"]},
            "output_format": {"type": "string", "enum": ["png", "jpeg", "webp", "svg"]},
            "output_compression": {"type": "integer", "minimum": 0, "maximum": 100},
            "n": {"type": "integer", "minimum": 1, "maximum": 10},
            "seed": {"type": "integer"},
            "image_path": {"type": "string", "description": "Reference image for image_to_image"},
            "image_paths": {"type": "array", "items": {"type": "string"}},
            "image_url": {"type": "string"},
            "image_urls": {"type": "array", "items": {"type": "string"}},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=100, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = [
        "prompt",
        "model",
        "aspect_ratio",
        "resolution",
        "size",
        "width",
        "height",
        "n",
        "seed",
        "generation_mode",
        "image_path",
        "image_paths",
    ]
    side_effects = ["writes image file(s) to output_path", "calls the OpenRouter Image API"]
    user_visible_verification = [
        "Open the generated image and check subject, framing and any rendered text"
    ]

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if client.api_key() else ToolStatus.UNAVAILABLE

    # ---- Cost ----

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        model = inputs.get("model") or DEFAULT_IMAGE_MODEL
        table = IMAGE_PRICES.get(model)
        if not table:
            raise PriceQuoteRequired(
                f"No verified OpenRouter image price for {model}; "
                "the response reports the billed cost in usage.cost"
            )
        count = float(inputs.get("n") or 1)
        unit = table.get("unit")
        if unit == "megapixel":
            megapixels = self._assumed_megapixels(inputs)
            return round(table["cost_usd"] * megapixels * count, 4)
        return round(table["cost_usd"] * count, 4)

    @staticmethod
    def _assumed_megapixels(inputs: dict[str, Any]) -> float:
        size = str(inputs.get("size") or "")
        if "x" in size:
            try:
                width, height = (float(part) for part in size.lower().split("x", 1))
                return max(0.01, width * height / 1_000_000)
            except ValueError:
                pass
        if inputs.get("width") and inputs.get("height"):
            return max(0.01, float(inputs["width"]) * float(inputs["height"]) / 1_000_000)
        return _TIER_MEGAPIXELS.get(str(inputs.get("resolution") or "1K").upper(), 1.0)

    # ---- Request ----

    def build_request(self, inputs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        prompt = (inputs.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt is required")
        model = inputs.get("model") or DEFAULT_IMAGE_MODEL
        if not model:
            raise ValueError("model is required")
        payload: dict[str, Any] = {"model": model, "prompt": prompt}
        for field in (
            "aspect_ratio",
            "resolution",
            "size",
            "quality",
            "background",
            "output_format",
            "output_compression",
            "n",
            "seed",
        ):
            if inputs.get(field) is not None:
                payload[field] = inputs[field]

        # The selector exposes pixel width/height; OpenRouter takes one explicit size.
        if not payload.get("size") and inputs.get("width") and inputs.get("height"):
            payload["size"] = f"{int(inputs['width'])}x{int(inputs['height'])}"

        references = self._references(inputs)
        if references:
            payload["input_references"] = references
        elif inputs.get("generation_mode") == "image_to_image":
            raise ValueError("image_to_image requires image_path/image_url references")
        return model, payload

    @staticmethod
    def _references(inputs: dict[str, Any]) -> list[dict[str, Any]]:
        values: list[str] = []
        for key in ("image_path", "image_url"):
            if inputs.get(key):
                values.append(inputs[key])
        for key in ("image_paths", "image_urls"):
            values.extend(inputs.get(key) or [])
        return [client.image_reference(value) for value in values]

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if not client.api_key():
            return ToolResult(success=False, error="OPENROUTER_API_KEY is not configured. " + self.install_instructions)
        try:
            model, payload = self.build_request(inputs)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))

        start = time.time()
        try:
            response = client.request_json("POST", "/images", payload=payload, timeout=600)
        except client.OpenRouterError as exc:
            return ToolResult(success=False, error=f"OpenRouter image request failed: {exc}")

        entries = response.get("data") or []
        if not entries:
            return ToolResult(success=False, error="OpenRouter returned no image data")

        media_type = entries[0].get("media_type") or "image/png"
        extension = _EXTENSIONS.get(media_type, "png")
        requested = Path(inputs.get("output_path") or f"{self.name}.{extension}")
        outputs: list[str] = []
        try:
            for index, entry in enumerate(entries):
                encoded = entry.get("b64_json")
                if not encoded:
                    raise ValueError("image entry carried no b64_json")
                target = (
                    requested
                    if index == 0
                    else requested.with_name(f"{requested.stem}_{index + 1}{requested.suffix or '.' + extension}")
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(base64.b64decode(encoded))
                outputs.append(str(target))
        except (ValueError, OSError) as exc:
            return ToolResult(success=False, error=f"Could not write image output: {exc}")

        cost = client.usage_cost(response)
        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "hosting_provider": "openrouter.ai",
                "model": model,
                "output": outputs[0],
                "outputs": outputs,
                "media_type": media_type,
                "count": len(outputs),
                "generation_mode": inputs.get("generation_mode") or "text_to_image",
                "aspect_ratio": payload.get("aspect_ratio"),
                "resolution": payload.get("resolution"),
                "request": {
                    key: value
                    for key, value in payload.items()
                    if key not in ("prompt", "input_references")
                },
                "reference_count": len(payload.get("input_references") or []),
                "usage": response.get("usage"),
                "cost_status": "billed" if cost is not None else "unquoted",
            },
            artifacts=outputs,
            cost_usd=cost,
            duration_seconds=round(time.time() - start, 2),
            seed=inputs.get("seed"),
            model=model,
        )
