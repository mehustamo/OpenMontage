"""Video generation through OpenRouter's async video API.

``POST /api/v1/videos`` starts a real paid job and returns a ``polling_url``;
``GET /api/v1/videos/{id}`` reports progress and ``GET .../content?index=0``
streams the finished MP4. Jobs are checkpointed to ``job_path`` and resumed with
``resume_job`` — a poll timeout must never trigger a second paid submission.

Model capabilities (durations, resolutions, aspect ratios, audio, frame images)
come from ``GET /api/v1/videos/models``; the checked-in snapshot lives in
``tools/provider_contracts/openrouter_media.json``.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tools import openrouter_client as client
from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    ResumeSupport,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)
from tools.openrouter_models import (
    DEFAULT_VIDEO_MODEL,
    estimate_video_cost,
    video_model,
    video_models,
)
from tools.provider_jobs import poll, save_job
from tools.provider_pricing import PriceQuoteRequired

_TERMINAL_FAILURES = {"failed", "cancelled", "canceled", "expired"}


def coerce_duration(value: Any) -> Optional[int]:
    """The video selector passes duration as a string ("5"); OpenRouter wants an int."""
    if value is None or value == "":
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"duration must be a whole number of seconds, got {value!r}") from exc


class OpenRouterVideo(BaseTool):
    name = "openrouter_video"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "openrouter"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.ASYNC
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
        "generate_video",
        "text_to_video",
        "image_to_video",
        "reference_to_video",
    ]
    supports = {
        "text_to_video": True,
        "image_to_video": True,
        "reference_to_video": True,
        "first_frame": True,
        "last_frame": True,
        "native_audio": True,
        "seed": True,
        "async_job": True,
        "resumable": True,
    }
    best_for = [
        "single-key text/image-to-video across Veo, Seedance, Kling, Wan and Grok",
        "jobs that report their own polling state and resume after a timeout",
        "native-audio clips on routes that advertise generate_audio",
    ]
    not_good_for = [
        "offline generation",
        "token-priced routes needing an exact pre-flight price",
    ]

    _MODELS = video_models()

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "description": "What the clip should show"},
            "model": {
                "type": "string",
                "default": DEFAULT_VIDEO_MODEL,
                "description": (
                    "OpenRouter video model id, e.g. x-ai/grok-imagine-video-1.5-lite, "
                    "google/veo-3.1-lite, bytedance/seedance-2.0-fast, kwaivgi/kling-v3.0-std"
                ),
            },
            "operation": {
                "type": "string",
                "enum": ["text_to_video", "image_to_video", "reference_to_video"],
                "default": "text_to_video",
            },
            "duration": {
                "type": ["integer", "string"],
                "minimum": 1,
                "description": "Clip length in seconds; the video selector passes it as a string",
            },
            "resolution": {"type": "string", "description": "e.g. 480p, 720p, 1080p, 4K"},
            "aspect_ratio": {"type": "string", "description": "e.g. 16:9, 9:16, 1:1"},
            "size": {"type": "string", "description": "Exact pixels, e.g. 1280x720"},
            "generate_audio": {"type": "boolean", "description": "Ask routes that support it for native audio"},
            "seed": {"type": "integer"},
            "first_frame_path": {"type": "string"},
            "first_frame_url": {"type": "string"},
            "last_frame_path": {"type": "string"},
            "last_frame_url": {"type": "string"},
            "reference_image_path": {"type": "string"},
            "reference_image_paths": {"type": "array", "items": {"type": "string"}},
            "reference_image_url": {"type": "string"},
            "reference_image_urls": {"type": "array", "items": {"type": "string"}},
            "reference_audio_url": {"type": "string"},
            "reference_audio_urls": {"type": "array", "items": {"type": "string"}},
            "reference_video_url": {"type": "string"},
            "previous_job_id": {
                "type": "string",
                "description": "Completed job to edit or extend on routes that support continuation",
            },
            "callback_url": {
                "type": "string",
                "description": "Optional HTTPS webhook for completion instead of polling",
            },
            "output_path": {"type": "string"},
            "job_path": {
                "type": "string",
                "description": "Where to checkpoint the submitted job so a timeout can be resumed",
            },
            "resume_job": {"type": "object", "description": "Job dict from a previous call; polls without resubmitting"},
            "poll_timeout": {"type": "number", "default": 900},
            "poll_interval": {"type": "number", "default": 10},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=500, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=0)
    resume_support = ResumeSupport.FROM_CHECKPOINT
    fallback_tools = ["video_selector"]

    idempotency_key_fields = [
        "prompt",
        "model",
        "operation",
        "duration",
        "resolution",
        "aspect_ratio",
        "generate_audio",
        "first_frame_path",
    ]
    side_effects = [
        "submits a paid OpenRouter video job",
        "writes the generated clip to output_path",
    ]
    user_visible_verification = [
        "Play the generated clip and check motion, framing and audio"
    ]

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if client.api_key() else ToolStatus.UNAVAILABLE

    # ---- Cost ----

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        model = inputs.get("model") or DEFAULT_VIDEO_MODEL
        row = video_model(model)
        if row is None:
            raise PriceQuoteRequired(f"{model} is not in the OpenRouter video catalog")
        duration = coerce_duration(inputs.get("duration"))
        if duration is None:
            supported = row.get("supported_durations") or []
            duration = min(supported) if supported else None
        estimate = estimate_video_cost(
            row,
            duration=duration,
            resolution=inputs.get("resolution"),
            generate_audio=inputs.get("generate_audio"),
        )
        if estimate is None:
            raise PriceQuoteRequired(
                f"{model} advertises token-based pricing; OpenRouter reports the billed "
                "cost on the completed job instead"
            )
        return estimate

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 60.0 + float(inputs.get("duration") or 5) * 15.0

    # ---- Request ----

    def validate_model_request(self, model: str, payload: dict[str, Any]) -> None:
        """Reject a request the selected model cannot serve (OpenRouter returns 400 too)."""
        row = video_model(model)
        if row is None:
            raise ValueError(
                f"{model} is not in the OpenRouter video catalog; "
                "refresh with scripts/openrouter_models_refresh.py"
            )
        durations = row.get("supported_durations")
        if durations and payload.get("duration") and payload["duration"] not in durations:
            raise ValueError(
                f"{model} supports durations {sorted(durations)}, not {payload['duration']}"
            )
        resolutions = row.get("supported_resolutions")
        if resolutions and payload.get("resolution") and payload["resolution"] not in resolutions:
            raise ValueError(
                f"{model} supports resolutions {resolutions}, not {payload['resolution']}"
            )
        ratios = row.get("supported_aspect_ratios")
        if ratios and payload.get("aspect_ratio") and payload["aspect_ratio"] not in ratios:
            raise ValueError(
                f"{model} supports aspect ratios {ratios}, not {payload['aspect_ratio']}"
            )

    def build_request(self, inputs: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        prompt = (inputs.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt is required")
        model = inputs.get("model") or DEFAULT_VIDEO_MODEL
        operation = inputs.get("operation") or "text_to_video"
        if operation not in ("text_to_video", "image_to_video", "reference_to_video"):
            raise ValueError(f"unsupported operation {operation!r}")

        payload: dict[str, Any] = {"model": model, "prompt": prompt}
        duration = coerce_duration(inputs.get("duration"))
        if duration is not None:
            payload["duration"] = duration
        for field in ("resolution", "aspect_ratio", "size", "seed", "generate_audio", "previous_job_id", "callback_url"):
            if inputs.get(field) is not None:
                payload[field] = inputs[field]

        frames = self._frame_images(inputs)
        if frames:
            payload["frame_images"] = frames
        references = self._references(inputs)
        if references:
            payload["input_references"] = references

        if operation == "image_to_video" and not (frames or references):
            raise ValueError(
                "image_to_video needs a first_frame_*, reference_image_* or previous_job_id input"
            )
        self.validate_model_request(model, payload)
        return model, operation, payload

    @staticmethod
    def _frame_images(inputs: dict[str, Any]) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        for frame_type, keys in (
            ("first_frame", ("first_frame_path", "first_frame_url", "image_path", "image_url")),
            ("last_frame", ("last_frame_path", "last_frame_url")),
        ):
            value = next((inputs.get(key) for key in keys if inputs.get(key)), None)
            if value:
                frames.append({**client.image_reference(value), "frame_type": frame_type})
        return frames

    @staticmethod
    def _references(inputs: dict[str, Any]) -> list[dict[str, Any]]:
        references: list[dict[str, Any]] = []
        images = [
            inputs[key]
            for key in ("reference_image_path", "reference_image_url")
            if inputs.get(key)
        ]
        for key in ("reference_image_paths", "reference_image_urls"):
            images.extend(inputs.get(key) or [])
        references.extend(client.image_reference(value) for value in images)
        audios = [
            inputs[key]
            for key in ("reference_audio_url",)
            if inputs.get(key)
        ]
        audios.extend(inputs.get("reference_audio_urls") or [])
        references.extend(
            {"type": "audio_url", "audio_url": {"url": value}} for value in audios
        )
        videos = [inputs[key] for key in ("reference_video_url",) if inputs.get(key)]
        references.extend(
            {"type": "video_url", "video_url": {"url": value}} for value in videos
        )
        return references

    # ---- Execution ----

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if not client.api_key():
            return ToolResult(
                success=False,
                error="OPENROUTER_API_KEY is not configured. " + self.install_instructions,
            )
        job = dict(inputs.get("resume_job") or {})
        try:
            if not job:
                model, operation, payload = self.build_request(inputs)
                submitted = client.request_json(
                    "POST", "/videos", payload=payload, timeout=300
                )
                job = {
                    "id": submitted.get("id"),
                    "polling_url": submitted.get("polling_url") or f"/videos/{submitted.get('id')}",
                    "status": submitted.get("status"),
                    "model": model,
                    "operation": operation,
                    "tool": self.name,
                    "submitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                if not job["id"]:
                    raise ValueError("OpenRouter returned no video job id")
                save_job(inputs.get("job_path"), job)
            if job.get("tool") != self.name:
                raise ValueError("Resume job belongs to a different tool")
            if not job.get("id"):
                raise ValueError("Resume job has no id")

            timeout = float(inputs.get("poll_timeout") or 900)
            interval = float(inputs.get("poll_interval") or 10)
            try:
                final = poll(
                    lambda: client.request_json(
                        "GET", f"/videos/{job['id']}", timeout=60
                    ),
                    timeout=timeout,
                    interval=interval,
                )
            except TimeoutError as exc:
                job["status"] = "pending"
                save_job(inputs.get("job_path"), job)
                return ToolResult(
                    success=False,
                    error=(
                        f"Video job {job['id']} is still pending after {timeout:.0f}s: {exc}. "
                        "Resume with resume_job — do not submit the request again."
                    ),
                    data={"resume_job": job, "job_id": job["id"], "cost_status": "pending"},
                )
            except RuntimeError as exc:
                save_job(inputs.get("job_path"), job)
                return ToolResult(
                    success=False,
                    error=str(exc),
                    data={"resume_job": job, "job_id": job["id"]},
                )

            status = str(final.get("status") or "").lower()
            if status in _TERMINAL_FAILURES:
                return ToolResult(
                    success=False,
                    error=f"OpenRouter video job {job['id']} ended as {status}: {final.get('error') or 'no detail'}",
                    data={"resume_job": job, "job_id": job["id"]},
                )

            urls = list(final.get("unsigned_urls") or final.get("urls") or [])
            if not urls:
                raise ValueError("Completed video job returned no output URL")
            requested = Path(inputs.get("output_path") or f"{self.name}.mp4")
            outputs = [
                client.download(
                    url,
                    requested if index == 0 else requested.with_name(f"{requested.stem}_{index + 1}{requested.suffix}"),
                )
                for index, url in enumerate(urls)
            ]
            cost = client.usage_cost(final)
            job["status"] = "completed"
            save_job(inputs.get("job_path"), job)
            return ToolResult(
                success=True,
                data={
                    "provider": self.provider,
                    "hosting_provider": "openrouter.ai",
                    "model": job["model"],
                    "operation": job["operation"],
                    "duration": final.get("duration"),
                    "resolution": final.get("resolution"),
                    "output": outputs[0],
                    "outputs": outputs,
                    "job": final,
                    "job_id": job["id"],
                    "resume_job": job,
                    "cost_status": "billed" if cost is not None else "unquoted",
                },
                artifacts=outputs,
                cost_usd=cost,
                model=job["model"],
            )
        except (ValueError, client.OpenRouterError) as exc:
            return ToolResult(
                success=False,
                error=str(exc),
                data={"resume_job": job} if job else {},
            )
