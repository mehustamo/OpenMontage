"""Contract tests for the OpenRouter media provider tools.

These tests verify the BaseTool contract and the OpenRouter-specific invariants
without a key and without any network call: identity, schemas, status reporting,
cost behavior, request shaping, job resumption and the checked-in catalogs.

Run: pytest tests/contracts/test_openrouter_tools.py -v
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from tools import openrouter_client
from tools.audio.openrouter_music import OpenRouterMusic
from tools.audio.openrouter_tts import OpenRouterTTS
from tools.base_tool import (
    BaseTool,
    ExecutionMode,
    ResumeSupport,
    ToolRuntime,
    ToolStatus,
    ToolTier,
)
from tools.graphics.openrouter_image import OpenRouterImage
from tools.openrouter_models import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_VIDEO_MODEL,
    catalog,
    estimate_video_cost,
    image_models,
    video_models,
)
from tools.provider_pricing import PriceQuoteRequired
from tools.tool_registry import registry
from tools.video.openrouter_video import OpenRouterVideo

TOOLS = [OpenRouterImage, OpenRouterVideo, OpenRouterTTS, OpenRouterMusic]
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EXPECTED_CAPABILITY = {
    OpenRouterImage: "image_generation",
    OpenRouterVideo: "video_generation",
    OpenRouterTTS: "tts",
    OpenRouterMusic: "music_generation",
}
EXPECTED_TIER = {
    OpenRouterImage: ToolTier.GENERATE,
    OpenRouterVideo: ToolTier.GENERATE,
    OpenRouterTTS: ToolTier.VOICE,
    OpenRouterMusic: ToolTier.VOICE,
}


@pytest.fixture(autouse=True)
def _no_key(monkeypatch):
    """No test may see a real key: status and execute paths are exercised explicitly."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)


# ------------------------------------------------------------------
# Contract compliance
# ------------------------------------------------------------------


@pytest.mark.parametrize("cls", TOOLS, ids=lambda c: c.name)
class TestContract:
    def test_inherits_base_tool(self, cls):
        assert issubclass(cls, BaseTool)

    def test_has_required_identity(self, cls):
        tool = cls()
        assert tool.name.startswith("openrouter_")
        assert tool.version
        assert tool.provider == "openrouter"

    def test_capability_and_tier(self, cls):
        tool = cls()
        assert tool.capability == EXPECTED_CAPABILITY[cls]
        assert tool.tier == EXPECTED_TIER[cls]

    def test_is_an_api_tool(self, cls):
        tool = cls()
        assert tool.runtime == ToolRuntime.API
        assert tool.resource_profile.network_required is True

    def test_declares_the_shared_api_key(self, cls):
        assert "env:OPENROUTER_API_KEY" in cls().dependencies

    def test_schemas_and_guidance_are_populated(self, cls):
        tool = cls()
        assert tool.input_schema.get("properties")
        assert "required" in tool.input_schema
        assert tool.capabilities
        assert tool.best_for
        assert tool.not_good_for
        assert tool.supports

    def test_output_path_is_documented(self, cls):
        assert "output_path" in cls().input_schema["properties"]

    def test_status_tracks_the_key(self, cls, monkeypatch):
        tool = cls()
        assert tool.get_status() == ToolStatus.UNAVAILABLE
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        assert tool.get_status() == ToolStatus.AVAILABLE

    def test_model_catalog_is_published_for_explicit_routing(self, cls):
        # filter_explicit_route drops a tool whose _MODELS lacks the requested id,
        # so every OpenRouter tool must advertise at least a default.
        assert cls._MODELS, f"{cls.name} publishes no model catalog"

    def test_agent_skill_pointer_resolves(self, cls):
        for skill in cls().agent_skills:
            assert (PROJECT_ROOT / ".agents" / "skills" / skill / "SKILL.md").exists()

    def test_execute_fails_cleanly_without_a_key(self, cls):
        result = cls().execute({})
        assert result.success is False
        assert "OPENROUTER_API_KEY" in (result.error or "")


# ------------------------------------------------------------------
# Client helpers
# ------------------------------------------------------------------


class TestClient:
    def test_usage_cost_reads_the_billed_amount(self):
        assert openrouter_client.usage_cost({"usage": {"cost": 0.014}}) == 0.014
        assert openrouter_client.usage_cost({"usage": {}}) is None
        assert openrouter_client.usage_cost({}) is None
        assert openrouter_client.usage_cost(None) is None

    def test_image_reference_inlines_a_local_file(self, tmp_path):
        image = tmp_path / "frame.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n payload")
        reference = openrouter_client.image_reference(str(image))
        assert reference["type"] == "image_url"
        url = reference["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == image.read_bytes()

    def test_image_reference_passes_urls_through(self):
        url = "https://example.com/frame.jpg"
        assert openrouter_client.image_reference(url)["image_url"]["url"] == url

    def test_image_reference_rejects_a_missing_file(self):
        with pytest.raises(openrouter_client.OpenRouterError):
            openrouter_client.image_reference("/nope/missing.png")

    def test_download_refuses_a_non_https_url(self, tmp_path):
        with pytest.raises(openrouter_client.OpenRouterError):
            openrouter_client.download("ftp://example.com/clip.mp4", tmp_path / "clip.mp4")

    def test_base_url_can_be_overridden(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://mirror.example.com/api/v1/")
        assert openrouter_client.api_url("/images") == "https://mirror.example.com/api/v1/images"


# ------------------------------------------------------------------
# Images
# ------------------------------------------------------------------


class TestImage:
    def test_default_model_is_in_the_catalog(self):
        assert DEFAULT_IMAGE_MODEL in image_models()
        assert DEFAULT_IMAGE_MODEL in OpenRouterImage._MODELS

    def test_cost_is_quoted_for_a_verified_route(self):
        assert OpenRouterImage().estimate_cost({"model": "black-forest-labs/flux.2-klein-4b"}) == 0.014

    def test_cost_scales_with_an_explicit_size(self):
        # 2048x2048 is 4.19 MP at $0.014/MP.
        cost = OpenRouterImage().estimate_cost({
            "model": "black-forest-labs/flux.2-klein-4b",
            "size": "2048x2048",
        })
        assert cost == pytest.approx(0.0587, rel=0.01)

    def test_unverified_model_raises_a_quote_instead_of_guessing(self):
        with pytest.raises(PriceQuoteRequired):
            OpenRouterImage().estimate_cost({"model": "google/gemini-3-pro-image"})

    def test_build_request_keeps_only_supported_fields(self):
        _, payload = OpenRouterImage().build_request({
            "prompt": "a lighthouse",
            "aspect_ratio": "16:9",
            "resolution": "1K",
            "negative_prompt": "ignored",
        })
        assert payload["aspect_ratio"] == "16:9"
        assert "negative_prompt" not in payload

    def test_image_to_image_requires_a_reference(self):
        with pytest.raises(ValueError):
            OpenRouterImage().build_request({"prompt": "edit", "generation_mode": "image_to_image"})

    def test_reference_images_become_input_references(self, tmp_path):
        image = tmp_path / "ref.png"
        image.write_bytes(b"png bytes")
        _, payload = OpenRouterImage().build_request({
            "prompt": "edit",
            "generation_mode": "image_to_image",
            "image_path": str(image),
        })
        assert payload["input_references"][0]["type"] == "image_url"

    def test_width_and_height_become_an_explicit_size(self):
        # image_selector exposes width/height rather than OpenRouter's single size field.
        _, payload = OpenRouterImage().build_request({
            "prompt": "a lighthouse",
            "width": 1280,
            "height": 720,
        })
        assert payload["size"] == "1280x720"
        assert OpenRouterImage().estimate_cost({
            "model": "black-forest-labs/flux.2-klein-4b",
            "width": 1024,
            "height": 1024,
        }) == pytest.approx(0.0147, rel=0.02)


# ------------------------------------------------------------------
# Video
# ------------------------------------------------------------------


class TestVideo:
    def test_default_model_is_in_the_catalog(self):
        assert DEFAULT_VIDEO_MODEL in video_models()
        assert DEFAULT_VIDEO_MODEL in OpenRouterVideo._MODELS

    def test_job_checkpointing_is_declared(self):
        tool = OpenRouterVideo()
        assert tool.resume_support == ResumeSupport.FROM_CHECKPOINT
        assert tool.execution_mode == ExecutionMode.ASYNC
        for field in ("job_path", "resume_job", "poll_timeout", "poll_interval"):
            assert field in tool.input_schema["properties"]

    def test_cost_estimate_uses_the_advertised_per_second_sku(self):
        estimate = OpenRouterVideo().estimate_cost({
            "model": "x-ai/grok-imagine-video-1.5-lite",
            "duration": 2,
            "resolution": "480p",
        })
        assert estimate == pytest.approx(0.04, rel=0.02)

    def test_audio_aware_sku_pricing_is_selected(self):
        row = video_models()["google/veo-3.1-lite"]
        assert estimate_video_cost(row, duration=4, resolution="720p", generate_audio=False) == pytest.approx(
            0.12, rel=0.02
        )
        assert estimate_video_cost(row, duration=4, resolution="720p", generate_audio=True) == pytest.approx(
            0.20, rel=0.02
        )

    def test_token_priced_models_raise_a_quote(self):
        with pytest.raises(PriceQuoteRequired):
            OpenRouterVideo().estimate_cost({"model": "bytedance/seedance-2.0-mini", "duration": 5})

    def test_unknown_model_fails_the_request_before_submitting(self):
        with pytest.raises(ValueError):
            OpenRouterVideo().build_request({"prompt": "x", "model": "not/a-model"})

    def test_unsupported_duration_is_rejected_locally(self):
        with pytest.raises(ValueError) as error:
            OpenRouterVideo().build_request({
                "prompt": "x",
                "model": "google/veo-3.1-lite",
                "duration": 7,
            })
        assert "durations" in str(error.value)

    def test_unsupported_resolution_is_rejected_locally(self):
        with pytest.raises(ValueError):
            OpenRouterVideo().build_request({
                "prompt": "x",
                "model": "minimax/hailuo-2.3",
                "resolution": "720p",
            })

    def test_duration_string_from_the_selector_is_accepted(self):
        # video_selector declares duration as a string ("5"), OpenRouter wants an int.
        _, _, payload = OpenRouterVideo().build_request({
            "prompt": "x",
            "model": "google/veo-3.1-lite",
            "duration": "4",
        })
        assert payload["duration"] == 4
        assert OpenRouterVideo().estimate_cost({
            "model": "google/veo-3.1-lite",
            "duration": "4",
            "resolution": "720p",
            "generate_audio": False,
        }) == pytest.approx(0.12, rel=0.02)

    def test_non_numeric_duration_is_rejected(self):
        with pytest.raises(ValueError):
            OpenRouterVideo().build_request({"prompt": "x", "duration": "four"})

    def test_image_to_video_needs_an_image(self):
        with pytest.raises(ValueError):
            OpenRouterVideo().build_request({"prompt": "x", "operation": "image_to_video"})

    def test_first_frame_becomes_a_frame_image(self, tmp_path):
        frame = tmp_path / "first.png"
        frame.write_bytes(b"png bytes")
        _, _, payload = OpenRouterVideo().build_request({
            "prompt": "x",
            "operation": "image_to_video",
            "first_frame_path": str(frame),
        })
        assert payload["frame_images"][0]["frame_type"] == "first_frame"

    def test_resume_job_from_another_tool_is_refused(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        result = OpenRouterVideo().execute({
            "prompt": "x",
            "resume_job": {"id": "job-1", "tool": "openrouter_image"},
        })
        assert result.success is False
        assert "different tool" in (result.error or "")


# ------------------------------------------------------------------
# Speech and music
# ------------------------------------------------------------------


class TestSpeech:
    def test_default_model_is_in_the_catalog(self):
        assert DEFAULT_TTS_MODEL in OpenRouterTTS._MODELS

    def test_a_voice_requiring_model_falls_back_to_the_catalog_sample(self):
        # The selector often calls without a voice; the catalog sample keeps that working.
        _, payload = OpenRouterTTS().build_request({"text": "hello", "model_id": DEFAULT_TTS_MODEL})
        assert payload["voice"] == "en_paul_neutral"

    def test_a_host_without_a_sample_voice_still_requires_one(self):
        with pytest.raises(ValueError):
            OpenRouterTTS().build_request({
                "text": "hello",
                "model_id": "elevenlabs/eleven-v4-turbo",
            })

    def test_request_shape_matches_the_endpoint(self):
        model, payload = OpenRouterTTS().build_request({
            "text": "hello",
            "voice": "en_paul_neutral",
            "response_format": "pcm",
        })
        assert model == DEFAULT_TTS_MODEL
        assert payload["input"] == "hello"
        assert payload["voice"] == "en_paul_neutral"
        assert payload["response_format"] == "pcm"

    def test_a_host_rejects_a_format_it_does_not_accept(self):
        # Voxtral takes mp3|pcm; wav is the one that used to slip through and 400.
        with pytest.raises(ValueError):
            OpenRouterTTS().build_request({"text": "hello", "response_format": "wav"})

    def test_response_format_is_always_sent(self):
        # Omitting it makes the endpoint stream raw PCM, which cannot be written
        # into an .mp3 file.
        _, payload = OpenRouterTTS().build_request({"text": "hello"})
        assert payload["response_format"] == "mp3"

    @pytest.mark.parametrize(
        "content_type,expected",
        [
            ("audio/mpeg", ("mp3", "mp3")),
            ("audio/pcm;rate=24000;channels=1", ("pcm", "pcm")),
            ("audio/wav", ("wav", "wav")),
            ("application/json", (None, None)),
            ("", (None, None)),
        ],
    )
    def test_delivered_format_follows_the_content_type(self, content_type, expected):
        from tools.audio.openrouter_tts import format_from_content_type

        assert format_from_content_type(content_type) == expected

    def test_gemini_tts_defaults_to_its_only_accepted_format(self):
        # Gemini TTS rejects mp3 with a 400; the catalog decides the default instead.
        model, payload = OpenRouterTTS().build_request({
            "text": "hei",
            "model_id": "google/gemini-3.8-flash-tts",
        })
        assert model == "google/gemini-3.8-flash-tts"
        assert payload["response_format"] == "pcm"

    def test_unsupported_format_for_a_host_fails_before_spending(self):
        with pytest.raises(ValueError) as error:
            OpenRouterTTS().build_request({
                "text": "hei",
                "model_id": "google/gemini-3.8-flash-tts",
                "response_format": "mp3",
            })
        assert "pcm" in str(error.value)

    def test_response_format_only_offers_verified_values(self):
        # Voxtral rejects wav/flac/opus/aac with a 400; mp3 and pcm are what works.
        assert OpenRouterTTS().input_schema["properties"]["response_format"]["enum"] == ["mp3", "pcm"]

    @pytest.mark.parametrize(
        "content_type,expected",
        [
            ("audio/pcm;rate=24000;channels=1", (24000, 1)),
            ("audio/pcm; rate=16000; channels=2", (16000, 2)),
            ("audio/pcm", (24000, 1)),
            ("", (24000, 1)),
            ("audio/pcm;rate=abc;channels=1", (24000, 1)),
        ],
    )
    def test_pcm_params_are_read_from_the_response(self, content_type, expected):
        from tools.audio.openrouter_tts import pcm_params

        assert pcm_params(content_type) == expected

    def test_pcm_is_wrapped_into_a_playable_wav(self):
        # Raw PCM has no header: downstream tools need a container, and mp3 would
        # be a needless first-generation lossy encode before the final mix.
        import wave
        from io import BytesIO

        from tools.audio.openrouter_tts import wrap_pcm_as_wav

        samples = b"\x00\x01" * 2400
        wrapped = wrap_pcm_as_wav(samples, rate=24000, channels=1)
        assert wrapped[:4] == b"RIFF" and wrapped[8:12] == b"WAVE"
        with wave.open(BytesIO(wrapped)) as handle:
            assert handle.getframerate() == 24000
            assert handle.getnchannels() == 1
            assert handle.readframes(handle.getnframes()) == samples

    def test_multi_speaker_turns_replace_text(self):
        _, payload = OpenRouterTTS().build_request({
            "voice": "en_paul_neutral",
            "turns": [
                {"text": "Hi Jane.", "voice": "A"},
                {"text": "Hi Paul.", "voice": "B"},
            ],
        })
        assert payload["input"][0]["voice"] == "A"
        assert payload["input"][1]["text"] == "Hi Paul."

    def test_price_is_reported_as_unquoted(self):
        with pytest.raises(PriceQuoteRequired):
            OpenRouterTTS().estimate_cost({"text": "hello"})


class TestMusic:
    def test_request_asks_for_streamed_audio_output(self):
        _, payload = OpenRouterMusic().build_request({
            "prompt": "warm ambient pad",
            "duration_seconds": 20,
        })
        assert payload["modalities"] == ["text", "audio"]
        assert payload["stream"] is True
        assert "20 seconds" in payload["messages"][0]["content"]

    def test_price_is_reported_as_unquoted(self):
        with pytest.raises(PriceQuoteRequired):
            OpenRouterMusic().estimate_cost({"prompt": "x"})

    def test_duration_is_reported_as_a_prompt_target_not_a_guarantee(self):
        schema = OpenRouterMusic().input_schema["properties"]
        assert "duration_seconds" in schema
        # The delivered length is reported back; a 75 s request returned ~31 s live.
        from tools.audio.openrouter_music import audio_duration_seconds

        assert audio_duration_seconds("/definitely/not/a/file.mp3") == 0.0


# ------------------------------------------------------------------
# Registry, selectors, catalogs, docs
# ------------------------------------------------------------------


class TestIntegration:
    def test_registry_discovers_all_four_tools(self):
        registry.discover()
        discovered = {name for name in registry.list_all() if name.startswith("openrouter_")}
        assert discovered == {"openrouter_image", "openrouter_video", "openrouter_tts", "openrouter_music"}

    def test_registry_reports_the_provider(self):
        registry.discover()
        assert {tool.name for tool in registry.get_by_provider("openrouter")} == {
            "openrouter_image",
            "openrouter_video",
            "openrouter_tts",
            "openrouter_music",
        }

    @pytest.mark.parametrize(
        "capability,expected",
        [
            ("image_generation", "openrouter_image"),
            ("video_generation", "openrouter_video"),
            ("tts", "openrouter_tts"),
            ("music_generation", "openrouter_music"),
        ],
    )
    def test_selectors_can_reach_the_tool_by_capability(self, capability, expected):
        registry.discover()
        names = {tool.name for tool in registry.get_by_capability(capability)}
        assert expected in names

    def test_checked_in_catalog_is_present_and_consistent(self):
        snapshot = catalog()
        assert snapshot.get("retrieved_at")
        assert image_models() and video_models()
        assert DEFAULT_IMAGE_MODEL in image_models()
        assert DEFAULT_VIDEO_MODEL in video_models()
        for entry in catalog()["video_models"]:
            assert entry.get("id")
            assert "pricing_skus" in entry

    def test_env_example_documents_the_key(self):
        text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        assert "OPENROUTER_API_KEY=" in text

    def test_skill_file_is_checked_in(self):
        skill = PROJECT_ROOT / ".agents" / "skills" / "openrouter-media" / "SKILL.md"
        body = skill.read_text(encoding="utf-8")
        assert body.startswith("---")
        assert "name: openrouter-media" in body

    def test_package_exposes_no_hardcoded_key(self):
        """No module under tools/ may embed a literal OpenRouter key."""
        for path in (PROJECT_ROOT / "tools").rglob("*.py"):
            assert "sk-or-v1-" not in path.read_text(encoding="utf-8"), path
