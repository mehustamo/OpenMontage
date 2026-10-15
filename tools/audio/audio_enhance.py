"""Audio enhancement tool for noise reduction and cleanup.

Provides noise reduction, normalization, and EQ via FFmpeg audio
filters. Optional pedalboard integration for higher-quality
processing when available.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    ToolResult,
    ToolStability,
    ToolTier,
)


# Codec implied by the output container. The tool defaulted to AAC for every
# extension, so an ``output_path`` ending in .wav produced an AAC stream wearing a
# WAV name: unreadable by audio_mixer, Remotion and any other consumer that trusts
# the extension.
_EXTENSION_CODECS = {
    ".wav": "pcm_s16le",
    ".wave": "pcm_s16le",
    ".mp3": "libmp3lame",
    ".flac": "flac",
    ".m4a": "aac",
    ".aac": "aac",
    ".opus": "libopus",
    ".ogg": "libvorbis",
    ".wma": "wmav2",
}
_LOSSLESS_CODECS = ("pcm_", "flac", "alac", "wavpack")


def codec_for_output(output_path: str | Path) -> str:
    """Pick the audio codec from the output extension (AAC when unknown)."""
    return _EXTENSION_CODECS.get(Path(output_path).suffix.lower(), "aac")


def is_lossless_codec(codec: str) -> bool:
    """Whether ``-b:a`` is meaningless for this codec."""
    return codec.startswith(_LOSSLESS_CODECS)


PRESETS = {
    "clean_speech": {
        "description": "Noise gate + highpass + compressor + limiter for clean dialogue",
        "af": (
            "highpass=f=80,"
            "lowpass=f=13000,"
            "agate=threshold=0.01:ratio=2:attack=5:release=50,"
            "acompressor=threshold=-20dB:ratio=3:attack=5:release=100,"
            "loudnorm=I=-16:LRA=11:TP=-1.5"
        ),
    },
    "noise_reduce": {
        "description": "Aggressive noise reduction for noisy environments",
        "af": (
            "afftdn=nf=-25:nt=w,"
            "highpass=f=100,"
            "loudnorm=I=-16:LRA=11:TP=-1.5"
        ),
    },
    "normalize_only": {
        "description": "Loudness normalization without other processing",
        "af": "loudnorm=I=-16:LRA=11:TP=-1.5",
    },
    "podcast": {
        "description": "Podcast-style processing: de-ess, compress, normalize",
        "af": (
            "highpass=f=80,"
            "acompressor=threshold=-18dB:ratio=4:attack=5:release=100:makeup=2,"
            "loudnorm=I=-16:LRA=7:TP=-1.5"
        ),
    },
    "broadcast": {
        "description": "Broadcast-standard processing with tight dynamics",
        "af": (
            "highpass=f=80,"
            "lowpass=f=15000,"
            "acompressor=threshold=-24dB:ratio=4:attack=5:release=80:makeup=3,"
            "alimiter=limit=0.95:attack=1:release=10,"
            "loudnorm=I=-24:LRA=7:TP=-2"
        ),
    },
    "voice_clarity": {
        "description": "Boost vocal presence with EQ and light compression",
        "af": (
            "highpass=f=80,"
            "equalizer=f=200:t=q:w=1.5:g=-3,"
            "equalizer=f=3000:t=q:w=1.0:g=3,"
            "equalizer=f=5000:t=q:w=1.5:g=2,"
            "acompressor=threshold=-20dB:ratio=2.5:attack=10:release=100,"
            "loudnorm=I=-16:LRA=11:TP=-1.5"
        ),
    },
}


class AudioEnhance(BaseTool):
    name = "audio_enhance"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "audio_processing"
    provider = "ffmpeg"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = ["cmd:ffmpeg"]
    install_instructions = "Install FFmpeg: https://ffmpeg.org/download.html"
    agent_skills = ["ffmpeg", "elevenlabs"]

    capabilities = [
        "noise_reduction",
        "normalization",
        "compression",
        "eq",
        "speech_cleanup",
    ]

    input_schema = {
        "type": "object",
        "required": ["input_path"],
        "properties": {
            "input_path": {"type": "string"},
            "output_path": {"type": "string"},
            "preset": {
                "type": "string",
                "enum": list(PRESETS.keys()),
                "default": "clean_speech",
            },
            "custom_af": {
                "type": "string",
                "description": "Custom FFmpeg audio filter string",
            },
            "audio_codec": {
                "type": "string",
                "description": (
                    "Audio codec; omitted means it is derived from the output "
                    "extension (.wav -> pcm_s16le, .mp3 -> libmp3lame, .flac -> flac, "
                    "else aac). An explicit value wins."
                ),
            },
            "audio_bitrate": {"type": "string", "default": "192k"},
            "sample_rate": {
                "type": "integer",
                "description": (
                    "Output sample rate. Omitted means 48000 for PCM/FLAC outputs "
                    "(loudnorm's own default is 192000, which bloats a lossless file "
                    "for no benefit) and the input rate otherwise."
                ),
            },
        },
    }

    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=500)
    idempotency_key_fields = ["input_path", "preset", "custom_af"]
    side_effects = ["writes enhanced audio/video to output_path"]
    user_visible_verification = [
        "Listen to enhanced audio and compare with original",
        "Verify speech is clear without artifacts or pumping",
    ]

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        input_path = Path(inputs["input_path"])
        if not input_path.exists():
            return ToolResult(success=False, error=f"Input not found: {input_path}")

        output_path = Path(
            inputs.get("output_path", str(input_path.with_stem(f"{input_path.stem}_enhanced")))
        )
        audio_codec = inputs.get("audio_codec") or codec_for_output(output_path)
        audio_bitrate = inputs.get("audio_bitrate", "192k")
        sample_rate = inputs.get("sample_rate")
        if sample_rate is None and is_lossless_codec(audio_codec):
            sample_rate = 48000

        af = inputs.get("custom_af")
        if not af:
            preset_name = inputs.get("preset", "clean_speech")
            preset = PRESETS.get(preset_name)
            if not preset:
                return ToolResult(success=False, error=f"Unknown preset: {preset_name}")
            af = preset["af"]

        start = time.time()

        # Determine if input is video or audio-only
        is_video = input_path.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm"}

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-af", af,
        ]
        if is_video:
            cmd.extend(["-c:v", "copy"])
        cmd.extend(["-c:a", audio_codec])
        if not is_lossless_codec(audio_codec):
            cmd.extend(["-b:a", audio_bitrate])
        if sample_rate:
            cmd.extend(["-ar", str(sample_rate)])
        cmd.append(str(output_path))

        try:
            self.run_command(cmd)
        except Exception as e:
            return ToolResult(success=False, error=f"FFmpeg failed: {e}")

        elapsed = time.time() - start

        return ToolResult(
            success=True,
            data={
                "input": str(input_path),
                "output": str(output_path),
                "preset": inputs.get("preset"),
                "filter": af,
                "audio_codec": audio_codec,
                "audio_bitrate": None if is_lossless_codec(audio_codec) else audio_bitrate,
                "sample_rate": sample_rate,
            },
            artifacts=[str(output_path)],
            duration_seconds=round(elapsed, 2),
        )

    @staticmethod
    def list_presets() -> dict[str, str]:
        """Return available presets and their descriptions."""
        return {name: p["description"] for name, p in PRESETS.items()}
