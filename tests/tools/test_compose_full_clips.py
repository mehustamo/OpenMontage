"""Assembly mode: lossless concat of already-finished clips.

Why this mode exists: `in_seconds`/`out_seconds` are *source offsets* in the re-encode
compose path but *timeline positions* in the Remotion composition, so switching runtimes
silently changes the picture. A composition whose cuts are whole, already-final clips at the
target geometry needs no re-encode at all: concatenating them with `-c:v copy` is lossless,
cannot drift, and is ~150x faster than re-encoding every frame (measured: 4.4 s vs 1620 s for
a 37 s vertical video).

These tests exercise the real FFmpeg path and assert the properties that make the mode worth
having — plus the cases it must refuse rather than silently fall back on.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.video.video_compose import VideoCompose

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)


def _make_clip(path: Path, w: int = 1080, h: int = 1920, seconds: float = 1.0,
               color: str = "teal", fps: int = 30) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         f"color=c={color}:s={w}x{h}:d={seconds}:r={fps}",
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
         "-g", "30", "-keyint_min", "30", str(path)],
        capture_output=True, check=True,
    )


def _make_wav(path: Path, seconds: float) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         f"sine=frequency=440:duration={seconds}", "-ar", "48000", "-ac", "2", str(path)],
        capture_output=True, check=True,
    )


def _seconds(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)]
    ).decode().strip()
    return float(out)


def _streams(path: Path) -> set[str]:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(path)]
    ).decode()
    return {line.strip() for line in out.splitlines() if line.strip()}


def _frame_digest(path: Path, at: float) -> str:
    """Hash one decoded frame, so 'copied verbatim' can be asserted rather than assumed."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{at:.3f}", "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    return hashlib.sha256(raw).hexdigest()


def _decisions(clips: list[Path], **extra) -> dict:
    cuts = [
        {"id": f"cut_{i:02d}", "source": str(clip), "in_seconds": 0.0,
         "out_seconds": _seconds(clip), "speed": 1.0,
         "transition_in": "cut", "transition_out": "cut"}
        for i, clip in enumerate(clips, start=1)
    ]
    decisions = {"version": "1.0", "render_runtime": "ffmpeg", "renderer_family": "animation-first",
                 "composition_mode": "sequential", "cuts": cuts, "overlays": [],
                 "subtitles": {"enabled": False}}
    decisions.update(extra)
    return decisions


def _run(decisions: dict, out: Path, **inputs):
    return VideoCompose().execute({
        "operation": "compose", "edit_decisions": decisions,
        "profile": "youtube_shorts", "output_path": str(out),
        "options": {"subtitle_burn": False}, **inputs})


def test_full_clip_assembly_is_lossless_with_no_drift(tmp_path):
    first, second = tmp_path / "a.mp4", tmp_path / "b.mp4"
    _make_clip(first, seconds=1.0, color="teal")
    _make_clip(second, seconds=1.5, color="purple")
    audio = tmp_path / "master.wav"
    _make_wav(audio, seconds=5.0)          # deliberately longer: the picture must win

    out = tmp_path / "assembled.mp4"
    result = _run(_decisions([first, second], assembly_mode="full-clips"), out,
                  audio_path=str(audio))

    assert result.success, result.error
    assert result.data["assembly_mode"] == "full-clips"
    assert result.data["video_copied"] is True
    assert result.data["cut_count"] == 2

    expected = _seconds(first) + _seconds(second)
    assert abs(_seconds(out) - expected) <= 0.05, "duration drifted"
    assert abs(result.data["duration_drift_seconds"]) <= 0.05
    # frames are bit-identical to the sources: nothing was re-encoded
    assert _frame_digest(out, 0.5) == _frame_digest(first, 0.5)
    assert _frame_digest(out, 1.7) == _frame_digest(second, 0.7)
    assert _streams(out) == {"video", "audio"}


def test_audio_can_come_from_edit_decisions_like_the_remotion_path(tmp_path):
    clip = tmp_path / "a.mp4"
    _make_clip(clip, seconds=1.0)
    audio = tmp_path / "master.wav"
    _make_wav(audio, seconds=0.8)

    decisions = _decisions([clip], assembly_mode="full-clips",
                           audio={"narration": {"src": str(audio), "volume": 1.0}})
    out = tmp_path / "assembled.mp4"
    result = _run(decisions, out)

    assert result.success, result.error
    assert result.data["has_audio"] is True
    assert _streams(out) == {"video", "audio"}


def test_geometry_mismatch_is_a_hard_error_not_a_silent_fallback(tmp_path):
    clip = tmp_path / "landscape.mp4"
    _make_clip(clip, w=640, h=360, seconds=1.0)

    out = tmp_path / "assembled.mp4"
    result = _run(_decisions([clip], assembly_mode="full-clips"), out)

    assert result.success is False
    assert "640x360" in result.error and "1080x1920" in result.error
    assert not out.exists(), "a rejected assembly must not leave a partial file"


def test_non_cut_transition_is_rejected(tmp_path):
    clip = tmp_path / "a.mp4"
    _make_clip(clip, seconds=1.0)
    decisions = _decisions([clip], assembly_mode="full-clips")
    decisions["cuts"][0]["transition_in"] = "fade"

    result = _run(decisions, tmp_path / "assembled.mp4")

    assert result.success is False
    assert "transition" in result.error.lower()


def test_unknown_assembly_mode_is_rejected(tmp_path):
    clip = tmp_path / "a.mp4"
    _make_clip(clip, seconds=1.0)

    result = _run(_decisions([clip], assembly_mode="magic"), tmp_path / "assembled.mp4")

    assert result.success is False
    assert "assembly_mode" in result.error


def test_without_assembly_mode_the_reencode_path_still_runs(tmp_path):
    """The mode is opt-in: existing compositions keep the frame-accurate trimming path."""
    clip = tmp_path / "a.mp4"
    _make_clip(clip, seconds=1.0)
    audio = tmp_path / "master.wav"
    _make_wav(audio, seconds=1.0)

    out = tmp_path / "composed.mp4"
    result = _run(_decisions([clip]), out, audio_path=str(audio))

    assert result.success, result.error
    assert "assembly_mode" not in (result.data or {})
    assert out.exists()
