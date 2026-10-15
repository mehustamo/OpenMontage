"""Contract tests for ``audio_enhance`` output-container handling.

The tool used to encode every output as AAC regardless of the file extension, so
an ``output_path`` ending in ``.wav`` produced an AAC stream under a WAV name —
unreadable by ``audio_mixer``, Remotion or anything else that trusts the suffix.
These tests pin the codec/rate resolution that fixed it, without running ffmpeg.
"""

from __future__ import annotations

from tools.audio.audio_enhance import (
    AudioEnhance,
    codec_for_output,
    is_lossless_codec,
)


class TestCodecForOutput:
    def test_wav_means_pcm(self):
        assert codec_for_output("/tmp/mix.wav") == "pcm_s16le"
        assert codec_for_output("/tmp/mix.WAVE") == "pcm_s16le"

    def test_lossy_containers_map_to_their_encoder(self):
        assert codec_for_output("/tmp/a.mp3") == "libmp3lame"
        assert codec_for_output("/tmp/a.flac") == "flac"
        assert codec_for_output("/tmp/a.m4a") == "aac"

    def test_unknown_extension_falls_back_to_aac(self):
        assert codec_for_output("/tmp/a.bin") == "aac"
        assert codec_for_output("/tmp/noextension") == "aac"


class TestLosslessDetection:
    def test_pcm_and_flac_are_lossless(self):
        assert is_lossless_codec("pcm_s16le")
        assert is_lossless_codec("flac")

    def test_lossy_codecs_are_not(self):
        assert not is_lossless_codec("aac")
        assert not is_lossless_codec("libmp3lame")


class TestSchema:
    def test_audio_codec_is_optional_with_an_explanation(self):
        schema = AudioEnhance().input_schema["properties"]
        assert "audio_codec" in schema
        assert "default" not in schema["audio_codec"]
        assert "extension" in schema["audio_codec"]["description"].lower()
        assert "sample_rate" in schema
