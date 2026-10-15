#!/usr/bin/env python3
"""Live smoke test for the OpenRouter media tools (spends real credits).

Mirrors ``scripts/atlas_media_smoke.py``: it exercises the tools through the
registry, writes artifacts to a scratch directory and prints the billed cost
OpenRouter reports. Nothing here runs in CI — a key and a budget are required.

    .venv/bin/python scripts/openrouter_media_smoke.py --all
    .venv/bin/python scripts/openrouter_media_smoke.py --image --tts
    .venv/bin/python scripts/openrouter_media_smoke.py --video --resume-after

Costs (2026-10-15, cheapest defaults):
    image  ~$0.014/image      video  $0.02/s at 480p on grok-imagine-video-1.5-lite
    tts    unquoted in the API response   music  unquoted in the API response
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.tool_registry import registry  # noqa: E402


def probe(path: str) -> str:
    """One-line ffprobe summary, or a byte count when ffprobe is unavailable."""
    target = Path(path)
    if not target.exists():
        return "missing"
    if shutil.which("ffprobe"):
        command = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_name,width,height",
            "-show_entries", "format=duration,size",
            "-of", "default=nw=1:nk=1", str(target),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        fields = [line for line in result.stdout.splitlines() if line.strip()]
        if fields:
            return " ".join(fields)
    return f"{target.stat().st_size} bytes"


def report(name: str, result) -> float:
    if result.success:
        cost = result.cost_usd
        cost_text = f"${cost:.4f}" if isinstance(cost, (int, float)) else result.data.get("cost_status", "unquoted")
        print(f"  OK   {name}: cost={cost_text} files={result.artifacts}")
        for artifact in result.artifacts:
            print(f"       {artifact} -> {probe(artifact)}")
        return float(cost or 0.0)
    print(f"  FAIL {name}: {result.error}")
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--tts", action="store_true")
    parser.add_argument("--music", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume-after", action="store_true", help="re-poll the finished video job without resubmitting")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--max-cost-usd", type=float, default=0.25)
    args = parser.parse_args()
    if not any((args.image, args.video, args.tts, args.music, args.all)):
        parser.error("choose at least one of --image --video --tts --music --all")

    workdir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="openrouter-smoke-"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"workdir: {workdir}")

    registry.discover()
    spent = 0.0
    video_job = None

    if args.image or args.all:
        print("image:")
        spent += report(
            "openrouter_image",
            registry.get("openrouter_image").execute({
                "prompt": "a small copper robot holding a clapperboard, studio light, plain backdrop",
                "aspect_ratio": "16:9",
                "resolution": "1K",
                "output_path": str(workdir / "image.jpg"),
            }),
        )

    if args.tts or args.all:
        print("tts:")
        spent += report(
            "openrouter_tts",
            registry.get("openrouter_tts").execute({
                "text": "OpenMontage puhuu nyt OpenRouterin kautta.",
                "voice": "en_paul_neutral",
                "output_path": str(workdir / "speech.mp3"),
            }),
        )

    if args.music or args.all:
        print("music:")
        spent += report(
            "openrouter_music",
            registry.get("openrouter_music").execute({
                "prompt": "sparse ambient piano, gentle pad, no percussion",
                "duration_seconds": 15,
                "output_path": str(workdir / "music.mp3"),
            }),
        )

    if args.video or args.all:
        print("video:")
        result = registry.get("openrouter_video").execute({
            "prompt": "a copper robot on a plain studio backdrop turns its head slowly",
            "duration": args.duration,
            "resolution": "480p",
            "aspect_ratio": "16:9",
            "generate_audio": False,
            "output_path": str(workdir / "clip.mp4"),
            "job_path": str(workdir / "clip.job.json"),
        })
        spent += report("openrouter_video", result)
        video_job = result.data.get("resume_job")
        if args.resume_after and video_job:
            print("video resume (no resubmission):")
            resumed = registry.get("openrouter_video").execute({
                "prompt": "resume probe",
                "resume_job": video_job,
                "output_path": str(workdir / "clip_resumed.mp4"),
            })
            report("openrouter_video(resume)", resumed)
        if video_job:
            (workdir / "video.job.json").write_text(json.dumps(video_job, indent=2), encoding="utf-8")

    print(f"reported spend: ${spent:.4f} (budget ${args.max_cost_usd:.2f})")
    if spent > args.max_cost_usd:
        print("note: spend exceeded --max-cost-usd; check the OpenRouter activity console", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
