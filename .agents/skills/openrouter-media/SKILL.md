---
name: openrouter-media
description: Generate images, video, speech and music through OpenRouter with one OPENROUTER_API_KEY. Use for openrouter_image, openrouter_video, openrouter_tts, openrouter_music, or when a production should stay on a single aggregator key across all media types.
---

# OpenRouter media

One key (`OPENROUTER_API_KEY`) reaches OpenRouter's dedicated generated-media
endpoints. Route full productions through `image_selector`, `video_selector` and
`tts_selector`; call the tools directly when the user names OpenRouter or an
exact model.

| Capability | Tool | Endpoint | Cost reporting |
|---|---|---|---|
| image | `openrouter_image` | `POST /api/v1/images` | exact billed cost in `usage.cost` |
| video | `openrouter_video` | `POST /api/v1/videos` + poll + content | `usage.cost` when returned, else unquoted |
| speech | `openrouter_tts` | `POST /api/v1/audio/speech` | none in the response (console is truth) |
| music | `openrouter_music` | `POST /api/v1/chat/completions`, `modalities=["text","audio"]` | none in the response |

## Preflight every paid call

1. Read the tool's `_MODELS` / `model_catalog`. Image and video catalogs come from
   `tools/provider_contracts/openrouter_media.json`; refresh with
   `scripts/openrouter_models_refresh.py` when a model id is missing.
2. Pass an exact `model`. An id absent from the catalog is filtered out by
   `filter_explicit_route` rather than substituted — never invent a slug.
3. Announce tool, exact model, request count and estimated cost before submitting.
4. Keep secrets in `.env`; never echo a key.

## Images

- Cheap default: `black-forest-labs/flux.2-klein-4b` (~$0.014 for 1 MP, JPEG or
  PNG per `output_format`). Quality routes: `google/gemini-3.1-flash-image`
  (Nano Banana 2), `google/gemini-3-pro-image`, `openai/gpt-image-2`,
  `bytedance-seed/seedream-5-0-pro`.
- Control shape with `aspect_ratio` + `resolution` (`512`, `768`, `1K`, `1.5K`,
  `2K`, `4K`) or an authoritative `size` such as `2048x2048`. A pixel `size`
  combined with a mismatching tier or ratio is rejected with a 400.
- `n` can request up to 10 images; single-image providers reject `n > 1`.
- Editing: pass `image_path`/`image_paths` (local files are inlined as base64
  data URLs) or `image_url`/`image_urls`, with `generation_mode=image_to_image`.
  Supported models include the Gemini image family, `gpt-image-*`, Seedream and
  `x-ai/grok-imagine-image-2.0`.
- There is no negative-prompt parameter anywhere on this API. Say what to avoid
  inside the prompt instead.
- Pricing shapes differ: some routes price per image, some per megapixel, some
  per output token. `estimate_cost` quotes the verified per-image routes and
  raises a quote for the rest; the completed call still reports the exact cost.

## Video

- `POST /api/v1/videos` starts a real paid job. Store `job_path`; on a poll
  timeout resume with the returned `resume_job` — never resubmit a paid request
  whose outcome is unknown.
- The tool validates `duration`, `resolution` and `aspect_ratio` against the
  model's advertised capabilities before submitting, so a 4K request to a
  720p-only route fails locally instead of burning a call.
- `image_to_video` takes `first_frame_path`/`first_frame_url` (and optionally
  `last_frame_path`/`last_frame_url`); `reference_to_video` takes
  `reference_image_*`, and audio/video references only reach routes that support
  them (Seedance 2+).
- `generate_audio=true` is honored only on routes advertising it (Veo, Kling v3,
  Wan 3.0, Seedance 2.x, Hailuo 3).
- Cheap first: `x-ai/grok-imagine-video-1.5-lite` ($0.02/s at 480p, $0.14/s at
  1080p), then `google/veo-3.1-lite` ($0.03/s without audio at 720p),
  `bytedance/seedance-2.0-fast`, `kwaivgi/kling-v3.0-std`.
- Token-priced routes (Seedance) have no pre-flight per-second price; the job
  reports the billed cost and the estimator returns a quote instead of a guess.
- Polling takes 30 s to a few minutes for short clips; size the timeout
  (`poll_timeout`, default 900 s) accordingly.

## Speech

- `POST /api/v1/audio/speech` returns raw audio, not JSON. The tool writes it and
  reports bytes plus format; the OpenRouter console is the only source of truth
  for what it cost.
- **Always send `response_format`.** Omitting it makes the endpoint stream raw
  `audio/pcm` (24 kHz mono) instead of the expected container, and a PCM payload
  written into a `.mp3` file only decodes by luck. The tool always sends it and
  names the file after the response's content type instead of assuming.
- Some hosts require an explicit `voice` (`mistralai/voxtral-mini-tts-2603` does).
  The catalog carries a sample voice per host, so a `tts_selector` call without a
  voice still renders; pass `voice` explicitly for any real production and check
  the language. `fish-audio/s2-pro` accepts a named voice or a cloned reference;
  `elevenlabs/eleven-v4-turbo` needs a real ElevenLabs voice id and has no sample
  in the catalog, so it fails fast instead of guessing.
- `turns=[{text, voice, instructions}]` produces two-speaker dialogue in one
  request on models that support multi-speaker input.
- Prefer `response_format=mp3` for delivery; `wav` when downstream tooling needs
  uncompressed input. `pcm` is written raw with a `.pcm` extension.

## Music

- Lyria models are chat models with audio output; audio requires streaming, which
  `openrouter_music` handles internally.
- `duration_seconds` is folded into the prompt as a target, not a guarantee.
  Verify the rendered length with ffprobe before relying on it for pacing.
- `google/lyria-3-clip-preview` is the cheap default;
  `google/lyria-3-pro-preview` for longer arrangements.

## Evidence and limits

Verified with live calls on 2026-10-15: image `black-forest-labs/flux.2-klein-4b`
($0.014), video `x-ai/grok-imagine-video-1.5-lite` 2 s at 480p ($0.04, 29 s),
speech `mistralai/voxtral-mini-tts-2603` (mp3 via `voice=en_paul_neutral`).
The image and video catalogs are snapshots of the live endpoints; the speech
catalog is hand-maintained because OpenRouter publishes no TTS model list.
Speech and music pricing is unquoted by design — read the OpenRouter activity
console before promising a per-unit price to the user.
