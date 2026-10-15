"""Shared transport for OpenRouter's generated-media APIs.

OpenRouter serves generated media on dedicated endpoints, all under one bearer key:

* images  ``POST /api/v1/images``           -> base64 bytes, billed cost in ``usage.cost``
* video   ``POST /api/v1/videos``           -> async job with a ``polling_url``
* speech  ``POST /api/v1/audio/speech``     -> raw audio bytestream
* music   ``POST /api/v1/chat/completions`` -> streamed audio deltas (``modalities=["text","audio"]``)

This module keeps the request/error/download handling in one place so each
adapter only has to describe its own request shape. It deliberately does not
retry a paid submission: OpenRouter bills a completed generation in full and
returns 502 for anything that did not complete, so the caller decides.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
APP_TITLE = "OpenMontage"
APP_REFERER = "https://github.com/calesthio/OpenMontage"


class OpenRouterError(RuntimeError):
    """An OpenRouter call failed. ``status`` is 0 for transport-level failures."""

    def __init__(self, message: str, *, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


def api_key() -> Optional[str]:
    return os.environ.get("OPENROUTER_API_KEY") or None


def base_url() -> str:
    return (os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def api_url(path: str) -> str:
    """Absolute URL for an API path; absolute inputs pass through unchanged."""
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return f"{base_url()}/{path.lstrip('/')}"


def headers(key: Optional[str] = None) -> dict[str, str]:
    resolved = key or api_key()
    if not resolved:
        raise OpenRouterError("OPENROUTER_API_KEY is not configured")
    return {
        "Authorization": f"Bearer {resolved}",
        "Content-Type": "application/json",
        "HTTP-Referer": APP_REFERER,
        "X-OpenRouter-Title": APP_TITLE,
    }


def _error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return (response.text or "").strip()[:300] or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
        if payload.get("message"):
            return str(payload["message"])
    return str(payload)[:300]


def request(method: str, path: str, *, payload: Any = None, key: Optional[str] = None, timeout: float = 180, stream: bool = False) -> requests.Response:
    """Issue an authenticated request and raise OpenRouterError on any non-2xx."""
    url = api_url(path)
    try:
        response = requests.request(
            method,
            url,
            headers=headers(key),
            json=payload,
            timeout=timeout,
            stream=stream,
        )
    except requests.RequestException as exc:
        raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc
    if response.status_code >= 400:
        raise OpenRouterError(
            _error_message(response), status=response.status_code
        )
    return response


def request_json(method: str, path: str, *, payload: Any = None, key: Optional[str] = None, timeout: float = 180) -> dict[str, Any]:
    response = request(method, path, payload=payload, key=key, timeout=timeout)
    try:
        return response.json()
    except ValueError as exc:
        raise OpenRouterError(f"OpenRouter returned a non-JSON body from {path}") from exc


def request_bytes(method: str, path: str, *, payload: Any = None, key: Optional[str] = None, timeout: float = 300) -> bytes:
    content, _ = request_media(method, path, payload=payload, key=key, timeout=timeout)
    return content


def request_media(
    method: str,
    path: str,
    *,
    payload: Any = None,
    key: Optional[str] = None,
    timeout: float = 300,
) -> tuple[bytes, str]:
    """Return ``(body, content_type)`` for a raw media endpoint.

    Callers that write generated audio must check the content type: the speech
    endpoint silently falls back to ``audio/pcm`` when ``response_format`` is
    omitted, and writing PCM into a ``.mp3`` file produces a file that only
    decodes by luck.
    """
    response = request(method, path, payload=payload, key=key, timeout=timeout)
    content = response.content
    if not content:
        raise OpenRouterError(f"OpenRouter returned an empty body from {path}")
    return content, (response.headers.get("content-type") or "").strip().lower()


def usage_cost(payload: Any) -> Optional[float]:
    """Exact billed cost of the call, when OpenRouter reports it."""
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    cost = usage.get("cost")
    if cost is None:
        return None
    try:
        return float(cost)
    except (TypeError, ValueError):
        return None


def image_reference(value: str) -> dict[str, Any]:
    """Build a ContentPartImage reference from a path, data URL or http(s) URL."""
    if not value:
        raise OpenRouterError("Empty image reference")
    if value.startswith("data:"):
        return {"type": "image_url", "image_url": {"url": value}}
    if value.startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": value}}
    path = Path(value)
    if not path.is_file():
        raise OpenRouterError(f"Reference image not found: {value}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def reference_image(value: str) -> dict[str, Any]:
    """Alias kept for symmetry with the video tool's naming."""
    return image_reference(value)


def download(url: str, path: Path, *, key: Optional[str] = None) -> str:
    """Download generated media to ``path`` atomically.

    Accepts base64 data URIs, public https URLs and OpenRouter content URLs
    (``.../api/v1/videos/<id>/content``), which require the bearer key.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "data"):
        raise OpenRouterError(f"Refusing to download from {parsed.scheme or 'relative'} URL")
    if url.startswith("data:"):
        header, _, body = url.partition(",")
        if ";base64" not in header:
            raise OpenRouterError("Only base64 data URIs are supported")
        content = base64.b64decode(body, validate=True)
    else:
        request_headers = headers(key) if parsed.netloc.endswith("openrouter.ai") else {}
        try:
            response = requests.get(url, headers=request_headers, timeout=300)
        except requests.RequestException as exc:
            raise OpenRouterError(f"Media download failed: {exc}") from exc
        if response.status_code >= 400:
            raise OpenRouterError(
                _error_message(response), status=response.status_code
            )
        content = response.content
    if not content:
        raise OpenRouterError("OpenRouter returned empty media")
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(content)
    temporary.replace(path)
    return str(path)
