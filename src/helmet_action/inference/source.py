from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


def redact_source(source: str | int) -> str:
    """Hide userinfo in RTSP/HTTP URLs so credentials never hit logs."""
    if isinstance(source, int):
        return str(source)
    text = str(source)
    parts = urlsplit(text)
    if parts.scheme.lower() in {"rtsp", "rtsps", "http", "https"} and parts.username:
        host = parts.hostname or ""
        netloc = host
        if parts.port:
            netloc = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    if "@" in text and "://" in text:
        scheme, rest = text.split("://", 1)
        return f"{scheme}://***@{rest.rsplit('@', 1)[-1]}"
    return text


def parse_video_source(source: str | int) -> str | int:
    """Map CLI source to OpenCV capture target.

    Existing files always win over integer webcam indices, so a file named "0"
    is not treated as camera 0. Bare digits that are not paths become webcam indices.
    """
    if isinstance(source, int):
        return source
    text = str(source).strip()
    if not text:
        raise ValueError("empty video source")
    lowered = text.lower()
    if lowered.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return text
    from pathlib import Path

    path = Path(text)
    if path.exists():
        return str(path.expanduser().resolve())
    if text.isdigit():
        return int(text)
    return text
