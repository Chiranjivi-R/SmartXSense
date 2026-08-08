"""Sanitize client-facing API error messages (no paths, traces, or artifact internals)."""

from __future__ import annotations

import re

_WIN_ABS = re.compile(
    r"[A-Za-z]:[\\/](?:[^\\/:*?\"<>|\r\n]+[\\/])*[^\\/:*?\"<>|\r\n]*"
)
_UNIX_ABS = re.compile(
    r"/(?:Users|home|var|tmp|opt|mnt|root|private|data)(?:/[^\s\"',:;]+)+"
)
_FILE_URI = re.compile(r"file://[^\s\"']+", re.IGNORECASE)
_TRACE = re.compile(
    r"Traceback \(most recent call last\)|File \"[^\"]+\", line \d+",
    re.IGNORECASE,
)
_ARTIFACT_LEAK = re.compile(
    r"\b(pickle|UnpicklingError|joblib\.load|marshal\.loads)\b",
    re.IGNORECASE,
)


def sanitize_client_error_message(
    message: str,
    *,
    fallback: str = "Request could not be processed",
) -> str:
    """Return a client-safe error string.

    Keeps useful validation text while redacting absolute filesystem paths,
    traceback fragments, and pickle/artifact internals.
    """
    text = str(message or "").strip()
    if not text:
        return fallback
    if _TRACE.search(text):
        return fallback
    if _ARTIFACT_LEAK.search(text):
        return "Model artifact could not be processed"

    cleaned = _WIN_ABS.sub("<path>", text)
    cleaned = _UNIX_ABS.sub("<path>", cleaned)
    cleaned = _FILE_URI.sub("<path>", cleaned)
    cleaned = re.sub(r"(?:\s*<path>\s*)+", " <path> ", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    if cleaned in {"", "<path>", ": <path>", "<path>."}:
        return fallback
    return cleaned[:500]
