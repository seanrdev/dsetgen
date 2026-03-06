"""Security and safety utilities for the ingestion layer.

Every file path that enters the framework passes through these checks
**before** any handler opens it.  This module is intentionally thin and
side-effect-free so that it is easy to audit.

Threat Model
------------
* **Directory traversal** — user-supplied (or scraped) file names that
  contain ``../`` or symlink tricks to escape the configured input root.
* **Memory bombs** — a 2 GB log file or a zip-bomb PDF that would OOM
  the process.
* **Encoding mismatches** — Latin-1 files labelled as UTF-8 that would
  produce mojibake or raise ``UnicodeDecodeError`` mid-stream.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import IO, Iterator

from doc2dataset.config import PipelineConfig
from doc2dataset.exceptions import (
    EncodingDetectionError,
    FileTooLargeError,
    PathTraversalError,
)

logger = logging.getLogger(__name__)


# ── Path safety ────────────────────────────────────────────────────────────

def validate_path(path: Path, config: PipelineConfig) -> None:
    """Assert *path* is safe to open.

    Raises
    ------
    PathTraversalError
        If the resolved path escapes the configured input root.
    FileTooLargeError
        If the file exceeds ``config.max_file_bytes``.
    FileNotFoundError
        If *path* does not exist or is not a regular file.
    """
    resolved = path.resolve()

    # 1. Existence & type
    if not resolved.is_file():
        raise FileNotFoundError(f"Not a regular file: {resolved}")

    # 2. Symlink-safe traversal check — the resolved path must live
    #    inside (or equal to) the configured input root.
    input_root = config.resolved_input_dir()
    try:
        resolved.relative_to(input_root)
    except ValueError:
        raise PathTraversalError(
            f"Path {resolved} escapes the input root {input_root}"
        ) from None

    # 3. Size ceiling
    size = resolved.stat().st_size
    if size > config.max_file_bytes:
        raise FileTooLargeError(
            f"{resolved} is {size:,} bytes — limit is {config.max_file_bytes:,}"
        )


def safe_resolve(raw: str, config: PipelineConfig) -> Path:
    """Resolve a potentially-untrusted string to a validated :class:`Path`.

    Use this when file paths come from an external manifest or API rather
    than from ``os.scandir``.
    """
    candidate = (config.resolved_input_dir() / raw).resolve()
    # Re-check that normalisation did not escape the sandbox
    try:
        candidate.relative_to(config.resolved_input_dir())
    except ValueError:
        raise PathTraversalError(
            f"Traversal attempt detected in supplied path: {raw!r}"
        ) from None
    return candidate


# ── Encoding fallback ──────────────────────────────────────────────────────

def open_with_fallback(
    path: Path,
    config: PipelineConfig,
) -> IO[str]:
    """Open *path* for reading, trying each encoding in the fallback chain.

    Returns the **first** file handle that does not raise on a probe read.
    The caller is responsible for closing the handle.

    Raises
    ------
    EncodingDetectionError
        If every encoding in the chain fails.
    """
    for encoding in config.encoding_fallback_chain:
        try:
            fh = open(path, "r", encoding=encoding, errors="strict")  # noqa: SIM115
            # Probe: read 8 KiB to trigger any decode error early.
            fh.read(config.stream_chunk_bytes)
            fh.seek(0)
            logger.debug("Opened %s with encoding %s", path, encoding)
            return fh
        except (UnicodeDecodeError, UnicodeError):
            logger.debug("Encoding %s failed for %s — trying next", encoding, path)
            continue
        except OSError:
            raise

    raise EncodingDetectionError(
        f"None of {config.encoding_fallback_chain} could decode {path}"
    )


# ── Streaming helpers ──────────────────────────────────────────────────────

def stream_lines(path: Path, config: PipelineConfig) -> Iterator[str]:
    """Yield lines from *path* without ever holding the full file in RAM.

    The caller can consume this iterator inside a for-loop and naturally
    get constant-memory behaviour.
    """
    fh = open_with_fallback(path, config)
    try:
        for line in fh:
            yield line
    finally:
        fh.close()


def stream_binary_chunks(
    path: Path,
    chunk_size: int = 64 * 1024,
) -> Iterator[bytes]:
    """Yield fixed-size binary chunks — useful for hashing or binary probes."""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            yield chunk
