"""Domain-specific exception hierarchy.

A structured exception tree lets callers differentiate transient failures
(retry-worthy) from permanent ones (skip-and-log) without catching bare
``Exception``.
"""

from __future__ import annotations


class dsetgenError(Exception):
    """Root exception for every error raised inside the framework."""


# ── Ingestion ──────────────────────────────────────────────────────────────

class IngestionError(dsetgenError):
    """Base for all file-ingestion failures."""


class UnsupportedFileTypeError(IngestionError):
    """No registered ingestor for the given file extension."""


class EncodingDetectionError(IngestionError):
    """Could not decode the file under any attempted encoding."""


class FileTooLargeError(IngestionError):
    """File exceeds the configured size ceiling."""


class CorruptedFileError(IngestionError):
    """File could not be parsed — likely truncated or structurally invalid."""


class PathTraversalError(IngestionError):
    """Attempted directory traversal detected in a user-supplied path."""


# ── Processing ─────────────────────────────────────────────────────────────

class ProcessingError(dsetgenError):
    """Base for normalisation / chunking failures."""


class ChunkingError(ProcessingError):
    """Token-aware chunker encountered an unrecoverable edge case."""


# ── LLM Orchestration ─────────────────────────────────────────────────────

class LLMError(dsetgenError):
    """Base for all LLM-adapter failures."""


class LLMTimeoutError(LLMError):
    """The LLM backend did not respond within the configured deadline."""


class LLMConnectionError(LLMError):
    """Could not reach the LLM backend (server down / network issue)."""


class LLMRateLimitError(LLMError):
    """The backend signalled that we should back off."""


class LLMMalformedResponseError(LLMError):
    """The LLM returned something that could not be parsed as valid output."""


# ── Output / State ─────────────────────────────────────────────────────────

class OutputError(dsetgenError):
    """Base for formatter / writer failures."""


class CheckpointError(dsetgenError):
    """Failed to read or write the checkpoint store."""
