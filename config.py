"""Centralised, immutable configuration for the pipeline.

Every tunable knob lives here so that nothing is scattered across modules.
Values can be overridden via environment variables (prefix ``D2D_``) or by
constructing a ``PipelineConfig`` directly in code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet


def _env(key: str, default: str) -> str:
    """Read an env var with a ``D2D_`` prefix, falling back to *default*."""
    return os.environ.get(f"D2D_{key}", default)


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration consumed by :class:`PipelineController`.

    Attributes are grouped by layer so the docstring doubles as quick
    reference documentation.
    """

    # ── Paths ──────────────────────────────────────────────────────────────
    input_dir: str = field(
        default_factory=lambda: _env("INPUT_DIR", "./data/raw"),
        metadata={"help": "Root directory (or glob) of source documents."},
    )
    output_path: str = field(
        default_factory=lambda: _env("OUTPUT_PATH", "./data/output/train.jsonl"),
        metadata={"help": "Destination file for the generated dataset."},
    )

    # ── Ingestion ──────────────────────────────────────────────────────────
    max_file_bytes: int = field(
        default_factory=lambda: int(_env("MAX_FILE_BYTES", str(500 * 1024 * 1024))),
        metadata={"help": "Hard ceiling per file (default 500 MB). Prevents memory bombs."},
    )
    allowed_extensions: FrozenSet[str] = frozenset(
        {".txt", ".md", ".pdf", ".docx", ".csv", ".json", ".html", ".htm"}
    )
    encoding_fallback_chain: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    stream_chunk_bytes: int = 8 * 1024  # 8 KiB read-ahead for streaming ingestors

    # ── Processing ─────────────────────────────────────────────────────────
    max_context_tokens: int = field(
        default_factory=lambda: int(_env("MAX_CONTEXT_TOKENS", "4096")),
        metadata={"help": "Context window budget for the target LLM."},
    )
    chunk_overlap_tokens: int = 128
    tokenizer_name: str = "cl100k_base"  # tiktoken encoding name (GPT-4 default)

    # ── LLM ────────────────────────────────────────────────────────────────
    ollama_base_url: str = field(
        default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    model_name: str = field(
        default_factory=lambda: _env("MODEL_NAME", "llama3.1:70b"),
    )
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 3
    llm_retry_backoff_base: float = 2.0  # Exponential backoff multiplier

    # ── Output ─────────────────────────────────────────────────────────────
    output_format: str = "jsonl"  # "jsonl" | "huggingface"
    jsonl_schema: str = "alpaca"  # "alpaca" | "sharegpt"

    # ── State / Checkpointing ──────────────────────────────────────────────
    checkpoint_db: str = field(
        default_factory=lambda: _env("CHECKPOINT_DB", "./state.db"),
        metadata={"help": "SQLite database for resume-from-failure."},
    )
    batch_commit_size: int = 50  # Flush checkpoint every N files

    # ── Concurrency ────────────────────────────────────────────────────────
    max_concurrent_llm_calls: int = field(
        default_factory=lambda: int(_env("MAX_CONCURRENT_LLM", "4")),
    )

    def resolved_input_dir(self) -> Path:
        """Return the input directory as an absolute :class:`Path`."""
        return Path(self.input_dir).expanduser().resolve()

    def resolved_output_path(self) -> Path:
        """Return the output file as an absolute :class:`Path`."""
        return Path(self.output_path).expanduser().resolve()
