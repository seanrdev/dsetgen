"""Central pipeline orchestrator.

The :class:`PipelineController` ties every layer together:

    Discover files → Ingest → Normalize → Chunk → LLM → Format → Checkpoint

It owns the **concurrency model** (``asyncio.Semaphore``-bounded fan-out
to the LLM) and the **fault-tolerance contract** (per-file catch-and-skip
with SQLite checkpointing so that a crash at file 45,000 of 50,000
resumes from 45,001).

Execution Flow
--------------
1. **File discovery** — walk ``config.input_dir``, filter by extension,
   subtract already-completed files from the checkpoint DB.
2. **Per-file processing** (possibly concurrent):
   a. Ingestor yields ``DocumentFragment`` objects.
   b. Normalizer cleans each fragment.
   c. Chunker splits fragments to fit the LLM context window.
   d. Each chunk is sent to the LLM adapter (with retries + backoff).
   e. The LLM response is written via the output formatter.
   f. The file is marked complete in the checkpoint DB.
3. **Finalize** — flush the formatter and close the checkpoint DB.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from dsetgen.config import PipelineConfig
from dsetgen.core.registry import (
    adapter_registry,
    formatter_registry,
    ingestor_registry,
)
from dsetgen.exceptions import (
    dsetgenError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    UnsupportedFileTypeError,
)
from dsetgen.ingestion.abstract_ingestor import AbstractIngestor
from dsetgen.llm.abstract_adapter import AbstractLLMAdapter
from dsetgen.output.base_formatter import BaseOutputFormatter
from dsetgen.processing.chunker import TokenAwareChunker
from dsetgen.processing.metadata import DocumentFragment
from dsetgen.processing.normalizer import Normalizer
from dsetgen.state.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


# ── Retry helper ───────────────────────────────────────────────────────────

async def _retry_with_backoff(
    coro_factory: Callable,
    *,
    max_retries: int,
    backoff_base: float,
    retry_on: tuple[Type[Exception], ...] = (LLMTimeoutError, LLMRateLimitError),
) -> Any:
    """Execute an async callable with exponential backoff on transient errors.

    Parameters
    ----------
    coro_factory:
        A zero-argument callable that returns an awaitable (so we get a
        fresh coroutine on each retry).
    max_retries:
        Maximum number of attempts (including the first).
    backoff_base:
        Multiplier for the exponential sleep (``backoff_base ** attempt``).
    retry_on:
        Tuple of exception types that trigger a retry.

    Returns the result of the first successful call.
    Raises the last exception if all retries are exhausted.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return await coro_factory()
        except retry_on as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                sleep_s = backoff_base ** attempt
                logger.warning(
                    "Retry %d/%d after %s — sleeping %.1fs",
                    attempt + 1,
                    max_retries,
                    type(exc).__name__,
                    sleep_s,
                )
                await asyncio.sleep(sleep_s)
    raise last_exc  # type: ignore[misc]


class PipelineController:
    """Top-level orchestrator for the Document-to-Dataset pipeline.

    Parameters
    ----------
    config:
        Fully populated :class:`PipelineConfig`.
    llm_adapter:
        An optional pre-constructed LLM adapter.  If ``None``, the
        controller will look up ``"ollama"`` in the adapter registry.
    formatter:
        An optional pre-constructed formatter.  If ``None``, one is
        created from ``config.output_format``.
    prompt_builder:
        A callable ``(DocumentFragment) → str`` that constructs the
        prompt sent to the LLM for each chunk.  **This is the primary
        extension point** — you supply the prompt logic, the framework
        handles everything else.
    response_parser:
        A callable ``(str, DocumentFragment) → Dict[str, Any]`` that
        converts the raw LLM output into a training record dict suitable
        for the formatter.
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        llm_adapter: Optional[AbstractLLMAdapter] = None,
        formatter: Optional[BaseOutputFormatter] = None,
        prompt_builder: Optional[Callable[[DocumentFragment], str]] = None,
        response_parser: Optional[
            Callable[[str, DocumentFragment], Dict[str, Any]]
        ] = None,
    ) -> None:
        self._config = config

        # ── Layers ─────────────────────────────────────────────────────────
        self._normalizer = Normalizer()
        self._chunker = TokenAwareChunker(config)
        self._checkpoint = CheckpointManager(config.checkpoint_db)

        # LLM adapter: injected or looked up from registry.
        if llm_adapter is not None:
            self._adapter = llm_adapter
        else:
            adapter_cls = adapter_registry.get("ollama")
            if adapter_cls is None:
                # Ensure handlers are imported (triggers registration).
                import dsetgen.llm.ollama_adapter  # noqa: F401

                adapter_cls = adapter_registry.get("ollama")
            assert adapter_cls is not None, "No LLM adapter registered"
            self._adapter = adapter_cls(config)

        # Output formatter: injected or looked up.
        if formatter is not None:
            self._formatter = formatter
        else:
            fmt_cls = formatter_registry.get(config.output_format)
            if fmt_cls is None:
                import dsetgen.output.jsonl_formatter  # noqa: F401
                import dsetgen.output.huggingface_formatter  # noqa: F401

                fmt_cls = formatter_registry.get(config.output_format)
            assert fmt_cls is not None, (
                f"No formatter registered for '{config.output_format}'"
            )
            self._formatter = fmt_cls(config)

        # User-supplied hooks (the "you plug in" part).
        self._prompt_builder = prompt_builder or self._default_prompt_builder
        self._response_parser = response_parser or self._default_response_parser

        # Concurrency limiter.
        self._semaphore = asyncio.Semaphore(config.max_concurrent_llm_calls)

    # ── Main entry point ───────────────────────────────────────────────────

    async def run(self) -> None:
        """Execute the full pipeline: discover → process → output.

        This method is **idempotent** — calling it again after a crash
        will skip already-completed files.
        """
        t0 = time.monotonic()
        logger.info("Pipeline starting — config: %s", self._config)

        # Ensure handler registrations are loaded.
        import dsetgen.ingestion.handlers  # noqa: F401

        # Boot subsystems.
        self._checkpoint.initialize()
        self._formatter.initialize()
        await self._adapter.startup()

        try:
            files = self._discover_files()
            completed = self._checkpoint.completed_files()
            pending = [f for f in files if str(f) not in completed]

            logger.info(
                "Discovered %d files — %d already done, %d to process",
                len(files),
                len(files) - len(pending),
                len(pending),
            )

            # Process files.  For I/O-bound LLM calls, we fan out with
            # a bounded semaphore.  For simplicity the file-level loop
            # is sequential (the concurrency lives inside _process_file
            # at the chunk level).
            for idx, file_path in enumerate(pending, start=1):
                logger.info(
                    "[%d/%d] Processing %s", idx, len(pending), file_path
                )
                await self._process_file(file_path)

        except KeyboardInterrupt:
            logger.warning("Interrupted — progress has been checkpointed")
        finally:
            self._formatter.finalize()
            await self._adapter.shutdown()
            stats = self._checkpoint.stats()
            self._checkpoint.close()
            elapsed = time.monotonic() - t0
            logger.info(
                "Pipeline finished in %.1fs — %s",
                elapsed,
                stats,
            )

    # ── File discovery ─────────────────────────────────────────────────────

    def _discover_files(self) -> List[Path]:
        """Walk the input directory and return all files with allowed extensions."""
        root = self._config.resolved_input_dir()
        if not root.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {root}")

        allowed = self._config.allowed_extensions
        files: List[Path] = []

        for dirpath, _, filenames in os.walk(root):
            for fname in sorted(filenames):
                p = Path(dirpath) / fname
                if p.suffix.lower() in allowed:
                    files.append(p)

        files.sort()  # Deterministic ordering for reproducible runs
        return files

    # ── Per-file processing ────────────────────────────────────────────────

    async def _process_file(self, file_path: Path) -> None:
        """Full lifecycle for a single file: ingest → normalize → chunk → LLM → write."""
        ext = file_path.suffix.lower()
        ingestor_cls = ingestor_registry.get(ext)
        if ingestor_cls is None:
            logger.warning("No ingestor for %s — skipping %s", ext, file_path)
            self._checkpoint.mark_failed(
                str(file_path), f"Unsupported extension: {ext}"
            )
            return

        ingestor: AbstractIngestor = ingestor_cls(self._config)

        # 1. Ingest — collect all fragments (safe_ingest handles errors).
        raw_fragments: List[DocumentFragment] = list(
            ingestor.safe_ingest(file_path)
        )
        if not raw_fragments:
            logger.warning("No content extracted from %s — skipping", file_path)
            self._checkpoint.mark_failed(str(file_path), "No content extracted")
            return

        # 2. Normalize.
        clean_fragments = self._normalizer.normalize_batch(raw_fragments)
        if not clean_fragments:
            logger.warning("All content was empty after normalisation — skipping %s", file_path)
            self._checkpoint.mark_failed(str(file_path), "Empty after normalisation")
            return

        # 3. Chunk.
        chunks = self._chunker.chunk_many(clean_fragments)
        logger.debug("%s → %d chunks", file_path, len(chunks))

        # 4. Send each chunk to the LLM and collect records.
        records: List[Dict[str, Any]] = []
        tasks = [self._process_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for chunk, result in zip(chunks, results):
            if isinstance(result, Exception):
                logger.error(
                    "LLM failed for chunk %s of %s: %s",
                    chunk.chunk_index,
                    file_path,
                    result,
                )
                continue
            if result is not None:
                records.append(result)

        # 5. Write records.
        for rec in records:
            self._formatter.write_record(rec)

        # 6. Checkpoint.
        self._checkpoint.mark_complete(
            str(file_path),
            num_chunks=len(chunks),
            num_records=len(records),
        )
        logger.info(
            "✓ %s — %d chunks → %d records",
            file_path.name,
            len(chunks),
            len(records),
        )

    async def _process_chunk(
        self, chunk: DocumentFragment
    ) -> Optional[Dict[str, Any]]:
        """Send one chunk to the LLM (with semaphore + retries) and parse."""
        async with self._semaphore:
            prompt = self._prompt_builder(chunk)

            raw_response = await _retry_with_backoff(
                lambda: self._adapter.generate(prompt),
                max_retries=self._config.llm_max_retries,
                backoff_base=self._config.llm_retry_backoff_base,
            )

            record = self._response_parser(raw_response, chunk)
            return record

    # ── Default hooks (overridable) ────────────────────────────────────────

    @staticmethod
    def _default_prompt_builder(fragment: DocumentFragment) -> str:
        """Placeholder prompt — replace with your own logic.

        This default asks the LLM to generate a Q&A pair from the chunk.
        It serves as a structural example; **you should override this**
        via the ``prompt_builder`` constructor parameter.
        """
        return (
            "You are a dataset-generation assistant.  Given the following "
            "text excerpt, generate one high-quality question-and-answer pair "
            "suitable for instruction-tuning.  Return ONLY valid JSON with "
            'keys "instruction", "input", and "output".\n\n'
            f"---\n{fragment.text}\n---"
        )

    @staticmethod
    def _default_response_parser(
        raw: str, fragment: DocumentFragment
    ) -> Dict[str, Any]:
        """Placeholder parser — attempts JSON extraction, attaches metadata.

        Override via the ``response_parser`` constructor parameter for
        custom schemas.
        """
        import json
        import re

        # Try direct parse.
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            # Extract first JSON object.
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    record = json.loads(match.group())
                except json.JSONDecodeError:
                    record = {"instruction": "", "input": "", "output": raw}
            else:
                record = {"instruction": "", "input": "", "output": raw}

        # Attach provenance metadata.
        record["_metadata"] = fragment.metadata.as_dict()
        return record
