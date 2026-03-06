"""Abstract base class that every file-type handler must implement.

Design Decisions
----------------
* **Strategy Pattern** — the pipeline holds a reference to
  ``AbstractIngestor`` and delegates to the concrete handler selected at
  runtime via the :mod:`~dsetgen.core.registry`.
* **Generator protocol** — ``ingest()`` yields pages/sections lazily so
  that multi-hundred-page PDFs never need to be held in memory at once.
* **Metadata sidecar** — every yielded text fragment is wrapped in
  :class:`~dsetgen.processing.metadata.DocumentFragment` which carries
  provenance (source path, page number, byte offset, timestamp).
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Iterator

from dsetgen.config import PipelineConfig
from dsetgen.processing.metadata import DocumentFragment

logger = logging.getLogger(__name__)


class AbstractIngestor(abc.ABC):
    """Contract that all concrete file handlers must honour.

    Parameters
    ----------
    config:
        The shared pipeline configuration — ingestors may read encoding
        chains, size limits, stream buffer sizes, etc.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    # ── Required overrides ─────────────────────────────────────────────────

    @abc.abstractmethod
    def ingest(self, path: Path) -> Iterator[DocumentFragment]:
        """Yield one :class:`DocumentFragment` per logical section of *path*.

        Implementations **must**:

        1. Validate the file (size, readability) via helpers in
           :mod:`~dsetgen.ingestion.security` *before* opening.
        2. Yield fragments lazily — **never** load an entire file into a
           single string.
        3. Populate :attr:`DocumentFragment.metadata` with at least
           ``source_path`` and ``page_or_section``.
        4. Raise a subclass of
           :class:`~dsetgen.exceptions.IngestionError` for
           unrecoverable problems (the pipeline will catch it and skip the
           file).

        Yields
        ------
        DocumentFragment
            Successive logical sections (pages, rows, JSON objects, etc.).
        """

    @abc.abstractmethod
    def supported_extensions(self) -> frozenset[str]:
        """Return the set of file extensions this handler can process.

        Example: ``frozenset({".txt"})``
        """

    # ── Optional hooks ─────────────────────────────────────────────────────

    def pre_validate(self, path: Path) -> None:
        """Hook executed before ``ingest()``.

        The default implementation runs the security checks (size ceiling,
        path traversal).  Override to add format-specific validation (e.g.
        magic-byte sniffing for PDFs).
        """
        from dsetgen.ingestion.security import validate_path

        validate_path(path, self._config)

    # ── Convenience ────────────────────────────────────────────────────────

    def safe_ingest(self, path: Path) -> Iterator[DocumentFragment]:
        """Run ``pre_validate`` then ``ingest``, logging but not raising.

        This is the entry-point called by the pipeline controller — it
        wraps exceptions so that a single bad file never kills the batch.
        """
        from dsetgen.exceptions import IngestionError

        try:
            self.pre_validate(path)
        except IngestionError:
            logger.exception("Validation failed for %s — skipping", path)
            return

        try:
            yield from self.ingest(path)
        except IngestionError:
            logger.exception("Ingestion failed for %s — skipping", path)
        except Exception:
            # Catch-all: unknown library errors should not crash the batch.
            logger.exception(
                "Unexpected error ingesting %s — skipping", path
            )
