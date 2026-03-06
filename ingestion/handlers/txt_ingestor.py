"""Plain-text file ingestor with streaming and encoding fallback."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from dsetgen.config import PipelineConfig
from dsetgen.core.registry import register_ingestor
from dsetgen.ingestion.abstract_ingestor import AbstractIngestor
from dsetgen.ingestion.security import stream_lines
from dsetgen.processing.metadata import DocumentFragment, FragmentMetadata

logger = logging.getLogger(__name__)

# Maximum number of lines to accumulate into one fragment before yielding.
_LINES_PER_FRAGMENT = 200


@register_ingestor(".txt")
class TxtIngestor(AbstractIngestor):
    """Stream ``.txt`` files line-by-line with automatic encoding fallback.

    Large files (e.g. multi-GB server logs) are never loaded in full.
    Lines are batched into fragments of ``_LINES_PER_FRAGMENT`` lines to
    keep downstream chunk sizes reasonable.
    """

    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".txt"})

    def ingest(self, path: Path) -> Iterator[DocumentFragment]:
        buffer: list[str] = []
        section_idx = 0

        for line in stream_lines(path, self._config):
            buffer.append(line)
            if len(buffer) >= _LINES_PER_FRAGMENT:
                yield self._flush(buffer, path, section_idx)
                buffer = []
                section_idx += 1

        if buffer:
            yield self._flush(buffer, path, section_idx)

    @staticmethod
    def _flush(
        lines: list[str], path: Path, section: int
    ) -> DocumentFragment:
        return DocumentFragment(
            text="".join(lines),
            metadata=FragmentMetadata(
                source_path=str(path),
                page_or_section=section,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
        )
