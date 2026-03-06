"""CSV ingestor — streaming row-by-row to handle arbitrarily large files."""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from doc2dataset.core.registry import register_ingestor
from doc2dataset.ingestion.abstract_ingestor import AbstractIngestor
from doc2dataset.ingestion.security import open_with_fallback
from doc2dataset.processing.metadata import DocumentFragment, FragmentMetadata

logger = logging.getLogger(__name__)

# Increase the CSV field-size limit for files with very wide columns.
csv.field_size_limit(10 * 1024 * 1024)  # 10 MB per cell


@register_ingestor(".csv")
class CsvIngestor(AbstractIngestor):
    """Stream ``.csv`` rows as individual fragments.

    Each row is rendered as ``"col1: val1 | col2: val2 | ..."`` so that
    column semantics are preserved in the plain-text representation.
    Batching (N rows per fragment) can be added by subclassing.
    """

    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".csv"})

    def ingest(self, path: Path) -> Iterator[DocumentFragment]:
        fh = open_with_fallback(path, self._config)
        try:
            reader = csv.DictReader(fh)
            for row_idx, row in enumerate(reader):
                text = " | ".join(
                    f"{k}: {v}" for k, v in row.items() if v
                )
                if not text.strip():
                    continue
                yield DocumentFragment(
                    text=text,
                    metadata=FragmentMetadata(
                        source_path=str(path),
                        page_or_section=row_idx,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        extra={"columns": list(row.keys())},
                    ),
                )
        finally:
            fh.close()
