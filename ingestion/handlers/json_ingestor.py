"""JSON ingestor — streaming parser for arbitrarily large JSON files.

Design
------
* For small files (< 50 MB): ``json.load`` into memory.
* For large files: ``ijson`` SAX-style parser yielding top-level items
  from arrays without loading the entire structure.
* Nested objects are flattened to a readable key–value string.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator

from dsetgen.core.registry import register_ingestor
from dsetgen.exceptions import CorruptedFileError
from dsetgen.ingestion.abstract_ingestor import AbstractIngestor
from dsetgen.processing.metadata import DocumentFragment, FragmentMetadata

logger = logging.getLogger(__name__)

_SMALL_FILE_THRESHOLD = 50 * 1024 * 1024  # 50 MB


def _flatten(obj: Any, prefix: str = "") -> str:
    """Recursively flatten a JSON object to ``"key: value"`` lines."""
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            new_key = f"{prefix}.{k}" if prefix else k
            parts.append(_flatten(v, new_key))
        return "\n".join(parts)
    if isinstance(obj, list):
        return f"{prefix}: {json.dumps(obj, ensure_ascii=False)}"
    return f"{prefix}: {obj}"


@register_ingestor(".json")
class JsonIngestor(AbstractIngestor):
    """Ingest ``.json`` files, yielding one fragment per top-level element."""

    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".json"})

    def ingest(self, path: Path) -> Iterator[DocumentFragment]:
        size = path.stat().st_size

        if size < _SMALL_FILE_THRESHOLD:
            yield from self._ingest_small(path)
        else:
            yield from self._ingest_streaming(path)

    def _ingest_small(self, path: Path) -> Iterator[DocumentFragment]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CorruptedFileError(f"Invalid JSON in {path}: {exc}") from exc

        items = data if isinstance(data, list) else [data]
        for idx, item in enumerate(items):
            text = _flatten(item) if isinstance(item, (dict, list)) else str(item)
            if not text.strip():
                continue
            yield DocumentFragment(
                text=text,
                metadata=FragmentMetadata(
                    source_path=str(path),
                    page_or_section=idx,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
            )

    def _ingest_streaming(self, path: Path) -> Iterator[DocumentFragment]:
        """Use ``ijson`` for SAX-style streaming of top-level array items."""
        try:
            import ijson
        except ImportError:
            logger.warning(
                "ijson not installed — falling back to full-load for %s "
                "(this may use significant memory for large files)",
                path,
            )
            yield from self._ingest_small(path)
            return

        with open(path, "rb") as fh:
            for idx, item in enumerate(ijson.items(fh, "item")):
                text = _flatten(item) if isinstance(item, (dict, list)) else str(item)
                if not text.strip():
                    continue
                yield DocumentFragment(
                    text=text,
                    metadata=FragmentMetadata(
                        source_path=str(path),
                        page_or_section=idx,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                )
