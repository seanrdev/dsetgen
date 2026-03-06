"""Hugging Face ``datasets``-compatible output formatter.

Records are accumulated in memory and flushed to a Parquet file (or
Arrow directory) at ``finalize()``.  This makes the output directly
loadable via ``datasets.load_from_disk()`` or ``datasets.Dataset.from_parquet()``.

For very large runs, records are flushed in shards of
``_SHARD_SIZE`` rows to bound peak memory usage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from doc2dataset.config import PipelineConfig
from doc2dataset.core.registry import register_formatter
from doc2dataset.output.base_formatter import BaseOutputFormatter

logger = logging.getLogger(__name__)

_SHARD_SIZE = 50_000  # Flush to disk every N records


@register_formatter("huggingface")
class HuggingFaceFormatter(BaseOutputFormatter):
    """Write training examples as a Parquet dataset loadable by HF ``datasets``.

    Falls back to JSONL if the ``datasets`` / ``pyarrow`` libraries are
    not installed.
    """

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._output_dir = Path(config.output_path).with_suffix("")  # directory
        self._buffer: List[Dict[str, Any]] = []
        self._shard_idx = 0
        self._count = 0

    def initialize(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("HF formatter initialised — output dir %s", self._output_dir)

    def write_record(self, record: Dict[str, Any]) -> None:
        self._buffer.append(record)
        self._count += 1
        if len(self._buffer) >= _SHARD_SIZE:
            self._flush_shard()

    def finalize(self) -> None:
        if self._buffer:
            self._flush_shard()
        logger.info(
            "HF formatter finalised — %d records in %d shards at %s",
            self._count,
            self._shard_idx,
            self._output_dir,
        )

    @property
    def records_written(self) -> int:
        return self._count

    # ── Internal ───────────────────────────────────────────────────────────

    def _flush_shard(self) -> None:
        try:
            from datasets import Dataset

            ds = Dataset.from_list(self._buffer)
            shard_path = self._output_dir / f"shard-{self._shard_idx:05d}"
            ds.save_to_disk(str(shard_path))
            logger.info("Wrote shard %d (%d rows)", self._shard_idx, len(self._buffer))
        except ImportError:
            # Fallback: dump as JSONL shard if HF datasets not available.
            shard_path = self._output_dir / f"shard-{self._shard_idx:05d}.jsonl"
            with open(shard_path, "w", encoding="utf-8") as fh:
                for rec in self._buffer:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.warning(
                "datasets library not installed — wrote JSONL fallback at %s",
                shard_path,
            )

        self._buffer.clear()
        self._shard_idx += 1
