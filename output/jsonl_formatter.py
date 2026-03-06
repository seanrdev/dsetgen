"""JSONL output formatter supporting Alpaca and ShareGPT schemas.

Each line is a self-contained JSON object.  The file is opened in
append mode so that interrupted runs can resume without overwriting
existing records.

Schemas
-------
* **Alpaca**: ``{"instruction": str, "input": str, "output": str}``
* **ShareGPT**: ``{"conversations": [{"from": "human"|"gpt", "value": str}, ...]}``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from dsetgen.config import PipelineConfig
from dsetgen.core.registry import register_formatter
from dsetgen.output.base_formatter import BaseOutputFormatter

logger = logging.getLogger(__name__)


@register_formatter("jsonl")
class JsonlFormatter(BaseOutputFormatter):
    """Append-mode JSONL writer.

    Supports ``alpaca`` and ``sharegpt`` schemas (selected via
    ``config.jsonl_schema``).
    """

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._path = Path(config.output_path)
        self._schema = config.jsonl_schema
        self._fh = None
        self._count = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "a", encoding="utf-8")  # noqa: SIM115
        logger.info("JSONL writer opened %s (schema=%s)", self._path, self._schema)

    def finalize(self) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None
        logger.info("JSONL writer finalised — %d records in %s", self._count, self._path)

    # ── Writing ────────────────────────────────────────────────────────────

    def write_record(self, record: Dict[str, Any]) -> None:
        if self._fh is None:
            self.initialize()

        # Validate minimal schema presence.
        if self._schema == "alpaca":
            assert "instruction" in record, (
                "Alpaca schema requires 'instruction' key"
            )
        elif self._schema == "sharegpt":
            assert "conversations" in record, (
                "ShareGPT schema requires 'conversations' key"
            )

        line = json.dumps(record, ensure_ascii=False)
        self._fh.write(line + "\n")
        self._count += 1

        # Periodic flush to avoid data loss on crash.
        if self._count % 100 == 0:
            self._fh.flush()

    @property
    def records_written(self) -> int:
        return self._count
