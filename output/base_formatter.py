"""Abstract base class for dataset output formatters.

Every formatter must implement two methods:

* ``write_record()`` — append a single training example.
* ``finalize()`` — flush buffers and close handles.

The Strategy Pattern lets the pipeline swap between JSONL, Parquet,
Hugging Face ``datasets``, or any custom format without touching the
controller logic.
"""

from __future__ import annotations

import abc
from typing import Any, Dict

from doc2dataset.config import PipelineConfig


class BaseOutputFormatter(abc.ABC):
    """Contract for all output formatters.

    Parameters
    ----------
    config:
        Pipeline configuration — formatters read ``output_path`` and
        format-specific settings.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    @abc.abstractmethod
    def write_record(self, record: Dict[str, Any]) -> None:
        """Persist a single training example.

        Parameters
        ----------
        record:
            A dict whose schema matches the target format (e.g. Alpaca's
            ``{"instruction": ..., "input": ..., "output": ...}``).
        """

    @abc.abstractmethod
    def finalize(self) -> None:
        """Flush any internal buffers and release file handles.

        Called exactly once by the pipeline controller after all records
        have been written.
        """

    # ── Optional hooks ─────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Called once before the first ``write_record()``.

        Override to open file handles, write headers, etc.
        """

    @property
    def records_written(self) -> int:
        """Return the number of records written so far (for progress logs)."""
        return 0
