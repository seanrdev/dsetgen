"""Provenance metadata carried alongside every text fragment.

Every piece of text that flows through the pipeline is wrapped in a
:class:`DocumentFragment`.  This ensures that the final training dataset
can trace each example back to a specific source file, page number, and
timestamp — critical for data auditing, deduplication, and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class FragmentMetadata:
    """Lightweight provenance record for a single text fragment.

    Attributes
    ----------
    source_path:
        Absolute path (or URI) of the originating file.
    page_or_section:
        Page number (for PDFs/DOCX), row index (CSV), array index (JSON),
        or a sequential section counter (TXT/MD/HTML).
    timestamp:
        ISO-8601 UTC string recording when the fragment was extracted.
    extra:
        Free-form bag for handler-specific metadata (e.g. ``bbox``
        coordinates from ``pdfplumber``, CSV column headers, etc.).
    """

    source_path: str = ""
    page_or_section: int | str = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict suitable for JSON output."""
        return {
            "source_path": self.source_path,
            "page_or_section": self.page_or_section,
            "timestamp": self.timestamp,
            **self.extra,
        }


@dataclass
class DocumentFragment:
    """A unit of text plus its provenance.

    This is the **lingua franca** of the pipeline — every layer consumes
    and produces ``DocumentFragment`` instances.
    """

    text: str
    metadata: FragmentMetadata = field(default_factory=FragmentMetadata)
    chunk_index: Optional[int] = None  # Set by the chunker

    @property
    def is_empty(self) -> bool:
        """True if the text is blank after stripping whitespace."""
        return not self.text or not self.text.strip()
