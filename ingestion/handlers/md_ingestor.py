"""Markdown file ingestor — strips syntax to yield clean plain text."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from dsetgen.config import PipelineConfig
from dsetgen.core.registry import register_ingestor
from dsetgen.ingestion.abstract_ingestor import AbstractIngestor
from dsetgen.ingestion.security import open_with_fallback
from dsetgen.processing.metadata import DocumentFragment, FragmentMetadata

logger = logging.getLogger(__name__)

# Lightweight regex-based Markdown stripping.  For heavier jobs, swap in
# ``mistune`` with a custom renderer that emits only text tokens.
_MD_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"!\[([^\]]*)\]\([^\)]+\)"), r"\1"),       # images
    (re.compile(r"\[([^\]]+)\]\([^\)]+\)"), r"\1"),         # links
    (re.compile(r"#{1,6}\s*"), ""),                          # headings
    (re.compile(r"[*_]{1,3}([^*_]+)[*_]{1,3}"), r"\1"),    # bold/italic
    (re.compile(r"`{1,3}[^`]*`{1,3}"), ""),                 # inline code
    (re.compile(r"^>\s?", re.MULTILINE), ""),                # blockquotes
    (re.compile(r"^[-*+]\s", re.MULTILINE), ""),             # unordered list
    (re.compile(r"^\d+\.\s", re.MULTILINE), ""),             # ordered list
    (re.compile(r"^---+$", re.MULTILINE), ""),               # horizontal rule
]


def _strip_markdown(text: str) -> str:
    for pattern, replacement in _MD_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


@register_ingestor(".md")
class MarkdownIngestor(AbstractIngestor):
    """Read ``.md`` files, strip Markdown syntax, yield clean fragments.

    Sections are split on double-newlines (paragraph boundaries) so that
    each fragment maps roughly to one logical section of the document.
    """

    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".md"})

    def ingest(self, path: Path) -> Iterator[DocumentFragment]:
        fh = open_with_fallback(path, self._config)
        try:
            raw = fh.read()
        finally:
            fh.close()

        clean = _strip_markdown(raw)
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", clean) if p.strip()]

        for idx, para in enumerate(paragraphs):
            yield DocumentFragment(
                text=para,
                metadata=FragmentMetadata(
                    source_path=str(path),
                    page_or_section=idx,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
            )
