"""HTML ingestor — strip tags and boilerplate to yield clean text."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from doc2dataset.core.registry import register_ingestor
from doc2dataset.ingestion.abstract_ingestor import AbstractIngestor
from doc2dataset.ingestion.security import open_with_fallback
from doc2dataset.processing.metadata import DocumentFragment, FragmentMetadata

logger = logging.getLogger(__name__)

# Tags whose content is never useful as training text.
_STRIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript"}


@register_ingestor(".html")
@register_ingestor(".htm")
class HtmlIngestor(AbstractIngestor):
    """Parse ``.html`` / ``.htm`` files and yield clean text fragments.

    Uses BeautifulSoup with the ``lxml`` parser (falls back to
    ``html.parser`` from stdlib if ``lxml`` is unavailable).  Boilerplate
    elements (nav, footer, scripts, styles) are stripped before extraction.
    """

    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".html", ".htm"})

    def ingest(self, path: Path) -> Iterator[DocumentFragment]:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise RuntimeError(
                "beautifulsoup4 is required — pip install beautifulsoup4"
            ) from exc

        fh = open_with_fallback(path, self._config)
        try:
            raw_html = fh.read()
        finally:
            fh.close()

        # Choose the fastest available parser.
        try:
            soup = BeautifulSoup(raw_html, "lxml")
        except Exception:
            soup = BeautifulSoup(raw_html, "html.parser")

        # Remove boilerplate tags entirely.
        for tag in soup.find_all(_STRIP_TAGS):
            tag.decompose()

        # Extract text, collapse whitespace.
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text)  # compress blank-line runs

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for idx, para in enumerate(paragraphs):
            yield DocumentFragment(
                text=para,
                metadata=FragmentMetadata(
                    source_path=str(path),
                    page_or_section=idx,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    extra={"title": (soup.title.string if soup.title else "")},
                ),
            )
