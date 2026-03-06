"""Text normalisation pipeline.

After ingestion, raw text still carries artefacts: stray HTML entities,
Markdown link remnants, excessive whitespace, PDF header/footer noise,
and non-printable control characters.  The :class:`Normalizer` applies a
configurable chain of micro-transforms to produce clean, uniform text
ready for the chunker.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Callable, List

from doc2dataset.processing.metadata import DocumentFragment

# Type alias for a single transform function.
TransformFn = Callable[[str], str]


# ── Individual transforms ──────────────────────────────────────────────────

def strip_html_entities(text: str) -> str:
    """Decode ``&amp;`` → ``&``, ``&lt;`` → ``<``, etc."""
    return html.unescape(text)


def strip_html_tags(text: str) -> str:
    """Remove any residual inline HTML tags (e.g. ``<br/>``, ``<b>``)."""
    return re.sub(r"<[^>]+>", " ", text)


def strip_markdown_links(text: str) -> str:
    """``[label](url)`` → ``label``."""
    return re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)


def strip_urls(text: str) -> str:
    """Remove bare http(s) URLs."""
    return re.sub(r"https?://\S+", "", text)


def collapse_whitespace(text: str) -> str:
    """Replace runs of spaces/tabs with a single space; normalise newlines."""
    text = re.sub(r"[^\S\n]+", " ", text)          # horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)          # 3+ newlines → 2
    return text.strip()


def strip_control_chars(text: str) -> str:
    """Remove non-printable control characters (except newline/tab)."""
    return "".join(
        ch for ch in text
        if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
    )


def strip_pdf_artefacts(text: str) -> str:
    """Heuristic removal of repeated page headers/footers.

    Common patterns: ``Page N of M``, standalone page numbers,
    and short lines repeated verbatim across pages.
    """
    text = re.sub(r"(?i)page\s+\d+\s*(of\s+\d+)?", "", text)
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    return text


# ── Default transform chain ───────────────────────────────────────────────

DEFAULT_TRANSFORMS: List[TransformFn] = [
    strip_html_entities,
    strip_html_tags,
    strip_markdown_links,
    strip_urls,
    strip_control_chars,
    strip_pdf_artefacts,
    collapse_whitespace,
]


class Normalizer:
    """Apply an ordered sequence of text transforms to document fragments.

    Parameters
    ----------
    transforms:
        Custom list of callables ``str → str``.  If ``None``, the module's
        :data:`DEFAULT_TRANSFORMS` chain is used.
    """

    def __init__(self, transforms: List[TransformFn] | None = None) -> None:
        self._transforms = transforms or list(DEFAULT_TRANSFORMS)

    def normalize(self, fragment: DocumentFragment) -> DocumentFragment:
        """Return a *new* fragment with all transforms applied to its text."""
        text = fragment.text
        for fn in self._transforms:
            text = fn(text)
        return DocumentFragment(
            text=text,
            metadata=fragment.metadata,
            chunk_index=fragment.chunk_index,
        )

    def normalize_batch(
        self, fragments: list[DocumentFragment]
    ) -> list[DocumentFragment]:
        """Normalize a list, discarding any that become empty."""
        results: list[DocumentFragment] = []
        for frag in fragments:
            cleaned = self.normalize(frag)
            if not cleaned.is_empty:
                results.append(cleaned)
        return results
