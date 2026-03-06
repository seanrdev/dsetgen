"""PDF ingestor — text-layer extraction with OCR fallback.

Strategy
--------
1. Try ``pdfplumber`` to extract text from the native text layer.
2. If a page yields no meaningful text (< 20 chars), assume it is a
   scanned image and fall back to ``pytesseract`` via ``pdf2image``.
3. Each page is emitted as a separate :class:`DocumentFragment` with its
   page number in the metadata — this gives downstream chunkers fine-
   grained control.

OCR Dependencies (optional)
----------------------------
* ``pytesseract`` — Python wrapper for Tesseract OCR.
* ``pdf2image`` — converts PDF pages to PIL images via ``poppler``.

If these are absent the handler will still work for text-layer PDFs; it
will log a warning and skip pages that need OCR.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from doc2dataset.core.registry import register_ingestor
from doc2dataset.exceptions import CorruptedFileError
from doc2dataset.ingestion.abstract_ingestor import AbstractIngestor
from doc2dataset.processing.metadata import DocumentFragment, FragmentMetadata

logger = logging.getLogger(__name__)

# Minimum characters a page must yield before we consider OCR.
_OCR_THRESHOLD = 20


def _try_import_ocr():
    """Lazy-import OCR dependencies, returning ``None`` if missing."""
    try:
        import pytesseract  # noqa: F811
        from pdf2image import convert_from_path  # noqa: F811

        return pytesseract, convert_from_path
    except ImportError:
        return None, None


@register_ingestor(".pdf")
class PdfIngestor(AbstractIngestor):
    """Extract text from ``.pdf`` files, page by page.

    Falls back to Tesseract OCR for pages that appear to be scanned images.
    """

    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".pdf"})

    def ingest(self, path: Path) -> Iterator[DocumentFragment]:
        try:
            import pdfplumber
        except ImportError as exc:
            raise RuntimeError(
                "pdfplumber is required for PDF ingestion — "
                "install it with: pip install pdfplumber"
            ) from exc

        pytesseract, convert_from_path = _try_import_ocr()

        try:
            pdf = pdfplumber.open(str(path))
        except Exception as exc:
            raise CorruptedFileError(
                f"pdfplumber could not open {path}: {exc}"
            ) from exc

        try:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()

                # ── OCR fallback for scanned pages ─────────────────────────
                if len(text) < _OCR_THRESHOLD:
                    if pytesseract is not None and convert_from_path is not None:
                        logger.info(
                            "Page %d of %s has sparse text — running OCR",
                            page_num,
                            path,
                        )
                        try:
                            images = convert_from_path(
                                str(path),
                                first_page=page_num,
                                last_page=page_num,
                                dpi=300,
                            )
                            if images:
                                text = pytesseract.image_to_string(images[0])
                        except Exception:
                            logger.exception(
                                "OCR failed on page %d of %s", page_num, path
                            )
                    else:
                        logger.warning(
                            "Page %d of %s needs OCR but pytesseract/pdf2image "
                            "not installed — skipping page",
                            page_num,
                            path,
                        )
                        continue

                if not text.strip():
                    continue

                yield DocumentFragment(
                    text=text,
                    metadata=FragmentMetadata(
                        source_path=str(path),
                        page_or_section=page_num,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        extra={"handler": "pdfplumber+ocr"},
                    ),
                )
        finally:
            pdf.close()
