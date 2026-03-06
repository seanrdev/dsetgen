"""Concrete file-type handlers.

Importing this package triggers registration for every built-in handler
so that :data:`~doc2dataset.core.registry.ingestor_registry` is populated
before the pipeline starts.
"""

from doc2dataset.ingestion.handlers.txt_ingestor import TxtIngestor  # noqa: F401
from doc2dataset.ingestion.handlers.md_ingestor import MarkdownIngestor  # noqa: F401
from doc2dataset.ingestion.handlers.pdf_ingestor import PdfIngestor  # noqa: F401
from doc2dataset.ingestion.handlers.docx_ingestor import DocxIngestor  # noqa: F401
from doc2dataset.ingestion.handlers.csv_ingestor import CsvIngestor  # noqa: F401
from doc2dataset.ingestion.handlers.json_ingestor import JsonIngestor  # noqa: F401
from doc2dataset.ingestion.handlers.html_ingestor import HtmlIngestor  # noqa: F401
