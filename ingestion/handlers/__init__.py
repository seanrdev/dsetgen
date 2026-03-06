"""Concrete file-type handlers.

Importing this package triggers registration for every built-in handler
so that :data:`~dsetgen.core.registry.ingestor_registry` is populated
before the pipeline starts.
"""

from dsetgen.ingestion.handlers.txt_ingestor import TxtIngestor  # noqa: F401
from dsetgen.ingestion.handlers.md_ingestor import MarkdownIngestor  # noqa: F401
from dsetgen.ingestion.handlers.pdf_ingestor import PdfIngestor  # noqa: F401
from dsetgen.ingestion.handlers.docx_ingestor import DocxIngestor  # noqa: F401
from dsetgen.ingestion.handlers.csv_ingestor import CsvIngestor  # noqa: F401
from dsetgen.ingestion.handlers.json_ingestor import JsonIngestor  # noqa: F401
from dsetgen.ingestion.handlers.html_ingestor import HtmlIngestor  # noqa: F401
