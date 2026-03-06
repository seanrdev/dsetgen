# Doc2Dataset — Document-to-Dataset Framework

## Architecture

```
doc2dataset/
├── __init__.py
├── config.py                          # Central configuration (dataclasses + env)
├── exceptions.py                      # Domain-specific exception hierarchy
│
├── core/
│   ├── __init__.py
│   ├── pipeline_controller.py         # Main orchestrator: Ingest → Process → LLM → Output
│   └── registry.py                    # Auto-discovery registry for ingestors/adapters
│
├── ingestion/
│   ├── __init__.py
│   ├── abstract_ingestor.py           # ABC for all file handlers
│   ├── security.py                    # Path sanitation, size guards, encoding fallback
│   └── handlers/
│       ├── __init__.py
│       ├── txt_ingestor.py            # .txt  (streaming, encoding fallback)
│       ├── md_ingestor.py             # .md   (strip markdown → plain text)
│       ├── pdf_ingestor.py            # .pdf  (text-layer + OCR fallback)
│       ├── docx_ingestor.py           # .docx (python-docx)
│       ├── csv_ingestor.py            # .csv  (streaming row-by-row)
│       ├── json_ingestor.py           # .json (ijson streaming parser)
│       └── html_ingestor.py           # .html (BeautifulSoup → plain text)
│
├── processing/
│   ├── __init__.py
│   ├── normalizer.py                  # Strip boilerplate, collapse whitespace
│   ├── chunker.py                     # Token-aware chunking (tiktoken / fallback)
│   └── metadata.py                    # Provenance dataclass (source, page, ts)
│
├── llm/
│   ├── __init__.py
│   ├── abstract_adapter.py            # ABC for any LLM backend
│   └── ollama_adapter.py              # Concrete: async HTTP to local Ollama
│
├── output/
│   ├── __init__.py
│   ├── base_formatter.py              # ABC for output formatters
│   ├── jsonl_formatter.py             # Alpaca / ShareGPT JSONL writer
│   └── huggingface_formatter.py       # HF datasets-compatible Arrow/Parquet
│
├── state/
│   ├── __init__.py
│   └── checkpoint.py                  # SQLite-backed checkpoint / resume logic
│
├── utils/
│   ├── __init__.py
│   └── logging.py                     # Structured logging setup
│
└── tests/
    ├── __init__.py
    └── ...                            # pytest test modules (not in scope)
```

## Quick Start

```python
from doc2dataset.config import PipelineConfig
from doc2dataset.core.pipeline_controller import PipelineController

config = PipelineConfig(
    input_dir="/data/raw_documents",
    output_path="/data/output/train.jsonl",
    ollama_base_url="http://localhost:11434",
    model_name="llama3.1:70b",
    max_context_tokens=4096,
    checkpoint_db="state.db",
)

pipeline = PipelineController(config)
await pipeline.run()          # Resumes from last checkpoint automatically
```

## Recommended Libraries

| Purpose | Library | Rationale |
|---|---|---|
| PDF text extraction | `pdfplumber` | Reliable text-layer extraction with bbox metadata |
| PDF OCR fallback | `pytesseract` + `pdf2image` | Industry-standard OCR when text layers are absent |
| DOCX parsing | `python-docx` | Mature, read-only access to .docx internals |
| HTML stripping | `beautifulsoup4` + `lxml` | Fast, forgiving HTML parser |
| Markdown stripping | `mistune` or regex | Lightweight MD → plain text |
| Token counting | `tiktoken` | OpenAI's fast BPE tokenizer; works offline for chunking |
| Streaming JSON | `ijson` | SAX-style JSON parser for multi-GB files |
| Async HTTP | `httpx` | Modern async client with timeout/retry built in |
| Checkpointing | `sqlite3` (stdlib) | Zero-dependency, ACID-compliant state store |
| HF output | `datasets` + `pyarrow` | Native Hugging Face ecosystem compatibility |
| Logging | `structlog` | Structured JSON logs for production observability |


Note, the base of this project is not written by me. The base/start of this project is created by AI. If you use this, please review. 
