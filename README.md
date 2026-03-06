# dsetgen — Document-to-Dataset Framework

## Architecture

```
dsetgen/
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
from dsetgen.config import PipelineConfig
from dsetgen.core.pipeline_controller import PipelineController
from dsetgen.processing.metadata import DocumentFragment
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


config = PipelineConfig(
    input_dir="/home/user/books/",
    output_path="/home/user/dataset/train.jsonl",
    llm_backend="openai",
    ollama_base_url="http://127.0.0.1:8080/v1",
    model_name="Some_Random_Model_name",
    max_context_tokens=4096,
    checkpoint_db="/home/user/checkpoint/state.db",
)


SYSTEM_PROMPT = """You are an expert dataset generator for LLM fine-tuning.
When given a text excerpt, produce one high-quality training example.
Return ONLY valid JSON with keys: "instruction", "input", "output".
No markdown fences, no explanation, no preamble."""


def my_prompt_builder(fragment: DocumentFragment) -> str:
    """Builds the user message for each chunk."""
    return (
        f"Generate one instruction-tuning example from this text:\n\n"
        f"---\n{fragment.text}\n---"
    )

pipeline = PipelineController(
    config,
    system_prompt=SYSTEM_PROMPT,
    prompt_builder=my_prompt_builder,
)
asyncio.run(pipeline.run())          # Resumes from last checkpoint automatically
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
