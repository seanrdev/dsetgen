"""Token-aware text chunking.

The LLM has a fixed context window.  The chunker's job is to split long
documents into pieces that:

1. Never exceed ``max_context_tokens`` (minus a safety margin reserved for
   the prompt template itself).
2. Overlap by ``chunk_overlap_tokens`` so that semantic continuity is
   maintained across chunk boundaries.
3. Prefer splitting at paragraph or sentence boundaries so that chunks
   are coherent.

Token Counting Strategy
-----------------------
* **Primary**: ``tiktoken`` with the encoding named in
  ``config.tokenizer_name`` (default ``cl100k_base``).  This is fast and
  accurate for GPT-family models; for LLaMA the counts will be a close
  upper-bound approximation — safe for budget enforcement.
* **Fallback**: Whitespace-split word count × 1.3 (heuristic ratio) if
  ``tiktoken`` is not installed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import List, Optional

from dsetgen.config import PipelineConfig
from dsetgen.processing.metadata import DocumentFragment

logger = logging.getLogger(__name__)

# ── Token counter abstraction ──────────────────────────────────────────────

_encoder = None  # Module-level cache
_encoder_resolved = False  # Track whether we've already attempted the import


def _get_encoder(name: str):
    """Lazy-load a tiktoken encoder, falling back to ``None``."""
    global _encoder, _encoder_resolved
    if _encoder_resolved:
        return _encoder
    _encoder_resolved = True
    try:
        import tiktoken

        _encoder = tiktoken.get_encoding(name)
        return _encoder
    except ImportError:
        logger.warning(
            "tiktoken not installed — using heuristic token counting. "
            "Install tiktoken for accurate counts: pip install tiktoken"
        )
        return None


def count_tokens(text: str, tokenizer_name: str = "cl100k_base") -> int:
    """Return the (approximate) token count of *text*."""
    enc = _get_encoder(tokenizer_name)
    if enc is not None:
        return len(enc.encode(text))
    # Heuristic fallback: ~1.3 tokens per whitespace-delimited word.
    return int(len(text.split()) * 1.3)


# ── Sentence splitter ─────────────────────────────────────────────────────

_SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"  # Split after terminal punctuation + space + cap
)


def _split_sentences(text: str) -> List[str]:
    """Best-effort sentence splitting (no heavy NLP dependency)."""
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


# ── Chunker ────────────────────────────────────────────────────────────────

class TokenAwareChunker:
    """Split :class:`DocumentFragment` objects so each fits the LLM window.

    Parameters
    ----------
    config:
        Pipeline config — reads ``max_context_tokens``,
        ``chunk_overlap_tokens``, and ``tokenizer_name``.
    prompt_budget_tokens:
        Tokens reserved for the system/user prompt template that wraps
        each chunk before being sent to the LLM.  Defaults to 512.
    """

    def __init__(
        self,
        config: PipelineConfig,
        prompt_budget_tokens: int = 512,
    ) -> None:
        self._max_tokens = config.max_context_tokens - prompt_budget_tokens
        self._overlap = config.chunk_overlap_tokens
        self._tokenizer = config.tokenizer_name

        if self._max_tokens <= 0:
            raise ValueError(
                f"max_context_tokens ({config.max_context_tokens}) must "
                f"exceed prompt_budget_tokens ({prompt_budget_tokens})"
            )

    def chunk(self, fragment: DocumentFragment) -> List[DocumentFragment]:
        """Split *fragment* into token-bounded chunks.

        If the fragment already fits, it is returned as-is (in a list).
        """
        total = count_tokens(fragment.text, self._tokenizer)
        if total <= self._max_tokens:
            return [replace(fragment, chunk_index=0)]

        # ── Greedy sentence-level packing ──────────────────────────────────
        sentences = _split_sentences(fragment.text)
        if not sentences:
            # No sentence boundaries — fall back to hard word-level split.
            return self._hard_split(fragment)

        chunks: List[DocumentFragment] = []
        current: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            stok = count_tokens(sentence, self._tokenizer)
            if stok > self._max_tokens:
                # Single sentence exceeds budget — hard-split it.
                if current:
                    chunks.append(
                        self._make_chunk(fragment, " ".join(current), len(chunks))
                    )
                    current, current_tokens = [], 0
                for sub in self._hard_split_text(sentence):
                    chunks.append(self._make_chunk(fragment, sub, len(chunks)))
                continue

            if current_tokens + stok > self._max_tokens:
                chunks.append(
                    self._make_chunk(fragment, " ".join(current), len(chunks))
                )
                # Overlap: keep the last few sentences.
                overlap_buf: List[str] = []
                overlap_tok = 0
                for s in reversed(current):
                    st = count_tokens(s, self._tokenizer)
                    if overlap_tok + st > self._overlap:
                        break
                    overlap_buf.insert(0, s)
                    overlap_tok += st
                current = overlap_buf
                current_tokens = overlap_tok

            current.append(sentence)
            current_tokens += stok

        if current:
            chunks.append(
                self._make_chunk(fragment, " ".join(current), len(chunks))
            )

        return chunks

    def chunk_many(
        self, fragments: List[DocumentFragment]
    ) -> List[DocumentFragment]:
        """Convenience: chunk a list of fragments, concatenating results."""
        out: List[DocumentFragment] = []
        for frag in fragments:
            out.extend(self.chunk(frag))
        return out

    # ── Internal helpers ───────────────────────────────────────────────────

    def _hard_split_text(self, text: str) -> List[str]:
        """Word-level split when no sentence boundary exists."""
        words = text.split()
        parts: List[str] = []
        buf: List[str] = []
        tok_count = 0
        for w in words:
            wt = count_tokens(w, self._tokenizer)
            if tok_count + wt > self._max_tokens and buf:
                parts.append(" ".join(buf))
                buf, tok_count = [], 0
            buf.append(w)
            tok_count += wt
        if buf:
            parts.append(" ".join(buf))
        return parts

    def _hard_split(self, frag: DocumentFragment) -> List[DocumentFragment]:
        parts = self._hard_split_text(frag.text)
        return [self._make_chunk(frag, p, i) for i, p in enumerate(parts)]

    @staticmethod
    def _make_chunk(
        parent: DocumentFragment, text: str, idx: int
    ) -> DocumentFragment:
        return DocumentFragment(
            text=text,
            metadata=parent.metadata,
            chunk_index=idx,
        )
