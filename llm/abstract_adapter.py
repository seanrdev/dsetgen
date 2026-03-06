"""Abstract base class for LLM backend adapters.

The Strategy Pattern in action: the pipeline holds an
``AbstractLLMAdapter`` reference and delegates to whatever concrete
backend is configured — Ollama, vLLM, llama.cpp, a cloud API, etc.

Every adapter must implement:

* ``generate()`` — single-shot text completion.
* ``generate_structured()`` — completion that is parsed and validated
  as JSON (for Q&A pair generation, etc.).
* ``health_check()`` — non-destructive liveness probe.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

from dsetgen.config import PipelineConfig


class AbstractLLMAdapter(abc.ABC):
    """Contract for all LLM backend adapters.

    Parameters
    ----------
    config:
        Pipeline-level configuration (URL, model name, timeouts, etc.).
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    # ── Required overrides ─────────────────────────────────────────────────

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send *prompt* to the LLM and return the completion text.

        Implementations must handle:

        * **Timeouts** → raise :class:`~dsetgen.exceptions.LLMTimeoutError`
        * **Connection failures** → raise :class:`~dsetgen.exceptions.LLMConnectionError`
        * **Rate-limit signals** → raise :class:`~dsetgen.exceptions.LLMRateLimitError`

        The pipeline's retry logic sits above this method.
        """

    @abc.abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Like :meth:`generate` but parse the response as JSON.

        Raises
        ------
        LLMMalformedResponseError
            If the LLM output cannot be parsed as valid JSON.
        """

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Return ``True`` if the backend is reachable and ready."""

    # ── Optional lifecycle hooks ───────────────────────────────────────────

    async def startup(self) -> None:
        """Called once when the pipeline starts (e.g. warm up connection pool)."""

    async def shutdown(self) -> None:
        """Called once when the pipeline finishes (e.g. close HTTP client)."""
